import hashlib
import os
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime
from enum import StrEnum
from pathlib import Path

import lightning as pl
import modal
import torch
import typer
from lightning.pytorch.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    ModelSummary,
    TQDMProgressBar,
)
from lightning.pytorch.loggers import TensorBoardLogger
from torch import nn, optim
from torch.nn.functional import relu
from torch.utils.data import DataLoader, Dataset
from torchmetrics.functional.classification import binary_auroc
from tqdm import tqdm

from files import (
    DATA_JSONL,
    EVAL_JSONL,
    FLAGS_JSONL,
    KEYWORDS_JSONL,
    LABELS_JSON,
    MODEL_DIR,
    PT_DIR,
    REPORT_DIR,
    RUNS_DIR,
    TERMS_JSONL,
    TEXT_ENC_CACHE_PT,
    TOOLS_DIR,
    TRAIN_JSONL,
    WEB_PUBLIC_DIR,
    PtFile,
)
from model.color import COLOR_SHIFT, energy_distance, rgb_to_oklab
from model.config import (
    COND_CRITIC_MISMATCH_WEIGHT,
    COND_CRITIC_RANDOM_WEIGHT,
    CONFIG_NAME,
    EARLY_STOP_MIN_DELTA_GAN,
    EARLY_STOP_PATIENCE_ENCODER,
    EARLY_STOP_PATIENCE_GAN,
    ENERGY_TRAIN_SAMPLE_SIZE,
    ENERGY_VAL_SAMPLE_SIZE,
    EPOCHS_COND_PROBE,
    EPOCHS_GAN,
    EPOCHS_TASK,
    GAN_BATCH_SIZE,
    GAN_LOSS_CRITIC,
    GAN_LOSS_ENERGY,
    GRAD_CLIP_CRITIC,
    GRAD_CLIP_GEN,
    INFONCE_TEMP_EMOJI,
    INFONCE_TEMP_STYLE,
    LOSS_WEIGHT_COLOR,
    LOSS_WEIGHT_EMOJI,
    LOSS_WEIGHT_STYLE,
    LR_ENCODER,
    LR_GAN_CRITIC,
    LR_GAN_GEN,
    SAMPLING_RATE_MAX,
    SAMPLING_RATE_MIN,
    SAMPLING_SOURCES,
    SEED,
    TASK_BATCH_SIZE,
    VAL_CHECK_INTERVAL,
)
from model.data import (
    eval_data_loader,
    eval_ds,
    rnd_color_tensor,
    sample_colors_tensor,
    train_data_loader,
    train_ds,
)
from model.export_onnx import export
from model.metric import GanMetric, LogStage, Metric, Source, Split, named_metric
from model.model import (
    ColorGen,
    Critic,
    EmojiHead,
    StyleHead,
    TextEncoder,
)
from model.runmeta import file_sha, load_pt, require_clean_tree, save_pt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


_CUDA = torch.cuda.is_available()
_DETERMINISTIC: bool | str = "warn" if _CUDA else True


def lse_infonce(
    logits: torch.Tensor,
    target: torch.Tensor,
    temp: float,
) -> torch.Tensor:
    has_pos = target.sum(dim=-1) > 0
    if not bool(has_pos.any()):
        return logits.new_zeros(())

    z = logits / temp
    all_lse = torch.logsumexp(z, dim=-1)
    pos_lse = torch.logsumexp(z.masked_fill(target == 0, float("-inf")), dim=-1)
    row_loss = all_lse - pos_lse

    return row_loss[has_pos].mean()


def _dequantize(colors: torch.Tensor) -> torch.Tensor:
    return (colors + (torch.rand_like(colors) - 0.5)).clamp(
        -COLOR_SHIFT, COLOR_SHIFT)


def _critic_probe_step(
    critic: Critic, cond: torch.Tensor, colors: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    n = colors.shape[0]
    shuffled = colors[torch.randperm(n, device=colors.device)]
    random_colors = rnd_color_tensor(n, device=colors.device)

    pair = _dequantize(torch.cat([colors, shuffled, random_colors], dim=0))
    cond_pair = torch.cat([cond, cond, cond], dim=0)
    score = critic(cond_pair, pair)
    real, shuf_score, rand_score = score.chunk(3, dim=0)

    loss = relu(1 - real).mean() \
        + relu(1 + shuf_score).mean() \
        + relu(1 + rand_score).mean()

    target = torch.cat([score.new_ones(n), score.new_zeros(n)])
    auroc_shuf = binary_auroc(
        torch.cat([real, shuf_score], dim=0).detach().squeeze(-1), target.long())
    auroc_rand = binary_auroc(
        torch.cat([real, rand_score], dim=0).detach().squeeze(-1), target.long())

    return loss, auroc_shuf, auroc_rand


def mrr(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    order = logits.argsort(dim=-1, descending=True)
    rel = target.gather(1, order)
    ranks = torch.arange(1, logits.size(-1) + 1, device=logits.device)
    return (rel / ranks).amax(dim=-1)


def acc_at_k(logits: torch.Tensor, target: torch.Tensor, k: int) -> torch.Tensor:
    order = logits.argsort(dim=-1, descending=True)
    rel = target.gather(1, order)
    return rel[:, :k].amax(dim=-1)


def _energy_subsample(x: torch.Tensor, n: int) -> torch.Tensor:
    if x.shape[0] <= n:
        return x
    return x[:n]


_DEFAULT_PT = PT_DIR


class Stage(StrEnum):
    gan = "gan"
    encoder = "encoder"
    cond = "cond"


class LitEncoder(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.save_hyperparameters()

        self.enc = TextEncoder()
        self.style = StyleHead()
        self.emoji = EmojiHead()
        self.critic = Critic()

        self.train_dataset = None

    def _log(self, name: str, val: torch.Tensor, bs: int) -> None:
        self.log(name, val, on_step=False, on_epoch=True,
                 prog_bar=True, batch_size=bs)

    def _step(self, batch, split: Split):
        text, emoji, style, source, colors, has_color = batch
        enc = self.enc(text)

        style_logits = self.style(enc)
        loss_style = lse_infonce(style_logits, style, INFONCE_TEMP_STYLE)
        self._log(
            named_metric(LogStage.ENC, Source.STYLE, Metric.LOSS, split),
            loss_style,
            style.size(0),
        )
        loss_style = LOSS_WEIGHT_STYLE * loss_style
        self._log(
            named_metric(LogStage.ENC, Source.STYLE, Metric.MRR, split),
            mrr(style_logits, style).mean(),
            style.size(0),
        )

        emoji_logits = self.emoji(enc)
        loss_emoji = lse_infonce(emoji_logits, emoji, INFONCE_TEMP_EMOJI)
        self._log(
            named_metric(LogStage.ENC, Source.EMOJI, Metric.LOSS, split),
            loss_emoji,
            emoji.size(0),
        )
        loss_emoji = LOSS_WEIGHT_EMOJI * loss_emoji
        has_e = emoji.sum(dim=-1) > 0
        n_e = int(has_e.sum())
        if n_e:
            rr = mrr(emoji_logits[has_e], emoji[has_e])
            self._log(
                named_metric(LogStage.ENC, Source.EMOJI, Metric.MRR, split),
                rr.mean(), n_e)

        if split == Split.VAL and n_e:
            self._log(
                named_metric(LogStage.ENC, Source.FULL_TEXT, Metric.ACC_1),
                acc_at_k(emoji_logits[has_e], emoji[has_e], 1).mean(),
                n_e,
            )

        if split == Split.TRAIN:
            for name, cfg in SAMPLING_SOURCES.items():
                if cfg.metric != named_metric(LogStage.ENC, name, Metric.ACC_1):
                    continue
                mask = torch.tensor(
                    [s == name for s in source], device=emoji.device
                )
                n = int(mask.sum())
                if n:
                    self._log(
                        named_metric(LogStage.ENC, name, Metric.ACC_1),
                        acc_at_k(emoji_logits[mask], emoji[mask], 1).mean(),
                        n,
                    )

        loss_critic = enc.new_zeros(())
        n_c = int(has_color.sum())
        if n_c > 1:
            cond_c = enc[has_color]
            colors_c = colors[has_color]
            loss_critic, auroc_shuf, auroc_rand = _critic_probe_step(
                self.critic, cond_c, colors_c)
            self._log(
                named_metric(LogStage.ENC, Source.COLOR, Metric.AUROC_SHUF, split),
                auroc_shuf, n_c)
            self._log(
                named_metric(LogStage.ENC, Source.COLOR, Metric.AUROC_RAND, split),
                auroc_rand, n_c)

        return (
            loss_style
            + loss_emoji
            + LOSS_WEIGHT_COLOR * loss_critic
        )

    def on_train_epoch_start(self):
        if self.train_dataset is None or self.train_dataset.rates is None:
            return
        metrics = self.trainer.callback_metrics
        for name, cfg in SAMPLING_SOURCES.items():
            key = cfg.metric
            if key in metrics:
                val = float(metrics[key])
                frac = (val - cfg.from_) / (cfg.to - cfg.from_)
                rate = SAMPLING_RATE_MAX - (
                    SAMPLING_RATE_MAX - SAMPLING_RATE_MIN) * frac
                rate = min(SAMPLING_RATE_MAX, max(SAMPLING_RATE_MIN, rate))
                self.train_dataset.rates.set(name, rate)
            self.log(
                named_metric(LogStage.ENC, name, Metric.RATE),
                self.train_dataset.rates.get(name),
            )

    def training_step(self, batch, batch_idx):
        return self._step(batch, Split.TRAIN)

    def validation_step(self, batch, batch_idx):
        self._step(batch, Split.VAL)

    def configure_optimizers(self):
        params = (
            list(self.enc.parameters())
            + list(self.style.parameters())
            + list(self.emoji.parameters())
            + list(self.critic.parameters())
        )
        return optim.Adam(params, lr=LR_ENCODER)


class LitColorGAN(pl.LightningModule):
    def __init__(self, critic: Critic):
        super().__init__()

        self.gen = ColorGen()
        self.critic = critic

        self.automatic_optimization = False
        self._val_cond: list[torch.Tensor] = []
        self._val_real: list[torch.Tensor] = []

    def on_train_epoch_start(self):
        self.gen.net[0].eval()

    def on_validation_epoch_start(self):
        self._val_cond.clear()
        self._val_real.clear()

    def validation_step(self, batch, batch_idx):
        cond, colors = batch
        self._val_cond.append(cond)
        self._val_real.append(colors)

    def on_validation_epoch_end(self):
        if not self._val_real:
            return

        self.gen.eval()
        with torch.no_grad():
            cond = torch.cat(self._val_cond)
            colors = torch.cat(self._val_real)

            fake = self.gen(cond)

            val_energy = energy_distance(
                rgb_to_oklab(_energy_subsample(colors, ENERGY_VAL_SAMPLE_SIZE)),
                rgb_to_oklab(_energy_subsample(fake, ENERGY_VAL_SAMPLE_SIZE)))
            self.log(GanMetric.ENERGY_VAL, val_energy, prog_bar=True)

    def training_step(self, batch, batch_idx):
        cond, colors = batch
        opt_gen, opt_critic = self.optimizers()  # type: ignore

        fake = self.gen(cond)

        # CRITIC
        random_colors = rnd_color_tensor(colors.shape[0], device=colors.device)
        cond_wrong = cond[torch.randperm(cond.shape[0], device=cond.device)]

        real = self.critic(cond, _dequantize(colors))
        fake_score = self.critic(cond, fake.detach())
        wrong_score = self.critic(cond_wrong, _dequantize(colors))
        random_score = self.critic(cond, _dequantize(random_colors))

        loss_critic = relu(1 - real).mean() \
            + COND_CRITIC_MISMATCH_WEIGHT * relu(1 + fake_score).mean() \
            + (1 - COND_CRITIC_MISMATCH_WEIGHT) * relu(1 + wrong_score).mean() \
            + COND_CRITIC_RANDOM_WEIGHT * relu(1 + random_score).mean()

        n = colors.shape[0]
        auroc_target = torch.cat([real.new_ones(n), real.new_zeros(n)])
        auroc_gen = binary_auroc(
            torch.cat([real, fake_score], dim=0).detach().squeeze(-1),
            auroc_target.long())
        auroc_shuf = binary_auroc(
            torch.cat([real, wrong_score], dim=0).detach().squeeze(-1),
            auroc_target.long())
        auroc_rand = binary_auroc(
            torch.cat([real, random_score], dim=0).detach().squeeze(-1),
            auroc_target.long())

        opt_critic.zero_grad()
        self.manual_backward(loss_critic)
        self.clip_gradients(
            opt_critic,  # type: ignore
            gradient_clip_val=GRAD_CLIP_CRITIC,
            gradient_clip_algorithm="norm")

        opt_critic.step()

        # GENERATOR
        gen_score = self.critic(cond, fake)
        loss_energy = energy_distance(
            rgb_to_oklab(_energy_subsample(fake, ENERGY_TRAIN_SAMPLE_SIZE)),
            rgb_to_oklab(_energy_subsample(colors, ENERGY_TRAIN_SAMPLE_SIZE)))

        loss_gen_critic = -gen_score.mean()

        loss_gen = \
            GAN_LOSS_CRITIC * loss_gen_critic \
            + GAN_LOSS_ENERGY * loss_energy

        opt_gen.zero_grad()

        self.manual_backward(loss_gen)
        self.clip_gradients(
            opt_gen,  # type: ignore
            gradient_clip_val=GRAD_CLIP_GEN,
            gradient_clip_algorithm="norm")

        opt_gen.step()

        self.log(GanMetric.COND_LOSS, loss_critic, prog_bar=True)
        self.log(GanMetric.COND_AUROC_GEN, auroc_gen, prog_bar=True)
        self.log(GanMetric.COND_AUROC_SHUF, auroc_shuf, prog_bar=True)
        self.log(GanMetric.COND_AUROC_RAND, auroc_rand, prog_bar=True)
        self.log(
            GanMetric.COND_MEAN_SCORE_REAL,
            real.detach().mean(), prog_bar=False)
        self.log(
            GanMetric.COND_MEAN_SCORE_FAKE,
            fake_score.detach().mean(), prog_bar=False)
        self.log(
            GanMetric.COND_MEAN_SCORE_WRONG,
            wrong_score.detach().mean(), prog_bar=False)
        self.log(
            GanMetric.COND_MEAN_SCORE_RANDOM,
            random_score.detach().mean(), prog_bar=False)
        self.log(
            GanMetric.GEN_LOSS_COND,
            loss_gen_critic, prog_bar=True)
        self.log(GanMetric.ENERGY_TRAIN, loss_energy, prog_bar=True)

    def configure_optimizers(self):
        opt_gen = optim.SGD(self.gen.parameters(), lr=LR_GAN_GEN)
        opt_critic = optim.SGD(self.critic.parameters(), lr=LR_GAN_CRITIC)

        # opt_gen = optim.Adam(
        #     self.gen.parameters(),
        #     lr=LR_GAN_GEN,
        #     betas=(0.5, 0.999))

        # opt_critic = optim.Adam(
        #     self.critic.parameters(),
        #     lr=LR_GAN_COND_CRITIC,
        #     betas=(0.5, 0.999))

        return [opt_gen, opt_critic]


class LitCondCriticProbe(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.critic = Critic()

    def _step(self, batch: tuple[torch.Tensor, torch.Tensor]):
        cond, colors = batch
        return _critic_probe_step(self.critic, cond, colors)

    def training_step(self, batch, batch_idx):
        loss, auroc_shuf, auroc_rand = self._step(batch)
        self.log(
            named_metric(LogStage.COND, Source.COLOR,
                         Metric.AUROC_SHUF, Split.TRAIN),
            auroc_shuf, on_step=False, on_epoch=True, prog_bar=True,
        )
        self.log(
            named_metric(LogStage.COND, Source.COLOR,
                         Metric.AUROC_RAND, Split.TRAIN),
            auroc_rand, on_step=False, on_epoch=True, prog_bar=True,
        )
        return loss

    def validation_step(self, batch, batch_idx):
        _, auroc_shuf, auroc_rand = self._step(batch)
        self.log(
            named_metric(LogStage.COND, Source.COLOR, Metric.AUROC_SHUF, Split.VAL),
            auroc_shuf, on_step=False, on_epoch=True, prog_bar=True,
        )
        self.log(
            named_metric(LogStage.COND, Source.COLOR, Metric.AUROC_RAND, Split.VAL),
            auroc_rand, on_step=False, on_epoch=True, prog_bar=True,
        )

    def configure_optimizers(self):
        return optim.SGD(self.critic.parameters(), lr=LR_GAN_CRITIC)


def _load(mod: nn.Module, path: Path) -> nn.Module:
    sd, meta = load_pt(path)
    mod.load_state_dict(sd)
    mod._pt_meta = meta  # type: ignore
    return mod


def _no_progress_bar() -> bool:
    return os.environ.get("EMOJIC_NO_PROGRESS_BAR") == "1"


def _load_critic(pt_dir: Path) -> Critic:
    critic_path = PtFile.CRITIC.in_dir(pt_dir)
    critic = Critic()
    if not critic_path.exists():
        return critic
    try:
        return _load(critic, critic_path)  # type: ignore
    except Exception:
        print(
            f"{critic_path}: does not match Critic, "
            "falling back to a fresh critic",
            flush=True,
        )
        return Critic()


def _pt_files_ok(pt_dir: Path) -> bool:
    checks: list[tuple[PtFile, nn.Module]] = [
        (PtFile.ENC, TextEncoder()),
        (PtFile.STYLE, StyleHead()),
        (PtFile.EMOJI, EmojiHead()),
    ]
    for name, mod in checks:
        path = name.in_dir(pt_dir)
        if not path.exists():
            return False
        try:
            _load(mod, path)
        except Exception:
            return False
    return True


def _train_encoder(ds, out_dir: Path) -> LitEncoder:
    dl = train_data_loader(data_set=ds, batch_size=TASK_BATCH_SIZE)
    val_dl = eval_data_loader()

    no_bar = _no_progress_bar()
    bar_cbs = [] if no_bar else [TQDMProgressBar()]

    monitor = named_metric(LogStage.ENC, Source.EMOJI, Metric.MRR, Split.VAL)
    ckpt = ModelCheckpoint(
        monitor=monitor, mode="max", save_top_k=1, filename="best-{step}"
    )
    trainer = pl.Trainer(
        devices="auto",
        accelerator="auto",
        logger=TensorBoardLogger(
            "runs", name=CONFIG_NAME, version=LogStage.ENC.value,
            default_hp_metric=False
        ),
        deterministic=_DETERMINISTIC,  # type: ignore
        max_epochs=EPOCHS_TASK,
        val_check_interval=min(VAL_CHECK_INTERVAL, len(dl)),
        enable_progress_bar=not no_bar,
        callbacks=[
            ckpt,
            EarlyStopping(monitor=monitor, mode="max",
                          patience=EARLY_STOP_PATIENCE_ENCODER),
            *bar_cbs,
            ModelSummary(),
        ],
    )

    mod = LitEncoder()
    mod.train_dataset = ds
    trainer.fit(mod, dl, val_dl)

    if ckpt.best_model_path:
        mod = LitEncoder.load_from_checkpoint(ckpt.best_model_path)

    save_pt(mod.enc.state_dict(), PtFile.ENC.in_dir(out_dir), stage="enc")
    save_pt(mod.style.state_dict(), PtFile.STYLE.in_dir(out_dir), stage="enc")
    save_pt(mod.emoji.state_dict(), PtFile.EMOJI.in_dir(out_dir), stage="enc")
    save_pt(
        mod.critic.state_dict(),
        PtFile.CRITIC.in_dir(out_dir), stage="enc")

    return mod


def _train_gan(
    enc: TextEncoder, enc_path: Path, critic: Critic,
    ds, val_ds, out_dir: Path,
) -> LitColorGAN:
    enc.requires_grad_(False)
    train_cond, val_cond = _encoded_texts(enc, enc_path, ds, val_ds)

    gan_dl = DataLoader(
        _CondColorDataset(train_cond, ds.colors),
        batch_size=GAN_BATCH_SIZE,
        shuffle=True,
        drop_last=True,
    )
    val_dl = DataLoader(
        _CondColorDataset(val_cond, val_ds.colors),
        batch_size=2000,
        shuffle=False,
        drop_last=False,
    )
    no_bar = _no_progress_bar()
    bar_cbs = [] if no_bar else [TQDMProgressBar()]

    monitor = GanMetric.ENERGY_VAL
    ckpt = ModelCheckpoint(
        monitor=monitor, mode="min", save_top_k=1,
        filename="best-gan-{step}"
    )
    trainer = pl.Trainer(
        devices="auto",
        accelerator="auto",
        logger=TensorBoardLogger(
            "runs",
            name=CONFIG_NAME,
            version=LogStage.GAN.value,
            default_hp_metric=False),

        deterministic=_DETERMINISTIC,  # type: ignore
        max_epochs=EPOCHS_GAN,
        enable_progress_bar=not no_bar,
        val_check_interval=min(VAL_CHECK_INTERVAL, len(ds)),
        callbacks=[
            ckpt,
            EarlyStopping(
                monitor=monitor,
                mode="min",
                patience=EARLY_STOP_PATIENCE_GAN,
                min_delta=EARLY_STOP_MIN_DELTA_GAN),

            *bar_cbs,
            ModelSummary(),
        ],
    )

    gan = LitColorGAN(critic)
    trainer.fit(gan, gan_dl, val_dl)

    if ckpt.best_model_path:
        gan = LitColorGAN.load_from_checkpoint(
            ckpt.best_model_path, critic=Critic(),
        )

    save_pt(gan.gen.state_dict(), PtFile.GEN.in_dir(out_dir), stage="gan")
    return gan


class _CondColorDataset(Dataset):
    def __init__(self, cond: torch.Tensor, colors: list):
        self.cond = cond
        self.colors = colors

    def __len__(self) -> int:
        return len(self.colors)

    def __getitem__(self, idx: int):
        return self.cond[idx], sample_colors_tensor(self.colors[idx])


def _encode_texts(enc: TextEncoder, text: torch.Tensor) -> torch.Tensor:
    enc.eval()
    device = torch.device(
        "cuda") if torch.cuda.is_available() else torch.device("cpu")
    enc.to(device)
    chunks = []
    starts = range(0, text.shape[0], GAN_BATCH_SIZE)
    with torch.no_grad():
        for i in tqdm(
            starts, desc="encoding text", disable=_no_progress_bar()
        ):
            chunks.append(enc(text[i:i + GAN_BATCH_SIZE].to(device)).cpu())
    return torch.cat(chunks)


def _text_enc_fingerprint(enc_path: Path) -> dict[str, str | None]:
    return {
        "enc_sha256": file_sha(enc_path),
        "train_sha256": file_sha(TRAIN_JSONL),
        "eval_sha256": file_sha(EVAL_JSONL),
    }


def _encoded_texts(
    enc: TextEncoder, enc_path: Path, ds, val_ds
) -> tuple[torch.Tensor, torch.Tensor]:
    fingerprint = _text_enc_fingerprint(enc_path)

    if TEXT_ENC_CACHE_PT.exists():
        cached, meta = load_pt(TEXT_ENC_CACHE_PT)
        if meta and all(meta.get(k) == v for k, v in fingerprint.items()):
            print(f"using cached text encodings: {TEXT_ENC_CACHE_PT}")
            return cached["train"], cached["val"]

    train_cond = _encode_texts(enc, ds.text)
    val_cond = _encode_texts(enc, val_ds.text)
    save_pt(
        {"train": train_cond, "val": val_cond}, TEXT_ENC_CACHE_PT, **fingerprint
    )
    return train_cond, val_cond


def _run_cond() -> None:
    pl.seed_everything(SEED, workers=True)
    torch.backends.cudnn.benchmark = _CUDA
    if _CUDA:
        torch.set_float32_matmul_precision("high")

    require_clean_tree()
    _DEFAULT_PT.mkdir(parents=True, exist_ok=True)

    enc_path = PtFile.ENC.in_dir(_DEFAULT_PT)
    if not enc_path.exists():
        sys.exit(
            f"{enc_path}: missing -- run `train --local` first to produce "
            "a trained encoder"
        )
    enc = _load(TextEncoder(), enc_path)  # type: ignore
    enc.requires_grad_(False)

    ds = train_ds(mix_sources=False)
    val_ds = eval_ds()
    train_cond, val_cond = _encoded_texts(enc, enc_path, ds, val_ds)  # type: ignore

    train_dl = DataLoader(
        _CondColorDataset(train_cond, ds.colors),
        batch_size=GAN_BATCH_SIZE,
        shuffle=True,
        drop_last=True,
    )
    val_dl = DataLoader(
        _CondColorDataset(val_cond, val_ds.colors),
        batch_size=2000,
        shuffle=False,
        drop_last=False,
    )

    no_bar = _no_progress_bar()
    bar_cbs = [] if no_bar else [TQDMProgressBar()]

    trainer = pl.Trainer(
        devices="auto",
        accelerator="auto",
        logger=TensorBoardLogger(
            "runs", name=CONFIG_NAME, version=LogStage.COND.value,
            default_hp_metric=False
        ),
        deterministic=_DETERMINISTIC,  # type: ignore
        max_epochs=EPOCHS_COND_PROBE,
        enable_progress_bar=not no_bar,
        callbacks=[*bar_cbs, ModelSummary()],
    )

    trainer.fit(LitCondCriticProbe(), train_dl, val_dl)


def _run_report_local(pt_dir: Path) -> None:
    subprocess.run(
        [sys.executable, "tools/report.py", "--pt", str(pt_dir)], check=True
    )


def _run_local(stage: Stage | None) -> None:
    pl.seed_everything(SEED, workers=True)
    torch.backends.cudnn.benchmark = _CUDA
    if _CUDA:
        torch.set_float32_matmul_precision("high")

    require_clean_tree()
    skip_report = os.environ.get("EMOJIC_SKIP_REPORT") == "1"
    _DEFAULT_PT.mkdir(parents=True, exist_ok=True)

    if stage == Stage.gan:
        if _pt_files_ok(_DEFAULT_PT):
            enc_path = PtFile.ENC.in_dir(_DEFAULT_PT)
            enc = _load(TextEncoder(), enc_path)
            critic = _load_critic(_DEFAULT_PT)
            _train_gan(
                enc, enc_path, critic,  # type: ignore
                train_ds(mix_sources=False),
                eval_ds(),
                _DEFAULT_PT,
            )
            export()
            if not skip_report:
                _run_report_local(_DEFAULT_PT)
            return
        print(
            f"{_DEFAULT_PT}: missing or corrupted pt files, "
            "falling back to a fresh full training",
            flush=True,
        )
        stage = None

    if stage == Stage.encoder:
        _train_encoder(train_ds(), _DEFAULT_PT)
        return

    ds = train_ds()
    mod = _train_encoder(ds, _DEFAULT_PT)

    critic = _load_critic(_DEFAULT_PT)
    _train_gan(
        mod.enc, PtFile.ENC.in_dir(_DEFAULT_PT), critic,
        train_ds(mix_sources=False),
        eval_ds(),
        _DEFAULT_PT,
    )
    export()
    if not skip_report:
        _run_report_local(_DEFAULT_PT)


CPU = 16
GPU_CPU = 8
DEFAULT_GPU = "T4"
GPU_TASK_BATCH_SIZE = 512
GPU_GAN_BATCH_SIZE = 1024
CPU_MEMORY_MIB = 8192
GPU_MEMORY_MIB = 16384
TIMEOUT_S = 60 * 180
REPO = "/repo"
VENV_PY = sys.executable
TB_PORT = 6006

WORKTREE_TAG = hashlib.sha1(str(Path.cwd().resolve()).encode()).hexdigest()[:10]
VOL_NAME = f"emojic-artifacts-{WORKTREE_TAG}"
ARTIFACTS = "/artifacts"

DEP_FILES = [Path(p) for p in ("pyproject.toml", "uv.lock",
                               ".python-version", "README.md")]
CODE_FILES = [
    Path("files.py"),
    MODEL_DIR / "__init__.py",
    MODEL_DIR / "color.py",
    MODEL_DIR / "config.py",
    MODEL_DIR / "data.py",
    MODEL_DIR / "kwtokens.py",
    MODEL_DIR / "metric.py",
    MODEL_DIR / "metrics.py",
    MODEL_DIR / "model.py",
    MODEL_DIR / "train.py",
    MODEL_DIR / "export_onnx.py",
    MODEL_DIR / "pred.py",
    MODEL_DIR / "runmeta.py",
    TOOLS_DIR / "report.py",
    LABELS_JSON,
    DATA_JSONL,
    TRAIN_JSONL,
    EVAL_JSONL,
    KEYWORDS_JSONL,
    TERMS_JSONL,
    FLAGS_JSONL,
]
COLLECT_TREES = [PT_DIR, RUNS_DIR, WEB_PUBLIC_DIR, REPORT_DIR]

modal_image = modal.Image.debian_slim(python_version="3.13").pip_install("uv")
for _name in DEP_FILES:
    modal_image = modal_image.add_local_file(_name, f"{REPO}/{_name}", copy=True)
for _name in CODE_FILES:
    modal_image = modal_image.add_local_file(_name, f"{REPO}/{_name}", copy=True)
modal_image = modal_image.run_commands(
    f"cd {REPO} && UV_PROJECT_ENVIRONMENT=/usr/local "
    "uv sync --frozen --no-default-groups --group dev --group gpu"
)
modal_image = modal_image.workdir(REPO)

modal_app = modal.App(f"emojic-train-{WORKTREE_TAG}", image=modal_image)
vol = modal.Volume.from_name(VOL_NAME, create_if_missing=True)


def _run_env(threads: int) -> dict[str, str]:
    return {
        **os.environ,
        "PYTHONUNBUFFERED": "1",
        "EMOJIC_NO_PROGRESS_BAR": "1",
        "EMOJIC_SKIP_REPORT": "1",
        "OMP_NUM_THREADS": str(threads),
        "MKL_NUM_THREADS": str(threads),
        "OPENBLAS_NUM_THREADS": str(threads),
        "NUMEXPR_NUM_THREADS": str(threads),
    }


def _stash(dst: Path) -> int:
    root, out = Path(REPO), dst
    n = 0
    for tree in COLLECT_TREES:
        base = root / tree
        if not base.is_dir():
            continue
        for path in base.rglob("*"):
            if path.is_file():
                rel = path.relative_to(root)
                (out / rel).parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(path, out / rel)
                n += 1
    return n


@modal_app.function(
    cpu=CPU,
    memory=CPU_MEMORY_MIB,
    timeout=TIMEOUT_S,
    volumes={ARTIFACTS: vol},
    include_source=False,
)
def train_remote(
    stage: str,
    threads: int,
    git_sha: str,
    run_time: str,
    gpu: str = "",
) -> dict[str, int]:
    env = _run_env(threads)
    env["EMOJIC_GIT_SHA"] = git_sha
    env["EMOJIC_RUN_TIME"] = run_time
    env["EMOJIC_DISPATCH_CHECKED"] = "1"
    if gpu:
        env["EMOJIC_TASK_BATCH_SIZE"] = str(GPU_TASK_BATCH_SIZE)
        env["EMOJIC_GAN_BATCH_SIZE"] = str(GPU_GAN_BATCH_SIZE)
        env["EMOJIC_DATA_WORKERS"] = "4"
        env["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

    code = 1
    try:
        tb = subprocess.Popen(
            [
                VENV_PY,
                "-m",
                "tensorboard.main",
                "--logdir",
                f"{REPO}/runs",
                "--host",
                "0.0.0.0",
                "--port",
                str(TB_PORT),
                "--reload_interval",
                "5",
            ],
            cwd=REPO,
            env=env,
        )
        try:
            with modal.forward(TB_PORT) as tunnel:
                print(f"TensorBoard: {tunnel.url}", flush=True)
                cmd = [VENV_PY, "-m", f"{MODEL_DIR}.train"]
                if stage:
                    cmd.append(stage)
                cmd.append("--local")
                proc = subprocess.Popen(
                    cmd,
                    cwd=REPO,
                    env=env,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                )
                for line in proc.stdout:  # type: ignore
                    sys.stdout.write(line)
                    sys.stdout.flush()
                code = proc.wait()
        finally:
            tb.terminate()
    finally:
        n = _stash(Path(ARTIFACTS))
        vol.commit()
        print(f"stashed {n} files to volume {VOL_NAME}", flush=True)
    if code != 0:
        raise RuntimeError(f"train.py {stage or 'all'} --local exited with {code}")
    return {"files": n}


def _modal(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "modal", *args], capture_output=True, text=True
    )


def _retrieve_and_cleanup() -> bool:
    staging = Path(tempfile.mkdtemp(prefix="emojic-modal-"))
    try:
        got = _modal("volume", "get", "--force", VOL_NAME, "/", str(staging))
        if got.returncode != 0:
            print(f"volume get failed, leaving {VOL_NAME} intact:")
            print(got.stdout, got.stderr)
            return False

        src = staging
        kids = list(staging.iterdir())
        if (
            len(kids) == 1
            and kids[0].is_dir()
            and kids[0].name
            not in {
                str(PT_DIR),
                str(RUNS_DIR),
                str(REPORT_DIR),
                "web",
            }
        ):
            src = kids[0]

        landed: list[str] = []
        for item in sorted(src.iterdir()):
            target = Path.cwd() / item.name
            if item.is_dir():
                shutil.copytree(item, target, dirs_exist_ok=True)
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(item, target)
            landed.append(item.name)

        print(f"retrieved into {Path.cwd()}: {', '.join(landed) or '(nothing)'}")
        if not landed:
            return False

        rm = _modal("volume", "delete", "-y", VOL_NAME)
        print(
            f"deleted volume {VOL_NAME}"
            if rm.returncode == 0
            else f"volume cleanup failed:\n{rm.stderr}"
        )
        return True
    finally:
        shutil.rmtree(staging, ignore_errors=True)


def _run_remote(stage: str, git_sha: str, run_time: str) -> dict[str, int]:
    fn = train_remote.with_options(
        gpu=DEFAULT_GPU, cpu=GPU_CPU, memory=GPU_MEMORY_MIB, timeout=TIMEOUT_S
    )
    return fn.remote(
        stage=stage,
        threads=GPU_CPU,
        git_sha=git_sha,
        run_time=run_time,
        gpu=DEFAULT_GPU,
    )


def _dispatch(stage: Stage | None) -> None:
    require_clean_tree()
    git_sha = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True
    ).stdout.strip()
    run_time = datetime.now().strftime("%Y%m%d-%H%M%S")
    remote_stage = stage or Stage.encoder
    suffix = ", then the GAN locally" if stage is None else ""
    print(
        f"Training {remote_stage.value} on Modal {DEFAULT_GPU} GPU{suffix}...",
        flush=True,
    )
    try:
        with modal.enable_output(), modal_app.run():
            print(_run_remote(remote_stage.value, git_sha, run_time))
    finally:
        landed = _retrieve_and_cleanup()
    if stage is None and landed:
        _run_local(Stage.gan)


_app = typer.Typer(
    context_settings={"help_option_names": ["-h", "--help"]},
)


@_app.command()
def cli(
    stage: Stage | None = typer.Argument(
        None,
        metavar="[gan|encoder|cond]",
        help="gan = train only the color GAN, using a pretrained encoder "
        "(enc.pt, style.pt, emoji.pt in pt/); always runs locally. "
        "encoder = train only the text encoder + heads, on Modal unless "
        "--local. "
        "cond = probe whether the color Critic has capacity to learn the "
        "conditional color distribution, training it on cached text "
        "encodings against shuffled (mismatched) and random-color pairs; "
        "always local. "
        "Omit to train the encoder (Modal unless --local), then the GAN, "
        "which always trains locally.",
    ),
    local: bool = typer.Option(
        False, "--local",
        help="Train the encoder stage on this machine instead of Modal "
        "(the GAN always trains locally).",
    ),
) -> None:
    """Train the emojic model, then export + report. The encoder trains
    on Modal (a T4 GPU) by default and --local runs it here instead; the
    GAN always trains locally. Aborts on a dirty git tree."""
    if stage == Stage.cond:
        _run_cond()
    elif stage == Stage.gan:
        _run_local(Stage.gan)
    elif local:
        _run_local(stage)
    else:
        _dispatch(stage)


if __name__ == "__main__":
    _app()
