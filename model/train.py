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
from torch.nn.functional import (
    cross_entropy,
    relu,
)

from files import (
    DATA_JSONL,
    EMOJI_EMBED_PT,
    EMOJI_PT,
    ENC_PT,
    EVAL_JSONL,
    FLAGS_JSONL,
    KEYWORDS_JSONL,
    LABELS_JSON,
    MODEL_DIR,
    PREVIEW_DIR,
    PT_DIR,
    REPORT_DIR,
    RUNS_DIR,
    STYLE_PT,
    TERMS_JSONL,
    TOOLS_DIR,
    TRAIN_JSONL,
    WEB_PUBLIC_DIR,
    PtFile,
)
from model.color import energy_distance, rgb_to_oklab
from model.config import (
    CONFIG_NAME,
    EARLY_STOP_MIN_DELTA_GAN,
    EARLY_STOP_PATIENCE_ENCODER,
    EARLY_STOP_PATIENCE_GAN,
    EMBED_SIZE_TEXT,
    EPOCHS_GAN,
    EPOCHS_TASK,
    GAN_BATCH_SIZE,
    GRAD_CLIP_CRITIC,
    GRAD_CLIP_GEN,
    INFONCE_TEMP_EMOJI,
    INFONCE_TEMP_STYLE,
    LOSS_WEIGHT_ENERGY,
    LR_ENCODER,
    LR_GAN_CRITIC,
    LR_GAN_GEN,
    MACRO_MIN_SUPPORT,
    SAMPLING_RATE_MAX,
    SAMPLING_RATE_MIN,
    SAMPLING_SOURCES,
    SEED,
    TASK_BATCH_SIZE,
    VAL_CHECK_INTERVAL,
)
from model.data import (
    SRC_FULL,
    eval_data_loader,
    train_data_loader,
    train_ds,
)
from model.export_onnx import export
from model.metric import GanMetric, Metric, Source, Split, named_metric
from model.metrics import macro_average
from model.model import (
    ColorCritic,
    ColorGen,
    EmojiEmbedding,
    EmojiHead,
    LangHead,
    StyleHead,
    TextEncoder,
)
from model.pred import rgb_to_hex
from model.runmeta import load_pt, require_clean_tree, run_meta, save_pt

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


def mrr(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    order = logits.argsort(dim=-1, descending=True)
    rel = target.gather(1, order)
    ranks = torch.arange(1, logits.size(-1) + 1, device=logits.device)
    return (rel / ranks).amax(dim=-1)


def acc_at_k(logits: torch.Tensor, target: torch.Tensor, k: int) -> torch.Tensor:
    order = logits.argsort(dim=-1, descending=True)
    rel = target.gather(1, order)
    return rel[:, :k].amax(dim=-1)


ALL_HEADS: tuple[str, ...] = ("style", "emoji", "lang")
_DEFAULT_PT = PT_DIR


class Stage(StrEnum):
    gan = "gan"
    energy = "energy"


class LitEncoder(pl.LightningModule):
    def __init__(self, heads: tuple[str, ...] = ALL_HEADS):
        super().__init__()
        self.save_hyperparameters()
        self.heads = tuple(heads)

        self.enc = TextEncoder()
        if "style" in self.heads:
            self.style = StyleHead()
        if "emoji" in self.heads:
            self.emoji_embed = EmojiEmbedding()
            self.emoji = EmojiHead()
        if "lang" in self.heads:
            self.lang = LangHead()

        self._val_e_rr: list[torch.Tensor] = []
        self._trn_e_rr: list[torch.Tensor] = []

        self._val_s_rr: list[torch.Tensor] = []
        self._val_s_tgt: list[torch.Tensor] = []
        self._trn_s_rr: list[torch.Tensor] = []
        self._trn_s_tgt: list[torch.Tensor] = []

        self.train_dataset = None

    def _log(self, name: str, val: torch.Tensor, bs: int) -> None:
        self.log(name, val, on_step=False, on_epoch=True,
                 prog_bar=True, batch_size=bs)

    def _step(self, batch, split: Split):
        text, emoji, style, _colors, lang, source = batch
        enc = self.enc(text)
        loss = enc.new_zeros(())

        if "style" in self.heads:
            style_logits = self.style(enc)
            loss_style = lse_infonce(style_logits, style, INFONCE_TEMP_STYLE)
            loss = loss + loss_style
            s_rr, s_tgt = (
                (self._val_s_rr, self._val_s_tgt)
                if split == Split.VAL
                else (self._trn_s_rr, self._trn_s_tgt)
            )
            s_rr.append(mrr(style_logits, style).detach())
            s_tgt.append(style.detach())

        if "emoji" in self.heads:
            q_txt = self.emoji(enc)
            emoji_logits = self.emoji_embed.score(q_txt)
            loss_emoji = lse_infonce(emoji_logits, emoji, INFONCE_TEMP_EMOJI)
            loss = loss + loss_emoji
            has_e = emoji.sum(dim=-1) > 0
            n_e = int(has_e.sum())
            if n_e:
                rr = mrr(emoji_logits[has_e], emoji[has_e])
                e_rr = self._val_e_rr if split == Split.VAL else self._trn_e_rr
                e_rr.append(rr.detach())

            if split == Split.VAL:
                for name, cfg in SAMPLING_SOURCES.items():
                    if cfg.metric != named_metric(name, Metric.ACC_1):
                        continue
                    mask = torch.tensor(
                        [s == name for s in source], device=emoji.device
                    )
                    n = int(mask.sum())
                    if n:
                        self._log(
                            named_metric(name, Metric.ACC_1),
                            acc_at_k(emoji_logits[mask], emoji[mask], 1).mean(),
                            n,
                        )
                full_mask = torch.tensor(
                    [s == SRC_FULL for s in source], device=emoji.device
                )
                n_full_e = int(full_mask.sum())
                if n_full_e:
                    self._log(
                        named_metric(Source.FULL_TEXT, Metric.ACC_1),
                        acc_at_k(
                            emoji_logits[full_mask], emoji[full_mask], 1
                        ).mean(),
                        n_full_e,
                    )

        if "lang" in self.heads:
            lang_logits = self.lang(enc)
            loss_lang = cross_entropy(lang_logits, lang)
            loss = loss + loss_lang

        return loss

    def on_validation_epoch_start(self):
        self._val_e_rr.clear()
        self._val_s_rr.clear()
        self._val_s_tgt.clear()

    def on_validation_epoch_end(self):
        self._epoch_metrics(Split.VAL)

    def on_train_epoch_start(self):
        self._trn_e_rr.clear()
        self._trn_s_rr.clear()
        self._trn_s_tgt.clear()

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
                named_metric(name, Metric.RATE),
                self.train_dataset.rates.get(name),
            )

    def on_train_epoch_end(self):
        self._epoch_metrics(Split.TRAIN)

    def _epoch_metrics(self, split: Split):
        if "emoji" in self.heads:
            e_rr = self._val_e_rr if split == Split.VAL else self._trn_e_rr
            if e_rr:
                emoji_mrr = torch.cat(e_rr).mean()
                self.log(
                    named_metric(Source.EMOJI, Metric.MRR, split),
                    emoji_mrr, prog_bar=True,
                )

        if "style" in self.heads:
            s_rr, s_tgt = (
                (self._val_s_rr, self._val_s_tgt)
                if split == Split.VAL
                else (self._trn_s_rr, self._trn_s_tgt)
            )
            if s_rr:
                style_macro, _, _ = macro_average(
                    torch.cat(s_rr), torch.cat(s_tgt), MACRO_MIN_SUPPORT
                )
                self.log(
                    named_metric(Source.STYLE, Metric.MRR, split),
                    style_macro, prog_bar=True,
                )

    def training_step(self, batch, batch_idx):
        return self._step(batch, Split.TRAIN)

    def validation_step(self, batch, batch_idx):
        self._step(batch, Split.VAL)

    def configure_optimizers(self):
        params = list(self.enc.parameters())
        if "emoji" in self.heads:
            params += list(self.emoji_embed.parameters())
        for h in self.heads:
            params += list(getattr(self, h).parameters())
        return optim.Adam(params, lr=LR_ENCODER)


class LitColorGAN(pl.LightningModule):
    def __init__(self, enc: TextEncoder, critic: ColorCritic):
        super().__init__()

        self.enc = enc.requires_grad_(False).eval()

        self.gen = ColorGen()
        self.critic = critic

        self.automatic_optimization = False
        self._val_text: list[torch.Tensor] = []
        self._val_real: list[torch.Tensor] = []

    def on_train_epoch_start(self):
        self.gen.net[0].eval()

    def on_train_batch_start(self, batch, batch_idx):
        self.enc.eval()

    def _cond(self, text: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return self.enc(text)

    def on_validation_epoch_start(self):
        self._val_text.clear()
        self._val_real.clear()

    def validation_step(self, batch, batch_idx):
        text, _, _, colors, *_ = batch
        self._val_text.append(text)
        self._val_real.append(colors)

    def on_validation_epoch_end(self):
        if not self._val_real:
            return

        self.gen.eval()
        with torch.no_grad():
            text = torch.cat(self._val_text)
            real = rgb_to_oklab(torch.cat(self._val_real))

            fake = rgb_to_oklab(self.gen(self.enc(text)))
            val = energy_distance(real, fake)
            self.log(GanMetric.ENERGY_VAL, val, prog_bar=True)

    def training_step(self, batch, batch_idx):
        text, _, _, colors, *_ = batch
        opt_gen, opt_critic = self.optimizers()  # type: ignore

        cond = self._cond(text)

        fake = self.gen(cond)

        # CRITIC
        pair = torch.cat([colors, fake.detach()], dim=0)
        cond_pair = torch.cat([cond, cond], dim=0)
        score = self.critic(cond_pair, pair)
        real, fake_score = score.chunk(2, dim=0)

        loss_critic = relu(1 - real).mean() + relu(1 + fake_score).mean()

        opt_critic.zero_grad()
        self.manual_backward(loss_critic)
        self.clip_gradients(
            opt_critic,  # type: ignore
            gradient_clip_val=GRAD_CLIP_CRITIC,
            gradient_clip_algorithm="norm")

        opt_critic.step()

        # GENERATOR
        gen_score = self.critic(cond, fake)
        loss_energy = energy_distance(rgb_to_oklab(fake), rgb_to_oklab(colors))

        loss_gen_critic = -gen_score.mean()

        loss_gen = \
            (1 - LOSS_WEIGHT_ENERGY) * loss_gen_critic \
            + LOSS_WEIGHT_ENERGY * loss_energy

        opt_gen.zero_grad()

        self.manual_backward(loss_gen)
        self.clip_gradients(
            opt_gen,  # type: ignore
            gradient_clip_val=GRAD_CLIP_GEN,
            gradient_clip_algorithm="norm")

        opt_gen.step()

        self.log(GanMetric.CRITIC_LOSS, loss_critic, prog_bar=True)
        self.log(GanMetric.GEN_LOSS, loss_gen_critic, prog_bar=True)
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
        #     lr=LR_GAN_CRITIC,
        #     betas=(0.5, 0.999))

        return [opt_gen, opt_critic]


class LitColorEnergy(pl.LightningModule):
    def __init__(self):
        super().__init__()

        self.gen = ColorGen()

        self._val_real: list[torch.Tensor] = []

    def _null_cond(self, colors: torch.Tensor) -> torch.Tensor:
        return torch.ones(
            colors.shape[0], EMBED_SIZE_TEXT,
            device=colors.device, dtype=colors.dtype)

    def on_validation_epoch_start(self):
        self._val_real.clear()

    def validation_step(self, batch, batch_idx):
        _, _, _, colors, *_ = batch
        self._val_real.append(colors)

    def on_validation_epoch_end(self):
        if not self._val_real:
            return

        self.gen.eval()
        with torch.no_grad():
            real_rgb = torch.cat(self._val_real)
            real = rgb_to_oklab(real_rgb)
            fake = rgb_to_oklab(self.gen(self._null_cond(real_rgb)))
            val = energy_distance(real, fake)
            self.log(GanMetric.ENERGY_VAL, val, prog_bar=True)

    def training_step(self, batch, batch_idx):
        _, _, _, colors, *_ = batch

        fake = self.gen(self._null_cond(colors))
        loss = energy_distance(rgb_to_oklab(fake), rgb_to_oklab(colors))

        self.log(GanMetric.ENERGY_TRAIN, loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return optim.SGD(self.gen.parameters(), lr=LR_GAN_GEN)


def _load(mod: nn.Module, path: Path) -> nn.Module:
    sd, meta = load_pt(path)
    mod.load_state_dict(sd)
    mod._pt_meta = meta  # type: ignore
    return mod


def _no_progress_bar() -> bool:
    return os.environ.get("EMOJIC_NO_PROGRESS_BAR") == "1"


def _pt_files_ok(pt_dir: Path) -> bool:
    checks: list[tuple[PtFile, nn.Module]] = [
        (PtFile.ENC, TextEncoder()),
        (PtFile.STYLE, StyleHead()),
        (PtFile.EMOJI, EmojiHead()),
        (PtFile.EMOJI_EMBED, EmojiEmbedding()),
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


def _train_encoder(ds, heads: tuple[str, ...], out_dir: Path) -> LitEncoder:
    dl = train_data_loader(data_set=ds, batch_size=TASK_BATCH_SIZE)
    val_dl = eval_data_loader()

    no_bar = _no_progress_bar()
    bar_cbs = [] if no_bar else [TQDMProgressBar()]

    monitor = named_metric(Source.EMOJI, Metric.MRR, Split.VAL)
    ckpt = ModelCheckpoint(
        monitor=monitor, mode="max", save_top_k=1, filename="best-{step}"
    )
    trainer = pl.Trainer(
        devices="auto",
        accelerator="auto",
        logger=TensorBoardLogger(
            "runs", name=CONFIG_NAME, version="enc", default_hp_metric=False
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

    mod = LitEncoder(heads=heads)
    mod.train_dataset = ds
    trainer.fit(mod, dl, val_dl)

    if ckpt.best_model_path:
        mod = LitEncoder.load_from_checkpoint(ckpt.best_model_path)

    save_pt(mod.enc.state_dict(), PtFile.ENC.in_dir(out_dir), stage="enc")
    if "emoji" in heads:
        save_pt(
            mod.emoji_embed.state_dict(),
            PtFile.EMOJI_EMBED.in_dir(out_dir),
            stage="enc",
        )
    for h in ALL_HEADS:
        if h in heads:
            save_pt(
                getattr(mod, h).state_dict(),
                PtFile(f"{h}.pt").in_dir(out_dir),
                stage="enc",
            )

    return mod


def _train_gan(
    enc: TextEncoder, critic: ColorCritic, ds, out_dir: Path
) -> LitColorGAN:
    val_dl = eval_data_loader(mix_sources=False)
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
            version="gan",
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

    gan = LitColorGAN(enc, critic)
    gan_dl = train_data_loader(data_set=ds, batch_size=GAN_BATCH_SIZE)
    trainer.fit(gan, gan_dl, val_dl)

    if ckpt.best_model_path:
        gan = LitColorGAN.load_from_checkpoint(
            ckpt.best_model_path, enc=enc, critic=ColorCritic()
        )

    save_pt(gan.gen.state_dict(), PtFile.GEN.in_dir(out_dir), stage="gan")
    return gan


_ENERGY_PREVIEW_CARD_TEXT = "What's on your mind?"


def _energy_preview_card(bg1: str, bg2: str, fg: str) -> str:
    return (
        '<div class="card" style="'
        f"background:linear-gradient(135deg,{bg1},{bg2});color:{fg}"
        f'">{_ENERGY_PREVIEW_CARD_TEXT}</div>'
    )


def _write_energy_preview(mod: LitColorEnergy, n: int = 50) -> Path:
    mod.gen.eval()
    with torch.no_grad():
        cond = mod._null_cond(torch.empty(n, 1))
        colors = mod.gen(cond)

    cards = "\n".join(
        _energy_preview_card(*rgb_to_hex(colors[i])) for i in range(n)
    )
    html = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>energy preview</title>
<style>
body {{ margin: 0; padding: 2em; background: #111; font-family: sans-serif; }}
.grid {{ display: grid; gap: 1em;
  grid-template-columns: repeat(auto-fill, minmax(180px, 1fr)); }}
.card {{ aspect-ratio: 1; border-radius: 12px; display: flex; align-items: center;
  justify-content: center; text-align: center; padding: 1em; box-sizing: border-box; }}
</style>
</head>
<body>
<div class="grid">
{cards}
</div>
</body>
</html>
"""
    PREVIEW_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%y-%m-%d-%H-%M")
    sha = run_meta()["sha"]
    out_path = PREVIEW_DIR / f"{ts}-{sha}.html"
    out_path.write_text(html, encoding="utf-8")
    return out_path


def _train_energy(ds) -> LitColorEnergy:
    val_dl = eval_data_loader(mix_sources=False)
    no_bar = _no_progress_bar()
    bar_cbs = [] if no_bar else [TQDMProgressBar()]

    trainer = pl.Trainer(
        devices="auto",
        accelerator="auto",
        logger=TensorBoardLogger(
            "runs", name=CONFIG_NAME, version="energy", default_hp_metric=False
        ),
        deterministic=_DETERMINISTIC,  # type: ignore
        max_epochs=EPOCHS_GAN,
        enable_progress_bar=not no_bar,
        val_check_interval=min(VAL_CHECK_INTERVAL, len(ds)),
        callbacks=[
            *bar_cbs,
            ModelSummary(),
            EarlyStopping(
                monitor=GanMetric.ENERGY_VAL,
                mode="min",
                patience=EARLY_STOP_PATIENCE_GAN,
                min_delta=EARLY_STOP_MIN_DELTA_GAN),
        ],
    )

    mod = LitColorEnergy()
    dl = train_data_loader(data_set=ds, batch_size=GAN_BATCH_SIZE)
    trainer.fit(mod, dl, val_dl)
    out_path = _write_energy_preview(mod)
    print(out_path)
    return mod


def _run_report_local(pt_dir: Path) -> None:
    subprocess.run(
        [sys.executable, "tools/report.py", "--pt", str(pt_dir)], check=True
    )


def _run_local(stage: Stage | None) -> None:
    pl.seed_everything(SEED, workers=True)
    torch.backends.cudnn.benchmark = _CUDA
    if _CUDA:
        torch.set_float32_matmul_precision("high")

    if stage == Stage.energy:
        _train_energy(train_ds(mix_sources=False))  # type: ignore
        return

    require_clean_tree()
    skip_report = os.environ.get("EMOJIC_SKIP_REPORT") == "1"
    _DEFAULT_PT.mkdir(parents=True, exist_ok=True)

    if stage == Stage.gan:
        if _pt_files_ok(_DEFAULT_PT):
            enc = _load(TextEncoder(), PtFile.ENC.in_dir(_DEFAULT_PT))
            critic = ColorCritic()
            _train_gan(enc, critic, train_ds(  # type: ignore
                mix_sources=False), _DEFAULT_PT)
            export()
            if not skip_report:
                _run_report_local(_DEFAULT_PT)
            return
        print(
            f"{_DEFAULT_PT}: missing or corrupted pt files, "
            "falling back to a fresh full training",
            flush=True,
        )

    ds = train_ds()
    mod = _train_encoder(ds, ALL_HEADS, _DEFAULT_PT)

    critic = ColorCritic()
    _train_gan(mod.enc, critic, train_ds(  # type: ignore
        mix_sources=False), _DEFAULT_PT)
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
    enc_bytes: bytes | None = None,
    style_bytes: bytes | None = None,
    emoji_bytes: bytes | None = None,
    emoji_embed_bytes: bytes | None = None,
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

    uploads = {
        ENC_PT: enc_bytes,
        STYLE_PT: style_bytes,
        EMOJI_PT: emoji_bytes,
        EMOJI_EMBED_PT: emoji_embed_bytes,
    }
    if any(v is not None for v in uploads.values()):
        Path(REPO, PT_DIR).mkdir(parents=True, exist_ok=True)
    for rel, data in uploads.items():
        if data is not None:
            Path(REPO, rel).write_bytes(data)

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
    pt_bytes: dict[str, bytes | None] = {
        "enc_bytes": None,
        "style_bytes": None,
        "emoji_bytes": None,
        "emoji_embed_bytes": None,
    }
    if stage == "gan":
        for name in (
            ENC_PT,
            STYLE_PT,
            EMOJI_PT,
            EMOJI_EMBED_PT,
        ):
            if not name.exists():
                raise typer.BadParameter(
                    f"{name} not found -- run `train --local` "
                    "(or fetch a Modal run) first"
                )
        pt_bytes = {
            "enc_bytes": ENC_PT.read_bytes(),
            "style_bytes": STYLE_PT.read_bytes(),
            "emoji_bytes": EMOJI_PT.read_bytes(),
            "emoji_embed_bytes": EMOJI_EMBED_PT.read_bytes(),
        }

    fn = train_remote.with_options(
        gpu=DEFAULT_GPU, cpu=GPU_CPU, memory=GPU_MEMORY_MIB, timeout=TIMEOUT_S
    )
    return fn.remote(
        stage=stage,
        threads=GPU_CPU,
        git_sha=git_sha,
        run_time=run_time,
        gpu=DEFAULT_GPU,
        **pt_bytes,
    )


def _dispatch(stage: Stage | None) -> None:
    require_clean_tree()
    git_sha = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True
    ).stdout.strip()
    run_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    stage_str = stage.value if stage else ""
    print(
        f"Training {stage_str or 'full pipeline'} on Modal {DEFAULT_GPU} GPU...",
        flush=True,
    )
    try:
        with modal.enable_output(), modal_app.run():
            print(_run_remote(stage_str, git_sha, run_time))
    finally:
        landed = _retrieve_and_cleanup()
    if landed:
        _run_report_local(_DEFAULT_PT)


_app = typer.Typer(
    context_settings={"help_option_names": ["-h", "--help"]},
)


@_app.command()
def cli(
    stage: Stage | None = typer.Argument(
        None,
        metavar="[gan|energy]",
        help="gan = train only the color GAN, using a pretrained encoder. "
        "energy = debug: fit ColorGen alone (no encoder, no critic). "
        "Omit to train the encoder then the GAN.",
    ),
    local: bool = typer.Option(
        False, "--local", help="Train on this machine instead of Modal."
    ),
) -> None:
    """Train the emojic model.

    Stages
      (none)   Text encoder (all heads), then the color GAN, then ONNX
               export + report.
      gan      Color GAN only: frozen pretrained encoder, generator and
               critic trained from scratch. Requires enc.pt, style.pt,
               emoji.pt, emoji_embed.pt in pt/. Then export + report.
      energy   Debug only: fit ColorGen directly against the energy
               distance, with a constant (all-ones) text embedding standing
               in for the (unused, untrained) encoder and no critic. No
               checkpoints, no export, no report -- just watch
               gan/energy/{train,val} in TensorBoard. Always runs locally,
               ignores --local.

    Location
      Runs on Modal (a T4 GPU) by default. --local runs here instead.
      A dirty git tree always aborts (except stage "energy", which never
      touches checkpoints).
    """
    if stage == Stage.energy:
        _run_local(stage)
        return

    if local:
        _run_local(stage)
    else:
        _dispatch(stage)


if __name__ == "__main__":
    _app()
