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
from torch.nn.functional import binary_cross_entropy_with_logits, normalize
from torch.utils.data import DataLoader, random_split

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from files import (
    COLORS_JSONL,
    DATA_JSONL,
    EMOJI_PT,
    ENC_PT,
    ENERGY_KEYWORDS_TXT,
    EVAL_JSONL,
    EXP_ENC_PT,
    EXP_GAN_PT,
    KEYWORDS_JSON,
    LABELS_JSON,
    MODEL_DIR,
    PT_DIR,
    STYLE_PT,
    TOOLS_DIR,
    TRAIN_JSONL,
)
from model.config import (
    CONFIG_NAME,
    EARLY_STOP_PATIENCE,
    ENERGY_Z_SAMPLES,
    EPOCHS_GAN,
    EPOCHS_TASK,
    GAN_BATCH_SIZE,
    GAN_CRITIC_LR,
    GAN_GEN_LR,
    GRAD_CLIP_CRITIC,
    GRAD_CLIP_GEN,
    INFONCE_TEMP,
    LR,
    SEED,
    TASK_BATCH_SIZE,
    TEXT_EMBED_SIZE,
    VAL_CHECK_INTERVAL,
)
from model.data import (
    EmojiDataset,
    eval_data_loader,
    read,
    train_data_loader,
    train_ds,
)
from model.export_onnx import export
from model.model import (
    ColorDsc,
    ColorGen,
    EmojiHead,
    StyleHead,
    TextEncoder,
    rgb_to_oklab,
)
from model.runmeta import load_pt, require_clean_tree, save_pt


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


def energy_distance(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    mode = "donot_use_mm_for_euclid_dist"
    xy = torch.cdist(x, y, compute_mode=mode).mean()
    xx = torch.cdist(x, x, compute_mode=mode).mean()
    yy = torch.cdist(y, y, compute_mode=mode).mean()
    return (2 * xy - xx - yy).clamp(min=0.0).sqrt()


class LitTask(pl.LightningModule):
    def __init__(self, heads: tuple[str, ...] = ("style", "emoji")):
        super().__init__()
        self.save_hyperparameters()

        self.heads = heads

        self.enc = TextEncoder()
        self.style = StyleHead()
        self.emoji = EmojiHead()

    def _step(self, batch, split):
        text, emoji, style, _ = batch

        enc = self.enc(text)
        loss = enc.new_zeros(())

        if "style" in self.heads:
            style_logits = self.style(enc)
            loss_style = lse_infonce(style_logits, style, INFONCE_TEMP)
            loss = loss + loss_style

            style_mrr = mrr(style_logits, style).mean()

            for name, val, bs in (
                (f"loss/s/{split}", loss_style, text.size(0)),
                (f"MRR/s/{split}", style_mrr, text.size(0)),
            ):
                self.log(name, val, on_step=False, on_epoch=True,
                         prog_bar=True, batch_size=bs)

        if "emoji" in self.heads:
            emoji_logits = self.emoji(enc)
            loss_emoji = lse_infonce(emoji_logits, emoji, INFONCE_TEMP)
            loss = loss + loss_emoji

            has_e = emoji.sum(dim=-1) > 0
            n_e = int(has_e.sum())
            if n_e:
                emoji_mrr = mrr(emoji_logits[has_e], emoji[has_e]).mean()
            else:
                emoji_mrr = torch.zeros((), device=emoji.device)

            for name, val, bs in (
                (f"loss/e/{split}", loss_emoji, text.size(0)),
                (f"MRR/e/{split}", emoji_mrr, max(n_e, 1)),
            ):
                self.log(name, val, on_step=False, on_epoch=True,
                         prog_bar=True, batch_size=bs)

        return loss

    def training_step(self, batch, batch_idx):
        return self._step(batch, "train")

    def validation_step(self, batch, batch_idx):
        self._step(batch, "val")

    def configure_optimizers(self):
        params = list(self.enc.parameters())
        for name in self.heads:
            params += list(getattr(self, name).parameters())
        return optim.Adam(params, lr=LR)


class LitColorGAN(pl.LightningModule):
    def __init__(self, enc: TextEncoder):
        super().__init__()

        self.enc = enc.requires_grad_(False).eval()

        self.gen = ColorGen()
        self.tst = ColorDsc()

        self.register_buffer(
            "z_bank",
            normalize(
                torch.randn(
                    ENERGY_Z_SAMPLES,
                    TEXT_EMBED_SIZE,
                    generator=torch.Generator().manual_seed(SEED),
                ),
                dim=-1,
            ),
        )

        self.automatic_optimization = False
        self._val_text: list[torch.Tensor] = []
        self._val_real: list[torch.Tensor] = []

    def on_train_epoch_start(self):
        self.enc.eval()

    def _cond(self, text: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return self.enc(text)

    def on_validation_epoch_start(self):
        self._val_text.clear()
        self._val_real.clear()

    def validation_step(self, batch, batch_idx):
        text, _, _, colors = batch
        self._val_text.append(text)
        self._val_real.append(colors)

    def _split_energy(self, pts: torch.Tensor) -> torch.Tensor:
        m = pts.size(0)
        half = m // 2
        perm = torch.randperm(m, generator=torch.Generator().manual_seed(SEED)).to(
            pts.device
        )
        return energy_distance(pts[perm[:half]], pts[perm[half: 2 * half]])

    def on_validation_epoch_end(self):
        if not self._val_real:
            return

        self.gen.eval()
        with torch.no_grad():
            text = torch.cat(self._val_text)
            real = rgb_to_oklab(torch.cat(self._val_real))
            n = text.size(0)
            z = self.z_bank[  # type: ignore
                torch.arange(n, device=self.device) % self.z_bank.size(
                    0)  # type: ignore
            ]

            fake = rgb_to_oklab(self.gen(self.enc(text), z))
            val = energy_distance(real, fake)
            self.log("energy/gan/val", val, prog_bar=True)

            if isinstance(self.logger, TensorBoardLogger):
                ref = self._split_energy(real)
                self.logger.experiment.add_scalar(
                    "energy/gan/ref", ref, self.global_step)

    def training_step(self, batch, batch_idx):
        text, _, _, colors = batch
        opt_gen, opt_tst = self.optimizers()  # type: ignore

        cond = self._cond(text)

        fake = self.gen(cond)

        both = torch.cat([colors, fake.detach()], dim=0)
        tst_real, tst_fake = self.tst(
            torch.cat([cond, cond], dim=0), both).chunk(2, dim=0)

        loss_tst_real = binary_cross_entropy_with_logits(
            tst_real, torch.ones_like(tst_real))

        loss_tst_fake = binary_cross_entropy_with_logits(
            tst_fake, torch.zeros_like(tst_fake)
        )

        loss_tst = loss_tst_real + loss_tst_fake

        opt_tst.zero_grad()
        self.manual_backward(loss_tst)
        self.clip_gradients(
            opt_tst,  # type: ignore
            gradient_clip_val=GRAD_CLIP_CRITIC,
            gradient_clip_algorithm="norm",
        )

        opt_tst.step()

        _, tst_fake = self.tst(
            torch.cat([cond, cond], dim=0),
            torch.cat([colors, fake], dim=0),
        ).chunk(2, dim=0)
        loss_gen = binary_cross_entropy_with_logits(
            tst_fake, torch.ones_like(tst_fake))

        opt_gen.zero_grad()
        self.manual_backward(loss_gen)
        self.clip_gradients(
            opt_gen,  # type: ignore
            gradient_clip_val=GRAD_CLIP_GEN,
            gradient_clip_algorithm="norm",
        )

        opt_gen.step()

        self.log("loss/gan/tst", loss_tst, prog_bar=True)
        self.log("loss/gan/gen", loss_gen, prog_bar=True)

    def configure_optimizers(self):
        opt_gen = optim.SGD(self.gen.parameters(), lr=GAN_GEN_LR)
        opt_tst = optim.SGD(self.tst.parameters(), lr=GAN_CRITIC_LR)

        return [opt_gen, opt_tst]


class LitColorExp(pl.LightningModule):
    def __init__(self):
        super().__init__()

        self.enc = TextEncoder()
        self.gen = ColorGen()
        self.tst = ColorDsc()

        self.automatic_optimization = False
        self._ep_text: list[torch.Tensor] = []
        self._ep_real: list[torch.Tensor] = []

    def on_train_epoch_start(self):
        self._ep_text.clear()
        self._ep_real.clear()

    def _split_energy(self, pts: torch.Tensor) -> torch.Tensor:
        m = pts.size(0)
        half = m // 2
        perm = torch.randperm(m, generator=torch.Generator().manual_seed(SEED)).to(
            pts.device
        )
        return energy_distance(pts[perm[:half]], pts[perm[half: 2 * half]])

    def on_train_epoch_end(self):
        if not self._ep_real:
            return

        was_training = self.gen.training
        self.gen.eval()
        with torch.no_grad():
            text = torch.cat(self._ep_text)
            real = rgb_to_oklab(torch.cat(self._ep_real))
            fake = rgb_to_oklab(self.gen(self.enc(text)))
            self.log("energy/gan/train", energy_distance(real, fake), prog_bar=True)

            if isinstance(self.logger, TensorBoardLogger):
                self.logger.experiment.add_scalar(
                    "energy/gan/ref", self._split_energy(real), self.global_step
                )
        if was_training:
            self.gen.train()

    def training_step(self, batch, batch_idx):
        text, _, _, colors = batch
        opt_gen, opt_tst = self.optimizers()  # type: ignore

        self._ep_text.append(text)
        self._ep_real.append(colors)

        cond = self.enc(text)
        cond_d = cond.detach()

        fake = self.gen(cond)

        both = torch.cat([colors, fake.detach()], dim=0)
        tst_real, tst_fake = self.tst(
            torch.cat([cond_d, cond_d], dim=0), both).chunk(2, dim=0)

        loss_tst_real = binary_cross_entropy_with_logits(
            tst_real, torch.ones_like(tst_real))
        loss_tst_fake = binary_cross_entropy_with_logits(
            tst_fake, torch.zeros_like(tst_fake))
        loss_tst = loss_tst_real + loss_tst_fake

        opt_tst.zero_grad()
        self.manual_backward(loss_tst)
        self.clip_gradients(
            opt_tst,  # type: ignore
            gradient_clip_val=GRAD_CLIP_CRITIC,
            gradient_clip_algorithm="norm",
        )
        opt_tst.step()

        _, tst_fake = self.tst(
            torch.cat([cond, cond], dim=0),
            torch.cat([colors, fake], dim=0),
        ).chunk(2, dim=0)
        loss_gen = binary_cross_entropy_with_logits(
            tst_fake, torch.ones_like(tst_fake))

        opt_gen.zero_grad()
        self.manual_backward(loss_gen)
        self.clip_gradients(
            opt_gen,  # type: ignore
            gradient_clip_val=GRAD_CLIP_GEN,
            gradient_clip_algorithm="norm",
        )
        opt_gen.step()

        self.log("loss/gan/tst", loss_tst, prog_bar=True)
        self.log("loss/gan/gen", loss_gen, prog_bar=True)

    def configure_optimizers(self):
        opt_gen = optim.SGD(
            list(self.gen.parameters()) + list(self.enc.parameters()), lr=GAN_GEN_LR
        )
        opt_tst = optim.SGD(self.tst.parameters(), lr=GAN_CRITIC_LR)
        return [opt_gen, opt_tst]


class LitCriticExp(pl.LightningModule):
    def __init__(self):
        super().__init__()

        self.enc = TextEncoder()
        self.tst = ColorDsc()

    def _step(self, batch, split):
        text, _, _, colors = batch

        if split == "val":
            shift = 1
        else:
            shift = int(torch.randint(1, text.size(0), (1,)).item())
        mismatch = colors.roll(shift, dims=0)

        cond = self.enc(text)
        cond2 = torch.cat([cond, cond], dim=0)
        pal = torch.cat([colors, mismatch], dim=0)
        logit_real, logit_fake = self.tst(cond2, pal).chunk(2, dim=0)

        loss_real = binary_cross_entropy_with_logits(
            logit_real, torch.ones_like(logit_real))
        loss_fake = binary_cross_entropy_with_logits(
            logit_fake, torch.zeros_like(logit_fake))
        loss = loss_real + loss_fake

        acc_real = (logit_real > 0).float().mean()
        acc_fake = (logit_fake < 0).float().mean()
        acc = 0.5 * (acc_real + acc_fake)

        bs = text.size(0)
        for name, val in (
            (f"loss/critic/{split}", loss),
            (f"acc/critic/{split}", acc),
            (f"acc/critic/real/{split}", acc_real),
            (f"acc/critic/fake/{split}", acc_fake),
        ):
            self.log(name, val, on_step=False, on_epoch=True,
                     prog_bar=True, batch_size=bs)

        return loss

    def training_step(self, batch, batch_idx):
        return self._step(batch, "train")

    def validation_step(self, batch, batch_idx):
        self._step(batch, "val")

    def configure_optimizers(self):
        return optim.Adam(
            list(self.enc.parameters()) + list(self.tst.parameters()), lr=1e-3)


class Model(StrEnum):
    emoji = "emoji"
    style = "style"
    task = "task"
    gan = "gan"
    all = "all"
    color_exp = "color-exp"
    critic = "critic"


TASK_HEADS: dict[str, tuple[str, ...]] = {
    "emoji": ("emoji",),
    "style": ("style",),
    "task": ("style", "emoji"),
    "all": ("style", "emoji"),
}


def _load(mod: nn.Module, path: str) -> nn.Module:
    sd, meta = load_pt(path)
    mod.load_state_dict(sd)
    mod._pt_meta = meta  # type: ignore
    return mod


def _no_progress_bar() -> bool:
    return os.environ.get("EMOJIC_NO_PROGRESS_BAR") == "1"


def _train_task(ds, stage: str, heads: tuple[str, ...]) -> LitTask:
    task_dl = train_data_loader(data_set=ds, batch_size=TASK_BATCH_SIZE)
    val_dl = eval_data_loader()

    no_bar = _no_progress_bar()
    progress_bar_cbs = [] if no_bar else [TQDMProgressBar()]

    if "emoji" in heads:
        task_monitor = "MRR/e/val"
    else:
        task_monitor = "MRR/s/val"

    task_ckpt = ModelCheckpoint(
        monitor=task_monitor, mode="max", save_top_k=1, filename="best-{step}"
    )

    task_trainer = pl.Trainer(
        devices="auto",
        accelerator="auto",
        logger=TensorBoardLogger(
            "runs", name=CONFIG_NAME, version=stage, default_hp_metric=False
        ),
        deterministic=True,
        max_epochs=EPOCHS_TASK,
        val_check_interval=min(VAL_CHECK_INTERVAL, len(task_dl)),
        enable_progress_bar=not no_bar,
        callbacks=[
            task_ckpt,
            EarlyStopping(monitor=task_monitor, mode="max",
                          patience=EARLY_STOP_PATIENCE),
            *progress_bar_cbs,
            ModelSummary(),
        ],
    )

    task = LitTask(heads=heads)
    task_trainer.fit(task, task_dl, val_dl)

    if task_ckpt.best_model_path:
        task = LitTask.load_from_checkpoint(task_ckpt.best_model_path)

    saved = [("enc", task.enc)]
    if "style" in heads:
        saved.append(("style", task.style))
    if "emoji" in heads:
        saved.append(("emoji", task.emoji))

    for name, mod in saved:
        save_pt(mod.state_dict(), f"{PT_DIR}/{name}.pt", stage=stage)

    return task


def _train_gan(enc: TextEncoder, ds) -> LitColorGAN:
    val_dl = eval_data_loader()
    no_bar = _no_progress_bar()
    progress_bar_cbs = [] if no_bar else [TQDMProgressBar()]

    gan_ckpt = ModelCheckpoint(
        monitor="energy/gan/val", mode="min", save_top_k=1, filename="best-gan-{step}"
    )

    gan_trainer = pl.Trainer(
        devices="auto",
        accelerator="auto",
        logger=TensorBoardLogger(
            "runs", name=CONFIG_NAME, version="gan", default_hp_metric=False
        ),
        deterministic=True,
        max_epochs=EPOCHS_GAN,
        enable_progress_bar=not no_bar,
        callbacks=[
            gan_ckpt,
            EarlyStopping(
                monitor="energy/gan/val", mode="min", patience=EARLY_STOP_PATIENCE
            ),
            *progress_bar_cbs,
            ModelSummary(),
        ],
    )

    gan = LitColorGAN(enc)
    gan_dl = train_data_loader(data_set=ds, batch_size=GAN_BATCH_SIZE)
    gan_trainer.fit(gan, gan_dl, val_dl)

    if gan_ckpt.best_model_path:
        gan = LitColorGAN.load_from_checkpoint(gan_ckpt.best_model_path, enc=enc)

    for name, mod in (
        ("gen", gan.gen),
        ("tst", gan.tst),
    ):
        save_pt(mod.state_dict(), f"{PT_DIR}/{name}.pt", stage="gan")

    return gan


def _run_color_exp() -> None:
    require_clean_tree()
    pl.seed_everything(SEED, workers=True)
    torch.backends.cudnn.benchmark = False

    if not Path(COLORS_JSONL).exists():
        raise typer.BadParameter(
            f"{COLORS_JSONL} not found -- run "
            f"`bun extract-colors {DATA_JSONL} > {COLORS_JSONL}` first"
        )

    ds = EmojiDataset(list(read(COLORS_JSONL)))
    dl = train_data_loader(data_set=ds, batch_size=GAN_BATCH_SIZE)

    no_bar = _no_progress_bar()
    progress_bar_cbs = [] if no_bar else [TQDMProgressBar()]

    trainer = pl.Trainer(
        devices="auto",
        accelerator="auto",
        logger=TensorBoardLogger(
            "runs", name=CONFIG_NAME, version="color-exp", default_hp_metric=False
        ),
        deterministic=True,
        max_epochs=EPOCHS_GAN,
        enable_progress_bar=not no_bar,
        enable_checkpointing=False,
        callbacks=[*progress_bar_cbs, ModelSummary()],
    )

    exp = LitColorExp()
    trainer.fit(exp, dl)

    save_pt(exp.enc.state_dict(), EXP_ENC_PT, stage="color-exp")
    save_pt(exp.gen.state_dict(), EXP_GAN_PT, stage="color-exp")


def _run_critic_exp() -> None:
    require_clean_tree()
    pl.seed_everything(SEED, workers=True)
    torch.backends.cudnn.benchmark = False

    if not Path(COLORS_JSONL).exists():
        raise typer.BadParameter(
            f"{COLORS_JSONL} not found -- run "
            f"`bun extract-colors {DATA_JSONL} > {COLORS_JSONL}` first"
        )

    ds = EmojiDataset(list(read(COLORS_JSONL)))
    n_val = max(1, len(ds) // 10)
    split = random_split(
        ds, [len(ds) - n_val, n_val],
        generator=torch.Generator().manual_seed(SEED))
    train_split, val_split = split

    dl = DataLoader(
        train_split, batch_size=GAN_BATCH_SIZE, shuffle=True,
        drop_last=True, num_workers=0)
    val_dl = DataLoader(
        val_split, batch_size=2000, shuffle=False,
        drop_last=False, num_workers=0)

    no_bar = _no_progress_bar()
    progress_bar_cbs = [] if no_bar else [TQDMProgressBar()]

    trainer = pl.Trainer(
        devices="auto",
        accelerator="auto",
        logger=TensorBoardLogger(
            "runs", name=CONFIG_NAME, version="critic-exp", default_hp_metric=False
        ),
        deterministic=True,
        max_epochs=EPOCHS_GAN,
        enable_progress_bar=not no_bar,
        enable_checkpointing=False,
        callbacks=[*progress_bar_cbs, ModelSummary()],
    )

    trainer.fit(LitCriticExp(), dl, val_dl)


def _run_report_local(model: Model) -> None:
    subprocess.run([sys.executable, "tools/report.py", "--pt", PT_DIR], check=True)


def _run_local(model: Model) -> None:
    require_clean_tree()
    pl.seed_everything(SEED, workers=True)
    torch.backends.cudnn.benchmark = False

    skip_report = os.environ.get("EMOJIC_SKIP_REPORT") == "1"

    if model == Model.gan:
        ds = train_ds()
        enc = _load(TextEncoder(), ENC_PT)
        _train_gan(enc, ds)  # type: ignore
        export()
        if not skip_report:
            _run_report_local(model)
        return

    ds = train_ds()
    task = _train_task(ds, model.value, TASK_HEADS[model.value])

    if model in (Model.emoji, Model.style, Model.task):
        if not skip_report:
            _run_report_local(model)
        return

    _train_gan(task.enc, ds)
    export()
    if not skip_report:
        _run_report_local(model)


CPU = 16
MEMORY_MIB = 16384
TIMEOUT_S = 60 * 60
TIMEOUT_S_ALL = 90 * 60
REPO = "/repo"
VENV_PY = sys.executable
TB_PORT = 6006

WORKTREE_TAG = hashlib.sha1(str(Path.cwd().resolve()).encode()).hexdigest()[:10]
VOL_NAME = f"emojic-artifacts-{WORKTREE_TAG}"
ARTIFACTS = "/artifacts"

DEP_FILES = ["pyproject.toml", "uv.lock", ".python-version", "README.md"]
CODE_FILES = [
    "files.py",
    f"{MODEL_DIR}/__init__.py",
    f"{MODEL_DIR}/config.py",
    f"{MODEL_DIR}/data.py",
    f"{MODEL_DIR}/model.py",
    f"{MODEL_DIR}/train.py",
    f"{MODEL_DIR}/export_onnx.py",
    f"{MODEL_DIR}/pred.py",
    f"{MODEL_DIR}/runmeta.py",
    f"{TOOLS_DIR}/report.py",
    LABELS_JSON,
    KEYWORDS_JSON,
    ENERGY_KEYWORDS_TXT,
    DATA_JSONL,
    TRAIN_JSONL,
    EVAL_JSONL,
]
COLLECT_TREES = [PT_DIR, "runs", "web/public", "report"]

modal_image = modal.Image.debian_slim(python_version="3.13").pip_install("uv")
for _name in DEP_FILES:
    modal_image = modal_image.add_local_file(_name, f"{REPO}/{_name}", copy=True)
modal_image = modal_image.run_commands(
    f"cd {REPO} && UV_PROJECT_ENVIRONMENT=/usr/local uv sync --frozen"
)
for _name in CODE_FILES:
    modal_image = modal_image.add_local_file(_name, f"{REPO}/{_name}", copy=True)
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


def _stash(dst: str) -> int:
    root, out = Path(REPO), Path(dst)
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
    memory=MEMORY_MIB,
    timeout=TIMEOUT_S,
    volumes={ARTIFACTS: vol},
    include_source=False,
)
def train_remote(
    model: str,
    threads: int,
    git_sha: str,
    run_time: str,
    enc_bytes: bytes | None = None,
    style_bytes: bytes | None = None,
    emoji_bytes: bytes | None = None,
) -> dict[str, int]:
    env = _run_env(threads)
    env["EMOJIC_GIT_SHA"] = git_sha
    env["EMOJIC_RUN_TIME"] = run_time
    env["EMOJIC_DISPATCH_CHECKED"] = "1"
    if enc_bytes is not None or style_bytes is not None or emoji_bytes is not None:
        Path(REPO, PT_DIR).mkdir(parents=True, exist_ok=True)
    if enc_bytes is not None:
        Path(REPO, ENC_PT).write_bytes(enc_bytes)
    if style_bytes is not None:
        Path(REPO, STYLE_PT).write_bytes(style_bytes)
    if emoji_bytes is not None:
        Path(REPO, EMOJI_PT).write_bytes(emoji_bytes)
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
                proc = subprocess.Popen(
                    [VENV_PY, f"{MODEL_DIR}/train.py", model, "--local"],
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
        n = _stash(ARTIFACTS)
        vol.commit()
        print(f"stashed {n} files to volume {VOL_NAME}", flush=True)
    if code != 0:
        raise RuntimeError(f"train.py {model} --local exited with {code}")
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
                PT_DIR,
                "runs",
                "report",
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


def _run_remote(
    model: Model, cpu: int, memory: int, git_sha: str, run_time: str
) -> dict[str, int]:
    enc_bytes = style_bytes = emoji_bytes = None
    if model == Model.gan:
        for name in (ENC_PT, STYLE_PT, EMOJI_PT):
            if not Path(name).exists():
                raise typer.BadParameter(
                    f"{name} not found -- run `train task --local` "
                    "(or fetch a Modal task run) first"
                )
        enc_bytes = Path(ENC_PT).read_bytes()
        style_bytes = Path(STYLE_PT).read_bytes()
        emoji_bytes = Path(EMOJI_PT).read_bytes()

    timeout = TIMEOUT_S_ALL if model == Model.all else TIMEOUT_S
    fn = train_remote
    if cpu != CPU or memory != MEMORY_MIB or timeout != TIMEOUT_S:
        fn = train_remote.with_options(cpu=cpu, memory=memory, timeout=timeout)
    return fn.remote(
        model=model.value,
        threads=cpu,
        git_sha=git_sha,
        run_time=run_time,
        enc_bytes=enc_bytes,
        style_bytes=style_bytes,
        emoji_bytes=emoji_bytes,
    )


def _dispatch(
    model: Model, cpu: int, memory: int, fetch_only: bool, need_app_ctx: bool
) -> None:
    if model in (Model.color_exp, Model.critic):
        raise typer.BadParameter(
            f"{model.value} runs locally only -- use `train {model.value}`")

    if fetch_only:
        if _retrieve_and_cleanup():
            _run_report_local(model)
        return

    require_clean_tree()
    git_sha = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True
    ).stdout.strip()
    run_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"Training {model.value} on Modal...", flush=True)
    try:
        with modal.enable_output():
            if need_app_ctx:
                with modal_app.run():
                    print(_run_remote(model, cpu, memory, git_sha, run_time))
            else:
                print(_run_remote(model, cpu, memory, git_sha, run_time))
    finally:
        landed = _retrieve_and_cleanup()
    if landed:
        _run_report_local(model)


_app = typer.Typer(
    context_settings={"help_option_names": ["-h", "--help"]},
)


@_app.command()
def cli(
    model: Model = typer.Argument(..., help="Which stage(s) to train."),
    local: bool = typer.Option(
        False, "--local", help="Train on this machine instead of Modal."
    ),
    cpu: int | None = typer.Option(None, help="Modal CPU count (Modal only)."),
    memory: int | None = typer.Option(
        None, help="Modal memory in MiB (Modal only)."),
) -> None:
    """Train emoji/style/task/gan/all: on Modal by default, or locally with --local."""
    if model in (Model.color_exp, Model.critic):
        if cpu is not None or memory is not None:
            raise typer.BadParameter(
                f"--cpu/--memory do not apply to {model.value} (local only)")
        if model == Model.color_exp:
            _run_color_exp()
        else:
            _run_critic_exp()
        return

    if local and (cpu is not None or memory is not None):
        raise typer.BadParameter(
            "--cpu/--memory only apply when dispatching to Modal")

    if local:
        _run_local(model)
    else:
        _dispatch(
            model, cpu or CPU, memory or MEMORY_MIB, fetch_only=False, need_app_ctx=True
        )


@modal_app.local_entrypoint()
def main(
    model: str, cpu: int = CPU, memory: int = MEMORY_MIB, fetch_only: bool = False
) -> None:
    _dispatch(Model(model), cpu, memory, fetch_only, need_app_ctx=False)


if __name__ == "__main__":
    _app()
