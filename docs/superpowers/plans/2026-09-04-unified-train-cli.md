# Unified train.py CLI Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace `train.py` + `train_gan.py` + `train-modal.py` + `loop_emoji.py` with a single Typer-based `train.py` (`train <task|gan|all> [--local]`), installed as a `train` console script with shell autocompletion, and update `CLAUDE.md` + `Taskfile.yml` to match.

**Architecture:** One file, `train.py`, keeps the existing `LitTask`/`LitColorGAN` Lightning modules and pure metric functions unchanged, but restructures the driving code into three layers: (1) `_train_task`/`_train_gan` — the actual local training, factored out of today's combined `main()`; (2) Modal scaffolding lifted from `train-modal.py` (image, volume, `train_remote` app function, artifact stash/retrieve) generalized to accept a `model` argument and, for `gan`, upload a locally-read `enc.pt`; (3) a Typer CLI (`train <model> [--local] [--cpu] [--memory]`) that either calls layer 1 directly or dispatches to layer 2 inside an ephemeral Modal app context. A separate `modal.App.local_entrypoint()` (`main`) exposes the same dispatch through `modal run train.py::main` for `Taskfile.yml`'s detached background workflow, which needs Modal's own `--detach` (not available through a plain blocking Python call).

**Tech Stack:** Python 3.13, `uv`, Typer (CLI + free shell completion), PyTorch Lightning, Modal SDK, hatchling (new build backend, needed only to make `train` an installed console script).

**Spec:** This plan document is self-contained; it was written directly from the user's spec (see conversation) plus two clarifying answers:
- `train gan` (no `--local`) gets **real** Modal support: local `enc.pt` bytes are uploaded into the remote container before training.
- `loop_emoji.py` is **deleted outright**, not folded into `train.py` — only a single train session is supported.

## Global Constraints

- Package management is `uv` only — never `pip install`; use `uv add`/`uv lock`/`uv sync`.
- No comments or docstrings in source, except: shebangs, `# type: ignore` / `# noqa`, and Typer command docstrings (the codebase already uses these as `--help` text — keep that pattern, don't add explanatory comments anywhere else).
- Never run a full training loop to "test" a change (`uv run python train.py ...` / `train ...` end-to-end). Verify with `ruff check` / `ruff format --check`, `uv run python test_runmeta.py`, `--help` output, and plain imports only.
- Lint/format gate before considering any task done: `uv run ruff check .` and `uv run ruff format --check .` (line length 93, rules E/F/I/UP/B).
- `uv run python train.py` / new `train` CLI must still abort on a dirty git tree via `runmeta.require_clean_tree()` — no override, matching today's behavior.
- All tool scripts assume the repo root as CWD — unchanged.
- Git commits/PRs from this session end with the attribution block already configured for this session (Co-Authored-By / Claude-Session line) — apply it to every commit made while executing this plan.

---

## File Structure

- **`pyproject.toml`** (modify) — add `[build-system]` (hatchling), `[project.scripts] train = "train:_app"`, and `[tool.hatch.build.targets.wheel]` listing exactly the modules `train.py` imports (`train.py`, `config.py`, `data.py`, `model.py`, `runmeta.py`), so `uv sync` installs a real `train` console script.
- **`uv.lock`** (regenerate) — `uv lock` after the `pyproject.toml` change.
- **`train.py`** (rewrite in place) — the single unified script. Keeps `f1`, `lse_infonce`, `mrr_at_k`, `energy_distance`, `ap_at_k`, `LitTask`, `LitColorGAN` verbatim from today's file. Adds: `Model` enum, `_load`, `_train_task`, `_train_gan`, `_run_local` (local training layer); Modal `CPU`/`MEMORY_MIB`/image/volume scaffolding, `train_remote`, `_run_env`, `_stash`, `_modal`, `_retrieve_and_cleanup`, `_run_remote`, `_dispatch` (Modal layer, generalized from `train-modal.py`); Typer `_app`/`cli` command and `modal_app.local_entrypoint() main` (CLI layer).
- **`train_gan.py`** (delete) — folded into `train.py`'s `gan` model / `_train_gan`.
- **`train-modal.py`** (delete) — folded into `train.py`'s Modal layer.
- **`loop_emoji.py`** (delete) — per clarification, not replaced.
- **`Taskfile.yml`** (modify) — `modal-train`, `modal-train:fetch`, `modal-train:list`/`:reports`/`:drop` tasks currently reference `train-modal.py`; repoint at `train.py::main`, fixing the `modal-train:fetch`'s existing dangling `train-modal.py::fetch` reference (no such function ever existed — the real fetch path is `main -- --fetch-only`) along the way.
- **`CLAUDE.md`** (modify) — rewrite the `train.py`/`train_gan.py`/`train-modal.py` bullets into one `train.py` bullet, delete the `loop_emoji.py` bullet and the "Training location is a free mix" paragraph, and fix every other line in "Environment & commands" / "Conventions" that names the old scripts.

---

### Task 1: Package `train.py` as an installed console script

**Files:**
- Modify: `pyproject.toml`
- Modify: `uv.lock` (via `uv lock`, not hand-edited)

**Interfaces:**
- Produces: a `train` executable in `.venv/bin/` (and reachable as `uv run train ...`) once Task 2 gives `train.py` a `_app` Typer instance.

- [ ] **Step 1: Add the build-system + console-script config**

Edit `pyproject.toml`:

```toml
[project]
name = "emojic"
version = "0.1.0"
description = "Add your description here"
readme = "README.md"
requires-python = ">=3.13"
dependencies = [
    "lightning>=2.6.5",
    "numpy>=2.4.6",
    "pyyaml>=6.0.3",
    "tensorboard>=2.21.0",
    'torch',
    "tqdm>=4.70.0",
    "typer>=0.27.2",
]

[project.scripts]
train = "train:_app"

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
include = ["train.py", "config.py", "data.py", "model.py", "runmeta.py"]
```

Leave `[[tool.uv.index]]`, `[tool.uv.sources]`, `[dependency-groups]`, `[tool.ruff]`, `[tool.ruff.lint]`, `[tool.ruff.lint.per-file-ignores]` exactly as they are today, below this.

- [ ] **Step 2: Regenerate the lockfile**

Run: `uv lock`
Expected: succeeds, `uv.lock` changes (adds hatchling + the project's own wheel metadata).

- [ ] **Step 3: Sync and sanity check the build**

Run: `uv sync`
Expected: succeeds. This step will only fully prove itself once Task 2 lands (`train.py` needs its Typer `_app` object to exist) — if `uv sync` fails now because `train.py` doesn't yet define `_app`, that's expected; re-run it at the end of Task 2's steps instead and note that here.

- [ ] **Step 4: Commit**

```bash
git add pyproject.toml uv.lock
git commit -m "build: package train.py as an installed \`train\` console script"
```

---

### Task 2: Rewrite train.py as the unified CLI

**Files:**
- Modify: `train.py` (full rewrite)

**Interfaces:**
- Consumes: `config.py` (`CONFIG_NAME`, `EARLY_STOP_PATIENCE`, `EMOJI_AP_K`, `ENERGY_Z_SAMPLES`, `EPOCHS_GAN`, `EPOCHS_TASK`, `GAN_BATCH_SIZE`, `GAN_LR`, `GRAD_CLIP`, `INFONCE_TEMP`, `LR`, `SEED`, `STYLE_AP_K`, `TASK_BATCH_SIZE`, `TEXT_EMBED_SIZE`, `VAL_CHECK_INTERVAL`); `data.py` (`eval_data_loader`, `train_data_loader`, `train_ds`); `model.py` (`ColorDsc`, `ColorGen`, `EmojiHead`, `StyleHead`, `TextEncoder`, `rgb_to_oklab`); `runmeta.py` (`load_pt`, `require_clean_tree`, `save_pt`); `export_onnx.py` (`export`).
- Produces: module-level `_app` (Typer instance, the `[project.scripts]` target from Task 1), `Model` enum (`task`/`gan`/`all`), `LitTask`, `LitColorGAN` (unchanged public shape — still constructed as `LitTask()` / `LitColorGAN(enc)`), `modal_app` (the `modal.App`), `train_remote` (the `@modal_app.function`), `main` (the `@modal_app.local_entrypoint()`, signature `main(model: str, cpu: int = CPU, memory: int = MEMORY_MIB, fetch_only: bool = False)`) — this is what `Taskfile.yml` (Task 4) and any direct `modal run train.py::main` call target.

- [ ] **Step 1: Write the new train.py**

```python
import hashlib
import os
import shutil
import subprocess
import sys
import tempfile
from enum import Enum
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

from config import (
    CONFIG_NAME,
    EARLY_STOP_PATIENCE,
    EMOJI_AP_K,
    ENERGY_Z_SAMPLES,
    EPOCHS_GAN,
    EPOCHS_TASK,
    GAN_BATCH_SIZE,
    GAN_LR,
    GRAD_CLIP,
    INFONCE_TEMP,
    LR,
    SEED,
    STYLE_AP_K,
    TASK_BATCH_SIZE,
    TEXT_EMBED_SIZE,
    VAL_CHECK_INTERVAL,
)
from data import (
    eval_data_loader,
    train_data_loader,
    train_ds,
)
from export_onnx import export
from model import (
    ColorDsc,
    ColorGen,
    EmojiHead,
    StyleHead,
    TextEncoder,
    rgb_to_oklab,
)
from runmeta import load_pt, require_clean_tree, save_pt


def f1(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return 2 * a * b / (a + b + 1e-8)


def lse_infonce(
    logits: torch.Tensor,
    target: torch.Tensor,
    temp: float,
) -> torch.Tensor:
    z = logits / temp
    all_lse = torch.logsumexp(z, dim=-1)
    pos_lse = torch.logsumexp(z.masked_fill(target == 0, float("-inf")), dim=-1)
    row_loss = all_lse - pos_lse
    has_pos = target.sum(dim=-1) > 0
    if not bool(has_pos.any()):
        return logits.new_zeros(())
    return row_loss[has_pos].mean()


def mrr_at_k(logits: torch.Tensor, target: torch.Tensor, k: int) -> torch.Tensor:
    k = min(k, logits.size(-1))
    topk = logits.topk(k, dim=-1).indices
    rel = target.gather(1, topk)
    ranks = torch.arange(1, k + 1, device=logits.device)
    return (rel / ranks).amax(dim=-1)


def energy_distance(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    mode = "donot_use_mm_for_euclid_dist"
    xy = torch.cdist(x, y, compute_mode=mode).mean()
    xx = torch.cdist(x, x, compute_mode=mode).mean()
    yy = torch.cdist(y, y, compute_mode=mode).mean()
    return (2 * xy - xx - yy).clamp(min=0.0).sqrt()


def ap_at_k(logits: torch.Tensor, target: torch.Tensor, k: int) -> torch.Tensor:
    k = min(k, logits.size(-1))
    topk = logits.topk(k, dim=-1).indices
    rel = target.gather(1, topk)
    ranks = torch.arange(1, k + 1, device=logits.device)
    prec = rel.cumsum(dim=-1) / ranks
    denom = target.sum(dim=-1).clamp(max=k).clamp(min=1.0)
    return (prec * rel).sum(dim=-1) / denom


class LitTask(pl.LightningModule):
    def __init__(self):
        super().__init__()

        self.enc = TextEncoder()
        self.style = StyleHead()
        self.emoji = EmojiHead()

    def _step(self, batch, split):
        text, emoji, style, _ = batch

        enc = self.enc(text)

        style_logits = self.style(enc)
        loss_style = lse_infonce(style_logits, style, INFONCE_TEMP)

        emoji_logits = self.emoji(enc)
        loss_emoji = lse_infonce(emoji_logits, emoji, INFONCE_TEMP)

        style_ap = ap_at_k(style_logits, style, STYLE_AP_K).mean()
        style_mrr = mrr_at_k(style_logits, style, STYLE_AP_K).mean()

        has_e = emoji.sum(dim=-1) > 0
        n_e = int(has_e.sum())
        if n_e:
            emoji_ap = ap_at_k(emoji_logits[has_e], emoji[has_e], EMOJI_AP_K).mean()
            emoji_mrr = mrr_at_k(
                emoji_logits[has_e], emoji[has_e], EMOJI_AP_K).mean()
        else:
            emoji_ap = torch.zeros((), device=emoji.device)
            emoji_mrr = torch.zeros((), device=emoji.device)

        for name, val, bs in (
            (f"loss/s/{split}", loss_style, text.size(0)),
            (f"loss/e/{split}", loss_emoji, text.size(0)),
            (f"mAP@{EMOJI_AP_K}/e/{split}", emoji_ap, max(n_e, 1)),
            (f"mAP@{STYLE_AP_K}/s/{split}", style_ap, text.size(0)),
            (f"MRR@{EMOJI_AP_K}/e/{split}", emoji_mrr, max(n_e, 1)),
            (f"MRR@{STYLE_AP_K}/s/{split}", style_mrr, text.size(0)),
        ):
            self.log(name, val, on_step=False, on_epoch=True,
                     prog_bar=True, batch_size=bs)

        return loss_style + loss_emoji

    def training_step(self, batch, batch_idx):
        return self._step(batch, "train")

    def validation_step(self, batch, batch_idx):
        self._step(batch, "val")

    def on_train_epoch_end(self):
        self._log_f1("train")

    def on_validation_epoch_end(self):
        self._log_f1("val")

    def _log_f1(self, split):
        m = self.trainer.callback_metrics
        a = m.get(f"mAP@{EMOJI_AP_K}/e/{split}")
        b = m.get(f"mAP@{STYLE_AP_K}/s/{split}")
        if a is not None and b is not None:
            self.log(f"f1/{split}", f1(a, b), prog_bar=True)

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=LR)


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
                torch.arange(
                    n, device=self.device) % self.z_bank.size(0)  # type: ignore
            ]

            fake = rgb_to_oklab(self.gen(self.enc(text), z))
            val = energy_distance(real, fake)
            self.log("energy/gan/val", val, prog_bar=True)

            gan_scalars = {"val": val, "ref": self._split_energy(real)}

            if isinstance(self.logger, TensorBoardLogger):
                w = self.logger.experiment
                w.add_scalars("energy/gan", gan_scalars, self.global_step)

    def training_step(self, batch, batch_idx):
        text, _, _, colors = batch
        opt_gen, opt_tst = self.optimizers()  # type: ignore

        cond = self._cond(text)

        fake = self.gen(cond)

        tst_real = self.tst(cond, colors)
        tst_fake = self.tst(cond, fake.detach())

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
            gradient_clip_val=GRAD_CLIP,
            gradient_clip_algorithm="norm",
        )

        opt_tst.step()

        tst_fake = self.tst(cond, fake)
        loss_gen = binary_cross_entropy_with_logits(
            tst_fake, torch.ones_like(tst_fake))

        opt_gen.zero_grad()
        self.manual_backward(loss_gen)
        self.clip_gradients(
            opt_gen,  # type: ignore
            gradient_clip_val=GRAD_CLIP,
            gradient_clip_algorithm="norm",
        )

        opt_gen.step()

        self.log("loss/gan/tst", loss_tst, prog_bar=True)
        self.log("loss/gan/gen", loss_gen, prog_bar=True)

    def configure_optimizers(self):
        opt_gen = optim.SGD(self.gen.parameters(), lr=GAN_LR)
        opt_tst = optim.SGD(self.tst.parameters(), lr=GAN_LR)

        return [opt_gen, opt_tst]


class Model(str, Enum):
    task = "task"
    gan = "gan"
    all = "all"


def _load(mod: nn.Module, path: str) -> nn.Module:
    sd, meta = load_pt(path)
    mod.load_state_dict(sd)
    mod._pt_meta = meta
    return mod


def _no_progress_bar() -> bool:
    return os.environ.get("EMOJIC_NO_PROGRESS_BAR") == "1"


def _train_task(ds) -> LitTask:
    task_dl = train_data_loader(data_set=ds, batch_size=TASK_BATCH_SIZE)
    val_dl = eval_data_loader()

    no_bar = _no_progress_bar()
    progress_bar_cbs = [] if no_bar else [TQDMProgressBar()]

    task_monitor = f"MRR@{EMOJI_AP_K}/e/val"

    task_ckpt = ModelCheckpoint(
        monitor=task_monitor, mode="max", save_top_k=1, filename="best-{step}"
    )

    task_trainer = pl.Trainer(
        devices="auto",
        accelerator="auto",
        logger=TensorBoardLogger(
            "runs", name=CONFIG_NAME, version="task", default_hp_metric=False
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

    task = LitTask()
    task_trainer.fit(task, task_dl, val_dl)

    if task_ckpt.best_model_path:
        task = LitTask.load_from_checkpoint(task_ckpt.best_model_path)

    for name, mod in (
        ("enc", task.enc),
        ("style", task.style),
        ("emoji", task.emoji),
    ):
        save_pt(mod.state_dict(), f"{name}.pt", stage="task")

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
        save_pt(mod.state_dict(), f"{name}.pt", stage="gan")

    return gan


def _run_local(model: Model) -> None:
    require_clean_tree()
    pl.seed_everything(SEED, workers=True)
    torch.backends.cudnn.benchmark = False

    if model == Model.gan:
        ds = train_ds()
        enc = _load(TextEncoder(), "enc.pt")
        _train_gan(enc, ds)  # type: ignore
        export()
        subprocess.run([sys.executable, "tools/report.py"], check=True)
        return

    ds = train_ds()
    task = _train_task(ds)

    if model == Model.task:
        subprocess.run(
            [sys.executable, "tools/report.py", "--only", "data,emoji,style"],
            check=True,
        )
        return

    _train_gan(task.enc, ds)
    export()
    subprocess.run([sys.executable, "tools/report.py"], check=True)


CPU = 16
MEMORY_MIB = 16384
TIMEOUT_S = 3600
REPO = "/repo"
VENV_PY = f"{REPO}/.venv/bin/python"
TB_PORT = 6006

WORKTREE_TAG = hashlib.sha1(str(Path.cwd().resolve()).encode()).hexdigest()[:10]
VOL_NAME = f"emojic-artifacts-{WORKTREE_TAG}"
ARTIFACTS = "/artifacts"

DEP_FILES = ["pyproject.toml", "uv.lock", ".python-version", "README.md"]
CODE_FILES = [
    "config.py",
    "data.py",
    "model.py",
    "train.py",
    "tools/report.py",
    "export_onnx.py",
    "run.py",
    "runmeta.py",
    "labels.json",
    "words.json",
    "energy_keywords.txt",
    "data.jsonl",
    "train.jsonl",
    "eval.jsonl",
]
COLLECT_GLOBS = ["*.pt"]
COLLECT_TREES = ["runs", "web/public", "report"]

modal_image = modal.Image.debian_slim(python_version="3.13").pip_install("uv")
for _name in DEP_FILES:
    modal_image = modal_image.add_local_file(_name, f"{REPO}/{_name}", copy=True)
modal_image = modal_image.run_commands(f"cd {REPO} && uv sync --frozen")
for _name in CODE_FILES:
    modal_image = modal_image.add_local_file(_name, f"{REPO}/{_name}", copy=True)

modal_app = modal.App(f"emojic-train-{WORKTREE_TAG}", image=modal_image)
vol = modal.Volume.from_name(VOL_NAME, create_if_missing=True)


def _run_env(threads: int) -> dict[str, str]:
    return {
        **os.environ,
        "PYTHONUNBUFFERED": "1",
        "EMOJIC_NO_PROGRESS_BAR": "1",
        "OMP_NUM_THREADS": str(threads),
        "MKL_NUM_THREADS": str(threads),
        "OPENBLAS_NUM_THREADS": str(threads),
        "NUMEXPR_NUM_THREADS": str(threads),
    }


def _stash(dst: str) -> int:
    root, out = Path(REPO), Path(dst)
    n = 0
    for pattern in COLLECT_GLOBS:
        for path in root.glob(pattern):
            if path.is_file():
                shutil.copy2(path, out / path.name)
                n += 1
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


@modal_app.function(cpu=CPU, memory=MEMORY_MIB, timeout=TIMEOUT_S, volumes={ARTIFACTS: vol})
def train_remote(
    model: str, threads: int, git_sha: str, enc_bytes: bytes | None = None
) -> dict[str, int]:
    env = _run_env(threads)
    env["EMOJIC_GIT_SHA"] = git_sha
    env["EMOJIC_DISPATCH_CHECKED"] = "1"
    if enc_bytes is not None:
        Path(REPO, "enc.pt").write_bytes(enc_bytes)
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
                    [VENV_PY, "train.py", model, "--local"],
                    cwd=REPO,
                    env=env,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                )
                for line in proc.stdout:
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


def _retrieve_and_cleanup() -> None:
    staging = Path(tempfile.mkdtemp(prefix="emojic-modal-"))
    try:
        got = _modal("volume", "get", "--force", VOL_NAME, "/", str(staging))
        if got.returncode != 0:
            print(f"volume get failed, leaving {VOL_NAME} intact:")
            print(got.stdout, got.stderr)
            return

        src = staging
        kids = list(staging.iterdir())
        if (
            len(kids) == 1
            and kids[0].is_dir()
            and kids[0].name
            not in {
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
            return

        rm = _modal("volume", "delete", "-y", VOL_NAME)
        print(
            f"deleted volume {VOL_NAME}"
            if rm.returncode == 0
            else f"volume cleanup failed:\n{rm.stderr}"
        )
    finally:
        shutil.rmtree(staging, ignore_errors=True)


def _run_remote(model: Model, cpu: int, memory: int, git_sha: str) -> dict[str, int]:
    enc_bytes = None
    if model == Model.gan:
        enc_path = Path("enc.pt")
        if not enc_path.exists():
            raise typer.BadParameter(
                "enc.pt not found -- run `train task --local` "
                "(or fetch a Modal task run) first"
            )
        enc_bytes = enc_path.read_bytes()

    fn = train_remote
    if cpu != CPU or memory != MEMORY_MIB:
        fn = train_remote.with_options(cpu=cpu, memory=memory)
    return fn.remote(model=model.value, threads=cpu, git_sha=git_sha, enc_bytes=enc_bytes)


def _dispatch(
    model: Model, cpu: int, memory: int, fetch_only: bool, need_app_ctx: bool
) -> None:
    if fetch_only:
        _retrieve_and_cleanup()
        return

    require_clean_tree()
    git_sha = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True
    ).stdout.strip()
    try:
        if need_app_ctx:
            with modal_app.run():
                print(_run_remote(model, cpu, memory, git_sha))
        else:
            print(_run_remote(model, cpu, memory, git_sha))
    finally:
        _retrieve_and_cleanup()


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
    memory: int | None = typer.Option(None, help="Modal memory in MiB (Modal only)."),
) -> None:
    """Train task/gan/all: on Modal by default, or locally with --local."""
    if local and (cpu is not None or memory is not None):
        raise typer.BadParameter("--cpu/--memory only apply when dispatching to Modal")

    if local:
        _run_local(model)
    else:
        _dispatch(model, cpu or CPU, memory or MEMORY_MIB, fetch_only=False, need_app_ctx=True)


@modal_app.local_entrypoint()
def main(
    model: str, cpu: int = CPU, memory: int = MEMORY_MIB, fetch_only: bool = False
) -> None:
    _dispatch(Model(model), cpu, memory, fetch_only, need_app_ctx=False)


if __name__ == "__main__":
    _app()
```

- [ ] **Step 2: Format and lint**

Run: `uv run ruff format .` then `uv run ruff check .`
Expected: both clean (fix any `E`/`F`/`I`/`UP`/`B` findings ruff reports; do not silence with `# noqa` unless a finding is a false positive).

- [ ] **Step 3: Verify the module imports and the CLI surface**

Run: `uv run python -c "import train"`
Expected: no error (proves the Modal scaffolding and all imports are import-time safe without live Modal credentials).

Run: `uv run python train.py --help`
Expected: shows `Usage: train.py [OPTIONS] {task|gan|all}` (or equivalent), lists `--local`, `--cpu`, `--memory`, and (since `add_completion` is left at its Typer default of `True`) `--install-completion` / `--show-completion`.

Run: `uv run python train.py task --help`
Expected: succeeds, shows the same options scoped to the `task` choice (Typer/Click renders per-command help identically here since there is one command).

- [ ] **Step 4: Finish the console-script install from Task 1**

Run: `uv sync` then `uv run train --help`
Expected: identical output to `uv run python train.py --help` — confirms the `[project.scripts]` entry point resolves.

- [ ] **Step 5: Commit**

```bash
git add train.py
git commit -m "feat: unify train.py/train_gan.py/train-modal.py into one Typer CLI"
```

---

### Task 3: Delete the superseded scripts

**Files:**
- Delete: `train_gan.py`
- Delete: `train-modal.py`
- Delete: `loop_emoji.py`
- Modify: `pyproject.toml` if either deleted file is referenced anywhere else in it (check first — expected: not referenced)

**Interfaces:**
- Consumes: nothing new.
- Produces: nothing new; this task only removes files now fully superseded by `train.py`.

- [ ] **Step 1: Confirm nothing else still imports the files being deleted**

Run: `grep -rn "train_gan\|train-modal\|loop_emoji" --include="*.py" .`
Expected: only self-references inside `train_gan.py`/`train-modal.py`/`loop_emoji.py` themselves (which are about to be deleted).

- [ ] **Step 2: Delete the files**

```bash
git rm train_gan.py train-modal.py loop_emoji.py
```

- [ ] **Step 3: Re-run the lint/format gate**

Run: `uv run ruff check .` and `uv run ruff format --check .`
Expected: both clean (no leftover references, no stale `# noqa` etc. tied to the deleted files).

- [ ] **Step 4: Commit**

```bash
git commit -m "chore: remove train_gan.py, train-modal.py, loop_emoji.py (superseded by train.py)"
```

---

### Task 4: Repoint Taskfile.yml at train.py

**Files:**
- Modify: `Taskfile.yml`

**Interfaces:**
- Consumes: `train.py`'s `main` local entrypoint (`modal_app.local_entrypoint()`, signature `main(model: str, cpu: int = CPU, memory: int = MEMORY_MIB, fetch_only: bool = False)`) from Task 2, reachable via `modal run train.py::main`.

- [ ] **Step 1: Update `modal-train`**

In the `modal-train` task, change:
```
        for f in train.jsonl eval.jsonl labels.json words.json; do
```
(unchanged), and change:
```
        setsid --fork sh -c "cd '$wt' && exec uv run modal run --detach train-modal.py::main" >"$log" 2>&1 </dev/null
```
to:
```
        setsid --fork sh -c "cd '$wt' && exec uv run modal run --detach train.py::main -- --model task" >"$log" 2>&1 </dev/null
```

(`--model task` preserves this shortcut's existing behavior — a detached task-stage-only Modal run; it never trained gan/all before.)

- [ ] **Step 2: Update `modal-train:fetch`**

Change:
```
          [ -f "${wt}train-modal.py" ] || continue
```
to:
```
          [ -f "${wt}train.py" ] || continue
```

Change:
```
          ( cd "$wt" && uv run modal run train-modal.py::fetch ) || echo "  fetch failed for $tag (continuing)"
```
to:
```
          ( cd "$wt" && uv run modal run train.py::main -- --model task --fetch-only ) || echo "  fetch failed for $tag (continuing)"
```

(This also fixes a pre-existing dangling reference — `train-modal.py` never defined a `fetch` function, so this task's `modal run train-modal.py::fetch` could never have worked. `--fetch-only` short-circuits before `model` is used, so `--model task` here is a required-but-inert placeholder.)

- [ ] **Step 3: Leave `modal-train:list`, `modal-train:reports`, `modal-train:drop` as-is**

They don't reference `train-modal.py` — confirm with:
Run: `grep -n "train-modal" Taskfile.yml`
Expected: no matches after Steps 1-2.

- [ ] **Step 4: Commit**

```bash
git add Taskfile.yml
git commit -m "chore: repoint Taskfile.yml modal-train tasks at train.py"
```

---

### Task 5: Update CLAUDE.md

**Files:**
- Modify: `CLAUDE.md`

**Interfaces:**
- Consumes: nothing (documentation only).

- [ ] **Step 1: Replace the `train.py` / `train_gan.py` / `train-modal.py` bullets and the "free mix" paragraph**

Find this block (four consecutive bullets, right after the intro paragraph, before the `tools/report.py` bullet):

```
- `train.py` — `LitTask` (encoder + style + emoji heads, both trained with `train.py:lse_infonce` Log-Sum-Exp InfoNCE; early-stop/checkpoint via `ModelCheckpoint`/`EarlyStopping` on `MRR@10/e/val` — emoji top-10 mean reciprocal rank from the hand-rolled `train.py:mrr_at_k`. Both `train.py:ap_at_k` (mAP) and `mrr_at_k` (MRR) are logged for each head at `EMOJI_AP_K`/`STYLE_AP_K` (10/5) from `config.py`, plus `f1/val` = harmonic mean of the two mAPs — no `torchmetrics`; emoji AP/MRR are measured only over rows carrying ≥1 emoji. **Planned:** also logging `acc@1`/`acc@10` (emoji) and `acc@1`/`acc@5` (style) on val — the stated priority metric set — is not wired into `train.py` yet; `acc@k` today lives only in the `tools/report.py` keyword probe) and `LitColorGAN` (color GAN on a **frozen** encoder; conditions on the encoder's text embedding, does not touch the classifier heads; GAN stage early-stop/checkpoint on `energy/gan/val`, min). `uv run python train.py` runs **both** stages back-to-back, reloads the best task checkpoint before the GAN stage, saves five loose state dicts — `enc.pt` / `style.pt` / `emoji.pt` after the task stage, `gen.pt` / `tst.pt` after the GAN stage (all gitignored) — then shells out to `export_onnx.py` to refresh `web/public/` and to `tools/report.py` to write the run report.
- `train_gan.py` — retrains just the GAN stage against a pre-saved `enc.pt`, re-saving `gen.pt` / `tst.pt`, then calls `export_onnx.py:export()` and shells out to `tools/report.py`.
- `train-modal.py` — `uv run modal run train-modal.py [--cpu N] [--memory MIB]` runs **only the task stage** of `train.py` on a Modal CPU box (it sets `EMOJIC_TASK_ONLY=1`, so `train.py` `sys.exit(0)`s after saving `enc.pt` / `style.pt` / `emoji.pt` — no GAN stage, no ONNX export), then always runs `tools/report.py --only data,emoji,style` against the fresh `.pt` (no `gen.pt` on the box, so the Colors section is skipped; its `report/<ts>-<sha>/` output is collected back), forwards a TensorBoard tunnel, and copies the artifacts back (`*.pt`, `runs/`, `report/`; `web/public/` is not regenerated). Code + data files are `add_local_file`d at image-build time, so run `bun run regen` **locally first** — Modal does not regen. To finish a Modal task run: re-run `train_gan.py` locally against the returned `enc.pt` to produce `gen.pt` / `tst.pt`, refresh `web/public/`, and write the full report. `train-modal.py`'s local entrypoint now aborts on a dirty git tree and forwards `EMOJIC_GIT_SHA` / `EMOJIC_DISPATCH_CHECKED` so Modal-trained `.pt` files carry the dispatch commit SHA.
- **Training location is a free mix.** Usual workflow: task stage on Modal (`train-modal.py`), GAN stage local (`train_gan.py` against the returned `enc.pt`). Full local is `train.py` (both stages). The entry points compose: `train.py` = both stages here, `train-modal.py` = **task stage only** on Modal, `train_gan.py` = GAN-only here. There is no full-Modal path — the GAN stage always runs locally.
```

Replace it with:

```
- `train.py` — single Typer CLI, `train <task|gan|all> [--local] [--cpu N] [--memory MIB]`, installed as a `train` console script (`[project.scripts]` in `pyproject.toml`; also runnable as `uv run python train.py ...` or `uv run train ...`) with free shell-completion (`train --install-completion`). `model` picks which stage(s) run: `task` (encoder + `LitTask`'s style/emoji heads, both trained with `train.py:lse_infonce` Log-Sum-Exp InfoNCE; early-stop/checkpoint via `ModelCheckpoint`/`EarlyStopping` on `MRR@10/e/val` — emoji top-10 mean reciprocal rank from the hand-rolled `train.py:mrr_at_k`. Both `train.py:ap_at_k` (mAP) and `mrr_at_k` (MRR) are logged for each head at `EMOJI_AP_K`/`STYLE_AP_K` (10/5) from `config.py`, plus `f1/val` = harmonic mean of the two mAPs — no `torchmetrics`; emoji AP/MRR are measured only over rows carrying ≥1 emoji. **Planned:** also logging `acc@1`/`acc@10` (emoji) and `acc@1`/`acc@5` (style) on val — the stated priority metric set — is not wired into `train.py` yet; `acc@k` today lives only in the `tools/report.py` keyword probe); `gan` (`LitColorGAN` — color GAN on a **frozen** encoder; conditions on the encoder's text embedding, does not touch the classifier heads; GAN stage early-stop/checkpoint on `energy/gan/val`, min — trained against a pre-saved `enc.pt`); `all` (task then gan, back to back, reloading the best task checkpoint before the GAN stage — the old bare `train.py`'s behavior). Location defaults to Modal (same machinery as the old `train-modal.py`: image build, artifact volume, TensorBoard tunnel, `EMOJIC_GIT_SHA`/`EMOJIC_DISPATCH_CHECKED` so Modal-trained `.pt` files carry the dispatch commit SHA); pass `--local` to run on this machine instead (`--cpu`/`--memory` then don't apply and are rejected). `task`/`all` train their own encoder wherever they run; `gan` needs a pre-trained `enc.pt` — locally it must already be on disk, and on Modal the local `enc.pt` bytes are uploaded into the remote container before training (so a from-scratch Modal `gan` run needs a prior `train task --local`, or `train task`'s artifacts already fetched back). After training, `task` alone runs `tools/report.py --only data,emoji,style` (no `gen.pt` yet, so Colors is skipped); `gan` and `all` refresh `web/public/` via `export_onnx.py` and run the full `tools/report.py`. `uv run modal run train.py::main -- --model <task|gan|all> [--cpu N] [--memory MIB] [--fetch-only]` reaches the same Modal dispatch directly through the Modal CLI — used by `Taskfile.yml`'s detached `modal-train` task, since that needs Modal's own `--detach` semantics rather than a plain blocking call; `--fetch-only` recovers artifacts from an interrupted run's volume without starting a new one. Code + data files are `add_local_file`d into the Modal image at build time, so run `bun run regen` **locally first** — Modal does not regen. Aborts on a dirty git tree in every mode (`runmeta.require_clean_tree`, no override).
```

- [ ] **Step 2: Delete the `loop_emoji.py` bullet**

Remove this line entirely (it sat right after the `run.py` bullet):

```
- `loop_emoji.py` — `uv run python loop_emoji.py [--iterations 5] [--target 0.95] [--rank 5] [--cpu N] [--memory MIB]`: the emoji train → report → upsample loop. Each iteration runs `train-modal.py`, then `tools/report.py --only emoji` and reads the newest `report/<ts>-<sha>/report.json`; if the `words.json` keyword `acc@5 ≥ --target` it stops, else it runs `bun run tools/data/upsample-emoji-test.ts --rank <rank>` (which reads that same `report.json`'s `emoji.keywords.words` and appends to `data.jsonl`) then `bun run regen`. On success it trains the color GAN locally (`train_gan.py`); on failure it prints the missed-keyword list. **Planned:** a matching color loop is not built yet.
```

- [ ] **Step 3: Fix the `tools/report.py` bullet's cross-reference**

Find:
```
Auto-runs at the end of `train.py` / `train_gan.py`; re-run it by hand after `bun run regen`.
```
Replace with:
```
Auto-runs at the end of every `train.py <task|gan|all> --local` run (and inside the Modal container for `task`/`gan`/`all` dispatched to Modal); re-run it by hand after `bun run regen`.
```

- [ ] **Step 4: Fix the "Environment & commands" bullet about `bun run regen` ordering**

Find:
```
- `train.jsonl` / `eval.jsonl` / `labels.json` are gitignored and produced by `bun run regen` from `data.jsonl`. A fresh checkout must run `bun run regen` **before** `uv run python train.py` / `run.py` / `export_onnx.py` / `train-modal.py` / `tools/report.py` / `loop_emoji.py` (they import `config`, which loads `labels.json`; `tools/report.py` and `loop_emoji.py` also read `train.jsonl`; `train-modal.py` `add_local_file`s `train.jsonl` / `eval.jsonl` / `labels.json` / `data.jsonl` / `words.json` / `energy_keywords.txt` at import) and before `npm test` in `web/` (`feelings.test.js` reads `labels.json`). The Pages deploy is unaffected — it builds from the committed `web/public/`.
```
Replace with:
```
- `train.jsonl` / `eval.jsonl` / `labels.json` are gitignored and produced by `bun run regen` from `data.jsonl`. A fresh checkout must run `bun run regen` **before** `uv run python train.py` / `run.py` / `export_onnx.py` / `tools/report.py` (they import `config`, which loads `labels.json`; `tools/report.py` also reads `train.jsonl`; `train.py`'s Modal path `add_local_file`s `train.jsonl` / `eval.jsonl` / `labels.json` / `data.jsonl` / `words.json` / `energy_keywords.txt` at import) and before `npm test` in `web/` (`feelings.test.js` reads `labels.json`). The Pages deploy is unaffected — it builds from the committed `web/public/`.
```

- [ ] **Step 5: Fix the "Full run" and "abort on dirty tree" bullets**

Find:
```
- Full run: `uv run python train.py` trains the task stage then the GAN stage (see `EPOCHS_TASK` / `EPOCHS_GAN` in `config.py`), writes `enc.pt` / `style.pt` / `emoji.pt` / `gen.pt` / `tst.pt`, and refreshes `web/public/` via `export_onnx.py`. Slow — don't use it as a smoke test. `train-modal.py` runs the task stage only on Modal (`enc.pt` / `style.pt` / `emoji.pt`, no GAN, no export); `train_gan.py` retrains only the GAN. `onnx` + `onnxscript` (dev deps) are required by `export_onnx.py`. `train.py` / `train_gan.py` auto-run `tools/report.py` at the end; re-run `uv run python tools/report.py` by hand after `bun run regen` or a Modal fetch to refresh `report/<ts>-<sha>/`.
- `uv run python train.py` / `train_gan.py` and `uv run modal run train-modal.py` abort immediately on an unclean git tree (`runmeta.require_clean_tree`, no override) — commit or stash first.
```
Replace with:
```
- Full run: `train all --local` (or `uv run python train.py all --local`) trains the task stage then the GAN stage (see `EPOCHS_TASK` / `EPOCHS_GAN` in `config.py`), writes `enc.pt` / `style.pt` / `emoji.pt` / `gen.pt` / `tst.pt`, and refreshes `web/public/` via `export_onnx.py`. Slow — don't use it as a smoke test. `train task --local` runs the task stage only, locally (`enc.pt` / `style.pt` / `emoji.pt`, no GAN, no export); `train gan --local` retrains only the GAN against a pre-saved `enc.pt`. Drop `--local` on any of these to dispatch the same stage(s) to a Modal CPU box instead (`--cpu`/`--memory` tune the box; `gan` on Modal uploads the local `enc.pt` into the container first). `onnx` + `onnxscript` (dev deps) are required by `export_onnx.py`. Every `train.py` run auto-runs `tools/report.py` at the end; re-run `uv run python tools/report.py` by hand after `bun run regen` or a Modal fetch to refresh `report/<ts>-<sha>/`.
- `train.py` (every mode, local or Modal) aborts immediately on an unclean git tree (`runmeta.require_clean_tree`, no override) — commit or stash first.
```

- [ ] **Step 6: Fix the repo-root script list**

Find:
```
- Non-model scripts live under `tools/data/`; the model stack — `config.py`, `data.py`, `model.py`, `train.py`, `train_gan.py`, `train-modal.py`, `run.py`, `export_onnx.py`, `loop_emoji.py`, `runmeta.py` — stays at the repo root. `tools/report.py` is the exception: it lives under `tools/` but imports the model stack (it prepends the repo root to `sys.path`). `test_model.py` is stale (imports a removed `Model` class). All tool scripts assume the repo root as CWD.
```
Replace with:
```
- Non-model scripts live under `tools/data/`; the model stack — `config.py`, `data.py`, `model.py`, `train.py`, `run.py`, `export_onnx.py`, `runmeta.py` — stays at the repo root. `tools/report.py` is the exception: it lives under `tools/` but imports the model stack (it prepends the repo root to `sys.path`). `test_model.py` is stale (imports a removed `Model` class). All tool scripts assume the repo root as CWD.
```

- [ ] **Step 7: Fix the "Conventions" section's verification bullet**

Find:
```
Behavioral verification is a full `uv run python train.py` (watch `MRR@10/e/val` — the early-stop/checkpoint metric — alongside `mAP@10/e/val` / `MRR@5/s/val` / `mAP@5/s/val` and `f1/val` in TensorBoard, and `energy/gan/val` vs. `energy/gan/ref` for the GAN stage; `enc.pt` etc. are overwritten even when a run is worse — trust the logs, not the files) plus `uv run python run.py` for a `pred.jsonl` spot-check (threshold-based `top_labels`: styles 1–3, emojis exactly 1) and `uv run python tools/report.py` for the full run report (`report/<ts>-<sha>/report.html` — emoji/style `acc@k`, `words.json` keyword probe, OKLab colour energy + per-colour hit-rate). Grow the corpus with `bun run train` and/or `bun run upsample-emojis` / `bun run upsample-emoji-test` / `bun run upsample-colors`, then `bun run regen`.
```
Replace with:
```
Behavioral verification is a full `train all --local` (watch `MRR@10/e/val` — the early-stop/checkpoint metric — alongside `mAP@10/e/val` / `MRR@5/s/val` / `mAP@5/s/val` and `f1/val` in TensorBoard, and `energy/gan/val` vs. `energy/gan/ref` for the GAN stage; `enc.pt` etc. are overwritten even when a run is worse — trust the logs, not the files) plus `uv run python run.py` for a `pred.jsonl` spot-check (threshold-based `top_labels`: styles 1–3, emojis exactly 1) and `uv run python tools/report.py` for the full run report (`report/<ts>-<sha>/report.html` — emoji/style `acc@k`, `words.json` keyword probe, OKLab colour energy + per-colour hit-rate). Grow the corpus with `bun run train` and/or `bun run upsample-emojis` / `bun run upsample-emoji-test` / `bun run upsample-colors`, then `bun run regen`.
```

- [ ] **Step 8: Grep for anything missed**

Run: `grep -n "train_gan\|train-modal\|loop_emoji\|EMOJIC_TASK_ONLY" CLAUDE.md`
Expected: no matches.

- [ ] **Step 9: Commit**

```bash
git add CLAUDE.md
git commit -m "docs: update CLAUDE.md for the unified train.py CLI"
```

---

### Task 6: Final verification pass

**Files:**
- None (verification only).

- [ ] **Step 1: Full lint/format gate**

Run: `uv run ruff check .` and `uv run ruff format --check .`
Expected: both clean.

- [ ] **Step 2: Python assertion suite**

Run: `uv run python test_runmeta.py`
Expected: prints `ok`.

- [ ] **Step 3: CLI surface, all three ways of invoking it**

Run: `uv run python train.py --help`, `uv run train --help`, `train --help` (if the venv is activated in the shell)
Expected: identical usage text in all three, listing `task|gan|all`, `--local`, `--cpu`, `--memory`, `--install-completion`, `--show-completion`.

- [ ] **Step 4: Modal local_entrypoint is discoverable**

Run: `uv run modal run train.py::main --help`
Expected: Modal's own generated help for `main`, listing `--model`, `--cpu`, `--memory`, `--fetch-only` (proves `Taskfile.yml`'s `modal run train.py::main -- ...` calls from Task 4 target a real, well-formed entrypoint).

- [ ] **Step 5: web/ test suite still passes (unaffected, but confirms `labels.json` / `meta.json` contracts weren't touched)**

Run: `cd web && npm test`
Expected: passes, same as before this change (only run if `bun run regen` has been run at least once so `labels.json` exists — if it errors solely on a missing `labels.json`, that's a pre-existing fresh-checkout requirement, not a regression from this plan; note it and move on).

- [ ] **Step 6: Confirm a clean tree**

Run: `git status`
Expected: clean (everything committed task-by-task above).

## Self-Review

**Spec coverage:**
- "`train <model> [--local]`, model one of task/gan/all" → Task 2's `cli()` command. ✓
- "default is Modal, `--local` for local machine" → Task 2's `cli()` local/`_dispatch` branch. ✓
- "remove all other training scripts and tools" (scoped by clarification to train_gan.py/train-modal.py/loop_emoji.py) → Task 3. ✓
- "Update CLAUDE.md" → Task 5. ✓
- "Use typer and install in the current project so I can run `train task --local` directly" → Task 1 (`[project.scripts]`) + Task 2 (`_app`). ✓
- "Add commandline autocomplete if possible (free with typer)" → Task 2 leaves `add_completion` at Typer's default `True` (every other script in the repo explicitly sets it `False` — this one deliberately doesn't) and Task 6 Step 3 verifies `--install-completion`/`--show-completion` are present. ✓
- Clarification: real Modal support for `gan` → Task 2's `_run_remote` reads and uploads local `enc.pt` bytes for `Model.gan`. ✓
- Clarification: delete `loop_emoji.py`, don't reimplement its loop → Task 3 deletes it; no loop logic appears anywhere in the new `train.py`. ✓
- Taskfile.yml, which referenced the deleted scripts, isn't explicitly named in the user's spec but breaks without a fix → Task 4 (in scope: nothing else references the deleted files per Task 3 Step 1's grep).

**Placeholder scan:** no TBD/TODO, no "add error handling", no "similar to Task N" — every step has literal file content or a literal command.

**Type consistency:** `Model` enum (`task`/`gan`/`all`) is defined once in Task 2 and used identically by `cli()`, `_run_local`, `_run_remote`, `_dispatch`, and `main()` (`Model(model)` conversion). `train_remote`'s signature (`model: str, threads: int, git_sha: str, enc_bytes: bytes | None`) matches exactly between its definition and both callers (`_run_remote`'s `fn.remote(...)` call — the only caller). `_dispatch`'s signature matches both call sites (`cli()` and `main()`).

---

**Plan complete and saved to `docs/superpowers/plans/2026-09-04-unified-train-cli.md`.** Two execution options:

**1. Subagent-Driven (recommended)** — I dispatch a fresh subagent per task, review between tasks, fast iteration

**2. Inline Execution** — Execute tasks in this session using executing-plans, batch execution with checkpoints

**Which approach?**
