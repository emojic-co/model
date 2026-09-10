import hashlib
import os
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime
from enum import StrEnum
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

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

from files import (
    CRITIC_PT,
    DATA_JSONL,
    EMOJI_PT,
    ENC_PT,
    ENERGY_KEYWORDS_TXT,
    EVAL_JSONL,
    FUSION_PT,
    KEYWORDS_JSON,
    LABELS_JSON,
    MODEL_DIR,
    PT_DIR,
    STYLE_PT,
    TOOLS_DIR,
    TRAIN_JSONL,
)
from model.color import rgb_to_oklab
from model.config import (
    CONFIG_NAME,
    EARLY_STOP_PATIENCE,
    ENERGY_WEIGHT,
    ENERGY_Z_SAMPLES,
    EPOCHS_GAN,
    EPOCHS_TASK,
    GAN_BATCH_SIZE,
    GAN_CRITIC_LR,
    GAN_GEN_LR,
    GAN_GEN_MARGIN,
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
    eval_data_loader,
    scatter_flex,
    train_data_loader,
    train_ds,
)
from model.export_onnx import export
from model.model import (
    ColorCritic,
    ColorGen,
    EmojiHead,
    FusionHead,
    StyleHead,
    TextEncoder,
)
from model.runmeta import load_pt, require_clean_tree, save_pt

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


def harmonic_mean(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    s = a + b
    return torch.where(s > 0, 2 * a * b / s, torch.zeros_like(s))


def roc_auc(pos: torch.Tensor, neg: torch.Tensor) -> torch.Tensor:
    if pos.numel() == 0 or neg.numel() == 0:
        return pos.new_zeros(())
    diff = pos[:, None] - neg[None, :]
    wins = torch.sign(diff).clamp(min=0.0)
    ties = (diff == 0).to(diff.dtype)
    return (wins + 0.5 * ties).mean()


AUC_MAX_SAMPLES = 4096


def _cap(x: torch.Tensor) -> torch.Tensor:
    if x.numel() <= AUC_MAX_SAMPLES:
        return x
    idx = torch.randperm(x.numel(), device=x.device)[:AUC_MAX_SAMPLES]
    return x[idx]


def buffered_auc(pos: list[torch.Tensor], neg: list[torch.Tensor]) -> torch.Tensor:
    return roc_auc(_cap(torch.cat(pos)), _cap(torch.cat(neg)))


def energy_distance(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    mode = "donot_use_mm_for_euclid_dist"
    xy = torch.cdist(x, y, compute_mode=mode).mean()
    xx = torch.cdist(x, x, compute_mode=mode).mean()
    yy = torch.cdist(y, y, compute_mode=mode).mean()
    return (2 * xy - xx - yy).clamp(min=0.0).sqrt()


ALL_HEADS: tuple[str, ...] = ("style", "emoji", "critic", "fusion")
_DEFAULT_PT = Path(PT_DIR)


class Stage(StrEnum):
    enc = "enc"
    gan = "gan"


def _parse_heads(csv: str | None) -> tuple[str, ...]:
    if csv is None:
        return ALL_HEADS
    got = {tok.strip() for tok in csv.split(",") if tok.strip()}
    bad = got - set(ALL_HEADS)
    if bad or not got:
        raise typer.BadParameter(
            f"--heads: {', '.join(sorted(bad)) or 'empty'} "
            f"(choose from {', '.join(ALL_HEADS)})"
        )
    if "fusion" in got and "emoji" not in got:
        raise typer.BadParameter("--heads: fusion requires emoji")
    return tuple(h for h in ALL_HEADS if h in got)


def _validate(
    stage: Stage | None,
    local: bool,
    heads: str | None,
    pt: Path,
    out: Path,
    gpu: str,
    cpu: bool,
) -> tuple[str, ...] | None:
    if heads is not None and stage != Stage.enc:
        raise typer.BadParameter("--heads is only valid with the 'enc' stage")
    if not local and (pt != _DEFAULT_PT or out != _DEFAULT_PT):
        raise typer.BadParameter(
            "--pt / -o must be the default (pt/) unless --local is set"
        )
    if gpu and cpu:
        raise typer.BadParameter("--gpu and --cpu are mutually exclusive")
    if gpu and local:
        raise typer.BadParameter(
            "--gpu picks a Modal GPU and can't be combined with --local"
        )
    return _parse_heads(heads) if stage == Stage.enc else None


class LitEncoder(pl.LightningModule):
    def __init__(self, heads: tuple[str, ...] = ALL_HEADS):
        super().__init__()
        self.save_hyperparameters()
        self.heads = tuple(heads)

        self.enc = TextEncoder()
        if "style" in self.heads:
            self.style = StyleHead()
        if "emoji" in self.heads:
            self.emoji = EmojiHead()
        if "critic" in self.heads:
            self.critic = ColorCritic()
        if "fusion" in self.heads:
            self.fusion = FusionHead()

        self._val_pos: list[torch.Tensor] = []
        self._val_neg: list[torch.Tensor] = []
        self._trn_pos: list[torch.Tensor] = []
        self._trn_neg: list[torch.Tensor] = []
        self._val_rr: list[torch.Tensor] = []
        self._trn_rr: list[torch.Tensor] = []

    def _log(self, name, val, bs):
        self.log(name, val, on_step=False, on_epoch=True,
                 prog_bar=True, batch_size=bs)

    def _step(self, batch, split):
        text, emoji, style, colors, flex_idx, flex_raw, flexq = batch
        enc = self.enc(text)
        loss = enc.new_zeros(())
        bs = text.size(0)

        if "style" in self.heads:
            style_logits = self.style(enc)
            loss_style = lse_infonce(style_logits, style, INFONCE_TEMP)
            loss = loss + loss_style
            self._log(f"loss/s/{split}", loss_style, bs)
            self._log(f"MRR/s/{split}", mrr(style_logits, style).mean(), bs)

        if "emoji" in self.heads:
            emoji_logits = self.emoji(enc)
            loss_emoji = lse_infonce(emoji_logits, emoji, INFONCE_TEMP)
            loss = loss + loss_emoji
            self._log(f"loss/e/{split}", loss_emoji, bs)
            has_e = emoji.sum(dim=-1) > 0
            n_e = int(has_e.sum())
            if n_e:
                rr = mrr(emoji_logits[has_e], emoji[has_e])
                emoji_mrr = rr.mean()
                buf = self._val_rr if split == "val" else self._trn_rr
                buf.append(rr.detach())
            else:
                emoji_mrr = torch.zeros((), device=emoji.device)
            self._log(f"MRR/e/{split}", emoji_mrr, max(n_e, 1))

        if "fusion" in self.heads:
            flex_dense = scatter_flex(flex_idx, flex_raw)
            fusion_logits = self.fusion(emoji_logits.detach(), flex_dense, flexq)
            loss_fusion = lse_infonce(fusion_logits, emoji, INFONCE_TEMP)
            loss = loss + loss_fusion
            self._log(f"loss/fusion/{split}", loss_fusion, bs)

            flex_score = flex_dense[..., 0]
            flex_logits = torch.where(
                flex_score > 0, flex_score, flex_score.new_full((), -1e9)
            )
            if n_e:
                frr = mrr(fusion_logits[has_e], emoji[has_e]).mean()
                xrr = mrr(flex_logits[has_e], emoji[has_e]).mean()
            else:
                frr = torch.zeros((), device=emoji.device)
                xrr = torch.zeros((), device=emoji.device)
            self._log(f"MRR/fusion/{split}", frr, max(n_e, 1))
            self._log(f"MRR/flex/{split}", xrr, max(n_e, 1))

        if "critic" in self.heads:
            shift = 1 if split == "val" else int(torch.randint(1, bs, (1,)).item())
            neg_colors = colors.roll(shift, dims=0)
            pos = self.critic(enc, colors)
            neg = self.critic(enc, neg_colors)
            loss_critic = binary_cross_entropy_with_logits(
                pos, torch.ones_like(pos)
            ) + binary_cross_entropy_with_logits(neg, torch.zeros_like(neg))
            loss = loss + loss_critic
            self._log(f"loss/critic/{split}", loss_critic, bs)
            pos_buf, neg_buf = (
                (self._val_pos, self._val_neg)
                if split == "val"
                else (self._trn_pos, self._trn_neg)
            )
            pos_buf.append(pos.detach().flatten())
            neg_buf.append(neg.detach().flatten())

        return loss

    def on_validation_epoch_start(self):
        self._val_pos.clear()
        self._val_neg.clear()
        self._val_rr.clear()

    def on_validation_epoch_end(self):
        self._epoch_metrics("val", self._val_pos, self._val_neg, self._val_rr)

    def on_train_epoch_start(self):
        self._trn_pos.clear()
        self._trn_neg.clear()
        self._trn_rr.clear()

    def on_train_epoch_end(self):
        self._epoch_metrics("train", self._trn_pos, self._trn_neg, self._trn_rr)

    def _epoch_metrics(self, split, pos_buf, neg_buf, rr_buf):
        auc = None
        if "critic" in self.heads and pos_buf:
            auc = buffered_auc(pos_buf, neg_buf)
            self.log(f"auc/critic/{split}", auc, prog_bar=True)
        if auc is not None and rr_buf:
            emoji_mrr = torch.cat(rr_buf).mean()
            self.log(f"F1/{split}", harmonic_mean(emoji_mrr, auc), prog_bar=True)

    def training_step(self, batch, batch_idx):
        return self._step(batch, "train")

    def validation_step(self, batch, batch_idx):
        self._step(batch, "val")

    def configure_optimizers(self):
        params = list(self.enc.parameters())
        for h in self.heads:
            params += list(getattr(self, h).parameters())
        return optim.Adam(params, lr=LR)


class LitColorGAN(pl.LightningModule):
    def __init__(self, enc: TextEncoder, critic: ColorCritic):
        super().__init__()

        self.enc = enc.requires_grad_(False).eval()

        self.gen = ColorGen()
        self.tst = critic

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
        text, _, _, colors, *_ = batch
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
        text, _, _, colors, *_ = batch
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

        gen_real, gen_fake = self.tst(
            torch.cat([cond, cond], dim=0),
            torch.cat([colors, fake], dim=0),
        ).chunk(2, dim=0)
        hinge = torch.relu(
            gen_real.detach() - gen_fake + GAN_GEN_MARGIN
        ).mean()
        energy = energy_distance(rgb_to_oklab(fake), rgb_to_oklab(colors))
        loss_gen = hinge + ENERGY_WEIGHT * energy

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
        self.log("energy/gan/train", energy, prog_bar=True)

    def configure_optimizers(self):
        opt_gen = optim.SGD(self.gen.parameters(), lr=GAN_GEN_LR)
        opt_tst = optim.SGD(self.tst.parameters(), lr=GAN_CRITIC_LR)

        return [opt_gen, opt_tst]


def _load(mod: nn.Module, path: str) -> nn.Module:
    sd, meta = load_pt(path)
    mod.load_state_dict(sd)
    mod._pt_meta = meta  # type: ignore
    return mod


def _no_progress_bar() -> bool:
    return os.environ.get("EMOJIC_NO_PROGRESS_BAR") == "1"


def _require_pt(folder: Path, names: list[str]) -> None:
    missing = [n for n in names if not (folder / n).exists()]
    if missing:
        raise typer.BadParameter(f"{folder}: missing {', '.join(missing)}")


def _train_encoder(ds, heads: tuple[str, ...], out_dir: Path) -> LitEncoder:
    dl = train_data_loader(data_set=ds, batch_size=TASK_BATCH_SIZE)
    val_dl = eval_data_loader()

    no_bar = _no_progress_bar()
    bar_cbs = [] if no_bar else [TQDMProgressBar()]

    monitor = (
        "MRR/fusion/val"
        if "fusion" in heads
        else "F1/val"
        if {"emoji", "critic"} <= set(heads)
        else "MRR/e/val"
        if "emoji" in heads
        else "MRR/s/val"
        if "style" in heads
        else "auc/critic/val"
    )
    ckpt = ModelCheckpoint(
        monitor=monitor, mode="max", save_top_k=1, filename="best-{step}"
    )
    trainer = pl.Trainer(
        devices="auto",
        accelerator="auto",
        logger=TensorBoardLogger(
            "runs", name=CONFIG_NAME, version="enc", default_hp_metric=False
        ),
        deterministic=_DETERMINISTIC,
        max_epochs=EPOCHS_TASK,
        val_check_interval=min(VAL_CHECK_INTERVAL, len(dl)),
        enable_progress_bar=not no_bar,
        callbacks=[
            ckpt,
            EarlyStopping(monitor=monitor, mode="max",
                          patience=EARLY_STOP_PATIENCE),
            *bar_cbs,
            ModelSummary(),
        ],
    )

    mod = LitEncoder(heads=heads)
    trainer.fit(mod, dl, val_dl)

    if ckpt.best_model_path:
        mod = LitEncoder.load_from_checkpoint(ckpt.best_model_path)

    save_pt(mod.enc.state_dict(), str(out_dir / "enc.pt"), stage="enc")
    for h in ALL_HEADS:
        if h in heads:
            save_pt(getattr(mod, h).state_dict(), str(
                out_dir / f"{h}.pt"), stage="enc")

    return mod


def _train_gan(
    enc: TextEncoder, critic: ColorCritic, ds, out_dir: Path
) -> LitColorGAN:
    val_dl = eval_data_loader()
    no_bar = _no_progress_bar()
    bar_cbs = [] if no_bar else [TQDMProgressBar()]

    ckpt = ModelCheckpoint(
        monitor="energy/gan/val", mode="min", save_top_k=1, filename="best-gan-{step}"
    )
    trainer = pl.Trainer(
        devices="auto",
        accelerator="auto",
        logger=TensorBoardLogger(
            "runs", name=CONFIG_NAME, version="gan", default_hp_metric=False
        ),
        deterministic=_DETERMINISTIC,
        max_epochs=EPOCHS_GAN,
        enable_progress_bar=not no_bar,
        callbacks=[
            ckpt,
            EarlyStopping(
                monitor="energy/gan/val", mode="min", patience=EARLY_STOP_PATIENCE
            ),
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

    save_pt(gan.gen.state_dict(), str(out_dir / "gen.pt"), stage="gan")
    return gan


def _run_report_local(pt_dir: Path) -> None:
    subprocess.run(
        [sys.executable, "tools/report.py", "--pt", str(pt_dir)], check=True
    )


def _run_local(
    stage: Stage | None,
    heads: tuple[str, ...] | None,
    pt_dir: Path,
    out_dir: Path,
) -> None:
    require_clean_tree()
    pl.seed_everything(SEED, workers=True)
    torch.backends.cudnn.benchmark = _CUDA
    if _CUDA:
        torch.set_float32_matmul_precision("high")
    skip_report = os.environ.get("EMOJIC_SKIP_REPORT") == "1"
    out_dir.mkdir(parents=True, exist_ok=True)

    if stage == Stage.gan:
        _require_pt(
            pt_dir, ["enc.pt", "critic.pt", "style.pt", "emoji.pt", "fusion.pt"]
        )
        enc = _load(TextEncoder(), str(pt_dir / "enc.pt"))
        critic = _load(ColorCritic(), str(pt_dir / "critic.pt"))
        _train_gan(enc, critic, train_ds(), out_dir)  # type: ignore
        if out_dir == _DEFAULT_PT:
            export()
        if not skip_report:
            _run_report_local(out_dir)
        return

    ds = train_ds()
    heads = heads or ALL_HEADS
    mod = _train_encoder(ds, heads, out_dir)

    if stage == Stage.enc:
        if not skip_report:
            _run_report_local(out_dir)
        return

    critic = _load(ColorCritic(), str(out_dir / "critic.pt"))
    _train_gan(mod.enc, critic, ds, out_dir)  # type: ignore
    if out_dir == _DEFAULT_PT:
        export()
    if not skip_report:
        _run_report_local(out_dir)


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

DEP_FILES = ["pyproject.toml", "uv.lock", ".python-version", "README.md"]
CODE_FILES = [
    "files.py",
    f"{MODEL_DIR}/__init__.py",
    f"{MODEL_DIR}/color.py",
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
    f"cd {REPO} && UV_PROJECT_ENVIRONMENT=/usr/local "
    "uv sync --frozen --no-default-groups --group dev --group gpu"
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
    memory=CPU_MEMORY_MIB,
    timeout=TIMEOUT_S,
    volumes={ARTIFACTS: vol},
    include_source=False,
)
def train_remote(
    stage: str,
    heads: str,
    threads: int,
    git_sha: str,
    run_time: str,
    gpu: str = "",
    enc_bytes: bytes | None = None,
    style_bytes: bytes | None = None,
    emoji_bytes: bytes | None = None,
    critic_bytes: bytes | None = None,
    fusion_bytes: bytes | None = None,
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
        CRITIC_PT: critic_bytes,
        FUSION_PT: fusion_bytes,
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
                cmd = [VENV_PY, f"{MODEL_DIR}/train.py"]
                if stage:
                    cmd.append(stage)
                if stage == "enc" and heads:
                    cmd += ["--heads", heads]
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
        n = _stash(ARTIFACTS)
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
    stage: str, heads: str, git_sha: str, run_time: str, gpu: str
) -> dict[str, int]:
    pt_bytes: dict[str, bytes | None] = {
        "enc_bytes": None,
        "style_bytes": None,
        "emoji_bytes": None,
        "critic_bytes": None,
        "fusion_bytes": None,
    }
    if stage == "gan":
        for name in (ENC_PT, STYLE_PT, EMOJI_PT, CRITIC_PT, FUSION_PT):
            if not Path(name).exists():
                raise typer.BadParameter(
                    f"{name} not found -- run `train enc --local` "
                    "(or fetch a Modal enc run) first"
                )
        pt_bytes = {
            "enc_bytes": Path(ENC_PT).read_bytes(),
            "style_bytes": Path(STYLE_PT).read_bytes(),
            "emoji_bytes": Path(EMOJI_PT).read_bytes(),
            "critic_bytes": Path(CRITIC_PT).read_bytes(),
            "fusion_bytes": Path(FUSION_PT).read_bytes(),
        }

    threads = GPU_CPU if gpu else CPU
    fn = train_remote
    if gpu:
        fn = train_remote.with_options(
            gpu=gpu, cpu=GPU_CPU, memory=GPU_MEMORY_MIB, timeout=TIMEOUT_S
        )
    return fn.remote(
        stage=stage,
        heads=heads,
        threads=threads,
        git_sha=git_sha,
        run_time=run_time,
        gpu=gpu,
        **pt_bytes,
    )


def _dispatch(stage: Stage | None, heads_csv: str, gpu: str) -> None:
    require_clean_tree()
    git_sha = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True
    ).stdout.strip()
    run_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    stage_str = stage.value if stage else ""
    where = f"Modal {gpu} GPU" if gpu else "Modal"
    print(f"Training {stage_str or 'full pipeline'} on {where}...", flush=True)
    try:
        with modal.enable_output(), modal_app.run():
            print(_run_remote(stage_str, heads_csv, git_sha, run_time, gpu))
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
        metavar="[enc|gan]",
        help="enc = stage 1 (encoder + heads). gan = stage 2 (color GAN). "
        "Omit to run both back to back.",
    ),
    local: bool = typer.Option(
        False, "--local", help="Train on this machine instead of Modal."
    ),
    heads: str | None = typer.Option(
        None,
        "--heads",
        help="Comma list from {style,emoji,critic}; only with 'enc'. "
        "Default: all three.",
    ),
    pt: Path = typer.Option(
        _DEFAULT_PT, "--pt", help="Folder to read warm-start .pt from (default pt/)."
    ),
    out: Path = typer.Option(
        _DEFAULT_PT,
        "-o",
        "--output",
        help="Folder to write .pt to (default pt/). Non-default skips the web export.",
    ),
    gpu: str = typer.Option(
        "",
        "--gpu",
        help="Modal GPU type (e.g. T4, L4, A10G); defaults to "
        f"{DEFAULT_GPU} on remote. Not valid with --local.",
    ),
    cpu: bool = typer.Option(
        False,
        "--cpu",
        help="Run the Modal job on a CPU box instead of the default GPU.",
    ),
) -> None:
    """Train the emojic model.

    Stages
      (none)   Stage 1 with all heads, then Stage 2, then ONNX export + report.
      enc      Stage 1 only: TextEncoder + the --heads subset. Writes
               enc.pt plus one .pt per head. No export.
      gan      Stage 2 only: frozen encoder + generator, critic warm-started
               from <--pt>/critic.pt. Requires enc.pt, critic.pt, style.pt,
               emoji.pt in --pt. Writes gen.pt, then export + report.

    Location
      Runs on Modal by default, on a GPU (a T4 unless --gpu <type> picks
      another; larger batches, pinned-memory loaders). --cpu runs the Modal
      job on a CPU box instead. --local runs here. --pt / -o may differ from
      pt/ only with --local. A dirty git tree always aborts.

    Heads (stage 1 eval / checkpoint monitor)
      emoji+critic -> F1/val (harmonic mean of MRR/e/val and auc/critic/val),
      else emoji -> MRR/e/val, style -> MRR/s/val, critic -> auc/critic/val;
      the first match in that order is the checkpoint + early-stop metric.
    """
    resolved = _validate(stage, local, heads, pt, out, gpu, cpu)
    if local:
        _run_local(stage, resolved, pt, out)
    else:
        _dispatch(stage, heads or "", "" if cpu else (gpu or DEFAULT_GPU))


if __name__ == "__main__":
    _app()
