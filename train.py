import os
import subprocess
import sys

import lightning as pl
import torch
from lightning.pytorch.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    ModelSummary,
    TQDMProgressBar,
)
from lightning.pytorch.loggers import TensorBoardLogger
from torch import optim
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
from model import (
    ColorDsc,
    ColorGen,
    EmojiHead,
    StyleHead,
    TextEncoder,
    rgb_to_oklab,
)
from runmeta import save_pt


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
            emoji_mrr = mrr_at_k(emoji_logits[has_e], emoji[has_e], EMOJI_AP_K).mean()
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
            self.log(name, val, on_step=False, on_epoch=True, prog_bar=True, batch_size=bs)

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
        return energy_distance(pts[perm[:half]], pts[perm[half : 2 * half]])

    def on_validation_epoch_end(self):
        if not self._val_real:
            return

        self.gen.eval()
        with torch.no_grad():
            text = torch.cat(self._val_text)
            real = rgb_to_oklab(torch.cat(self._val_real))
            n = text.size(0)
            z = self.z_bank[  # type: ignore
                torch.arange(n, device=self.device) % self.z_bank.size(0)
            ]  # type: ignore

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

        loss_tst_real = binary_cross_entropy_with_logits(tst_real, torch.ones_like(tst_real))

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
        loss_gen = binary_cross_entropy_with_logits(tst_fake, torch.ones_like(tst_fake))

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


if __name__ == "__main__":
    pl.seed_everything(SEED, workers=True)
    torch.backends.cudnn.benchmark = False

    ds = train_ds()
    task_dl = train_data_loader(data_set=ds, batch_size=TASK_BATCH_SIZE)
    val_dl = eval_data_loader()

    no_progress_bar = os.environ.get("EMOJIC_NO_PROGRESS_BAR") == "1"
    progress_bar_cbs = [] if no_progress_bar else [TQDMProgressBar()]

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
        enable_progress_bar=not no_progress_bar,
        callbacks=[
            task_ckpt,
            EarlyStopping(monitor=task_monitor, mode="max", patience=EARLY_STOP_PATIENCE),
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

    if os.environ.get("EMOJIC_TASK_ONLY") == "1":
        sys.exit(0)

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
        enable_progress_bar=not no_progress_bar,
        callbacks=[
            gan_ckpt,
            EarlyStopping(
                monitor="energy/gan/val", mode="min", patience=EARLY_STOP_PATIENCE
            ),
            *progress_bar_cbs,
            ModelSummary(),
        ],
    )

    gan = LitColorGAN(task.enc)
    gan_dl = train_data_loader(data_set=ds, batch_size=GAN_BATCH_SIZE)
    gan_trainer.fit(gan, gan_dl, val_dl)

    if gan_ckpt.best_model_path:
        gan = LitColorGAN.load_from_checkpoint(gan_ckpt.best_model_path, enc=task.enc)

    for name, mod in (
        ("gen", gan.gen),
        ("tst", gan.tst),
    ):
        save_pt(mod.state_dict(), f"{name}.pt", stage="gan")

    subprocess.run([sys.executable, "export_onnx.py"], check=True)
