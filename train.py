import subprocess
import sys

import lightning as pl
import torch
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
from torch import optim
from torch.nn.functional import binary_cross_entropy_with_logits

from config import (
    CONFIG_NAME,
    EARLY_STOP_PATIENCE,
    EMOJI_AP_K,
    EPOCHS_TASK,
    FOCAL_ALPHA,
    FOCAL_GAMMA,
    GAN_LR,
    GRAD_CLIP,
    LR,
    SEED,
    STYLE_AP_K,
    STYLES,
    TASK_BATCH_SIZE,
    VAL_CHECK_INTERVAL,
)
from data import eval_data_loader, train_data_loader, train_ds
from model import ColorDsc, ColorGen, EmojiHead, StyleHead, TextEncoder

POS_WEIGHT_CLAMP = 10.0


def f1(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return 2 * a * b / (a + b + 1e-8)


def pos_weight(multihot: torch.Tensor) -> torch.Tensor:
    pos = multihot.sum(dim=0).clamp(min=1.0)
    neg = multihot.size(0) - pos
    return (neg / pos).sqrt().clamp(max=POS_WEIGHT_CLAMP)


def focal_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    alpha: float,
    gamma: float,
) -> torch.Tensor:
    ce = binary_cross_entropy_with_logits(logits, target, reduction="none")
    p = torch.sigmoid(logits)
    p_t = p * target + (1 - p) * (1 - target)
    alpha_t = alpha * target + (1 - alpha) * (1 - target)
    loss = alpha_t * (1 - p_t) ** gamma * ce
    return loss.sum() / target.sum().clamp(min=1.0)


def ap_at_k(logits: torch.Tensor, target: torch.Tensor, k: int) -> torch.Tensor:
    k = min(k, logits.size(-1))
    topk = logits.topk(k, dim=-1).indices
    rel = target.gather(1, topk)
    ranks = torch.arange(1, k + 1, device=logits.device)
    prec = rel.cumsum(dim=-1) / ranks
    denom = target.sum(dim=-1).clamp(max=k).clamp(min=1.0)
    return (prec * rel).sum(dim=-1) / denom


class LitTask(pl.LightningModule):
    def __init__(self, style_pos_weight=None):
        super().__init__()

        self.enc = TextEncoder()
        self.style = StyleHead()
        self.emoji = EmojiHead()

        self.register_buffer(
            "style_pos_weight",
            torch.ones(len(STYLES)) if style_pos_weight is None else style_pos_weight)

    def _step(self, batch, split):
        text, emoji, style, _ = batch

        enc = self.enc(text)

        style_logits = self.style(enc)
        loss_style = binary_cross_entropy_with_logits(
            style_logits, style,
            pos_weight=self.style_pos_weight)  # type: ignore

        emoji_logits = self.emoji(enc)
        loss_emoji = focal_loss(emoji_logits, emoji, FOCAL_ALPHA, FOCAL_GAMMA)

        style_ap = ap_at_k(style_logits, style, STYLE_AP_K).mean()

        has_e = emoji.sum(dim=-1) > 0
        n_e = int(has_e.sum())
        if n_e:
            emoji_ap = ap_at_k(emoji_logits[has_e], emoji[has_e], EMOJI_AP_K).mean()
        else:
            emoji_ap = torch.zeros((), device=emoji.device)

        for name, val, bs in (
            (f"loss/s/{split}", loss_style, text.size(0)),
            (f"loss/e/{split}", loss_emoji, text.size(0)),
            (f"mAP@{EMOJI_AP_K}/e/{split}", emoji_ap, max(n_e, 1)),
            (f"mAP@{STYLE_AP_K}/s/{split}", style_ap, text.size(0)),
        ):
            self.log(
                name, val,
                on_step=False, on_epoch=True, prog_bar=True,
                batch_size=bs)

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

        self.automatic_optimization = False

    def on_train_epoch_start(self):
        self.enc.eval()

    def _cond(self, text: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return self.enc(text)

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
            tst_fake, torch.zeros_like(tst_fake))

        loss_tst = loss_tst_real + loss_tst_fake

        opt_tst.zero_grad()
        self.manual_backward(loss_tst)
        self.clip_gradients(
            opt_tst,  # type: ignore
            gradient_clip_val=GRAD_CLIP, gradient_clip_algorithm="norm")

        opt_tst.step()

        tst_fake = self.tst(cond, fake)
        loss_gen = binary_cross_entropy_with_logits(
            tst_fake, torch.ones_like(tst_fake))

        opt_gen.zero_grad()
        self.manual_backward(loss_gen)
        self.clip_gradients(
            opt_gen,  # type: ignore
            gradient_clip_val=GRAD_CLIP, gradient_clip_algorithm="norm")

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

    task_ckpt = ModelCheckpoint(
        monitor="f1/val",
        mode="max",
        save_top_k=1,
        filename="best-{step}")

    task_trainer = pl.Trainer(
        devices="auto",
        accelerator="auto",
        logger=TensorBoardLogger(
            "runs", name=CONFIG_NAME, version="task", default_hp_metric=False),
        deterministic=True,
        max_epochs=EPOCHS_TASK,
        val_check_interval=min(VAL_CHECK_INTERVAL, len(task_dl)),
        callbacks=[
            task_ckpt,
            EarlyStopping(
                monitor="f1/val", mode="max", patience=EARLY_STOP_PATIENCE),
        ],
    )

    task = LitTask(style_pos_weight=pos_weight(ds.style))  # type: ignore
    task_trainer.fit(task, task_dl, val_dl)

    if task_ckpt.best_model_path:
        task = LitTask.load_from_checkpoint(task_ckpt.best_model_path)

    for name, mod in (
        ("enc", task.enc),
        ("style", task.style),
        ("emoji", task.emoji),
    ):
        torch.save(mod.state_dict(), f"{name}.pt")

    # gan_trainer = pl.Trainer(
    #     devices="auto",
    #     accelerator="auto",
    #     logger=TensorBoardLogger(
    #         "runs", name=CONFIG_NAME, version="gan", default_hp_metric=False),
    #     deterministic=True,
    #     max_epochs=EPOCHS_GAN,
    #     enable_checkpointing=False,
    # )

    # gan = LitColorGAN(task.enc)
    # gan_dl = train_data_loader(data_set=ds, batch_size=GAN_BATCH_SIZE)
    # gan_trainer.fit(gan, gan_dl)

    # for name, mod in (
    #     ("gen", gan.gen),
    #     ("tst", gan.tst),
    # ):
    #     torch.save(mod.state_dict(), f"{name}.pt")

    subprocess.run([sys.executable, "export_onnx.py"], check=True)
