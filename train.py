import lightning as pl
import torch
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
from torch import optim
from torch.nn.functional import binary_cross_entropy_with_logits
from torchmetrics.classification import MultilabelAveragePrecision

from config import (
    CONFIG_NAME,
    EARLY_STOP_PATIENCE,
    EMOJIS,
    GAN_EPOCHS,
    GAN_LR,
    GRAD_CLIP,
    LR,
    SEED,
    STYLES,
    TASK_EPOCHS,
    VAL_CHECK_INTERVAL,
)
from data import eval_data_loader, train_data_loader
from model import ColorDsc, ColorGen, EmojiHead, StyleHead, TextEncoder

POS_WEIGHT_CLAMP = 10.0


def f1(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return 2 * a * b / (a + b + 1e-8)


def pos_weight(multihot: torch.Tensor) -> torch.Tensor:
    pos = multihot.sum(dim=0).clamp(min=1.0)
    neg = multihot.size(0) - pos
    return (neg / pos).sqrt().clamp(max=POS_WEIGHT_CLAMP)


class LitTask(pl.LightningModule):
    def __init__(self, emoji_pos_weight=None, style_pos_weight=None):
        super().__init__()

        self.enc = TextEncoder()
        self.style = StyleHead()
        self.emoji = EmojiHead()

        self.register_buffer(
            "emoji_pos_weight",
            torch.ones(len(EMOJIS)) if emoji_pos_weight is None else emoji_pos_weight)
        self.register_buffer(
            "style_pos_weight",
            torch.ones(len(STYLES)) if style_pos_weight is None else style_pos_weight)

        self.emoji_ap_train = MultilabelAveragePrecision(
            num_labels=len(EMOJIS), average="macro")
        self.emoji_ap_val = MultilabelAveragePrecision(
            num_labels=len(EMOJIS), average="macro")
        self.style_ap_train = MultilabelAveragePrecision(
            num_labels=len(STYLES), average="macro")
        self.style_ap_val = MultilabelAveragePrecision(
            num_labels=len(STYLES), average="macro")

    def _step(self, batch, split):
        text, emoji, style, _ = batch

        enc = self.enc(text)

        style_logits = self.style(enc)
        loss_style = binary_cross_entropy_with_logits(
            style_logits, style,
            pos_weight=self.style_pos_weight)  # type: ignore

        emoji_logits = self.emoji(enc)
        loss_emoji = binary_cross_entropy_with_logits(
            emoji_logits, emoji,
            pos_weight=self.emoji_pos_weight)  # type: ignore

        emoji_ap = self.emoji_ap_train if split == "train" else self.emoji_ap_val
        style_ap = self.style_ap_train if split == "train" else self.style_ap_val
        style_ap.update(style_logits, style.int())

        has_e = emoji.sum(dim=-1) > 0
        n_e = int(has_e.sum())
        if n_e:
            e_logits = emoji_logits[has_e]
            e_target = emoji[has_e]
            emoji_ap.update(e_logits, e_target.int())
            top10 = e_logits.topk(10, dim=-1).indices
            hit10 = e_target.gather(1, top10).amax(dim=-1).mean()
        else:
            hit10 = torch.zeros((), device=emoji.device)

        for name, val, bs in (
            (f"loss/s/{split}", loss_style, text.size(0)),
            (f"loss/e/{split}", loss_emoji, text.size(0)),
            (f"hit10/e/{split}", hit10, max(n_e, 1)),
            (f"mAP/e/{split}", emoji_ap, max(n_e, 1)),
            (f"mAP/s/{split}", style_ap, text.size(0)),
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
        a = m.get(f"mAP/e/{split}")
        b = m.get(f"mAP/s/{split}")
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

        self.log("loss/tst", loss_tst, prog_bar=True)
        self.log("loss/gen", loss_gen, prog_bar=True)

    def configure_optimizers(self):
        opt_gen = optim.SGD(self.gen.parameters(), lr=GAN_LR)
        opt_tst = optim.SGD(self.tst.parameters(), lr=GAN_LR)

        return [opt_gen, opt_tst]


if __name__ == "__main__":
    pl.seed_everything(SEED, workers=True)
    torch.backends.cudnn.benchmark = False

    dl = train_data_loader()
    val_dl = eval_data_loader()

    ds = dl.dataset
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
        max_epochs=TASK_EPOCHS,
        val_check_interval=min(VAL_CHECK_INTERVAL, len(dl)),
        callbacks=[
            task_ckpt,
            EarlyStopping(
                monitor="f1/val", mode="max", patience=EARLY_STOP_PATIENCE),
        ],
    )

    task = LitTask(pos_weight(ds.emoji), pos_weight(ds.style))  # type: ignore
    task_trainer.fit(task, dl, val_dl)

    if task_ckpt.best_model_path:
        task = LitTask.load_from_checkpoint(task_ckpt.best_model_path)

    for name, mod in (
        ("enc", task.enc),
        ("style", task.style),
        ("emoji", task.emoji),
    ):
        torch.save(mod.state_dict(), f"{name}.pt")

    gan_trainer = pl.Trainer(
        devices="auto",
        accelerator="auto",
        logger=TensorBoardLogger(
            "runs", name=CONFIG_NAME, version="gan", default_hp_metric=False),
        deterministic=True,
        max_epochs=GAN_EPOCHS,
        enable_checkpointing=False,
    )

    gan = LitColorGAN(task.enc)
    gan_trainer.fit(gan, dl)

    for name, mod in (
        ("gen", gan.gen),
        ("tst", gan.tst),
    ):
        torch.save(mod.state_dict(), f"{name}.pt")
