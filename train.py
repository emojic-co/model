import lightning as pl
import torch
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
from torch import nn, optim
from torch.nn.functional import binary_cross_entropy_with_logits, cross_entropy

from config import (
    CONFIG_NAME,
    EARLY_STOP_PATIENCE,
    EPOCHS,
    GAN_LR,
    GRAD_CLIP,
    INFONCE_TEMP,
    LR,
    SEED,
    VAL_CHECK_INTERVAL,
)
from data import eval_data_loader, train_data_loader
from model import ColorGen, ColorTst, EmojiHead, FeelingHead, TextEncoder


def f1(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return 2 * a * b / (a + b + 1e-8)


class LitGAN(pl.LightningModule):
    def __init__(self):
        super().__init__()

        self.enc = TextEncoder()
        self.gen = ColorGen()
        self.tst = ColorTst()

        self.feels = FeelingHead()
        self.emoji = EmojiHead()

        self.automatic_optimization = False
        self.feeling_ce = nn.CrossEntropyLoss()

    def training_step(self, batch, batch_idx):
        text, emoji, feels, colors = batch
        opt_gen, opt_tst, opt_task = self.optimizers()  # type: ignore

        enc = self.enc(text)

        feels_pred = self.feels(enc)
        loss_feel = self.feeling_ce(feels_pred, feels)

        q, emoji_vec = self.emoji(enc)
        emoji_logits = q @ emoji_vec.t()
        loss_emoji = cross_entropy(emoji_logits / INFONCE_TEMP, emoji)

        opt_task.zero_grad()
        self.manual_backward(loss_feel + loss_emoji)
        self.clip_gradients(
            opt_task,  # type: ignore
            gradient_clip_val=GRAD_CLIP, gradient_clip_algorithm="norm")

        opt_task.step()

        cond = q.detach()

        z = self.gen.sample_z(text.size(0), text.device)
        fake = self.gen(cond, z)
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

        acc = (feels_pred.argmax(dim=-1) == feels).float().mean()
        top5 = feels_pred.topk(5, dim=-1).indices
        acc5 = (top5 == feels.unsqueeze(1)).any(dim=-1).float().mean()

        top10 = emoji_logits.topk(10, dim=-1).indices
        acc10 = (top10 == emoji.unsqueeze(1)).any(dim=-1).float().mean()

        self.log("loss/tst", loss_tst, prog_bar=True)
        self.log("loss/gen", loss_gen, prog_bar=True)
        self.log(
            "loss/f/train", loss_feel,
            on_step=False, on_epoch=True, prog_bar=True,
            batch_size=text.size(0))
        self.log(
            "acc/f/train", acc,
            on_step=False, on_epoch=True, prog_bar=True,
            batch_size=text.size(0))
        self.log(
            "acc5/f/train", acc5,
            on_step=False, on_epoch=True, prog_bar=True,
            batch_size=text.size(0))
        self.log(
            "loss/e/train", loss_emoji,
            on_step=False, on_epoch=True, prog_bar=True,
            batch_size=text.size(0))
        self.log(
            "acc10/e/train", acc10,
            on_step=False, on_epoch=True, prog_bar=True,
            batch_size=text.size(0))

    def validation_step(self, batch, batch_idx):
        text, emoji, feels, _ = batch

        enc = self.enc(text)

        feels_pred = self.feels(enc)
        loss_feel = self.feeling_ce(feels_pred, feels)

        q, emoji_vec = self.emoji(enc)
        emoji_logits = q @ emoji_vec.t()
        loss_emoji = cross_entropy(emoji_logits / INFONCE_TEMP, emoji)

        acc = (feels_pred.argmax(dim=-1) == feels).float().mean()
        top5 = feels_pred.topk(5, dim=-1).indices
        acc5 = (top5 == feels.unsqueeze(1)).any(dim=-1).float().mean()

        top10 = emoji_logits.topk(10, dim=-1).indices
        acc10 = (top10 == emoji.unsqueeze(1)).any(dim=-1).float().mean()

        self.log(
            "loss/f/val", loss_feel,
            on_step=False, on_epoch=True, prog_bar=True,
            batch_size=text.size(0))
        self.log(
            "acc/f/val", acc,
            on_step=False, on_epoch=True, prog_bar=True,
            batch_size=text.size(0))
        self.log(
            "acc5/f/val", acc5,
            on_step=False, on_epoch=True, prog_bar=True,
            batch_size=text.size(0))
        self.log(
            "loss/e/val", loss_emoji,
            on_step=False, on_epoch=True, prog_bar=True,
            batch_size=text.size(0))
        self.log(
            "acc10/e/val", acc10,
            on_step=False, on_epoch=True, prog_bar=True,
            batch_size=text.size(0))

    def on_train_epoch_end(self):
        self._log_f1("train")

    def on_validation_epoch_end(self):
        self._log_f1("val")

    def _log_f1(self, split):
        m = self.trainer.callback_metrics
        a = m.get(f"acc5/f/{split}")
        b = m.get(f"acc10/e/{split}")
        if a is not None and b is not None:
            self.log(f"f1/{split}", f1(a, b), prog_bar=True)

    def configure_optimizers(self):
        opt_gen = optim.SGD(self.gen.parameters(), lr=GAN_LR)
        opt_tst = optim.SGD(self.tst.parameters(), lr=GAN_LR)

        opt_task = optim.Adam(
            list(self.enc.parameters())
            + list(self.feels.parameters())
            + list(self.emoji.parameters()),
            lr=LR)

        return [opt_gen, opt_tst, opt_task]


if __name__ == "__main__":
    pl.seed_everything(SEED, workers=True)
    torch.backends.cudnn.benchmark = False

    logger = TensorBoardLogger(
        "runs",
        name=CONFIG_NAME,
        version="",
        default_hp_metric=False)

    ckpt = ModelCheckpoint(
        monitor="f1/val",
        mode="max",
        save_top_k=1,
        filename="best-{step}")
    early_stop = EarlyStopping(
        monitor="f1/val",
        mode="max",
        patience=EARLY_STOP_PATIENCE)

    trainer = pl.Trainer(
        devices="auto",
        accelerator="auto",
        logger=logger,
        deterministic=True,
        max_epochs=EPOCHS,
        val_check_interval=VAL_CHECK_INTERVAL,
        callbacks=[ckpt, early_stop],
    )

    model = LitGAN()
    dl = train_data_loader()
    val_dl = eval_data_loader()

    trainer.fit(model, dl, val_dl)

    if ckpt.best_model_path:
        model = LitGAN.load_from_checkpoint(ckpt.best_model_path)

    torch.save(model.state_dict(), "gan.pt")
    for name, mod in (
        ("enc", model.enc),
        ("gen", model.gen),
        ("tst", model.tst),
        ("feels", model.feels),
        ("emoji", model.emoji),
    ):
        torch.save(mod.state_dict(), f"{name}.pt")
