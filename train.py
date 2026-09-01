import lightning as pl
import torch
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
from torch import nn, optim
from torch.nn.functional import binary_cross_entropy_with_logits, cross_entropy

from config import (
    CONFIG_NAME,
    EARLY_STOP_PATIENCE,
    GAN_EPOCHS,
    GAN_LR,
    GRAD_CLIP,
    INFONCE_TEMP,
    LR,
    SEED,
    TASK_EPOCHS,
    VAL_CHECK_INTERVAL,
)
from data import eval_data_loader, train_data_loader
from model import ColorGen, ColorTst, EmojiHead, FeelingHead, TextEncoder


def f1(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return 2 * a * b / (a + b + 1e-8)


class LitTask(pl.LightningModule):
    def __init__(self):
        super().__init__()

        self.enc = TextEncoder()
        self.feels = FeelingHead()
        self.emoji = EmojiHead()

        self.feeling_ce = nn.CrossEntropyLoss()

    def _step(self, batch, split):
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

        for name, val in (
            (f"loss/f/{split}", loss_feel),
            (f"acc/f/{split}", acc),
            (f"acc5/f/{split}", acc5),
            (f"loss/e/{split}", loss_emoji),
            (f"acc10/e/{split}", acc10),
        ):
            self.log(
                name, val,
                on_step=False, on_epoch=True, prog_bar=True,
                batch_size=text.size(0))

        return loss_feel + loss_emoji

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
        a = m.get(f"acc5/f/{split}")
        b = m.get(f"acc10/e/{split}")
        if a is not None and b is not None:
            self.log(f"f1/{split}", f1(a, b), prog_bar=True)

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=LR)


class LitColorGAN(pl.LightningModule):
    def __init__(self, enc: TextEncoder, emoji: EmojiHead):
        super().__init__()

        self.enc = enc.requires_grad_(False).eval()
        self.emoji = emoji.requires_grad_(False).eval()

        self.gen = ColorGen()
        self.tst = ColorTst()

        self.automatic_optimization = False

    def on_train_epoch_start(self):
        self.enc.eval()
        self.emoji.eval()

    def _cond(self, text: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            q, _ = self.emoji(self.enc(text))
        return q

    def training_step(self, batch, batch_idx):
        text, _, _, colors = batch
        opt_gen, opt_tst = self.optimizers()  # type: ignore

        cond = self._cond(text)

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
        val_check_interval=VAL_CHECK_INTERVAL,
        callbacks=[
            task_ckpt,
            EarlyStopping(
                monitor="f1/val", mode="max", patience=EARLY_STOP_PATIENCE),
        ],
    )

    task = LitTask()
    task_trainer.fit(task, dl, val_dl)

    if task_ckpt.best_model_path:
        task = LitTask.load_from_checkpoint(task_ckpt.best_model_path)

    for name, mod in (
        ("enc", task.enc),
        ("feels", task.feels),
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

    gan = LitColorGAN(task.enc, task.emoji)
    gan_trainer.fit(gan, dl)

    for name, mod in (
        ("gen", gan.gen),
        ("tst", gan.tst),
    ):
        torch.save(mod.state_dict(), f"{name}.pt")
