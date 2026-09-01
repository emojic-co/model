from math import ceil, log2

import lightning as pl
import torch
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
from torch import nn, optim
from torch.nn.functional import binary_cross_entropy_with_logits, normalize, tanh
from torch.nn.utils import spectral_norm as sn

from config import (
    CHAR_EMBED_SIZE,
)
from data import (
    COLOR_DIM,
    EMOJIS,
    FEELINGS,
    VOCAB_SIZE,
    eval_data_loader,
    train_data_loader,
)

EMOJI_EMBED_SIZE = ceil(6 * log2(len(EMOJIS)))
TEXT_EMBED_SIZE = 96
SEED = 42


class TextEncoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.char_embed = nn.Embedding(VOCAB_SIZE, CHAR_EMBED_SIZE)

        def conv_sn(*, k: int, i: int, o: int) -> nn.Sequential:
            return nn.Sequential(
                sn(nn.Conv1d(i, o, kernel_size=k, bias=False)),
                # nn.BatchNorm1d(o)
            )

        def conv_sn_relu(*, k: int, i: int, o: int) -> nn.Sequential:
            return nn.Sequential(
                conv_sn(k=k, i=i, o=o),
                nn.LeakyReLU(negative_slope=0.1))

        def pool_conv_sn_relu(*, k: int, i: int, o: int) -> nn.Sequential:
            return nn.Sequential(
                nn.MaxPool1d(kernel_size=2, stride=2),
                conv_sn_relu(k=k, i=i, o=o))

        cs = [32, 64, TEXT_EMBED_SIZE]
        io = zip(cs[:-1], cs[1:], strict=True)

        self.encoder = nn.Sequential(
            conv_sn_relu(k=3, i=CHAR_EMBED_SIZE, o=cs[0]),
            *[
                pool_conv_sn_relu(k=3, i=i, o=o)
                for i, o in io])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.char_embed(x).transpose(1, 2)
        out = self.encoder(out)
        return torch.max(out, dim=-1).values


class ColorTst(nn.Module):
    def __init__(self):
        super().__init__()

        cs = [COLOR_DIM + TEXT_EMBED_SIZE, 128, 64, 32, 16]
        io = zip(cs[:-1], cs[1:], strict=True)
        self.net = nn.Sequential(
            *[
                nn.Sequential(
                    sn(nn.Conv1d(i, o, kernel_size=1, bias=True)),
                    nn.LeakyReLU(negative_slope=0.2)
                )
                for i, o in io
            ],
            nn.Conv1d(cs[-1], 1, kernel_size=1, bias=True),
        )

    def forward(self, text_embedding: torch.Tensor, colors: torch.Tensor) -> torch.Tensor:
        x = torch.cat(
            [colors.unsqueeze(-1), text_embedding.unsqueeze(-1)], dim=1)
        return self.net(x)


class ColorGen(nn.Module):
    def __init__(self):
        super().__init__()

        self.z_dim = 16
        self.head = nn.Linear(self.z_dim + TEXT_EMBED_SIZE, COLOR_DIM)

    def sample_z(self, n: int, device: torch.device | None = None) -> torch.Tensor:
        return torch.randn(n, self.z_dim, device=device)

    def forward(
        self,
        text_embedding: torch.Tensor,
        z: torch.Tensor | None = None
    ) -> torch.Tensor:
        if z is None:
            z = self.sample_z(
                text_embedding.size(0),
                text_embedding.device)

        colors = self.head(torch.cat([z, text_embedding], dim=-1))
        colors = tanh(colors) * 127.5

        return colors


class FeelingHead(nn.Module):
    def __init__(self):
        super().__init__()

        self.dropout = nn.Dropout(p=0.4)
        self.net = nn.Linear(
            TEXT_EMBED_SIZE,
            len(FEELINGS))

    def forward(self, text_embedding: torch.Tensor) -> torch.Tensor:
        out = self.dropout(text_embedding)
        out = self.net(out)

        return out


class EmojiHead(nn.Module):
    def __init__(self):
        super().__init__()

        self.net = nn.Linear(
            TEXT_EMBED_SIZE,
            EMOJI_EMBED_SIZE)

    def forward(self, text_embedding: torch.Tensor) -> torch.Tensor:
        out = self.net(text_embedding)
        out = tanh(out)
        return normalize(out, dim=-1)


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
        opt_gen, opt_tst, opt_feel = self.optimizers()  # type: ignore

        enc = self.enc(text)
        enc_d = enc.detach()

        feels_pred = self.feels(enc)
        loss_feel = self.feeling_ce(feels_pred, feels)

        opt_feel.zero_grad()
        self.manual_backward(loss_feel)
        self.clip_gradients(
            opt_feel,  # type: ignore
            gradient_clip_val=1.0, gradient_clip_algorithm="norm")
        opt_feel.step()

        z = self.gen.sample_z(text.size(0), text.device)
        fake = self.gen(enc_d, z)
        tst_real = self.tst(enc_d, colors)
        tst_fake = self.tst(enc_d, fake.detach())

        loss_tst_real = binary_cross_entropy_with_logits(
            tst_real, torch.ones_like(tst_real))

        loss_tst_fake = binary_cross_entropy_with_logits(
            tst_fake, torch.zeros_like(tst_fake))

        loss_tst = loss_tst_real + loss_tst_fake

        opt_tst.zero_grad()
        self.manual_backward(loss_tst)
        self.clip_gradients(
            opt_tst,  # type: ignore
            gradient_clip_val=1.0, gradient_clip_algorithm="norm")

        opt_tst.step()

        tst_fake = self.tst(enc_d, fake)
        loss_gen = binary_cross_entropy_with_logits(
            tst_fake, torch.ones_like(tst_fake))

        opt_gen.zero_grad()
        self.manual_backward(loss_gen)
        self.clip_gradients(
            opt_gen,  # type: ignore
            gradient_clip_val=1.0, gradient_clip_algorithm="norm")

        opt_gen.step()

        acc = (feels_pred.argmax(dim=-1) == feels).float().mean()
        top5 = feels_pred.topk(5, dim=-1).indices
        acc5 = (top5 == feels.unsqueeze(1)).any(dim=-1).float().mean()

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

    def validation_step(self, batch, batch_idx):
        text, _, feels, _ = batch

        enc = self.enc(text)

        feels_pred = self.feels(enc)
        loss_feel = self.feeling_ce(feels_pred, feels)

        acc = (feels_pred.argmax(dim=-1) == feels).float().mean()
        top5 = feels_pred.topk(5, dim=-1).indices
        acc5 = (top5 == feels.unsqueeze(1)).any(dim=-1).float().mean()

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

    def configure_optimizers(self):
        opt_gen = optim.SGD(self.gen.parameters(), lr=0.01)
        opt_tst = optim.SGD(self.tst.parameters(), lr=0.01)
        opt_feel = optim.Adam(
            list(self.enc.parameters()) + list(self.feels.parameters()),
            lr=0.01)
        return [opt_gen, opt_tst, opt_feel]


if __name__ == "__main__":
    pl.seed_everything(SEED, workers=True)
    torch.backends.cudnn.benchmark = False

    logger = TensorBoardLogger("runs", name="color_critic")

    ckpt = ModelCheckpoint(
        monitor="acc5/f/val",
        mode="max",
        save_top_k=1,
        filename="best-{step}")
    early_stop = EarlyStopping(
        monitor="acc5/f/val",
        mode="max",
        patience=20)

    trainer = pl.Trainer(
        devices="auto",
        accelerator="auto",
        logger=logger,
        deterministic=True,
        max_epochs=100,
        val_check_interval=100,
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
