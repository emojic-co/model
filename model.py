from math import ceil, log2

import lightning as pl
import torch
from lightning.pytorch.loggers import TensorBoardLogger
from torch import nn, optim
from torch.nn.functional import binary_cross_entropy_with_logits, tanh

from config import (
    CHAR_EMBED_SIZE,
    conv,
)
from data import COLOR_DIM, EMOJIS, VOCAB_SIZE, train_data_loader

EMOJI_EMBED_SIZE = ceil(6 * log2(len(EMOJIS)))

SEED = 42


def conv_bn(*, k: int, i: int, o: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv1d(i, o, kernel_size=k, bias=False),
        nn.BatchNorm1d(o),
    )


def conv_bn_relu(*, k: int, i: int, o: int) -> nn.Sequential:
    return nn.Sequential(
        conv_bn(k=k, i=i, o=o),
        nn.LeakyReLU(negative_slope=0.1),
    )


def pool_conv_bn_relu(*, k: int, i: int, o: int) -> nn.Sequential:
    return nn.Sequential(
        nn.MaxPool1d(kernel_size=2, stride=2),
        conv_bn_relu(k=k, i=i, o=o),
    )


class Encoder(nn.Module):
    def __init__(self, config: list[conv]):
        super().__init__()

        self.char_embed = nn.Embedding(VOCAB_SIZE, CHAR_EMBED_SIZE)

        k, o = config[0]

        self.encoder = nn.Sequential(
            conv_bn_relu(k=k, i=CHAR_EMBED_SIZE, o=o),
            *[
                pool_conv_bn_relu(k=k_layer, i=i_chan, o=o_chan)
                for (_, i_chan), (k_layer, o_chan) in zip(
                    config[:-1], config[1:], strict=True
                )
            ],
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.char_embed(x).transpose(1, 2)
        out = self.encoder(out)
        return torch.max(out, dim=-1).values


class MLP(nn.Module):
    # Expecting (B, D, 1) input
    def __init__(self, cs: list[int]):
        super().__init__()
        self.net = nn.Sequential(
            *[
                conv_bn_relu(k=1, i=i, o=o)
                for i, o in zip(cs[:-2], cs[1:-1], strict=True)
            ],
            nn.Conv1d(cs[-2], cs[-1], kernel_size=1, bias=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(2)


class ColorTst(nn.Module):
    # Input is a pair of text and 3 RGB colors, output is a single score
    def __init__(self):
        super().__init__()

        # self.encoder = Encoder([(3, dim)])
        self.net = MLP([COLOR_DIM, 64, 64, 64, 1])

    def forward(self, text: torch.Tensor, colors: torch.Tensor) -> torch.Tensor:
        # enc = self.encoder(text)
        logit = self.net(colors.unsqueeze(-1))
        return logit


class ColorGen(nn.Module):
    def __init__(self):
        super().__init__()

        dim = 128
        self.z_dim = 16
        self.encoder = Encoder([(3, dim)])
        self.head = nn.Linear(dim + self.z_dim, COLOR_DIM)

    def sample_z(self, n: int, device: torch.device | None = None) -> torch.Tensor:
        return torch.randn(n, self.z_dim, device=device)

    def forward(
        self, text: torch.Tensor, z: torch.Tensor | None = None
    ) -> torch.Tensor:
        enc = self.encoder(text)
        if z is None:
            z = self.sample_z(enc.size(0), enc.device)
        colors = self.head(torch.cat([enc, z], dim=-1))
        colors = tanh(colors) * 127.5

        return colors


class LitGAN(pl.LightningModule):
    def __init__(self):
        super().__init__()

        self.gen = ColorGen()
        self.tst = ColorTst()

        self.automatic_optimization = False

    def training_step(self, batch, batch_idx):
        text, _, _, colors = batch
        opt_gen, opt_tst = self.optimizers()  # type: ignore

        z = self.gen.sample_z(text.size(0), text.device)
        fake = self.gen(text, z)
        tst_real = self.tst(text, colors)
        tst_fake = self.tst(text, fake)

        # TST
        loss_tst_real = binary_cross_entropy_with_logits(
            tst_real, torch.ones_like(tst_real))

        loss_tst_fake = binary_cross_entropy_with_logits(
            tst_fake.detach(), torch.zeros_like(tst_fake))

        loss_tst = loss_tst_real + loss_tst_fake

        opt_tst.zero_grad()
        self.manual_backward(loss_tst)
        opt_tst.step()

        # GEN
        tst_fake = self.tst((text, fake))
        loss_gen = binary_cross_entropy_with_logits(
            tst_fake, torch.ones_like(tst_fake))

        opt_gen.zero_grad()
        self.manual_backward(loss_gen)
        opt_gen.step()

        # LOG
        self.log("loss/tst", loss_tst, prog_bar=True)
        self.log("loss/gen", loss_gen, prog_bar=True)

    def configure_optimizers(self):
        # Betas (0.5, 0.999) and lower learning rates for GAN stability
        opt_gen = optim.Adam(self.gen.parameters(), lr=0.0002, betas=(0.5, 0.999))
        opt_tst = optim.Adam(self.tst.parameters(), lr=0.0002, betas=(0.5, 0.999))
        return [opt_gen, opt_tst]


if __name__ == "__main__":
    pl.seed_everything(SEED, workers=True)
    torch.backends.cudnn.benchmark = False

    logger = TensorBoardLogger("runs", name="color_critic")

    trainer = pl.Trainer(
        devices="auto",
        accelerator="auto",
        logger=logger,
        max_epochs=3,
        deterministic=True,
    )

    model = LitGAN()
    dl = train_data_loader()

    trainer.fit(model, dl)

    torch.save(model.gen.state_dict(), "gen.pt")
