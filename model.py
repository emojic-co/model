from math import ceil, log2

import pytorch_lightning as pl
import torch
from lightning.pytorch.loggers import TensorBoardLogger
from torch import nn, optim
from torch.nn.functional import binary_cross_entropy_with_logits

from config import (
    CHAR_EMBED_SIZE,
    conv,
)
from data import COLOR_DIM, EMOJIS, VOCAB_SIZE, train_data_loader

EMOJI_EMBED_SIZE = ceil(6 * log2(len(EMOJIS)))


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


class ColorCritic(nn.Module):
    # Input is a pair of text and 3 RGB colors, output is a single score
    def __init__(self):
        super().__init__()

        dim = 64
        self.encoder = Encoder([(3, dim)])
        self.net = MLP([dim + COLOR_DIM, 32, 1])

    def forward(self, x: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        text, colors = x
        enc = self.encoder(text)
        logit = self.net(torch.cat([enc, colors], dim=-1).unsqueeze(-1))
        return logit


class LitColorCritic(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = ColorCritic()

    def training_step(self, batch, batch_idx):
        text, _, _, colors = batch

        fake = torch.randint_like(colors, 0, 256, dtype=torch.float32) - 127.5

        out_real = self.model((text, colors))
        out_fake = self.model((text, fake))

        loss_real = binary_cross_entropy_with_logits(
            out_real, torch.ones_like(out_real)
        )
        loss_fake = binary_cross_entropy_with_logits(
            out_fake, torch.zeros_like(out_fake)
        )

        loss = loss_real + loss_fake
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=0.001)


if __name__ == "__main__":
    logger = TensorBoardLogger("runs", name="color_critic")
    trainer = pl.Trainer(
        devices="auto",
        accelerator="auto",
        max_epochs=10,
    )

    model = LitColorCritic()
    dl = train_data_loader()

    trainer.fit(model, dl)
