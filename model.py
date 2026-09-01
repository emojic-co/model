from math import ceil, log2

import pytorch_lightning as pl
import torch
from torch import nn, optim
from torch.nn.functional import binary_cross_entropy, sigmoid

from config import (
    CHAR_EMBED_SIZE,
    conv,
)
from data import COLOR_DIM, EMOJIS, VOCAB_SIZE

EMOJI_EMBED_SIZE = ceil(6 * log2(len(EMOJIS)))


def conv_bn_relu(*, k, i, o):
    return nn.Sequential(
        nn.Conv1d(
            i,
            o,
            kernel_size=k,
            bias=False),

        nn.BatchNorm1d(o),
        nn.LeakyReLU(negative_slope=0.1),
    )


def pool_conv_bn_relu(*, k, i, o):
    return nn.Sequential(
        nn.MaxPool1d(kernel_size=2, stride=2),
        conv_bn_relu(
            k=k,
            i=i,
            o=o),
    )


class Encoder(nn.Module):
    def __init__(self, config: list[conv]):
        super().__init__()

        self.char_embed = nn.Embedding(
            VOCAB_SIZE,
            CHAR_EMBED_SIZE)

        k, o = config[0]

        self.encoder = nn.Sequential(
            conv_bn_relu(k=k, i=CHAR_EMBED_SIZE, o=o),
            *[
                pool_conv_bn_relu(k=k, i=i, o=o)
                for (k, i), (k, o) in
                zip(config[:-1], config[1:], strict=True)]
        )

    def forward(self, x):
        out = self.char_embed(x).transpose(1, 2)
        out = self.encoder(out)

        return torch.max(out, dim=-1).values


class MLP(nn.Module):
    # Expecting (B, D, 1) input
    def __init__(self, cs: list[int]):
        super().__init__()
        self.net = nn.Sequential(*[
            conv_bn_relu(k=1, i=i, o=o)
            for i, o in zip(cs[:-1], cs[1:], strict=True)
        ]
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


class ColorCritic(nn.Module):
    # Input is a pair of text and 3 RGB colors, output is a single score
    def __init__(self):
        super().__init__()

        dim = 64
        self.encoder = Encoder([(3, dim)])
        self.net = MLP([dim + COLOR_DIM, 32, 1])

    def forward(self, x):
        text, colors = x
        enc = self.encoder(text)
        logit = self.net(torch.cat([enc, colors], dim=-1).unsqueeze(-1))

        return sigmoid(logit)


class LitColorCritic(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = ColorCritic()

    def training_step(self, batch):
        text, emoji, feeling, colors = batch

        assert (emoji)
        assert (feeling)

        fake = torch.tensor(
            torch.randint_like(colors, 0, 256) - 127.5,
            dtype=torch.float32)

        out_real = self.model((text, colors))
        out_fake = self.model((text, fake))

        loss_real = binary_cross_entropy(out_real, torch.ones_like(out_real))
        loss_fake = binary_cross_entropy(out_fake, torch.zeros_like(out_fake))

        return loss_real + loss_fake

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=0.001)
