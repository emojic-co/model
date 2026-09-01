from math import ceil, log2

import torch
from torch import nn
from torch.nn.functional import normalize, sigmoid

from config import (
    CHAR_EMBED_SIZE,
    DROPOUT_COLOR,
    DROPOUT_EMOJI,
    DROPOUT_FEELING,
    ENCODER,
    conv,
)
from data import COLOR_DIM, EMOJIS, FEELING, VOCAB_SIZE

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


class Model(nn.Module):
    def __init__(self):
        super().__init__()

        self.char_embed = nn.Embedding(VOCAB_SIZE, CHAR_EMBED_SIZE)
        k, o = ENCODER[0]

        # Create initial layer block
        layers = [*layer(
            kernel=k,
            in_channels=CHAR_EMBED_SIZE,
            out_channels=o)]

        # Append MaxPool + downstream layers in a loop
        for (_, i), (k, o) in zip(ENCODER[:-1], ENCODER[1:], strict=True):
            layers.append(nn.MaxPool1d(kernel_size=2, stride=2))
            layers.extend(layer(
                kernel=k,
                in_channels=i,
                out_channels=o))

        self.net = nn.Sequential(*layers)

        _, o = ENCODER[-1]
        self.feeling_dropout = nn.Dropout(DROPOUT_FEELING)
        self.feeling = nn.Linear(o, len(FEELING))

        self.emoji_dropout = nn.Dropout(DROPOUT_EMOJI)
        self.emoji = nn.Linear(o, EMOJI_EMBED_SIZE)
        self.emoji_embed = nn.Embedding(len(EMOJIS), EMOJI_EMBED_SIZE)

        self.color_dropout = nn.Dropout(DROPOUT_COLOR)
        self.color = nn.Linear(o, COLOR_DIM)

    def forward(self, x):
        out = self.char_embed(x).transpose(1, 2)
        out = self.net(out)

        pooled = torch.max(out, dim=-1).values

        feeling_do = self.feeling_dropout(pooled)
        emoji_do = self.emoji_dropout(pooled)
        color_do = self.color_dropout(pooled)

        return (
            self.feeling(feeling_do),
            normalize(self.emoji(emoji_do), p=2, dim=-1),
            normalize(self.emoji_embed.weight, p=2, dim=-1),
            self.color(color_do),
        )
