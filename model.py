from math import ceil, log2

import torch
from torch import nn
from torch.nn.functional import normalize

from config import CHAR_EMBED_SIZE, CONV, DROPOUT_COLOR, DROPOUT_EMOJI, DROPOUT_FEELING
from data import COLOR_DIM, EMOJIS, FEELING, VOCAB_SIZE

EMOJI_EMBED_SIZE = ceil(6 * log2(len(EMOJIS)))


def layer(*, kernel, in_channels, out_channels):
    return [
        nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel,
            bias=False),

        nn.BatchNorm1d(out_channels),
        nn.LeakyReLU(negative_slope=0.1),
    ]


class Model(nn.Module):
    def __init__(self):
        super().__init__()

        self.char_embed = nn.Embedding(VOCAB_SIZE, CHAR_EMBED_SIZE)
        k, o = CONV[0]

        # Create initial layer block
        layers = [*layer(
            kernel=k,
            in_channels=CHAR_EMBED_SIZE,
            out_channels=o)]

        # Append MaxPool + downstream layers in a loop
        for (_, i), (k, o) in zip(CONV[:-1], CONV[1:], strict=True):
            layers.append(nn.MaxPool1d(kernel_size=2, stride=2))
            layers.extend(layer(
                kernel=k,
                in_channels=i,
                out_channels=o))

        self.net = nn.Sequential(*layers)

        _, o = CONV[-1]
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
