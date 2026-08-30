from itertools import chain

import torch
from torch import nn
from torch.nn.functional import normalize

from config import (
    CHAR_EMBED_SIZE,
    CONV,
    DROPOUT,
    EMOJI_EMBED_SIZE,
)
from data import EMOJIS, FEELING, VOCAB_SIZE


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

        self.net = nn.Sequential(
            *layer(
                kernel=k,
                in_channels=CHAR_EMBED_SIZE,
                out_channels=o,
            ),
            *chain.from_iterable(
                layer(
                    kernel=k,
                    in_channels=i,
                    out_channels=o,
                )
                for (_, i), (k, o) in zip(CONV[:-1], CONV[1:], strict=True)
            )
        )

        _, o = CONV[-1]
        self.feeling_dropout = nn.Dropout(DROPOUT)
        self.feeling = nn.Linear(o, len(FEELING))

        self.emoji = nn.Linear(o, EMOJI_EMBED_SIZE)
        self.emoji_embed = nn.Embedding(len(EMOJIS), EMOJI_EMBED_SIZE)

    def forward(self, x):
        out = self.char_embed(x).transpose(1, 2)
        out = self.net(out)  # (B, channels, T)

        pooled = torch.max(out, dim=-1).values  # (B, channels)

        return (
            self.feeling(self.feeling_dropout(pooled)),
            normalize(self.emoji(pooled), p=2, dim=-1),
            normalize(self.emoji_embed.weight, p=2, dim=-1),
        )
