import torch
from torch import nn
from torch.nn.functional import normalize

from config import (
    CHANNELS,
    CHAR_EMBED_SIZE,
    EMOJI_EMBED_SIZE,
    KERNEL,
    POOL_1D_SIZE,
)
from data import EMOJIS, FEELING, VOCAB_SIZE


class Layer(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super().__init__(
            nn.Conv1d(
                in_channels,
                out_channels,
                kernel_size=KERNEL,
                bias=False),

            nn.BatchNorm1d(out_channels),
            nn.LeakyReLU(negative_slope=0.1),
            nn.AvgPool1d(
                kernel_size=POOL_1D_SIZE,
                stride=POOL_1D_SIZE),
        )


class Model(nn.Module):
    def __init__(self):
        super().__init__()

        self.char_embed = nn.Embedding(VOCAB_SIZE, CHAR_EMBED_SIZE)
        in_channels = [CHAR_EMBED_SIZE, *CHANNELS[:-1]]
        self.conv = nn.Sequential(
            *(Layer(i, o) for i, o in zip(in_channels, CHANNELS, strict=True))
        )

        self.feeling = nn.Linear(CHANNELS[-1], len(FEELING))

        self.emoji = nn.Linear(CHANNELS[-1], EMOJI_EMBED_SIZE)
        self.emoji_embed = nn.Embedding(len(EMOJIS), EMOJI_EMBED_SIZE)

    def forward(self, x):
        out = self.char_embed(x).transpose(1, 2)
        out = self.conv(out)  # (B, channels, T)

        out = torch.max(out, dim=-1).values  # (B, channels)

        return (
            self.feeling(out),
            normalize(self.emoji(out), p=2, dim=-1),
            normalize(self.emoji_embed.weight, p=2, dim=-1),
        )
