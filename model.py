import torch
from torch import nn
from torch.nn.functional import normalize

from config import (
    CHANNELS,
    CHAR_EMBED_SIZE,
    EMOJI_EMBED_SIZE,
)
from data import EMOJIS, FEELING, PAD_IDX, VOCAB_SIZE


class Model(nn.Module):
    def __init__(self):
        super().__init__()

        self.char_embed = nn.Embedding(
            VOCAB_SIZE, CHAR_EMBED_SIZE, padding_idx=PAD_IDX)

        self.conv = nn.Sequential(
            nn.Conv1d(
                in_channels=CHAR_EMBED_SIZE,
                out_channels=CHANNELS[0],
                kernel_size=3,
                stride=1,
                padding=0,
                bias=False,
            ),

            nn.ReLU(),

            *[
                nn.Sequential(
                    nn.Conv1d(
                        in_channels=i,
                        out_channels=o,
                        kernel_size=3,
                        stride=1,
                        padding=0,
                        bias=False,
                    ),

                    nn.ReLU(),
                )
                for i, o in zip(CHANNELS[:-1], CHANNELS[1:], strict=False)
            ]
        )

        self.feeling = nn.Conv1d(
            kernel_size=1,
            in_channels=CHANNELS[-1],
            out_channels=len(FEELING))

        self.emoji = nn.Conv1d(
            kernel_size=1,
            in_channels=CHANNELS[-1],
            out_channels=EMOJI_EMBED_SIZE)

        self.emoji_embed = nn.Embedding(len(EMOJIS), EMOJI_EMBED_SIZE)

    def forward(self, x):
        # (B, T) long -> (B, T, CHAR_EMBED_SIZE) -> (B, CHAR_EMBED_SIZE, T)
        # for Conv1d. PAD_IDX rows stay a fixed zero vector (padding_idx),
        # matching the old one-hot path that sliced off the PAD channel.
        out = self.char_embed(x).transpose(1, 2)
        out = self.conv(out)
        out = torch.max(out, dim=-1).values.unsqueeze(-1)  # (B, CHANNELS[-1], 1)

        return (
            self.feeling(out).squeeze(-1),
            normalize(self.emoji(out).squeeze(-1), p=2, dim=-1),
            normalize(self.emoji_embed.weight, p=2, dim=-1))
