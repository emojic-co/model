import torch
from torch import nn
from torch.nn.functional import normalize

from config import (
    CHANNELS,
    CHAR_EMBED_SIZE,
    EMOJI_EMBED_SIZE,
    KERNEL_1,
)
from data import EMOJIS, FEELING, PAD_IDX, VOCAB_SIZE

CS1, CS2 = CHANNELS


class Model(nn.Module):
    def __init__(self):
        super().__init__()

        self.char_embed = nn.Embedding(
            VOCAB_SIZE, CHAR_EMBED_SIZE, padding_idx=PAD_IDX)

        self.conv = nn.Sequential(
            nn.Conv1d(
                in_channels=CHAR_EMBED_SIZE,
                out_channels=CS1,
                kernel_size=KERNEL_1,
                stride=2,
                padding=0,
                bias=False,
            ),

            nn.ReLU(),

            nn.Conv1d(
                in_channels=CS1,
                out_channels=CS2,
                kernel_size=KERNEL_1,
                stride=2,
                padding=0,
                bias=False,
            ),

            nn.ReLU(),
        )

        self.feeling = nn.Conv1d(
            kernel_size=1,
            in_channels=CS2,
            out_channels=len(FEELING))

        self.emoji = nn.Conv1d(
            kernel_size=1,
            in_channels=CS2,
            out_channels=EMOJI_EMBED_SIZE)

        self.emoji_embed = nn.Embedding(len(EMOJIS), EMOJI_EMBED_SIZE)

    def forward(self, x):
        # (B, T) long -> (B, T, CHAR_EMBED_SIZE) -> (B, CHAR_EMBED_SIZE, T)
        # for Conv1d. PAD_IDX rows stay a fixed zero vector (padding_idx),
        # matching the old one-hot path that sliced off the PAD channel.
        out = self.char_embed(x).transpose(1, 2)
        out = self.conv(out)
        out = torch.max(out, dim=-1).values.unsqueeze(-1)  # (B, CS2, 1)

        return (
            self.feeling(out).squeeze(-1),
            normalize(self.emoji(out).squeeze(-1), p=2, dim=-1),
            normalize(self.emoji_embed.weight, p=2, dim=-1))
