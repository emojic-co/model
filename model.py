import torch
from torch import nn
from torch.nn import functional as F

from config import (
    CHANNELS,
    EMOJI_EMBED_SIZE,
    HIDDEN,
    KERNEL_1,
)
from data import EMOJIS, FEELING, VOCAB_SIZE


class Model(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv1d(
                in_channels=VOCAB_SIZE - 1,
                out_channels=CHANNELS,
                kernel_size=KERNEL_1,
                padding=0,
                bias=False,
            ),
            nn.ReLU(),

            nn.MaxPool1d(
                kernel_size=3,
                stride=2),
        )

        self.lstm = nn.LSTM(
            input_size=CHANNELS,
            hidden_size=HIDDEN,
            num_layers=1,
            batch_first=True,
            bidirectional=False,
        )

        self.feeling = nn.Linear(HIDDEN, len(FEELING))

        self.emoji_proj = nn.Linear(HIDDEN, EMOJI_EMBED_SIZE)
        self.emoji_embed = nn.Embedding(len(EMOJIS), EMOJI_EMBED_SIZE)

    def forward(self, x):
        out = F \
            .one_hot(x, VOCAB_SIZE)[:, :, 1:] \
            .transpose(1, 2) \
            .to(torch.float32)

        out = self.conv(out)
        _, (h, _) = self.lstm(out.transpose(1, 2))
        # out = torch.max(out, dim=1).values

        return (
            self.feeling(h[-1]),
            self.emoji_proj(h[-1]),
            self.emoji_embed.weight)
