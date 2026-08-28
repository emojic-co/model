import torch
from torch import nn
from torch.nn import functional as F

from config import (
    CHANNELS_1,
    CHANNELS_2,
    KERNEL_1,
    KERNEL_2,
)
from data import FEELING, VOCAB_SIZE


class Model(nn.Module):
    def __init__(self):
        super().__init__()

        # Vocab is one-hot, not a learned embedding: forward() expands each
        # char index to a one-hot vector and drops channel 0 (PAD_IDX), so a
        # pad step is all zeros -- exactly what padding_idx=0 gave before.
        #
        # bias=False so a conv window lying entirely in the pad region produces
        # exactly 0 (pad steps are zero vectors). That keeps pad steps from
        # winning the global max below, so no pad mask is needed.
        self.net = nn.Sequential(
            nn.Conv1d(
                in_channels=VOCAB_SIZE - 1,
                out_channels=CHANNELS_1,
                kernel_size=KERNEL_1,
                padding=0,
                bias=False,
            ),
            nn.LeakyReLU(),

            nn.MaxPool1d(
                kernel_size=3,
                stride=2),

            nn.Conv1d(
                in_channels=CHANNELS_1,
                out_channels=CHANNELS_2,
                kernel_size=KERNEL_2,
                padding=0,
                bias=False,
            ),
            nn.LeakyReLU(),
        )

        self.feeling = nn.Linear(CHANNELS_2, len(FEELING))

    def forward(self, x):
        out = F.one_hot(x, VOCAB_SIZE)[..., 1:].float()  # drop PAD channel
        out = out.permute(0, 2, 1)
        out = self.net(out)

        out = torch.max(out, dim=2).values  # (batch, CHANNELS_2)

        return self.feeling(out)
