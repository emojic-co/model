from torch import nn

from config import (
    CHANNELS,
    EMBED_SIZE,
    HIDDEN,
    KERNEL_1,
)
from data import FEELING, VOCAB_SIZE


class Model(nn.Module):
    def __init__(self):
        super().__init__()

        self.embed = nn.Embedding(
            VOCAB_SIZE,
            EMBED_SIZE,
            padding_idx=0)

        self.conv = nn.Sequential(
            nn.Conv1d(
                in_channels=EMBED_SIZE,
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

    def forward(self, x):
        out = self.embed(x).transpose(1, 2)  # (B, EMBED_SIZE, T)

        out = self.conv(out)
        _, (h, _) = self.lstm(out.transpose(1, 2))
        # out = torch.max(out, dim=1).values

        return self.feeling(h[-1])
