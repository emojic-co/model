from torch import nn

from config import (
    CHANNELS,
    CHAR_EMBED_SIZE,
    EMOJI_EMBED_SIZE,
    HIDDEN,
    KERNEL_1,
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
        # (B, T) long -> (B, T, CHAR_EMBED_SIZE) -> (B, CHAR_EMBED_SIZE, T)
        # for Conv1d. PAD_IDX rows stay a fixed zero vector (padding_idx),
        # matching the old one-hot path that sliced off the PAD channel.
        out = self.char_embed(x).transpose(1, 2)

        out = self.conv(out)
        _, (h, _) = self.lstm(out.transpose(1, 2))
        # out = torch.max(out, dim=1).values

        return (
            self.feeling(h[-1]),
            self.emoji_proj(h[-1]),
            self.emoji_embed.weight)
