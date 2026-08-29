import torch
from torch import nn

from config import (
    CHANNELS,
    CHAR_EMBED_SIZE,
)
from data import FEELING, VOCAB_SIZE


class Model(nn.Module):
    def __init__(self):
        super().__init__()

        # PAD ("·", index 0) is just another vocab char: it gets its own
        # trainable embedding row like every other char (no padding_idx), and
        # nothing downstream masks the padded tail.
        self.char_embed = nn.Embedding(
            VOCAB_SIZE,
            CHAR_EMBED_SIZE)

        self.conv = nn.Sequential(
            nn.Conv1d(
                in_channels=CHAR_EMBED_SIZE,
                out_channels=CHANNELS[0],
                kernel_size=2,
                bias=False
            ),

            nn.BatchNorm1d(CHANNELS[0]),
            nn.ReLU(),

            *[
                nn.Sequential(
                    nn.MaxPool1d(
                        kernel_size=2,
                        stride=2),

                    nn.Conv1d(
                        in_channels=i,
                        out_channels=o,
                        kernel_size=2,
                        bias=False
                    ),

                    nn.BatchNorm1d(o),
                    nn.ReLU(),
                )
                for i, o in zip(CHANNELS[:-1], CHANNELS[1:], strict=False)
            ]
        )

        self.feeling = nn.Linear(
            CHANNELS[-1],
            len(FEELING))

        # self.emoji = nn.Linear(CHANNELS[-1], EMOJI_EMBED_SIZE)

        # self.emoji_embed = nn.Embedding(len(EMOJIS), EMOJI_EMBED_SIZE)

    def forward(self, x):
        # (B, T) long -> (B, T, CHAR_EMBED_SIZE) -> (B, CHAR_EMBED_SIZE, T)
        # for Conv1d. PAD is not special: its embedding is learned and the
        # padded tail is left in, so the global max over time sees every
        # timestep, pad-contaminated ones included.
        out = self.char_embed(x).transpose(1, 2)
        out = self.conv(out)  # (B, CHANNELS[-1], L)

        out = torch.max(out, dim=-1).values  # (B, CHANNELS[-1])

        return (
            self.feeling(out),
            None,
            None,
            # normalize(self.emoji(out), p=2, dim=-1),
            # normalize(self.emoji_embed.weight, p=2, dim=-1)
        )
