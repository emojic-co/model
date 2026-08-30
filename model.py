import torch
from torch import nn
from torch.nn.functional import normalize

from config import (
    CHANNELS,
    CHAR_EMBED_SIZE,
    EMOJI_EMBED_SIZE,
    KERNELS,
)
from data import EMOJIS, FEELING, VOCAB_SIZE


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
                kernel_size=KERNELS[0],
                bias=False
            ),

            nn.BatchNorm1d(CHANNELS[0]),
            nn.LeakyReLU(negative_slope=0.1),

            *[
                nn.Sequential(
                    nn.MaxPool1d(
                        kernel_size=2,
                        stride=2),

                    nn.Conv1d(
                        in_channels=i,
                        out_channels=o,
                        kernel_size=k,
                        bias=False
                    ),

                    nn.BatchNorm1d(o),
                    nn.LeakyReLU(negative_slope=0.1),
                )
                for i, o, k in zip(
                    CHANNELS[:-1],
                    CHANNELS[1:],
                    KERNELS[1:],
                    strict=True)
            ]
        )

        # One order-sensitive layer over the conv feature sequence. The global
        # max-pool below is otherwise a bag of n-grams -- word order and "not"
        # are invisible to it (Anxious collapses into Love/Sad, `not happy`
        # still reads Happy). A single bidirectional GRU reads the CHANNELS[-1]
        # conv features left-to-right and right-to-left; the heads pool its
        # 2*CHANNELS[-1] output.
        self.gru = nn.GRU(
            input_size=CHANNELS[-1],
            hidden_size=CHANNELS[-1],
            num_layers=1,
            bidirectional=True,
            batch_first=True,
        )

        self.feeling = nn.Linear(
            2 * CHANNELS[-1],
            len(FEELING))

        self.emoji = nn.Linear(2 * CHANNELS[-1], EMOJI_EMBED_SIZE)
        self.emoji_embed = nn.Embedding(len(EMOJIS), EMOJI_EMBED_SIZE)

    def forward(self, x):
        # (B, T) long -> (B, T, CHAR_EMBED_SIZE) -> (B, CHAR_EMBED_SIZE, T)
        # for Conv1d. PAD is not special: its embedding is learned and the
        # padded tail is left in, so the global max over time sees every
        # timestep, pad-contaminated ones included.
        out = self.char_embed(x).transpose(1, 2)
        out = self.conv(out)  # (B, CHANNELS[-1], L)

        # (B, C, L) -> (B, L, C) for the GRU, read the whole sequence (the
        # pad-contaminated tail included, as above), then max-pool over time
        # on the bidirectional 2*C output.
        out, _ = self.gru(out.transpose(1, 2))  # (B, L, 2*CHANNELS[-1])
        out = torch.max(out, dim=1).values  # (B, 2*CHANNELS[-1])

        return (
            self.feeling(out),
            normalize(self.emoji(out), p=2, dim=-1),
            normalize(self.emoji_embed.weight, p=2, dim=-1)
        )
