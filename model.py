import torch
from torch import nn
from torch.nn.functional import normalize

from config import (
    CHANNELS,
    CHAR_EMBED_SIZE,
    EMOJI_EMBED_SIZE,
)
from data import EMOJIS, FEELING, VOCAB_SIZE


class Layer(nn.Sequential):
    """One bigram block: Conv1d(k=2) -> BatchNorm1d -> LeakyReLU -> MaxPool1d(2, 2).

    The kernel-2 conv (stride 1, no padding) drops one time step; the stride-2
    pool then halves what is left (floor division -- an odd length loses its
    last step). Stacked, these blocks build a hierarchy of character n-grams.
    """

    def __init__(self, in_channels, out_channels):
        super().__init__(
            nn.Conv1d(in_channels, out_channels, kernel_size=2, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.LeakyReLU(negative_slope=0.1),
            nn.MaxPool1d(kernel_size=2, stride=2),
        )


class Model(nn.Module):
    def __init__(self):
        super().__init__()

        # PAD ("·", index 0) is just another vocab char: it gets its own
        # trainable embedding row like every other char (no padding_idx), and
        # nothing downstream masks the padded tail.
        self.char_embed = nn.Embedding(VOCAB_SIZE, CHAR_EMBED_SIZE)

        # Stack the config.CHANNELS bigram blocks, threading each block's output
        # channel count into the next. Every block halves the time axis (see
        # Layer), so after N blocks T is roughly MAX_TEXT_LEN >> N; the global
        # max over time in forward then collapses whatever remains to one vector
        # per sequence.
        in_channels = [CHAR_EMBED_SIZE, *CHANNELS[:-1]]
        self.conv = nn.Sequential(
            *(Layer(i, o) for i, o in zip(in_channels, CHANNELS, strict=True))
        )

        self.feeling = nn.Linear(CHANNELS[-1], len(FEELING))

        self.emoji = nn.Linear(CHANNELS[-1], EMOJI_EMBED_SIZE)
        self.emoji_embed = nn.Embedding(len(EMOJIS), EMOJI_EMBED_SIZE)

    def forward(self, x):
        # (B, T) long -> (B, T, CHAR_EMBED_SIZE) -> (B, CHAR_EMBED_SIZE, T)
        # for Conv1d. PAD is not special: its embedding is learned and the
        # padded tail is left in, so the global max over time sees every
        # timestep, pad-contaminated ones included.
        out = self.char_embed(x).transpose(1, 2)
        out = self.conv(out)  # (B, channels, T)

        out = torch.max(out, dim=-1).values  # (B, channels)

        return (
            self.feeling(out),
            normalize(self.emoji(out), p=2, dim=-1),
            normalize(self.emoji_embed.weight, p=2, dim=-1),
        )
