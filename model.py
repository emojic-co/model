import torch
from torch import nn
from torch.nn.functional import normalize

from config import (
    CHAR_EMBED_SIZE,
    EMOJI_EMBED_SIZE,
    MODEL,
)
from data import EMOJIS, FEELING, VOCAB_SIZE


class InceptionLayer(nn.Module):
    """One ``config.MODEL`` layer: parallel Conv1d branches, outputs concatenated.

    Each ``(kernel_size, out_channels)`` pair in the layer spec becomes an
    independent branch of Conv1d -> BatchNorm1d -> LeakyReLU. Every branch uses
    ``padding="same"`` (stride 1) so, whatever its kernel size, it returns the
    same time length as its input -- that is what lets the branch outputs be
    concatenated along the channel axis. The layer's output channel count is the
    sum of the branches' ``out_channels``.
    """

    def __init__(self, in_channels, branches):
        super().__init__()
        self.branches = nn.ModuleList(
            nn.Sequential(
                nn.Conv1d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    padding="same",
                    bias=False,
                ),
                nn.BatchNorm1d(out_channels),
                nn.LeakyReLU(negative_slope=0.1),
            )
            for kernel_size, out_channels in branches
        )
        self.out_channels = sum(out_channels for _, out_channels in branches)

    def forward(self, x):
        return torch.cat([branch(x) for branch in self.branches], dim=1)


class Model(nn.Module):
    def __init__(self):
        super().__init__()

        # PAD ("·", index 0) is just another vocab char: it gets its own
        # trainable embedding row like every other char (no padding_idx), and
        # nothing downstream masks the padded tail.
        self.char_embed = nn.Embedding(VOCAB_SIZE, CHAR_EMBED_SIZE)

        # Stack the config.MODEL layers, threading each layer's concatenated
        # channel count into the next. "same" padding keeps the time axis at
        # MAX_TEXT_LEN the whole way through -- there is no pooling between
        # layers; the only length reduction is the global max over time below.
        channels = CHAR_EMBED_SIZE
        layers = []
        for branches in MODEL:
            layer = InceptionLayer(channels, branches)
            layers.append(layer)
            channels = layer.out_channels
        self.conv = nn.Sequential(*layers)

        self.feeling = nn.Linear(channels, len(FEELING))

        self.emoji = nn.Linear(channels, EMOJI_EMBED_SIZE)
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
