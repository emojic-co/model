
import torch
from torch import nn
from torch.nn.functional import (
    normalize,
    tanh,
)
from torch.nn.utils import spectral_norm as sn

from model.color import COLOR_SHIFT
from model.config import (
    CHAR_EMBED_SIZE,
    CRITIC_COLOR_CHANNELS,
    CRITIC_TEXT_CHANNELS,
    DROPOUT_CRITIC,
    DROPOUT_EMOJI,
    DROPOUT_STYLE,
    EMOJI_EMBED_SIZE,
    ENCODER_CHANNELS,
    ENCODER_DILATION,
    ENCODER_KERNEL_SIZE,
    GEN_CHANNELS,
    RELU_SLOPE,
    STYLE_EMBED_SIZE,
    TEXT_EMBED_SIZE,
    Z_WEIGHT,
)
from model.data import COLOR_DIM, EMOJIS, PAD_IDX, STYLES, VOCAB_SIZE


class TextEncoderBlock(nn.Module):
    def __init__(self, i: int, o: int, dilation: int):
        super().__init__()
        self.net = nn.Sequential(
            sn(nn.Conv1d(
                i, o,
                kernel_size=ENCODER_KERNEL_SIZE,
                padding=dilation * (ENCODER_KERNEL_SIZE // 2),
                dilation=dilation,
                bias=True)),

            nn.LeakyReLU(negative_slope=RELU_SLOPE))

    def forward(self, x: torch.Tensor):
        return self.net(x)


class TextEncoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.char_embed = nn.Embedding(
            VOCAB_SIZE, CHAR_EMBED_SIZE, padding_idx=PAD_IDX)

        cs = ENCODER_CHANNELS
        io = zip([CHAR_EMBED_SIZE, *cs[:-1]], cs, ENCODER_DILATION, strict=True)

        self.blocks = nn.ModuleList(
            [TextEncoderBlock(i=i, o=o, dilation=d) for i, o, d in io])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.char_embed(x).transpose(1, 2)
        n = (x != PAD_IDX).sum(dim=1, keepdim=True)
        pooled = []
        for block in self.blocks:
            out = block(out)
            keep = torch.arange(out.shape[-1], device=x.device) < n
            masked = out.masked_fill(~keep.unsqueeze(1), float("-inf"))
            pooled.append(torch.max(masked, dim=-1).values)
        return torch.cat(pooled, dim=-1)


class StyleHead(nn.Module):
    def __init__(self):
        super().__init__()

        self.net = nn.Sequential(
            nn.Dropout(p=DROPOUT_STYLE),
            nn.Linear(TEXT_EMBED_SIZE, STYLE_EMBED_SIZE, bias=False))

        self.embed = nn.Embedding(len(STYLES), STYLE_EMBED_SIZE)
        self.bias = nn.Parameter(torch.zeros(len(STYLES)))

    def forward(self, text_embedding: torch.Tensor) -> torch.Tensor:
        s = self.net(text_embedding)
        return s @ self.embed.weight.t() + self.bias


class EmojiHead(nn.Module):
    def __init__(self):
        super().__init__()

        self.net = nn.Sequential(
            nn.Dropout(p=DROPOUT_EMOJI),
            nn.Linear(TEXT_EMBED_SIZE, EMOJI_EMBED_SIZE, bias=False))

        self.embed = nn.Embedding(len(EMOJIS), EMOJI_EMBED_SIZE)
        self.bias = nn.Parameter(torch.zeros(len(EMOJIS)))

    def forward(self, text_embedding: torch.Tensor) -> torch.Tensor:
        q = self.net(text_embedding)
        return q @ self.embed.weight.t() + self.bias


# GAN
class ColorGen(nn.Module):
    def __init__(self):
        super().__init__()

        io = zip(GEN_CHANNELS[:-1], GEN_CHANNELS[1:], strict=True)
        self.net = nn.Sequential(
            nn.Linear(TEXT_EMBED_SIZE, GEN_CHANNELS[0], bias=False),
            nn.LeakyReLU(negative_slope=RELU_SLOPE),
            *[
                nn.Sequential(
                    nn.Linear(i, o, bias=False),
                    nn.BatchNorm1d(o),
                    nn.LeakyReLU(negative_slope=RELU_SLOPE)
                )
                for i, o in io
            ],
            nn.Linear(GEN_CHANNELS[-1], COLOR_DIM),
        )

    def forward(
        self,
        cond: torch.Tensor,
        z: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if z is None:
            z = torch.randn_like(cond)
        z = normalize(z, dim=-1)
        seed = (1 - Z_WEIGHT) * normalize(cond) + Z_WEIGHT * z
        colors = self.net(seed)
        colors = tanh(colors) * COLOR_SHIFT

        return colors


def _critic_branch(in_dim: int, channels: list[int]) -> nn.Sequential:
    cs = [in_dim, *channels]
    io = zip(cs[:-1], cs[1:], strict=True)
    return nn.Sequential(
        *[
            nn.Sequential(
                sn(nn.Linear(i, o, bias=False)),
                nn.BatchNorm1d(o),
                nn.LeakyReLU(negative_slope=RELU_SLOPE)
            )
            for i, o in io
        ]
    )


class ColorCritic(nn.Module):
    def __init__(self):
        super().__init__()

        self.color_net = _critic_branch(COLOR_DIM, CRITIC_COLOR_CHANNELS)
        self.text_net = nn.Sequential(
            nn.Dropout(p=DROPOUT_CRITIC),
            _critic_branch(TEXT_EMBED_SIZE, CRITIC_TEXT_CHANNELS))
        self.proj = nn.Linear(
            CRITIC_TEXT_CHANNELS[-1], CRITIC_COLOR_CHANNELS[-1], bias=False)
        self.out = nn.Linear(CRITIC_COLOR_CHANNELS[-1], 1, bias=True)

    def forward(self, cond: torch.Tensor, colors: torch.Tensor) -> torch.Tensor:
        c = self.color_net(colors / COLOR_SHIFT)
        t = self.text_net(normalize(cond))

        # return self.out(c) + (self.proj(t) * c).sum(dim=-1, keepdim=True)
        return (self.proj(t) * c).sum(dim=-1, keepdim=True)
