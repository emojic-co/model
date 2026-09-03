
import torch
from torch import nn
from torch.nn.functional import (
    max_pool1d,
    normalize,
    tanh,
)
from torch.nn.utils import spectral_norm as sn

from config import (
    CHAR_EMBED_SIZE,
    CRITIC_CHANNELS,
    DROPOUT_EMOJI,
    DROPOUT_STYLE,
    EMOJI_EMBED_SIZE,
    ENCODER_CHANNELS,
    GEN_CHANNELS,
    RELU_SLOPE,
    STYLE_EMBED_SIZE,
    TEXT_EMBED_SIZE,
    Z_WEIGHT,
)
from data import COLOR_DIM, EMOJIS, PAD_IDX, STYLES, VOCAB_SIZE


class TextEncoderBlock(nn.Module):
    def __init__(self, i: int, o: int):
        super().__init__()
        self.net = nn.Sequential(
            sn(nn.Conv1d(i, o, kernel_size=3, padding=1, bias=True)),
            nn.LeakyReLU(negative_slope=RELU_SLOPE))
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)

    def forward(
        self, x: torch.Tensor, mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        x = x.masked_fill(mask[:, None, :], 0.0)
        out = self.net(x)
        out = out.masked_fill(mask[:, None, :], float("-inf"))
        out = self.pool(out)

        valid = max_pool1d(
            (~mask).float()[:, None, :], kernel_size=2, stride=2).squeeze(1)
        return out, valid == 0


class TextEncoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.char_embed = nn.Embedding(VOCAB_SIZE, CHAR_EMBED_SIZE)

        cs = ENCODER_CHANNELS
        io = zip([CHAR_EMBED_SIZE, *cs[:-1]], cs, strict=True)

        self.blocks = nn.ModuleList(
            [TextEncoderBlock(i=i, o=o) for i, o in io])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mask = x == PAD_IDX
        out = self.char_embed(x).transpose(1, 2)

        for block in self.blocks:
            out, mask = block(out, mask)

        out = out.masked_fill(mask[:, None, :], float("-inf"))
        return torch.max(out, dim=-1).values


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
COLOR_SCALE = 127.5

_LIN_TO_LMS = torch.tensor([
    [0.4122214708, 0.5363325363, 0.0514459929],
    [0.2119034982, 0.6806995451, 0.1073969566],
    [0.0883024619, 0.2817188376, 0.6299787005],
])
_LMS_TO_LAB = torch.tensor([
    [0.2104542553, 0.7936177850, -0.0040720468],
    [1.9779984951, -2.4285922050, 0.4505937099],
    [0.0259040371, 0.7827717662, -0.8086757660],
])
_LAB_TO_LMS = torch.tensor([
    [1.0, 0.3963377774, 0.2158037573],
    [1.0, -0.1055613458, -0.0638541728],
    [1.0, -0.0894841775, -1.2914855480],
])
_LMS_TO_LIN = torch.tensor([
    [4.0767416621, -3.3077115913, 0.2309699292],
    [-1.2684380046, 2.6097574011, -0.3413193965],
    [-0.0041960863, -0.7034186147, 1.7076147010],
])


def _srgb_to_linear(c: torch.Tensor) -> torch.Tensor:
    return torch.where(
        c <= 0.04045, c / 12.92, ((c.clamp(min=0.0) + 0.055) / 1.055) ** 2.4)


def _linear_to_srgb(c: torch.Tensor) -> torch.Tensor:
    return torch.where(
        c <= 0.0031308, c * 12.92, 1.055 * c.clamp(min=0.0) ** (1 / 2.4) - 0.055)


def rgb_to_oklab(rgb: torch.Tensor) -> torch.Tensor:
    shape = rgb.shape
    c = ((rgb + COLOR_SCALE) / 255.0).clamp(0.0, 1.0).reshape(*shape[:-1], -1, 3)
    lms = _srgb_to_linear(c) @ _LIN_TO_LMS.to(c).t()
    lms_ = lms.sign() * lms.abs().clamp(min=1e-12) ** (1 / 3)
    return (lms_ @ _LMS_TO_LAB.to(c).t()).reshape(shape)


def oklab_to_rgb(lab: torch.Tensor) -> torch.Tensor:
    shape = lab.shape
    x = lab.reshape(*shape[:-1], -1, 3)
    lms = (x @ _LAB_TO_LMS.to(x).t()) ** 3
    c = _linear_to_srgb(lms @ _LMS_TO_LIN.to(x).t())
    return (c.clamp(0.0, 1.0) * 255.0 - COLOR_SCALE).reshape(shape)


class ColorGen(nn.Module):
    def __init__(self):
        super().__init__()

        io = zip(GEN_CHANNELS[:-1], GEN_CHANNELS[1:], strict=True)
        self.net = nn.Sequential(
            nn.Linear(TEXT_EMBED_SIZE, GEN_CHANNELS[0], bias=False),
            nn.BatchNorm1d(GEN_CHANNELS[0]),
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
        colors = tanh(colors) * COLOR_SCALE

        return colors


class ColorDsc(nn.Module):
    def __init__(self):
        super().__init__()

        cs = [COLOR_DIM + TEXT_EMBED_SIZE, *CRITIC_CHANNELS]
        io = zip(cs[:-1], cs[1:], strict=True)
        self.net = nn.Sequential(
            *[
                nn.Sequential(
                    sn(nn.Conv1d(i, o, kernel_size=1, bias=True)),
                    # nn.Conv1d(i, o, kernel_size=1, bias=True),
                    nn.LeakyReLU(negative_slope=RELU_SLOPE)
                )
                for i, o in io
            ],
            nn.Conv1d(cs[-1], 1, kernel_size=1, bias=True),
        )

    def forward(self, cond: torch.Tensor, colors: torch.Tensor) -> torch.Tensor:
        c = rgb_to_oklab(colors)

        x = torch.cat([
            c.unsqueeze(-1),
            normalize(cond).unsqueeze(-1)], dim=1)

        return self.net(x)
