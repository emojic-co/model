
import torch
from torch import nn
from torch.nn.functional import (
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
    EMOJI_HIDDEN_LAYERS,
    ENCODER_KERNEL,
    GEN_CHANNELS,
    RELU_SLOPE,
    STYLE_EMBED_SIZE,
    TEXT_EMBED_SIZE,
    TEXT_ENCODER_CHANNELS,
    Z_WEIGHT,
)
from data import COLOR_DIM, EMOJIS, STYLES, VOCAB_SIZE


class TextEncoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.char_embed = nn.Embedding(VOCAB_SIZE, CHAR_EMBED_SIZE)

        def conv_norm(*, k: int, i: int, o: int) -> nn.Sequential:
            return nn.Sequential(
                nn.Conv1d(i, o, kernel_size=k, bias=False),
                nn.BatchNorm1d(o)
            )

        def conv_norm_relu(*, k: int, i: int, o: int) -> nn.Sequential:
            return nn.Sequential(
                conv_norm(k=k, i=i, o=o),
                nn.LeakyReLU(negative_slope=RELU_SLOPE))

        def pool_conv_norm_relu(*, k: int, i: int, o: int) -> nn.Sequential:
            return nn.Sequential(
                nn.MaxPool1d(kernel_size=2, stride=2),
                conv_norm_relu(k=k, i=i, o=o))

        cs = TEXT_ENCODER_CHANNELS
        io = zip(cs[:-1], cs[1:], strict=True)

        self.encoder = nn.Sequential(
            conv_norm_relu(k=ENCODER_KERNEL, i=CHAR_EMBED_SIZE, o=cs[0]),
            *[
                pool_conv_norm_relu(k=ENCODER_KERNEL, i=i, o=o)
                for i, o in io])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.char_embed(x).transpose(1, 2)
        out = self.encoder(out)
        return torch.max(out, dim=-1).values


class StyleHead(nn.Module):
    def __init__(self):
        super().__init__()

        self.net = nn.Sequential(
            nn.Dropout(p=DROPOUT_STYLE),
            nn.Linear(TEXT_EMBED_SIZE, STYLE_EMBED_SIZE, bias=False)
        )
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
            nn.Linear(TEXT_EMBED_SIZE, EMOJI_EMBED_SIZE, bias=False),
            *[
                nn.Sequential(
                    nn.LeakyReLU(negative_slope=RELU_SLOPE),
                    nn.Linear(EMOJI_EMBED_SIZE, EMOJI_EMBED_SIZE, bias=False))

                for _ in range(EMOJI_HIDDEN_LAYERS)])

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
            nn.Linear(TEXT_EMBED_SIZE, GEN_CHANNELS[0]),
            nn.LeakyReLU(negative_slope=RELU_SLOPE),
            *[
                nn.Sequential(
                    nn.Linear(i, o),
                    nn.LeakyReLU(negative_slope=RELU_SLOPE)
                )
                for i, o in io
            ],
            nn.Linear(GEN_CHANNELS[-1], COLOR_DIM),
        )

    def forward(
        self,
        cond: torch.Tensor,
    ) -> torch.Tensor:
        z = normalize(torch.randn_like(cond), dim=-1)
        seed = (1 - Z_WEIGHT) * normalize(cond) + Z_WEIGHT * z
        colors = self.net(seed)
        colors = tanh(colors) * 127.5

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
                    nn.LeakyReLU(negative_slope=RELU_SLOPE)
                )
                for i, o in io
            ],
            nn.Conv1d(cs[-1], 1, kernel_size=1, bias=True),
        )

    def forward(self, cond: torch.Tensor, colors: torch.Tensor) -> torch.Tensor:
        x = torch.cat(
            [colors.unsqueeze(-1), cond.unsqueeze(-1)], dim=1)
        return self.net(x)
