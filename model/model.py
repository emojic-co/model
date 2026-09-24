
import torch
import torch.nn as nn
from torch.nn.functional import (
    normalize,
    tanh,
)
from torch.nn.utils.parametrizations import spectral_norm as sn

from model.color import COLOR_SHIFT
from model.config import (
    DROPOUT,
    EMBED_SIZE_CHAR,
    EMBED_SIZE_COLOR,
    EMBED_SIZE_EMOJI,
    EMBED_SIZE_STYLE,
    EMBED_SIZE_TEXT,
    ENCODER_CHANNELS,
    ENCODER_DILATION,
    ENCODER_KERNEL_SIZE,
    GEN_HIDDEN_SIZE,
    RELU_SLOPE,
    Z_WEIGHT,
)
from model.data import COLOR_DIM, EMOJIS, PAD_IDX, STYLES, VOCAB_SIZE


def blk(i: int, o: int):
    return [
        nn.Linear(i, o, bias=False),
        nn.LayerNorm(o),
        nn.LeakyReLU(negative_slope=RELU_SLOPE)]


class TextEncoderBlock(nn.Module):
    def __init__(self, i: int, o: int, dilation: int, num_groups: int = 8):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(
                i, o,
                kernel_size=ENCODER_KERNEL_SIZE,
                padding=dilation * (ENCODER_KERNEL_SIZE // 2),
                dilation=dilation,
                bias=False  # Norm layer provides affine bias
            ),
            nn.GroupNorm(num_groups=1, num_channels=o),
            nn.LeakyReLU(negative_slope=RELU_SLOPE)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TextEncoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.char_embed = nn.Embedding(
            VOCAB_SIZE, EMBED_SIZE_CHAR, padding_idx=PAD_IDX)

        cs = ENCODER_CHANNELS
        io = zip([EMBED_SIZE_CHAR, *cs[:-1]], cs, ENCODER_DILATION, strict=True)

        self.blocks = nn.ModuleList(
            [TextEncoderBlock(i=i, o=o, dilation=d) for i, o, d in io])

        self.proj = nn.Sequential(
            nn.Dropout(p=DROPOUT),
            *blk(sum(cs), EMBED_SIZE_TEXT))

    def pool(self, x: torch.Tensor) -> torch.Tensor:
        out = self.char_embed(x).transpose(1, 2)
        n = (x != PAD_IDX).sum(dim=1, keepdim=True)
        pooled = []
        for block in self.blocks:
            out = block(out)
            keep = torch.arange(out.shape[-1], device=x.device) < n
            masked = out.masked_fill(~keep.unsqueeze(1), float("-inf"))
            pooled.append(torch.max(masked, dim=-1).values)

        return torch.cat(pooled, dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(self.pool(x))


class StyleHead(nn.Module):
    def __init__(self):
        super().__init__()

        self.embed = nn.Embedding(len(STYLES), EMBED_SIZE_STYLE)
        self.bias = nn.Parameter(torch.zeros(len(STYLES)))
        self.net = nn.Linear(
            EMBED_SIZE_TEXT,
            EMBED_SIZE_STYLE, bias=False)

    def forward(self, text_embedding: torch.Tensor) -> torch.Tensor:
        s = self.net(text_embedding)
        return s @ self.embed.weight.t() + self.bias


class EmojiHead(nn.Module):
    def __init__(self):
        super().__init__()

        self.net = nn.Linear(EMBED_SIZE_TEXT, EMBED_SIZE_EMOJI, bias=False)
        self.embed = nn.Embedding(len(EMOJIS), EMBED_SIZE_EMOJI)
        self.bias = nn.Parameter(torch.zeros(len(EMOJIS)))

    def score(self, q: torch.Tensor) -> torch.Tensor:
        return q @ self.embed.weight.t() + self.bias

    def forward(self, text_embedding: torch.Tensor) -> torch.Tensor:
        return self.score(self.net(text_embedding))


# GAN
class ColorGen(nn.Module):
    def __init__(self):
        super().__init__()

        self.net = nn.Sequential(
            *blk(EMBED_SIZE_TEXT, GEN_HIDDEN_SIZE),
            *blk(GEN_HIDDEN_SIZE, GEN_HIDDEN_SIZE),
            nn.Linear(GEN_HIDDEN_SIZE, COLOR_DIM))

    def forward(
        self,
        cond: torch.Tensor,
        z: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if z is None:
            z = torch.randn_like(cond, device=cond.device, dtype=cond.dtype)

        assert cond.shape == z.shape, \
            f"cond and z must have the same shape, got {cond.shape} and {z.shape}"

        z = normalize(z, dim=-1)
        cond = normalize(cond, dim=-1)

        seed = (1 - Z_WEIGHT) * cond + Z_WEIGHT * z

        colors = self.net(seed)
        return tanh(colors) * COLOR_SHIFT


def cblk(i: int, o: int):
    return [
        sn(nn.Linear(i, o)),
        nn.LeakyReLU(RELU_SLOPE)]


class ColorEmbedding(nn.Module):
    def __init__(self):
        super().__init__()

        self.net = nn.Sequential(
            *cblk(COLOR_DIM, EMBED_SIZE_COLOR),
            *cblk(EMBED_SIZE_COLOR, EMBED_SIZE_COLOR),
            *cblk(EMBED_SIZE_COLOR, EMBED_SIZE_COLOR))

    def forward(self, colors: torch.Tensor) -> torch.Tensor:
        return self.net(colors)


class ColorCritic(nn.Module):
    def __init__(self):
        super().__init__()

        self.color_critic = nn.Sequential(
            sn(nn.Linear(EMBED_SIZE_COLOR, 1)))

    def forward(self, color_embedding: torch.Tensor):

        return self.color_critic(color_embedding)


class CondColorCritic(nn.Module):
    def __init__(self):
        super().__init__()

        self.text_embedding = nn.Sequential(
            *cblk(EMBED_SIZE_TEXT, EMBED_SIZE_COLOR))

    def forward(self, cond: torch.Tensor, color_embedding: torch.Tensor):
        t = self.text_embedding(cond)

        return (t * color_embedding).sum(dim=-1, keepdim=True)
