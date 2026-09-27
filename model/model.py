
import torch
from torch import nn
from torch.nn.functional import normalize
from torch.nn.utils import spectral_norm as sn

from model.config import (
    DROPOUT,
    EMBED_SIZE_CHAR,
    EMBED_SIZE_EMOJI,
    EMBED_SIZE_STYLE,
    EMBED_SIZE_TEXT,
    ENCODER_CHANNELS,
    ENCODER_DILATION,
    ENCODER_KERNEL_SIZE,
    HIDDEN_SIZE_CRITIC,
    HIDDEN_SIZE_GEN,
    RELU_SLOPE,
    Z_DIM,
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
def gblk(i: int, o: int):
    return [
        nn.Linear(i, o, bias=False),
        # nn.LayerNorm(o),
        nn.LeakyReLU(RELU_SLOPE)]


class ColorGen(nn.Module):
    def __init__(self):
        super().__init__()

        self.z_net = nn.Sequential(
            *gblk(Z_DIM, HIDDEN_SIZE_GEN),
        )

        self.text_net = nn.Sequential(
            *gblk(EMBED_SIZE_TEXT, HIDDEN_SIZE_GEN),
        )

        self.mlp = nn.Sequential(
            *gblk(HIDDEN_SIZE_GEN, HIDDEN_SIZE_GEN),
            nn.Linear(HIDDEN_SIZE_GEN, COLOR_DIM)
        )

    def forward(self, cond: torch.Tensor) -> torch.Tensor:
        z = torch.randn(
            (len(cond), Z_DIM),
            device=cond.device,
            dtype=cond.dtype)

        z = self.z_net(z)
        t = self.text_net(cond)

        self.last_norm_z = z.detach().norm(dim=-1).mean()
        self.last_norm_t = t.detach().norm(dim=-1).mean()

        z = normalize(z, dim=-1)
        t = normalize(t, dim=-1)

        raw = self.mlp(t + Z_WEIGHT * z)
        # Output directly in OKLab space: L bounded to [0, 1], a/b unbounded.
        shape = raw.shape
        raw = raw.reshape(*shape[:-1], -1, 3)
        colors = torch.cat([raw[..., :1].sigmoid(), raw[..., 1:]], dim=-1)
        return colors.reshape(shape)


def cblk(i: int, o: int):
    return [
        sn(nn.Linear(i, o, bias=False)),
        nn.LayerNorm(o),
        nn.LeakyReLU(RELU_SLOPE)]


class Critic(nn.Module):
    def __init__(self):
        super().__init__()

        self.text_net = nn.Sequential(
            *cblk(EMBED_SIZE_TEXT, HIDDEN_SIZE_CRITIC)
        )

        self.color_net = nn.Sequential(
            *cblk(COLOR_DIM, HIDDEN_SIZE_CRITIC),
        )

        self.color_class = nn.Sequential(
            # *cblk(HIDDEN_SIZE_CRITIC, HIDDEN_SIZE_CRITIC),
            sn(nn.Linear(HIDDEN_SIZE_CRITIC, 1)),
        )

        self.text_color_class = nn.Sequential(
            # *cblk(HIDDEN_SIZE_CRITIC, HIDDEN_SIZE_CRITIC),
            sn(nn.Linear(HIDDEN_SIZE_CRITIC, 1))
        )

    def forward(self, cond: torch.Tensor, colors: torch.Tensor) -> torch.Tensor:
        """`colors` is expected in OKLab space (see `ColorGen.forward` /
        `model.color.rgb_to_oklab`)."""
        t = self.text_net(cond)
        c = self.color_net(colors)

        return \
            self.color_class(c) + \
            self.text_color_class(t * c)
