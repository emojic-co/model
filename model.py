
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
    CRITIC_RELU_SLOPE,
    DROPOUT_FEELING,
    EMOJI_EMBED_SIZE,
    ENCODER_KERNEL,
    ENCODER_RELU_SLOPE,
    GEN_Z_DIM,
    POOL_1D_SIZE,
    TEXT_EMBED_SIZE,
    TEXT_ENCODER_CHANNELS,
)
from data import COLOR_DIM, EMOJIS, FEELINGS, VOCAB_SIZE


class TextEncoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.char_embed = nn.Embedding(VOCAB_SIZE, CHAR_EMBED_SIZE)

        def conv_sn(*, k: int, i: int, o: int) -> nn.Sequential:
            return nn.Sequential(
                sn(nn.Conv1d(i, o, kernel_size=k, bias=False)),
                # nn.BatchNorm1d(o)
            )

        def conv_sn_relu(*, k: int, i: int, o: int) -> nn.Sequential:
            return nn.Sequential(
                conv_sn(k=k, i=i, o=o),
                nn.LeakyReLU(negative_slope=ENCODER_RELU_SLOPE))

        def pool_conv_sn_relu(*, k: int, i: int, o: int) -> nn.Sequential:
            return nn.Sequential(
                nn.MaxPool1d(kernel_size=POOL_1D_SIZE, stride=POOL_1D_SIZE),
                conv_sn_relu(k=k, i=i, o=o))

        cs = TEXT_ENCODER_CHANNELS
        io = zip(cs[:-1], cs[1:], strict=True)

        self.encoder = nn.Sequential(
            conv_sn_relu(k=ENCODER_KERNEL, i=CHAR_EMBED_SIZE, o=cs[0]),
            *[
                pool_conv_sn_relu(k=ENCODER_KERNEL, i=i, o=o)
                for i, o in io])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.char_embed(x).transpose(1, 2)
        out = self.encoder(out)
        return torch.max(out, dim=-1).values


class ColorTst(nn.Module):
    def __init__(self):
        super().__init__()

        cs = [COLOR_DIM + TEXT_EMBED_SIZE, *CRITIC_CHANNELS]
        io = zip(cs[:-1], cs[1:], strict=True)
        self.net = nn.Sequential(
            *[
                nn.Sequential(
                    sn(nn.Conv1d(i, o, kernel_size=1, bias=True)),
                    nn.LeakyReLU(negative_slope=CRITIC_RELU_SLOPE)
                )
                for i, o in io
            ],
            nn.Conv1d(cs[-1], 1, kernel_size=1, bias=True),
        )

    def forward(self, text_embedding: torch.Tensor, colors: torch.Tensor) -> torch.Tensor:
        x = torch.cat(
            [colors.unsqueeze(-1), text_embedding.unsqueeze(-1)], dim=1)
        return self.net(x)


class ColorGen(nn.Module):
    def __init__(self):
        super().__init__()

        self.z_dim = GEN_Z_DIM
        self.head = nn.Linear(self.z_dim + TEXT_EMBED_SIZE, COLOR_DIM)

    def sample_z(self, n: int, device: torch.device | None = None) -> torch.Tensor:
        return torch.randn(n, self.z_dim, device=device)

    def forward(
        self,
        text_embedding: torch.Tensor,
        z: torch.Tensor | None = None
    ) -> torch.Tensor:
        if z is None:
            z = self.sample_z(
                text_embedding.size(0),
                text_embedding.device)

        colors = self.head(torch.cat([z, text_embedding], dim=-1))
        colors = tanh(colors) * 127.5

        return colors


class FeelingHead(nn.Module):
    def __init__(self):
        super().__init__()

        self.dropout = nn.Dropout(p=DROPOUT_FEELING)
        self.net = nn.Linear(
            TEXT_EMBED_SIZE,
            len(FEELINGS))

    def forward(self, text_embedding: torch.Tensor) -> torch.Tensor:
        out = self.dropout(text_embedding)
        out = self.net(out)

        return out


class EmojiHead(nn.Module):
    def __init__(self):
        super().__init__()

        self.net = nn.Linear(
            TEXT_EMBED_SIZE,
            EMOJI_EMBED_SIZE)
        self.embed = nn.Embedding(len(EMOJIS), EMOJI_EMBED_SIZE)

    def forward(
        self,
        text_embedding: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        q = normalize(tanh(self.net(text_embedding)), dim=-1)
        emoji_vec = normalize(self.embed.weight, dim=-1)
        return q, emoji_vec
