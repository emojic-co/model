import torch
from torch import nn
from torch.nn import functional as F

from config import CHANNELS, EMBED_SIZE, EMOJI_EMBED_SIZE, NUM_LAYERS
from data import EMOJIS, FEELING, PAD_IDX, VOCAB_SIZE


class Model(nn.Module):
    def __init__(self):
        super().__init__()

        self.embedding = nn.Embedding(
            num_embeddings=VOCAB_SIZE,
            embedding_dim=EMBED_SIZE,
            padding_idx=PAD_IDX,
        )

        layers = [
            nn.Conv1d(
                in_channels=EMBED_SIZE,
                out_channels=CHANNELS,
                kernel_size=3,
                padding=1,
            ),
            nn.ReLU(),
        ]

        for _ in range(NUM_LAYERS - 1):
            layers.append(
                nn.Conv1d(
                    in_channels=CHANNELS,
                    out_channels=CHANNELS,
                    kernel_size=3,
                    padding=1,
                )
            )
            layers.append(nn.ReLU())

        self.net = nn.Sequential(*layers)

        # Emoji head: a learnable embedding per emoji, scored contrastively.
        # The text encoding is projected into the emoji space and matched
        # against every emoji vector by cosine similarity (CLIP-style). With a
        # plain cross-entropy on the true emoji this is InfoNCE against all
        # emojis as negatives -- pulls the matching pair together, pushes the
        # rest apart.
        self.text_proj = nn.Linear(
            CHANNELS,
            EMOJI_EMBED_SIZE)

        self.emoji_embedding = nn.Embedding(
            len(EMOJIS),
            EMOJI_EMBED_SIZE)

        # log temperature, initialised to ln(1 / 0.07) as in CLIP
        self.logit_scale = nn.Parameter(torch.tensor(2.6593))

        self.feeling = nn.Linear(CHANNELS, len(FEELING))

    def forward(self, x):
        pad_mask = (x == PAD_IDX).unsqueeze(1)  # (batch, 1, seq_len)

        out = self.embedding(x)
        out = out.permute(0, 2, 1)  # (batch, embed_dim, seq_len)
        out = self.net(out)
        out = out.masked_fill(pad_mask, -1e9)
        out = torch.max(out, dim=2).values  # (batch, H_SIZE)

        text_vec = F.normalize(self.text_proj(out), dim=-1)
        emoji_vec = F.normalize(self.emoji_embedding.weight, dim=-1)
        emoji_logits = self.logit_scale.exp() * text_vec @ emoji_vec.t()

        return (
            emoji_logits,
            self.feeling(out),
        )
