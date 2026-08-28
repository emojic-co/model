import torch
from torch import nn

from config import EMBED_SIZE, H_SIZE, NUM_LAYERS
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
                out_channels=H_SIZE,
                kernel_size=3,
                padding=1,
            ),
            nn.ReLU(),
        ]

        for _ in range(NUM_LAYERS):
            layers.append(
                nn.Conv1d(
                    in_channels=H_SIZE,
                    out_channels=H_SIZE,
                    kernel_size=3,
                    padding=1,
                )
            )
            layers.append(nn.ReLU())

        self.net = nn.Sequential(*layers)

        self.emoji = nn.Linear(H_SIZE, len(EMOJIS))
        self.feeling = nn.Linear(H_SIZE, len(FEELING))

    def forward(self, x):
        pad_mask = (x == PAD_IDX).unsqueeze(1)  # (batch, 1, seq_len)

        out = self.embedding(x)
        out = out.permute(0, 2, 1)  # (batch, embed_dim, seq_len)
        out = self.net(out)
        out = out.masked_fill(pad_mask, -1e9)
        out = torch.max(out, dim=2).values  # (batch, H_SIZE)

        return (
            self.emoji(out),
            self.feeling(out),
        )
