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

        self.conv = nn.Sequential(
            nn.Conv1d(
                in_channels=EMBED_SIZE,
                out_channels=H_SIZE,
                kernel_size=3,
                padding=1,
            ),
            nn.ReLU(),
        )

        self.rnn = nn.RNN(
            input_size=H_SIZE,
            hidden_size=H_SIZE,
            num_layers=NUM_LAYERS,
            batch_first=True,
        )

        self.emoji = nn.Linear(H_SIZE, len(EMOJIS))
        self.feeling = nn.Linear(H_SIZE, len(FEELING))

    def forward(self, x):
        lengths = (x != PAD_IDX).sum(dim=1).clamp(min=1)

        out = self.embedding(x).permute(0, 2, 1)
        out = self.conv(out)
        out = out.masked_fill((x == PAD_IDX).unsqueeze(1), 0)
        out = out.permute(0, 2, 1)

        # 4. RNN over the full padded sequence, then read the hidden state at
        # each row's last real character. This is identical to h_n[-1] from a
        # packed sequence (the RNN is causal, so trailing pad steps can't affect
        # an earlier index), but unlike pack_padded_sequence it traces to ONNX
        # via the legacy exporter (torch.onnx.export(dynamo=False)).
        out, _ = self.rnn(out)  # (batch, seq, H_SIZE)
        idx = (lengths - 1).view(-1, 1, 1).expand(-1, 1, out.size(2))
        last_step = out.gather(1, idx).squeeze(1)  # (batch, H_SIZE)

        # 6. Classification heads
        return (
            self.emoji(last_step),
            self.feeling(last_step),
        )
