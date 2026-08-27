import json
import re

import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset

from config import EMBED_SIZE, EPOCHS, H_SIZE, MAX_TEXT_LEN

# INPUT
# Index 0 is reserved for padding; real characters are numbered from 1.
PAD = '·'
CHARS = PAD + "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ.,!?;:()[]{}<>@#$%^&* "
PAD_IDX = 0
VOCAB_SIZE = len(CHARS) + 1

# OUTPUTS
feeling = [
    "Happy",
    "Excited",
    "Calm",
    "Sad",
    "Angry",
    "Anxious",
    "Neutral",
]

# TODO: decouple emojis from feelings, pick 30 popular diverse emojis and backfill the data.jsonl with them.
emojis = (
    "😀😂🥹😍🤔"  # Expressions & Feelings
    "🥳😎😭💀🔥"
    "❤️💯✨👍👏"  # Symbols & Gestures
    "🙌🙏💪🧠👀"
    "🐶🐱🦁🦉🐙"  # Animals & Nature
    "🌲🌺🌈☀️⭐"
    "🍕🌮🍣☕🍺"  # Food & Drink
    "⚽🎉🚀✈️🎸"  # Activities & Travel
    "💡💎📱🎁🔒"  # Objects & Tools
    "🌍🏆🎨🔮📍"  # Places & Concepts
)
char2idx = {char: i for i, char in enumerate(CHARS)}
feeling2idx = {f: i for i, f in enumerate(feeling)}
emoji2idx = {e: i for i, e in enumerate(emojis)}


def squash_oklab(raw: torch.Tensor) -> torch.Tensor:
    """Map a raw (..., 3) tensor into Oklab range: L in [0, 1], a/b in [-0.4, 0.4]."""
    L = torch.sigmoid(raw[..., :1])
    ab = 0.4 * torch.tanh(raw[..., 1:])
    return torch.cat([L, ab], dim=-1)


class Model(nn.Module):
    def __init__(self, *, embed_dim: int, hidden_dim: int):
        super().__init__()
        self.embedding = nn.Embedding(
            VOCAB_SIZE,
            embed_dim,
            padding_idx=PAD_IDX)

        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.emoji = nn.Linear(hidden_dim, len(emojis))
        self.feeling = nn.Linear(hidden_dim, len(feeling))
        # Each head emits raw (L, a, b); `squash_oklab` maps them into Oklab range.
        self.bg1 = nn.Linear(hidden_dim, 3)  # gradient background start color
        self.bg2 = nn.Linear(hidden_dim, 3)  # gradient background end color
        self.text_color = nn.Linear(hidden_dim, 3)  # foreground text color

    def forward(self, x):
        # True length of each row (chars before padding); clamp so an all-pad
        # row (empty text) still packs with length 1.
        lengths = (x != PAD_IDX).sum(dim=1).clamp(min=1)
        emb = self.embedding(x)  # (batch_size, seq_len, embed_dim)
        packed = nn.utils.rnn.pack_padded_sequence(
            emb, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        _, (h_n, _) = self.lstm(packed)
        # h_n[-1] is the hidden state at each row's last real character.
        last_step = h_n[-1]  # (batch_size, hidden_dim)

        return (
            self.emoji(last_step),
            self.feeling(last_step),
            squash_oklab(self.bg1(last_step)),
            squash_oklab(self.bg2(last_step)),
            squash_oklab(self.text_color(last_step)),
        )


def normalize(text: str) -> str:
    # Collapse any whitespace run to a single space, then drop non-vocab chars.
    text = re.sub(r"\s+", " ", text).strip()
    return "".join(c for c in text if c in char2idx)


def encode(text: str) -> torch.Tensor:
    """Normalize `text` and return a (1, MAX_TEXT_LEN) long tensor of char indices."""
    text = normalize(text)
    indices = [char2idx[c] for c in text[:MAX_TEXT_LEN]]
    padding = [PAD_IDX] * (MAX_TEXT_LEN - len(indices))
    return torch.tensor([indices + padding], dtype=torch.long)


class MultiTaskDataset(Dataset):
    def __init__(self, data):
        self.data = data
        self.data_len = len(data)

    def __len__(self):
        return self.data_len

    def __getitem__(self, idx):
        (
            text,
            emoji_target,
            feeling_target,
            bg1_oklab,
            bg2_oklab,
            text_color_oklab,
        ) = self.data[idx]

        x_tensor = encode(text).squeeze(0)

        return (
            x_tensor,
            torch.tensor(emoji2idx[emoji_target], dtype=torch.long),
            torch.tensor(feeling2idx[feeling_target], dtype=torch.long),
            torch.tensor(bg1_oklab, dtype=torch.float32),
            torch.tensor(bg2_oklab, dtype=torch.float32),
            torch.tensor(text_color_oklab, dtype=torch.float32),
        )


def load_data(*, path: str) -> MultiTaskDataset:
    """Read a .jsonl file of samples and return a MultiTaskDataset.

    Each line must be a JSON object with keys: text, emoji, feeling, bg1, bg2,
    text_color (bg1/bg2/text_color are Oklab [L, a, b] triples).
    """
    data = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            data.append(
                (
                    row["text"],
                    row["emoji"],
                    row["feeling"],
                    row["bg1"],
                    row["bg2"],
                    row["text_color"],
                )
            )
    return MultiTaskDataset(data)


def train(
    *,
    model: Model,
    data,
    lr,
    epochs,
    batch_size,
):

    optimizer = optim.Adam(model.parameters(), lr=lr)
    dataloader = DataLoader(data, batch_size=batch_size, shuffle=True)

    criterion_ce = nn.CrossEntropyLoss()
    criterion_mse = nn.MSELoss()

    model.train()

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total params: {total_params:,}")
    print("Starting training loop...\n")
    # TODO: add tqdm progress for the epoch progress
    for epoch in range(1, epochs + 1):
        total_loss = 0.0

        for (
            x,
            target_emoji,
            target_feeling,
            target_bg1,
            target_bg2,
            target_text_color,
        ) in dataloader:
            optimizer.zero_grad()

            (
                emoji_logits,
                feeling_logits,
                pred_bg1,
                pred_bg2,
                pred_text_color,
            ) = model(x)

            # Losses for discrete classes
            loss_emoji = criterion_ce(emoji_logits, target_emoji)
            loss_feeling = criterion_ce(feeling_logits, target_feeling)

            # Euclidean MSE Loss in Oklab space mirrors true perceptual distance
            loss_bg1 = criterion_mse(pred_bg1, target_bg1)
            loss_bg2 = criterion_mse(pred_bg2, target_bg2)
            loss_text_color = criterion_mse(pred_text_color, target_text_color)

            # Combined multi-task loss
            loss = (
                loss_emoji
                + loss_feeling
                + (loss_bg1 + loss_bg2 + loss_text_color) * 5.0
            )

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        if epoch % 5 == 0 or epoch == 1:
            print(f"Epoch [{epoch:02d}/{epochs}] - Loss: {avg_loss:.4f}")

    print("\nTraining completed successfully.")


@torch.no_grad()
def evaluate(model: Model, data) -> dict:
    """Return emoji/feeling accuracy and Oklab MSE over `data`."""
    model.eval()
    dataloader = DataLoader(data, batch_size=32)
    criterion_mse = nn.MSELoss(reduction="sum")

    n = 0
    emoji_correct = 0
    feeling_correct = 0
    color_sq_err = 0.0
    for (
        x,
        target_emoji,
        target_feeling,
        target_bg1,
        target_bg2,
        target_text_color,
    ) in dataloader:
        emoji_logits, feeling_logits, pred_bg1, pred_bg2, pred_text_color = model(
            x)
        emoji_correct += (emoji_logits.argmax(dim=-1)
                          == target_emoji).sum().item()
        feeling_correct += (
            (feeling_logits.argmax(dim=-1) == target_feeling).sum().item()
        )
        color_sq_err += criterion_mse(pred_bg1, target_bg1).item()
        color_sq_err += criterion_mse(pred_bg2, target_bg2).item()
        color_sq_err += criterion_mse(pred_text_color,
                                      target_text_color).item()
        n += x.size(0)

    return {
        "n": n,
        "emoji_acc": emoji_correct / n,
        "feeling_acc": feeling_correct / n,
        "color_mse": color_sq_err / (n * 9),  # 3 colors x 3 channels
    }


def predict(model: Model, text: str) -> dict:
    """Run inference for a single string, returning a plain-dict result."""
    model.eval()
    text = normalize(text)[:MAX_TEXT_LEN]
    with torch.no_grad():
        emoji_logits, feeling_logits, bg1, bg2, text_color = model(
            encode(text))
    return {
        "text": text,
        "emoji": emojis[emoji_logits.argmax(dim=-1).item()],
        "feeling": feeling[feeling_logits.argmax(dim=-1).item()],
        "bg1": bg1.squeeze(0).tolist(),
        "bg2": bg2.squeeze(0).tolist(),
        "text_color": text_color.squeeze(0).tolist(),
    }


if __name__ == "__main__":
    torch.manual_seed(0)

    dataset = load_data(path="data.jsonl")

    n_test = 100
    n_train = len(dataset) - n_test
    perm = torch.randperm(
        len(dataset), generator=torch.Generator().manual_seed(0)
    ).tolist()
    raw = dataset.data

    train_set = MultiTaskDataset([raw[i] for i in perm[:n_train]])
    test_set = MultiTaskDataset([raw[i] for i in perm[n_train:]])

    print(f"Train: {n_train}  Test: {n_test}\n")

    model = Model(embed_dim=EMBED_SIZE, hidden_dim=H_SIZE)

    train(
        model=model,
        data=train_set,
        lr=0.005,
        batch_size=8,
        epochs=EPOCHS,
    )

    metrics = evaluate(model, test_set)
    print(
        f"\nTest ({metrics['n']}): "
        f"emoji_acc={metrics['emoji_acc']:.2f}  "
        f"feeling_acc={metrics['feeling_acc']:.2f}  "
        f"color_mse={metrics['color_mse']:.4f}"
    )

    torch.save(model.state_dict(), "model.pt")
    print("Saved model to model.pt")

    sample_text = "party time!"
    result = predict(model, sample_text)
    print(f"\nInference on '{sample_text}':")
    print(f"Predicted Emoji:    {result['emoji']}")
    print(f"Predicted Feeling:  {result['feeling']}")
    print(f"Predicted Oklab 1:  {result['bg1']}")
    print(f"Predicted Oklab 2:  {result['bg2']}")
    print(f"Predicted TextClr:  {result['text_color']}")
