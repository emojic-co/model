import json
import re

import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset, random_split

# INPUT
vocab = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ.,!?;:()[]{}<>@#$%^&* "

# OUTPUTS
feeling = [
    "Happy",
    "Excited",
    "Calm",
    "Sad",
    "Angry",
    "Anxious",
]

# De-duplicate emojis string while retaining unique items
emojis = sorted(
    set(
        "😀😂😍😡😰🥰😎🤔😅😭🥳🙃👍👎👏🙏💪🔥💯❤️✨⭐🎉🚀🍕🍔🍟🍦🍩🍺🍷☕"
        "🏀⚽🎮🎲🎸🎨✈️🚗🚲🌴🌈☀️🌙🐶🐱🦁🐼🦊🍎🍌🥑🌶️🍿🍻🥂🏆🎯🎶🎤💡"
        "🔑📌⚡💥👑💍💎💖💔💤🤖👽💀👻💩🎃🔮🚢⛵🚨🏈⚾🎾🎱🎷🎹🎺🥁📱💻"
        "🎥📷📸🔍🔦🕯️💰⚖️🛒🎁🎈🎊✉️📦📍🔒🔓❤️‍🔥"
        # --- 100 additional popular & diverse emojis ---
        "🥳🤩😜🙈🙉🙊👋🤝🙌💅🧠👀👄🔥🌊🌸🌹🌻🌺🌾🍃🥦🍄🍉🍓🥭🍇🥥🧀"
        "🥞🥨🥓🥩🍗🌭🥪🌮🌯🍣🍜🍲🍡🧋🍵🍾🍹✈️🚁🚀🛸🚜🏎️🏍️⛵🚢🗺️⛵"
        "🗼🗽🗿🏰🎡🎢🎪🎨🎭🎫🎖️🏆🏅⚽🏀🏈🎾🏐🏉🎱🎯🧘‍♀️🏄‍♂️🏊‍♂️🏋️‍♂️🚴‍♂️"
        "🧗‍♂️🐾🦩🦄🐬🐳🐙🐉🌵🌲🪵💫🌟⚡💥🔥✨🎈🎉🎊🎋🎍🎏🧸🔮🧿"
    )
)

char2idx = {char: i for i, char in enumerate(vocab)}
feeling2idx = {f: i for i, f in enumerate(feeling)}
emoji2idx = {e: i for i, e in enumerate(emojis)}


class Model(nn.Module):
    def __init__(self, *, embed_dim: int, hidden_dim: int):
        super().__init__()
        self.embedding = nn.Embedding(len(vocab), embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.emoji = nn.Linear(hidden_dim, len(emojis))
        self.feeling = nn.Linear(hidden_dim, len(feeling))
        # Oklab values for gradient background start color (L, a, b)
        self.bg1 = nn.Linear(hidden_dim, 3)
        # Oklab values for gradient background end color (L, a, b)
        self.bg2 = nn.Linear(hidden_dim, 3)
        # Oklab values for text color (L, a, b)
        self.text_color = nn.Linear(hidden_dim, 3)

    def forward(self, x, state=None):
        x = self.embedding(x)  # (batch_size, seq_len, embed_dim)
        out, (h_n, c_n) = self.lstm(x, state)

        # Take the hidden state of the final time step
        last_step = out[:, -1, :]  # (batch_size, hidden_dim)
        # (batch_size, len(unique_emojis))
        emoji_logits = self.emoji(last_step)
        feeling_logits = self.feeling(last_step)  # (batch_size, len(feeling))
        bg1 = self.bg1(last_step)  # (batch_size, 3) -> Oklab (L, a, b)
        bg2 = self.bg2(last_step)  # (batch_size, 3) -> Oklab (L, a, b)
        # (batch_size, 3) -> Oklab (L, a, b)
        text_color = self.text_color(last_step)

        return emoji_logits, feeling_logits, bg1, bg2, text_color, (h_n, c_n)


def normalize(text: str) -> str:
    # Collapse any whitespace run to a single space, then drop non-vocab chars.
    text = re.sub(r"\s+", " ", text).strip()
    return "".join(c for c in text if c in char2idx)


def encode(text: str, max_len=16) -> torch.Tensor:
    """Normalize `text` and return a (1, max_len) long tensor of char indices."""
    text = normalize(text)
    indices = [char2idx.get(c, 0) for c in text[:max_len]]
    padding = [0] * (max_len - len(indices))
    return torch.tensor([indices + padding], dtype=torch.long)


class MultiTaskDataset(Dataset):
    def __init__(self, data, max_len: int):
        self.data = data
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        (
            text,
            emoji_target,
            feeling_target,
            bg1_oklab,
            bg2_oklab,
            text_color_oklab,
        ) = self.data[idx]

        x_tensor = encode(text, self.max_len).squeeze(0)

        return (
            x_tensor,
            torch.tensor(emoji2idx[emoji_target], dtype=torch.long),
            torch.tensor(feeling2idx[feeling_target], dtype=torch.long),
            torch.tensor(bg1_oklab, dtype=torch.float32),
            torch.tensor(bg2_oklab, dtype=torch.float32),
            torch.tensor(text_color_oklab, dtype=torch.float32),
        )


def load_data(*, path: str, max_len: int) -> MultiTaskDataset:
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
    return MultiTaskDataset(data, max_len=max_len)


def train(
    *,
    model: Model,
    data,
    epochs,
    batch_size=16,
):

    optimizer = optim.Adam(model.parameters(), lr=0.01)
    dataloader = DataLoader(data, batch_size=batch_size, shuffle=True)
    criterion_ce = nn.CrossEntropyLoss()
    criterion_mse = nn.MSELoss()

    model.train()

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total params: {total_params:,}")
    print("Starting training loop...\n")
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
                _,
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
        emoji_logits, feeling_logits, pred_bg1, pred_bg2, pred_text_color, _ = model(
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


def predict(model: Model, text: str, max_len: int = 16) -> dict:
    """Run inference for a single string, returning a plain-dict result."""
    model.eval()
    with torch.no_grad():
        emoji_logits, feeling_logits, bg1, bg2, text_color, _ = model(
            encode(text, max_len)
        )
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

    dataset = load_data(
        path="data.jsonl",
        max_len=32
    )

    n_test = max(1, len(dataset) // 5)
    n_train = len(dataset) - n_test
    train_set, test_set = random_split(
        dataset, [n_train, n_test], generator=torch.Generator().manual_seed(0)
    )
    print(f"Train: {n_train}  Test: {n_test}\n")

    model = Model(
        embed_dim=8,
        hidden_dim=16
    )

    train(
        model=model,
        data=train_set,
        epochs=200,)

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
