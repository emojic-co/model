import re

import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset

# INPUT
vocab = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ.,!?;:()[]{}<>@#$%^&* '

# OUTPUTS
feeling = [
    'Happy',
    'Excited',
    'Calm',
    'Sad',
    'Angry',
    'Anxious',
]

# De-duplicate emojis string while retaining unique items
emojis = sorted(set(
    "😀😂😍😡😰🥰😎🤔😅😭🥳🙃👍👎👏🙏💪🔥💯❤️✨⭐🎉🚀🍕🍔🍟🍦🍩🍺🍷☕🏀⚽🎮🎲🎸🎨✈️🚗🚲🌴🌈☀️🌙⭐🐶🐱🦁🐼🦊🍎🍌🥑🌶️🍿🍻🥂🏆🎯🎶🎤💡🔑📌⚡💥🎉👑💍💎💖💔💤🤖👽💀👻💩🎃🔮🚀🚢⛵🚗🚲🚨🏆⚽🏀🏈⚾🎾🎱🎮🎯🎲🎨🎤🎶🎷🎸🎹🎺🥁📱💻🎥📷📸🔍💡🔦🕯️💰💎⚖️🛒🎁🎈🎉🎊✉️📦📌📍🔑🔒🔓❤️‍🔥💖"))

char2idx = {char: i for i, char in enumerate(vocab)}
feeling2idx = {f: i for i, f in enumerate(feeling)}
emoji2idx = {e: i for i, e in enumerate(emojis)}


class Model(nn.Module):
    def __init__(self, embed_dim=8, hidden_dim=16):
        super().__init__()
        self.embedding = nn.Embedding(len(vocab), embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.emoji = nn.Linear(hidden_dim, len(emojis))
        self.feeling = nn.Linear(hidden_dim, len(feeling))
        # Oklab values for gradient background start color (L, a, b)
        self.bg1 = nn.Linear(hidden_dim, 3)
        # Oklab values for gradient background end color (L, a, b)
        self.bg2 = nn.Linear(hidden_dim, 3)

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

        return emoji_logits, feeling_logits, bg1, bg2, (h_n, c_n)


def normalize(text: str) -> str:
    # Collapse any whitespace run to a single space, then drop non-vocab chars.
    text = re.sub(r"\s+", " ", text).strip()
    return "".join(c for c in text if c in char2idx)


class MultiTaskDataset(Dataset):
    def __init__(self, data, max_len=16):
        self.data = data
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text, emoji_target, feeling_target, bg1_oklab, bg2_oklab = self.data[idx]

        text = normalize(text)

        indices = [char2idx.get(c, 0) for c in text[:self.max_len]]
        padding = [0] * (self.max_len - len(indices))
        x_tensor = torch.tensor(indices + padding, dtype=torch.long)

        return (
            x_tensor,
            torch.tensor(emoji2idx[emoji_target], dtype=torch.long),
            torch.tensor(feeling2idx[feeling_target], dtype=torch.long),
            torch.tensor(bg1_oklab, dtype=torch.float32),
            torch.tensor(bg2_oklab, dtype=torch.float32),
        )


def train(
    *,
        model: Model,
        data: MultiTaskDataset,
        epochs=20,
        batch_size=8,
):

    optimizer = optim.Adam(model.parameters(), lr=0.005)
    dataloader = DataLoader(data, batch_size=batch_size, shuffle=True)
    criterion_ce = nn.CrossEntropyLoss()
    criterion_mse = nn.MSELoss()

    model.train()

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total params: {total_params:,}")
    print("Starting training loop...\n")
    for epoch in range(1, epochs + 1):
        total_loss = 0.0

        for x, target_emoji, target_feeling, target_bg1, target_bg2 in dataloader:
            optimizer.zero_grad()

            emoji_logits, feeling_logits, pred_bg1, pred_bg2, _ = model(x)

            # Losses for discrete classes
            loss_emoji = criterion_ce(emoji_logits, target_emoji)
            loss_feeling = criterion_ce(feeling_logits, target_feeling)

            # Euclidean MSE Loss in Oklab space mirrors true perceptual distance
            loss_bg1 = criterion_mse(pred_bg1, target_bg1)
            loss_bg2 = criterion_mse(pred_bg2, target_bg2)

            # Combined multi-task loss
            loss = loss_emoji + loss_feeling + (loss_bg1 + loss_bg2) * 5.0

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        if epoch % 5 == 0 or epoch == 1:
            print(f"Epoch [{epoch:02d}/{epochs}] - Loss: {avg_loss:.4f}")

    print("\nTraining completed successfully.")


if __name__ == "__main__":
    # Sample training data tuple: (text, emoji, feeling, bg1_oklab, bg2_oklab)
    # Oklab format: [L (0..1), a (-0.4..0.4), b (-0.4..0.4)]
    raw_data = [
        ("party time!", "🎉", "Excited", [
         0.75, 0.15, 0.18], [0.85, 0.05, 0.22]),
        ("so calm today", "🌙", "Calm", [
         0.35, -0.05, -0.15], [0.45, -0.08, -0.10]),
        ("i love this", "😍", "Happy", [0.80, 0.12, 0.15], [0.90, 0.02, 0.18]),
        ("very frustrated", "😡", "Angry", [
         0.45, 0.25, 0.15], [0.30, 0.20, 0.10]),
        ("so nervous...", "😰", "Anxious", [
         0.50, -0.10, -0.05], [0.40, -0.05, -0.15]),
        ("feeling down", "😭", "Sad", [
         0.30, -0.02, -0.12], [0.20, -0.05, -0.08]),
    ] * 16

    dataset = MultiTaskDataset(raw_data, max_len=16)
    model = Model(embed_dim=16, hidden_dim=32)

    train(model=model, data=dataset)
    # Quick evaluation check
    model.eval()
    with torch.no_grad():
        sample_text = "party time!"
        indices = [char2idx.get(c, 0) for c in sample_text[:16]]
        padding = [0] * (16 - len(indices))
        input_tensor = torch.tensor([indices + padding], dtype=torch.long)

        pred_emoji, pred_feeling, pred_bg1, pred_bg2, _ = model(input_tensor)

        top_emoji = emojis[pred_emoji.argmax(dim=-1).item()]
        top_feeling = feeling[pred_feeling.argmax(dim=-1).item()]

        print(f"\nInference on '{sample_text}':")
        print(f"Predicted Emoji:   {top_emoji}")
        print(f"Predicted Feeling: {top_feeling}")
        print(f"Predicted Oklab 1: {pred_bg1.squeeze().tolist()}")
        print(f"Predicted Oklab 2: {pred_bg2.squeeze().tolist()}")
