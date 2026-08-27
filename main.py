import json
import re
from pathlib import Path

import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from config import (
    BATCH_SIZE,
    EMBED_SIZE,
    EPOCHS,
    GRAD_CLIP,
    H_SIZE,
    KEYWORDS_UPSAMPLE,
    LR,
    MAX_TEXT_LEN,
    NUM_LAYERS,
)

# INPUT
# Index 0 is reserved for padding; real characters are numbered from 1.
PAD = "·"
CHARS = PAD + "abcdefghijklmnopqrstuvwxyz!?:()@$%&* "
PAD_IDX = 0
# CHARS already includes PAD at index 0, so the vocab is exactly its length.
VOCAB_SIZE = len(CHARS)

# OUTPUTS
# The label sets live in labels.json so main.py and gen_data.ts share one
# source of truth. `feelings` are the 7 that all appear in data.jsonl;
# `emojis` is a fixed 80-emoji palette, fully decoupled from feelings (every
# emoji is paired with every feeling). The emoji list is stored explicitly so
# multi-codepoint glyphs like ❤️, ☀️, ⛈️, 🕊️ index as a single unit.
_LABELS = json.loads(
    (Path(__file__).parent / "labels.json").read_text(encoding="utf-8")
)
feeling = _LABELS["feelings"]
EMOJIS = _LABELS["emojis"]

char2idx = {char: i for i, char in enumerate(CHARS)}
feeling2idx = {f: i for i, f in enumerate(feeling)}
emoji2idx = {e: i for i, e in enumerate(EMOJIS)}

# Colors are no longer a learned model output. Each feeling maps to a fixed
# Oklab palette: bg1/bg2 are the gradient background, text_color the foreground.
# Values are [L (0..1), a (-0.4..0.4), b (-0.4..0.4)]; consumed by server.py /
# the web page. Warm/bright for Happy/Excited, cool for Calm, muted/dark for
# Sad/Anxious, dark saturated red for Angry, neutral grey for Neutral.
FEELING_PALETTE = {
    "Happy": ([0.90, 0.02, 0.13], [0.82, 0.06, 0.16], [0.22, 0.03, 0.06]),
    "Excited": ([0.80, 0.12, 0.10], [0.70, 0.16, 0.14], [0.97, 0.0, 0.02]),
    "Calm": ([0.88, -0.05, -0.04], [0.80, -0.06, -0.08], [0.28, -0.02, -0.03]),
    "Sad": ([0.55, -0.02, -0.09], [0.45, -0.02, -0.11], [0.95, -0.01, -0.02]),
    "Angry": ([0.48, 0.18, 0.09], [0.38, 0.16, 0.07], [0.97, 0.02, 0.01]),
    "Anxious": ([0.60, -0.03, -0.06], [0.50, 0.02, -0.04], [0.95, 0.0, -0.01]),
    "Neutral": ([0.92, 0.0, 0.0], [0.85, 0.0, 0.0], [0.25, 0.0, 0.0]),
}


def feeling_colors(feeling_name: str) -> dict:
    """Map a feeling to its gradient + text colors (Oklab [L, a, b] triples).

    Returns a dict with keys bg1, bg2, text_color. Unknown feelings fall back
    to the Neutral palette.
    """
    bg1, bg2, text_color = FEELING_PALETTE.get(
        feeling_name, FEELING_PALETTE["Neutral"])
    return {"bg1": list(bg1), "bg2": list(bg2), "text_color": list(text_color)}


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(
            VOCAB_SIZE, EMBED_SIZE, padding_idx=PAD_IDX
        )

        self.conv = nn.Conv1d(
            in_channels=EMBED_SIZE,
            out_channels=H_SIZE,
            kernel_size=3,
            padding=1,
        )

        self.relu = nn.ReLU()

        self.rnn = nn.GRU(
            input_size=H_SIZE,
            hidden_size=H_SIZE,
            num_layers=NUM_LAYERS,
            batch_first=True,
        )

        self.emoji = nn.Linear(H_SIZE, len(EMOJIS))
        self.feeling = nn.Linear(H_SIZE, len(feeling))

    def forward(self, x):
        # 1. Compute valid character lengths before padding
        lengths = (x != PAD_IDX).sum(dim=1).clamp(min=1)

        # 2. Embedding: (batch, seq_len) -> (batch, seq_len, embed_dim)
        emb = self.embedding(x)

        # 3. Conv1d: permute to (batch, embed_dim, seq_len)
        conv_in = emb.permute(0, 2, 1)
        conv_out = self.relu(self.conv(conv_in))

        # 4. Permute back for RNN: (batch, seq_len, h_size)
        rnn_in = conv_out.permute(0, 2, 1)

        # 5. Pack padded sequence
        packed = nn.utils.rnn.pack_padded_sequence(
            rnn_in, lengths.cpu(), batch_first=True, enforce_sorted=False
        )

        # 6. RNN forward pass
        _, h_n = self.rnn(packed)

        # h_n shape: (num_layers, batch, hidden_dim)
        last_step = h_n[-1]

        return (
            self.emoji(last_step),
            self.feeling(last_step),
        )


def normalize(text: str) -> str:
    # Collapse whitespace, lowercase, then drop any character not in the model
    # vocab.
    text = re.sub(r"\s+", " ", text).strip().lower()
    return "".join(c for c in text if c in char2idx)


def encode(text: str, *, normalized: bool = False) -> torch.Tensor:
    """Return a (1, MAX_TEXT_LEN) long tensor of char indices for `text`.

    `text` is run through `normalize` first unless `normalized=True` (the caller
    has already normalized it).
    """
    if not normalized:
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
        text, emoji_target, feeling_target = self.data[idx]

        x_tensor = encode(text).squeeze(0)

        return (
            x_tensor,
            torch.tensor(emoji2idx[emoji_target], dtype=torch.long),
            torch.tensor(feeling2idx[feeling_target], dtype=torch.long),
        )


def _read_rows(path: str) -> list[tuple[str, str, str]]:
    """Read a .jsonl file into a list of (text, emoji, feeling) tuples.

    Each line must be a JSON object with keys: text, emoji, feeling.
    """
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            rows.append((row["text"], row["emoji"], row["feeling"]))
    return rows


def load_data(
    *,
    path: str = "data.jsonl",
    keywords_path: str = "keywords.jsonl",
    keywords_upsample: int = KEYWORDS_UPSAMPLE,
) -> MultiTaskDataset:
    """Load the main corpus plus the keyword corpus into one MultiTaskDataset.

    The keyword rows (`keywords_path`) are duplicated `keywords_upsample` times
    so the thin keyword corpus carries more weight during training; 0 leaves it
    out entirely.
    """
    data = _read_rows(path)
    if keywords_upsample:
        data.extend(_read_rows(keywords_path) * keywords_upsample)
    return MultiTaskDataset(data)


def run_name() -> str:
    """Build a TensorBoard run name that encodes the training configuration."""
    return f"emb{EMBED_SIZE}-h{H_SIZE}-l{NUM_LAYERS}-lr{LR}-bs{BATCH_SIZE}-ep{EPOCHS}"


def train(
    *,
    model: Model,
    data,
    writer: SummaryWriter | None = None,
):
    optimizer = optim.Adam(model.parameters(), lr=LR)
    dataloader = DataLoader(data, batch_size=BATCH_SIZE, shuffle=True)
    criterion_ce = nn.CrossEntropyLoss()

    model.train()

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total params: {total_params:,}")
    print("Starting training loop...\n")
    epoch_bar = tqdm(range(1, EPOCHS + 1), desc="Training", unit="epoch")
    for epoch in epoch_bar:
        total_loss = 0.0
        total_emoji_loss = 0.0
        total_feeling_loss = 0.0

        for x, target_emoji, target_feeling in dataloader:
            optimizer.zero_grad()

            emoji_logits, feeling_logits = model(x)

            # Losses for the two discrete heads.
            loss_emoji = criterion_ce(emoji_logits, target_emoji)
            loss_feeling = criterion_ce(feeling_logits, target_feeling)

            loss = loss_emoji + loss_feeling

            loss.backward()
            # Clip gradient norm to tame the loss spikes that Adam + a noisy
            # batch produce late in training.
            nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()

            total_loss += loss.item()
            total_emoji_loss += loss_emoji.item()
            total_feeling_loss += loss_feeling.item()

        n_batches = len(dataloader)
        avg_loss = total_loss / n_batches
        epoch_bar.set_postfix(loss=f"{avg_loss:.4f}")

        if writer is not None:
            writer.add_scalar("loss/total", avg_loss, epoch)
            writer.add_scalar(
                "loss/emoji", total_emoji_loss / n_batches, epoch)
            writer.add_scalar(
                "loss/feeling", total_feeling_loss / n_batches, epoch)

    print("\nTraining completed successfully.")


@torch.no_grad()
def evaluate(model: Model, data) -> dict:
    """Return emoji/feeling accuracy over `data`."""
    model.eval()
    dataloader = DataLoader(data, batch_size=32)

    n = 0
    emoji_correct = 0
    feeling_correct = 0
    for x, target_emoji, target_feeling in dataloader:
        emoji_logits, feeling_logits = model(x)
        emoji_correct += (emoji_logits.argmax(dim=-1)
                          == target_emoji).sum().item()
        feeling_correct += (
            (feeling_logits.argmax(dim=-1) == target_feeling).sum().item()
        )
        n += x.size(0)

    return {
        "n": n,
        "emoji_acc": emoji_correct / n,
        "feeling_acc": feeling_correct / n,
    }


def predict(model: Model, text: str) -> dict:
    """Run inference for a single string, returning a plain-dict result.

    Colors are not predicted; they are looked up from the predicted feeling.
    """
    model.eval()
    text = normalize(text)
    with torch.no_grad():
        emoji_logits, feeling_logits = model(encode(text, normalized=True))
    feeling_name = feeling[feeling_logits.argmax(dim=-1).item()]
    return {
        "text": text,
        "emoji": EMOJIS[emoji_logits.argmax(dim=-1).item()],
        "feeling": feeling_name,
        **feeling_colors(feeling_name),
    }


if __name__ == "__main__":
    torch.manual_seed(0)

    dataset = load_data()

    n_test = min(200, len(dataset) // 10)
    n_train = len(dataset) - n_test
    perm = torch.randperm(
        len(dataset), generator=torch.Generator().manual_seed(0)
    ).tolist()
    raw = dataset.data

    train_set = MultiTaskDataset([raw[i] for i in perm[:n_train]])
    test_set = MultiTaskDataset([raw[i] for i in perm[n_train:]])

    print(f"Train: {n_train}  Test: {n_test}\n")

    model = Model()

    name = run_name()
    writer = SummaryWriter(log_dir=str(Path("runs") / name))
    print(f"TensorBoard run: runs/{name}\n")

    train(
        model=model,
        data=train_set,
        writer=writer,
    )

    metrics = evaluate(model, test_set)
    print(
        f"\nTest ({metrics['n']}): "
        f"emoji_acc={metrics['emoji_acc']:.2f}  "
        f"feeling_acc={metrics['feeling_acc']:.2f}"
    )
    writer.add_scalar("test/emoji_acc", metrics["emoji_acc"])
    writer.add_scalar("test/feeling_acc", metrics["feeling_acc"])
    writer.add_hparams(
        {
            "embed_size": EMBED_SIZE,
            "h_size": H_SIZE,
            "num_layers": NUM_LAYERS,
            "max_text_len": MAX_TEXT_LEN,
            "lr": LR,
            "batch_size": BATCH_SIZE,
            "epochs": EPOCHS,
        },
        {
            "test/emoji_acc": metrics["emoji_acc"],
            "test/feeling_acc": metrics["feeling_acc"],
        },
        run_name=".",
    )
    writer.close()

    torch.save(model.state_dict(), "model.pt")
    print("Saved model to model.pt")

    sample_text = "party time!"
    result = predict(model, sample_text)
    print(f"\nInference on '{sample_text}':")
    print(f"Predicted Emoji:    {result['emoji']}")
    print(f"Predicted Feeling:  {result['feeling']}")
    print(f"Background Gradient: {result['bg1']} -> {result['bg2']}")
    print(f"Text Color:         {result['text_color']}")
