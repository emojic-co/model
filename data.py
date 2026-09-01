import json
import re
from dataclasses import dataclass

import torch
from torch.utils.data import DataLoader, Dataset

from config import BATCH_SIZE, MAX_TEXT_LEN

TRAIN_PATH = "train.jsonl"
EVAL_PATH = "eval.jsonl"

PAD = "·"
PAD_IDX = 0
CHARS = PAD + "abcdefghijklmnopqrstuvwxyz!?:()@$%&* "
VOCAB_SIZE = len(CHARS)

with open('labels.json', encoding='utf-8') as f:
    LABELS = json.load(f)

FEELING = LABELS["feelings"]
EMOJIS = LABELS["emojis"]


char2idx = {char: i for i, char in enumerate(CHARS)}
feeling2idx = {f: i for i, f in enumerate(FEELING)}
emoji2idx = {e: i for i, e in enumerate(EMOJIS)}

COLOR_DIM = 9


def hex2rgb(h: str) -> tuple[int, int, int]:
    h = h.lstrip("#")
    return (
        int(h[0:2], 16),
        int(h[2:4], 16),
        int(h[4:6], 16))


def colors2tensor(colors: list[str]) -> torch.Tensor:
    vals = [c for h in colors for c in hex2rgb(h)]
    return torch.tensor(vals, dtype=torch.float32) - 127.5


def rnd_color_tensor() -> torch.Tensor:
    return torch.randint(0, 256, (COLOR_DIM,), dtype=torch.float32) - 127.5


def normalize(text: str) -> str:
    text = re.sub(r"\s+", " ", text).strip().lower()
    text = re.sub(r'(.)\1{2,}', r'\1\1', text)
    return "".join(c for c in text if c in char2idx)


def text_to_tensor(text: str) -> torch.Tensor:
    assert len(text) <= MAX_TEXT_LEN
    idxs = [char2idx[c] for c in text]
    idxs.extend([PAD_IDX] * (MAX_TEXT_LEN - len(idxs)))
    return torch.tensor(idxs, dtype=torch.long)


def emoji_to_tensor(emoji: str) -> torch.Tensor:
    idx = emoji2idx[emoji]
    return torch.tensor(idx, dtype=torch.long)


def feeling_to_tensor(feeling: str) -> torch.Tensor:
    idx = feeling2idx[feeling]
    return torch.tensor(idx, dtype=torch.long)


@dataclass
class record:
    text: str
    emoji: str
    feeling: str
    colors: list[str]


def read(path):
    def read_jsonl():
        with open(path, encoding='utf-8') as f:
            for line in f:
                yield json.loads(line)

    for d in read_jsonl():
        match d:
            case {
                "text": text,
                "emoji": emoji,
                "feeling": feeling,
                'bg': bg,
                'fg': fg
            }:
                text = normalize(text)

                if not text or len(text) > MAX_TEXT_LEN:
                    continue

                if emoji not in EMOJIS:
                    continue

                if feeling not in FEELING:
                    continue

                yield record(text, emoji, feeling, [*bg, fg])


class EmojiDataset(Dataset):
    def __init__(self, path):
        records = list(read(path))
        self.text = torch.stack([text_to_tensor(r.text) for r in records])
        self.emoji = torch.stack([emoji_to_tensor(r.emoji) for r in records])
        self.feeling = torch.stack([feeling_to_tensor(r.feeling) for r in records])
        self.colors = torch.stack([colors2tensor(r.colors) for r in records])

    def __len__(self):
        return len(self.text)

    def __getitem__(self, idx):
        return self.text[idx], self.emoji[idx], self.feeling[idx], self.colors[idx]


def train_data_loader():
    return DataLoader(
        EmojiDataset(TRAIN_PATH),
        batch_size=BATCH_SIZE,
        shuffle=True,
        drop_last=True,
        num_workers=0,
    )


def eval_data_loader():
    return DataLoader(
        EmojiDataset(EVAL_PATH),
        batch_size=BATCH_SIZE,
        shuffle=False,
        drop_last=False,
        num_workers=0,
    )


if __name__ == "__main__":
    for r, _ in zip(read(TRAIN_PATH), range(3), strict=False):
        print(r)
