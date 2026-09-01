import json
import re

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


def colors2tensor(bg: tuple[str, str], fg: str) -> torch.Tensor:
    vals = [c for h in (*bg, fg) for c in hex2rgb(h)]
    return torch.tensor(vals, dtype=torch.float32) - 127.5


def rnd_color_tensor() -> torch.Tensor:
    return torch.randint(0, 256, (COLOR_DIM,), dtype=torch.float32) - 127.5


def normalize(text: str) -> str:
    text = re.sub(r"\s+", " ", text).strip().lower()
    text = re.sub(r'(.)\1{2,}', r'\1\1', text)
    return "".join(c for c in text if c in char2idx)


def text_to_tensor(text: str) -> torch.Tensor:
    return torch.tensor(
        [char2idx[c] for c in text],
        dtype=torch.long)


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
                yield (text, emoji, feeling, [*bg, fg])


class EmojiDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def data_sets():
    train_data, eval_data = split()
    return EmojiDataset(train_data), EmojiDataset(eval_data)


def collate_fn(batch):
    texts, emojis, feelings, colors = zip(*batch, strict=False)

    padded_texts = torch.full(
        (len(texts), MAX_TEXT_LEN), PAD_IDX, dtype=torch.long)
    for i, t in enumerate(texts):
        padded_texts[i, : t.size(0)] = t[:MAX_TEXT_LEN]

    target_emojis = torch.tensor(emojis, dtype=torch.long)
    target_feelings = torch.tensor(feelings, dtype=torch.long)
    target_colors = torch.stack(list(colors))

    return padded_texts, target_emojis, target_feelings, target_colors


def train_data_loader(ds: EmojiDataset):
    return DataLoader(
        ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        drop_last=True,
        collate_fn=collate_fn,
        num_workers=0,
    )


def eval_data_loader(ds: EmojiDataset):
    return DataLoader(
        ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        drop_last=False,
        collate_fn=collate_fn,
        num_workers=0,
    )


if __name__ == "__main__":
    for r, _ in zip(read(TRAIN_PATH), range(3), strict=False):
        print(r)
