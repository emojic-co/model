import json
import random
import re

import torch
from torch.utils.data import DataLoader, Dataset

from config import BATCH_SIZE, MAX_TEXT_LEN, TEST_LEN

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


def normalize(text: str) -> str:
    text = re.sub(r"\s+", " ", text).strip().lower()
    text = re.sub(r'(.)\1{2,}', r'\1\1', text)
    return "".join(c for c in text if c in char2idx)


def text_to_tensor(text: str) -> torch.Tensor:
    return torch.tensor(
        [char2idx[c] for c in text],
        dtype=torch.long)


def read():
    """Load data.jsonl, dropping any record the current label set can't train.

    labels.json (see gen_labels.ts) is the closed vocabulary: a record is kept
    only if its feeling and emoji are both in it and its normalized text fits
    MAX_TEXT_LEN. Rows are never removed from data.jsonl itself -- filtering is
    purely a runtime concern.
    """
    with open('data.jsonl', encoding='utf-8') as f:
        rows = [json.loads(line) for line in f]

    out = []
    for d in rows:
        if d["feeling"] not in feeling2idx or d["emoji"] not in emoji2idx:
            continue
        text = normalize(d["text"])
        if len(text) > MAX_TEXT_LEN:
            continue
        out.append((
            text_to_tensor(text),
            emoji2idx[d["emoji"]],
            feeling2idx[d["feeling"]]))
    return out


def split():
    data = read()

    # Seed the generator and shuffle a copy of the list
    rng = random.Random(42)
    rng.shuffle(data)

    # Compute split index
    train_data = data[TEST_LEN:]
    eval_data = data[:TEST_LEN]

    return train_data, eval_data


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
    """Right-pad every text to MAX_TEXT_LEN, matching the ONNX/browser path."""
    texts, emojis, feelings = zip(*batch, strict=False)

    padded_texts = torch.full(
        (len(texts), MAX_TEXT_LEN), PAD_IDX, dtype=torch.long)
    for i, t in enumerate(texts):
        padded_texts[i, : t.size(0)] = t[:MAX_TEXT_LEN]

    target_emojis = torch.tensor(emojis, dtype=torch.long)
    target_feelings = torch.tensor(feelings, dtype=torch.long)

    return padded_texts, target_emojis, target_feelings


def train_data_loader(ds: EmojiDataset):
    return DataLoader(
        ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        drop_last=True,
        collate_fn=collate_fn
    )
