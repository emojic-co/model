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


def normalize(text: str) -> str:
    text = re.sub(r"\s+", " ", text).strip().lower()
    text = re.sub(r'(.)\1{2,}', r'\1\1', text)
    return "".join(c for c in text if c in char2idx)


def text_to_tensor(text: str) -> torch.Tensor:
    return torch.tensor(
        [char2idx[c] for c in text],
        dtype=torch.long)


def _read_jsonl(path):
    rows, bad = [], 0
    with open(path, encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                bad += 1
    if bad:
        print(f"{path}: skipped {bad} malformed line(s)")
    return rows


def _load(path):
    rows = _read_jsonl(path)

    out = []
    for d in rows:
        feeling = d["feeling"]
        if feeling not in feeling2idx or d["emoji"] not in emoji2idx:
            continue
        text = normalize(d["text"])
        if len(text) > MAX_TEXT_LEN:
            continue
        out.append((
            text,
            (text_to_tensor(text),
             emoji2idx[d["emoji"]],
             feeling2idx[feeling])))
    return out


def split():
    eval_pairs = _load(EVAL_PATH)
    eval_keys = {key for key, _ in eval_pairs}
    eval_data = [sample for _, sample in eval_pairs]

    train_data = [
        sample for key, sample in _load(TRAIN_PATH) if key not in eval_keys
    ]

    assert eval_data, f"{EVAL_PATH} is empty or missing"
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
