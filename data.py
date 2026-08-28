import json
import random
import re

import torch
from torch.nn.utils.rnn import pad_sequence
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
    with open('data.jsonl', encoding='utf-8') as f:
        data = [json.loads(line) for line in f]
        return [(
            text_to_tensor(normalize(d["text"])[:MAX_TEXT_LEN]),
            emoji2idx[d["emoji"]],
            feeling2idx[d["feeling"]]) for d in data]


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
    """Dynamically pad text sequences to the max length in the current batch."""
    # Unpack samples from MultiTaskDataset
    texts, emojis, feelings = zip(*batch, strict=False)

    # pad_sequence handles variable length tensors and pads with PAD_IDX
    padded_texts = pad_sequence(
        list(texts),
        batch_first=True,
        padding_value=PAD_IDX)

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
