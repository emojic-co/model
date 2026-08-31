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

COLOR_MEAN = torch.tensor(
    [0.809946, 0.006661, 0.018747, 0.721956, 0.019043,
     0.013285, 0.370698, 0.008323, 0.001856],
    dtype=torch.float32,
)
COLOR_STD = torch.tensor(
    [0.148909, 0.040569, 0.057178, 0.131150, 0.058753,
     0.060955, 0.242255, 0.033875, 0.035447],
    dtype=torch.float32,
)


def _srgb_to_linear(c: float) -> float:
    return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4


def hex_to_oklab(h: str) -> tuple[float, float, float]:
    h = h.lstrip("#")
    r, g, b = (_srgb_to_linear(int(h[i:i + 2], 16) / 255) for i in (0, 2, 4))

    lm = 0.4122214708 * r + 0.5363325363 * g + 0.0514459929 * b
    mm = 0.2119034982 * r + 0.6806995451 * g + 0.1073969566 * b
    sm = 0.0883024619 * r + 0.2817188376 * g + 0.6299787005 * b

    lm, mm, sm = lm ** (1 / 3), mm ** (1 / 3), sm ** (1 / 3)

    return (
        0.2104542553 * lm + 0.7936177850 * mm - 0.0040720468 * sm,
        1.9779984951 * lm - 2.4285922050 * mm + 0.4505937099 * sm,
        0.0259040371 * lm + 0.7827717662 * mm - 0.8086757660 * sm,
    )


def color_to_tensor(bg: list[str], fg: str) -> torch.Tensor:
    vals = [c for h in (*bg, fg) for c in hex_to_oklab(h)]
    return (torch.tensor(vals, dtype=torch.float32) - COLOR_MEAN) / COLOR_STD


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
        if "bg" not in d or "fg" not in d:
            continue
        text = normalize(d["text"])
        if len(text) > MAX_TEXT_LEN:
            continue
        out.append((
            text,
            (text_to_tensor(text),
             emoji2idx[d["emoji"]],
             feeling2idx[feeling],
             color_to_tensor(d["bg"], d["fg"]))))
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
