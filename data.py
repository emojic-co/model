import json
import re
from dataclasses import dataclass

import torch
from torch.utils.data import DataLoader, Dataset

from config import EMOJIS, MAX_TEXT_LEN, STYLES

TRAIN_PATH = "train.jsonl"
EVAL_PATH = "eval.jsonl"

PAD = "·"
PAD_IDX = 0
CHARS = PAD + "abcdefghijklmnopqrstuvwxyz!?:()@$%&* "
VOCAB_SIZE = len(CHARS)


char2idx = {char: i for i, char in enumerate(CHARS)}
style2idx = {s: i for i, s in enumerate(STYLES)}
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


def multi_hot(items: list[str], index: dict[str, int], size: int) -> torch.Tensor:
    out = torch.zeros(size, dtype=torch.float32)
    for it in items:
        i = index.get(it)
        if i is not None:
            out[i] = 1.0
    return out


def emojis_to_tensor(emojis: list[str]) -> torch.Tensor:
    return multi_hot(emojis, emoji2idx, len(EMOJIS))


def styles_to_tensor(styles: list[str]) -> torch.Tensor:
    return multi_hot(styles, style2idx, len(STYLES))


@dataclass
class record:
    text: str
    emojis: list[str]
    styles: list[str]
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
                "emojis": emojis,
                "styles": styles,
                'bg': bg,
                'fg': fg
            }:
                text = normalize(text)

                if not text or len(text) > MAX_TEXT_LEN:
                    continue

                emojis = [e for e in emojis.split() if e in emoji2idx]
                styles = [s for s in styles if s in style2idx]

                if not styles:
                    continue

                yield record(text, emojis, styles, [*bg, fg])


def load_energy_keywords(path: str) -> list[str]:
    try:
        with open(path, encoding="utf-8") as f:
            return [w for line in f if (w := line.strip())]
    except FileNotFoundError:
        return []


def keyword_index(
    keywords: list[str],
    *,
    max_texts: int,
    min_texts: int,
    seed: int,
) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
    kws = [k.lower() for k in keywords]
    hits: dict[str, list[record]] = {k: [] for k in kws}
    for r in read(TRAIN_PATH):
        for k in kws:
            if k in r.text:
                hits[k].append(r)

    g = torch.Generator().manual_seed(seed)
    out: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
    for k in kws:
        rows = hits[k]
        if len(rows) < min_texts:
            print(f"energy keyword {k!r}: {len(rows)} matches < {min_texts}, skipped")
            continue
        if len(rows) > max_texts:
            idx = torch.randperm(len(rows), generator=g)[:max_texts].tolist()
            rows = [rows[i] for i in idx]
        text_ids = torch.stack([text_to_tensor(r.text) for r in rows])
        palettes = torch.stack([colors2tensor(r.colors) for r in rows])
        out[k] = (text_ids, palettes)
    return out


class EmojiDataset(Dataset):
    def __init__(self, records: list[record]):
        self.text = torch.stack([text_to_tensor(r.text) for r in records])
        self.emoji = torch.stack([emojis_to_tensor(r.emojis) for r in records])
        self.style = torch.stack([styles_to_tensor(r.styles) for r in records])
        self.colors = torch.stack([colors2tensor(r.colors) for r in records])

    def __len__(self):
        return len(self.text)

    def __getitem__(self, idx):
        return self.text[idx], self.emoji[idx], self.style[idx], self.colors[idx]


def train_ds():
    return EmojiDataset(list(read(TRAIN_PATH)))


def train_data_loader(
    *, data_set: EmojiDataset,
        batch_size: int):

    return DataLoader(
        data_set,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=0,
    )


def eval_data_loader():
    return DataLoader(
        EmojiDataset(list(read(EVAL_PATH))),
        batch_size=2000,
        shuffle=False,
        drop_last=False,
        num_workers=0,
    )
