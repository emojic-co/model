import json
import os
import re
from dataclasses import dataclass

import torch
from torch.utils.data import DataLoader, Dataset

from files import CLDR_JSONL, EMOJILIB_JSONL, EVAL_JSONL, KEYWORDS_JSONL, TRAIN_JSONL
from model.config import (
    EMOJIS,
    KEYWORDS_SAMPLING_RATE,
    MAX_EMOJIS_PER_SAMPLE,
    MAX_TEXT_LEN,
    STYLES,
)

TRAIN_PATH = TRAIN_JSONL
EVAL_PATH = EVAL_JSONL
CLDR_PATH = CLDR_JSONL
EMOJILIB_PATH = EMOJILIB_JSONL
KEYWORDS_PATH = KEYWORDS_JSONL

PAD = "·"
PAD_IDX = 0
CHARS = PAD + "abcdefghijklmnopqrstuvwxyz0123456789!?:()@$%&* "
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


def sampled_emojis_to_tensor(emojis: list[str]) -> torch.Tensor:
    if len(emojis) > MAX_EMOJIS_PER_SAMPLE:
        perm = torch.randperm(len(emojis))[:MAX_EMOJIS_PER_SAMPLE].tolist()
        emojis = [emojis[i] for i in perm]
    return emojis_to_tensor(emojis)


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


def _row_kw(row: dict) -> torch.Tensor:
    pairs = row.get("kw")
    out = torch.zeros(len(EMOJIS), dtype=torch.float32)
    for i, v in pairs or []:
        out[int(i)] = float(v)
    return out


def _keywords_pool():
    try:
        recs = [r for r in read(KEYWORDS_PATH) if r.emojis]
    except FileNotFoundError:
        return None
    if not recs:
        return None
    return (
        torch.stack([text_to_tensor(r.text) for r in recs]),
        [r.emojis for r in recs],
        torch.stack([styles_to_tensor(r.styles) for r in recs]),
        torch.stack([colors2tensor(r.colors) for r in recs]),
    )


CLDR_MIN_KEYWORD_LEN = 3
EMOJILIB_MIN_KEYWORD_LEN = 3
KEYWORDS_MIN_KEYWORD_LEN = 3


def _keyword_pool(
    path: str, min_keyword_len: int
) -> tuple[torch.Tensor, torch.Tensor] | None:
    words: dict[str, list[str]] = {}
    try:
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                d = json.loads(line)
                word = str(d.get("text", ""))
                if len(word) < min_keyword_len or not re.search(r"[a-zA-Z]", word):
                    continue
                targets = words.setdefault(word, [])
                for e in str(d.get("emojis", "")).split():
                    if e not in targets:
                        targets.append(e)
    except FileNotFoundError:
        return None
    rows = [(w, exp) for w, exp in words.items() if any(e in emoji2idx for e in exp)]
    if not rows:
        return None
    return (
        torch.stack([text_to_tensor(normalize(w)) for w, _ in rows]),
        torch.stack([emojis_to_tensor(exp) for _, exp in rows]),
    )


def cldr_keyword_pool() -> tuple[torch.Tensor, torch.Tensor] | None:
    return _keyword_pool(CLDR_PATH, CLDR_MIN_KEYWORD_LEN)


def emojilib_keyword_pool() -> tuple[torch.Tensor, torch.Tensor] | None:
    return _keyword_pool(EMOJILIB_PATH, EMOJILIB_MIN_KEYWORD_LEN)


def keywords_keyword_pool() -> tuple[torch.Tensor, torch.Tensor] | None:
    return _keyword_pool(KEYWORDS_PATH, KEYWORDS_MIN_KEYWORD_LEN)


class EmojiDataset(Dataset):
    def __init__(self, records: list[record], mix_keywords: bool = False):
        self.text = torch.stack([text_to_tensor(r.text) for r in records])
        self.emoji_lists = [r.emojis for r in records]
        self.style = torch.stack([styles_to_tensor(r.styles) for r in records])
        self.colors = torch.stack([colors2tensor(r.colors) for r in records])
        self.keywords = (
            _keywords_pool() if mix_keywords and KEYWORDS_SAMPLING_RATE > 0 else None
        )

    def __len__(self):
        return len(self.text)

    def __getitem__(self, idx):
        if self.keywords is not None and torch.rand(1).item() < KEYWORDS_SAMPLING_RATE:
            text, emoji_lists, style, colors = self.keywords
            j = int(torch.randint(len(text), (1,)).item())
            return text[j], sampled_emojis_to_tensor(emoji_lists[j]), style[j], colors[j]
        return (
            self.text[idx],
            sampled_emojis_to_tensor(self.emoji_lists[idx]),
            self.style[idx],
            self.colors[idx],
        )


def train_ds():
    return EmojiDataset(list(read(TRAIN_PATH)), mix_keywords=True)


PIN_MEMORY = torch.cuda.is_available()
DATA_WORKERS = int(os.environ.get("EMOJIC_DATA_WORKERS", "0"))


def _loader_kwargs() -> dict:
    kw: dict = {"num_workers": DATA_WORKERS, "pin_memory": PIN_MEMORY}
    if DATA_WORKERS > 0:
        kw["persistent_workers"] = True
        kw["prefetch_factor"] = 4
    return kw


def train_data_loader(
    *, data_set: EmojiDataset,
        batch_size: int):

    return DataLoader(
        data_set,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        **_loader_kwargs(),
    )


def eval_data_loader():
    return DataLoader(
        EmojiDataset(list(read(EVAL_PATH)), mix_keywords=True),
        batch_size=2000,
        shuffle=False,
        drop_last=False,
        **_loader_kwargs(),
    )
