import json
import os
import re
from dataclasses import dataclass, field

import torch
from torch.utils.data import DataLoader, Dataset

from files import EVAL_JSONL, TRAIN_JSONL
from model.config import EMOJIS, MAX_TEXT_LEN, STYLES

FLEX_MAX_K = 32
FLEX_RAW_DIM = 10
FLEXQ_DIM = 5
_FLEXQ_KEYS = ("tokens", "matched", "sum", "max", "cand")

TRAIN_PATH = TRAIN_JSONL
EVAL_PATH = EVAL_JSONL

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


def styles_to_tensor(styles: list[str]) -> torch.Tensor:
    return multi_hot(styles, style2idx, len(STYLES))


@dataclass
class record:
    text: str
    emojis: list[str]
    styles: list[str]
    colors: list[str]
    flexsearch: list = field(default_factory=list)
    flexq: dict = field(default_factory=dict)


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

                yield record(
                    text,
                    emojis,
                    styles,
                    [*bg, fg],
                    d.get("flexsearch") or [],
                    d.get("flexq") or {})


def _row_flex(row):
    if isinstance(row, dict):
        fs = row.get("flexsearch") or []
        fq = row.get("flexq") or {}
    else:
        fs = row.flexsearch or []
        fq = row.flexq or {}

    idx = torch.full((FLEX_MAX_K,), -1, dtype=torch.long)
    raw = torch.zeros(FLEX_MAX_K, FLEX_RAW_DIM, dtype=torch.float32)
    pos = 0
    for entry in fs:
        if pos >= FLEX_MAX_K:
            break
        vi = emoji2idx.get(entry[0])
        if vi is None:
            continue
        score, score_norm, exact, fuzzy, best_idf, n_kw, kw_len, word_len, overlap = (
            entry[1:]
        )
        idx[pos] = vi
        raw[pos] = torch.tensor(
            [
                score,
                score_norm,
                exact,
                fuzzy,
                best_idf,
                1.0 / (pos + 1),
                n_kw,
                kw_len,
                word_len,
                overlap,
            ],
            dtype=torch.float32,
        )
        pos += 1

    flexq = torch.tensor(
        [float(fq.get(k, 0.0)) for k in _FLEXQ_KEYS], dtype=torch.float32
    )
    return idx, raw, flexq


def scatter_flex(flex_idx: torch.Tensor, flex_raw: torch.Tensor) -> torch.Tensor:
    b = flex_idx.size(0)
    v = len(EMOJIS)
    dense = flex_raw.new_zeros(b, v, FLEX_RAW_DIM)
    safe = flex_idx.clamp(min=0)
    mask = (flex_idx >= 0).unsqueeze(-1)
    src = flex_raw * mask
    dense.scatter_add_(1, safe.unsqueeze(-1).expand(-1, -1, FLEX_RAW_DIM), src)
    return dense


def load_energy_keywords(path: str) -> list[str]:
    try:
        with open(path, encoding="utf-8") as f:
            return [w for line in f if (w := line.strip())]
    except FileNotFoundError:
        return []


def load_emoji_keywords(path: str) -> tuple[torch.Tensor, torch.Tensor]:
    try:
        with open(path, encoding="utf-8") as f:
            words = json.load(f)
    except FileNotFoundError:
        words = {}

    if not words:
        return (
            torch.empty(0, MAX_TEXT_LEN, dtype=torch.long),
            torch.empty(0, len(EMOJIS), dtype=torch.float32))

    text = torch.stack([text_to_tensor(normalize(w)) for w in words])
    target = torch.stack([emojis_to_tensor(exp) for exp in words.values()])
    return text, target


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
        flex = [_row_flex(r) for r in records]
        self.flex_idx = torch.stack([f[0] for f in flex])
        self.flex_raw = torch.stack([f[1] for f in flex])
        self.flexq = torch.stack([f[2] for f in flex])

    def __len__(self):
        return len(self.text)

    def __getitem__(self, idx):
        return (
            self.text[idx],
            self.emoji[idx],
            self.style[idx],
            self.colors[idx],
            self.flex_idx[idx],
            self.flex_raw[idx],
            self.flexq[idx],
        )


def train_ds():
    return EmojiDataset(list(read(TRAIN_PATH)))


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
        EmojiDataset(list(read(EVAL_PATH))),
        batch_size=2000,
        shuffle=False,
        drop_last=False,
        **_loader_kwargs(),
    )
