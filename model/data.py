import json
import os
import re
from dataclasses import dataclass
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset

from files import EVAL_JSONL, TRAIN_JSONL
from model.config import (
    EMOJIS,
    MAX_TEXT_LEN,
    SAMPLING_RATE_MAX,
    SAMPLING_SOURCES,
    STYLES,
)

TRAIN_PATH = TRAIN_JSONL
EVAL_PATH = EVAL_JSONL

SRC_FULL = "full"

PAD = "·"
PAD_IDX = 0
HEBREW = "אבגדהוזחטיכלמנסעפצקרשתךםןףץ"
CHARS = ''.join([
    PAD,
    "abcdefghijklmnopqrstuvwxyz",
    HEBREW,
    "0123456789",
    "!?:()@$%&*",
    " ",])

VOCAB_SIZE = len(CHARS)


char2idx = {char: i for i, char in enumerate(CHARS)}
style2idx = {s: i for i, s in enumerate(STYLES)}
emoji2idx = {e: i for i, e in enumerate(EMOJIS)}

COLOR_DIM = 9
AB_RANGE = 0.4


def hex2rgb(h: str) -> tuple[int, int, int]:
    h = h.lstrip("#")
    return (
        int(h[0:2], 16),
        int(h[2:4], 16),
        int(h[4:6], 16))


def _srgb_to_linear(c: float) -> float:
    return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4


def _linear_to_srgb(c: float) -> float:
    v = 12.92 * c if c <= 0.0031308 else 1.055 * max(c, 0.0) ** (1 / 2.4) - 0.055
    return v * 255


def srgb_to_oklab(rgb: tuple[int, int, int]) -> tuple[float, float, float]:
    r = _srgb_to_linear(rgb[0] / 255)
    g = _srgb_to_linear(rgb[1] / 255)
    b = _srgb_to_linear(rgb[2] / 255)
    l_ = (0.4122214708 * r + 0.5363325363 * g + 0.0514459929 * b) ** (1 / 3)
    m_ = (0.2119034982 * r + 0.6806995451 * g + 0.1073969566 * b) ** (1 / 3)
    s_ = (0.0883024619 * r + 0.2817188376 * g + 0.6299787005 * b) ** (1 / 3)
    return (
        0.2104542553 * l_ + 0.793617785 * m_ - 0.0040720468 * s_,
        1.9779984951 * l_ - 2.428592205 * m_ + 0.4505937099 * s_,
        0.0259040371 * l_ + 0.7827717662 * m_ - 0.808675766 * s_,
    )


def oklab_to_srgb(lab: tuple[float, float, float]) -> tuple[float, float, float]:
    L, a, b = lab
    l_ = (L + 0.3963377774 * a + 0.2158037573 * b) ** 3
    m_ = (L - 0.1055613458 * a - 0.0638541728 * b) ** 3
    s_ = (L - 0.0894841775 * a - 1.291485548 * b) ** 3
    return (
        _linear_to_srgb(4.0767416621 * l_ - 3.3077115913 * m_ + 0.2309699292 * s_),
        _linear_to_srgb(-1.2684380046 * l_ + 2.6097574011 * m_ - 0.3413193965 * s_),
        _linear_to_srgb(-0.0041960863 * l_ - 0.7034186147 * m_ + 1.707614701 * s_),
    )


def hex_to_color_unit(h: str) -> tuple[float, float, float]:
    """sRGB hex -> model-space unit triple: rgb2oklab, clamp a/b to
    +/-AB_RANGE, map to [-1, 1] (L = 2*L-1, a/b = v/AB_RANGE)."""
    L, a, b = srgb_to_oklab(hex2rgb(h))
    a = max(-AB_RANGE, min(AB_RANGE, a))
    b = max(-AB_RANGE, min(AB_RANGE, b))
    return (2 * L - 1, a / AB_RANGE, b / AB_RANGE)


def color_unit_to_hex(unit: tuple[float, float, float]) -> str:
    """Model-space unit triple -> sRGB hex (see Critic.forward / ColorGen)."""
    n0, n1, n2 = unit
    L = (n0 + 1) / 2
    a = n1 * AB_RANGE
    b = n2 * AB_RANGE
    r, g, b_ = oklab_to_srgb((L, a, b))

    def clamp(v: float) -> int:
        return max(0, min(255, round(v)))

    return f"#{clamp(r):02x}{clamp(g):02x}{clamp(b_):02x}"


def colors2tensor(colors: list[str]) -> torch.Tensor:
    vals = [v for h in colors for v in hex_to_color_unit(h)]
    return torch.tensor(vals, dtype=torch.float32)


def color_tensor_to_hexes(vec9: torch.Tensor) -> list[str]:
    assert vec9.shape == (9,), "Input tensor must be of shape (9,)"
    flat = vec9.tolist()
    return [color_unit_to_hex(tuple(flat[i:i + 3])) for i in range(0, 9, 3)]


def sample_colors_tensor(colors: list[list[str]]) -> torch.Tensor:
    if not colors:
        return torch.zeros(COLOR_DIM)
    i = int(torch.randint(len(colors), (1,)).item())
    return colors2tensor(colors[i])


def normalize(text: str) -> str:
    text = re.sub(r"\s+", " ", text.lower()).strip()
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
    colors: list[list[str]]


def _parse_record(d: dict) -> record | None:
    match d:
        case {"text": text, "emojis": emojis, "styles": styles}:
            text = normalize(text)

            if not text or len(text) > MAX_TEXT_LEN:
                return None

            emojis = [e for e in emojis.split() if e in emoji2idx]
            styles = [s for s in styles if s in style2idx]

            colors = [
                [*c["bg"], c["fg"]]
                for c in d.get("colors") or []
                if isinstance(c, dict) and c.get("bg") and c.get("fg")
            ]

            return record(text, emojis, styles, colors)
    return None


def read(path: Path):
    def read_jsonl():
        with path.open(encoding='utf-8') as f:
            for line in f:
                yield json.loads(line)

    for d in read_jsonl():
        r = _parse_record(d)
        if r is not None:
            yield r


def read_color_keywords(path: Path) -> list[tuple[str, record]]:
    out: list[tuple[str, record]] = []
    try:
        with path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                d = json.loads(line)
                r = _parse_record({**d, "emojis": "", "styles": []})
                if r is not None and r.colors:
                    out.append((d["keyword"], r))
    except FileNotFoundError:
        pass
    return out


_Pool = tuple[torch.Tensor, list[list[str]], torch.Tensor]


def _load_pool(path: Path) -> _Pool | None:
    recs: list[record] = []
    try:
        with path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                r = _parse_record(json.loads(line))
                if r is not None and r.emojis:
                    recs.append(r)
    except FileNotFoundError:
        return None
    if not recs:
        return None
    return (
        torch.stack([text_to_tensor(r.text) for r in recs]),
        [r.emojis for r in recs],
        torch.stack([styles_to_tensor(r.styles) for r in recs]),
    )


def _sample_pool(pool: _Pool, src: str) -> tuple:
    text, emoji_lists, style = pool
    j = int(torch.randint(len(text), (1,)).item())
    return (
        text[j],
        emojis_to_tensor(emoji_lists[j]),
        style[j],
        src,
        torch.zeros(COLOR_DIM),
        False,
    )


class SamplingRates:
    def __init__(self, names: list[str], base_rate: float):
        self.base_rate = base_rate
        self._current = {name: base_rate for name in names}

    def get(self, name: str) -> float:
        return self._current[name]

    def set(self, name: str, rate: float) -> None:
        self._current[name] = rate


class EmojiDataset(Dataset):
    def __init__(
        self,
        records: list[record],
        mix_sources: bool = False,
    ):
        self.text = torch.stack([text_to_tensor(r.text) for r in records])
        self.text_str = [r.text for r in records]
        self.emoji_lists = [r.emojis for r in records]
        self.style = torch.stack([styles_to_tensor(r.styles) for r in records])
        self.colors = [r.colors for r in records]

        self.rates = (
            SamplingRates(list(SAMPLING_SOURCES), SAMPLING_RATE_MAX)
            if mix_sources
            else None
        )
        self.pools: dict[str, _Pool] = {}
        if mix_sources:
            for name, source in SAMPLING_SOURCES.items():
                pool = _load_pool(source.path)
                if pool is not None:
                    self.pools[name] = pool

    def __len__(self):
        return len(self.text)

    def __getitem__(self, idx):
        r = torch.rand(1).item()
        if self.rates is not None:
            cum = 0.0
            for name, pool in self.pools.items():
                cum += self.rates.get(name)
                if r < cum:
                    return _sample_pool(pool, name)
        colors = self.colors[idx]
        return (
            self.text[idx],
            emojis_to_tensor(self.emoji_lists[idx]),
            self.style[idx],
            SRC_FULL,
            sample_colors_tensor(colors),
            bool(colors),
        )


def train_ds(mix_sources: bool = True):
    return EmojiDataset(list(read(TRAIN_PATH)), mix_sources=mix_sources)


PIN_MEMORY = torch.cuda.is_available()
DATA_WORKERS = int(os.environ.get("EMOJIC_DATA_WORKERS", "4"))


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


def eval_ds() -> EmojiDataset:
    return EmojiDataset(list(read(EVAL_PATH)), mix_sources=False)


def eval_data_loader():
    return DataLoader(
        eval_ds(),
        batch_size=2000,
        shuffle=False,
        drop_last=False,
        **_loader_kwargs(),
    )
