import json
import re
from pathlib import Path

FUZZY_MIN_LEN = 4
MIN_FUZZY_SCORE = 0.66
STOPWORDS = set(
    (
        "a an the to of in on at is it its i you we they he she this that for and or but"
        " not with my your me am are was were be been being do does did have has had"
        " will would can could just so if"
    ).split()
)
_NON = re.compile(r"[^a-z0-9\s]")
_WS = re.compile(r"\s+")


def query_tokens(text: str) -> list[str]:
    t = _NON.sub(" ", text.lower())
    return [w for w in _WS.split(t) if len(w) >= 2 and w not in STOPWORDS]


def _overlap(w: str, k: str) -> float:
    if w == k:
        return 1.0
    if len(w) < FUZZY_MIN_LEN or len(k) < FUZZY_MIN_LEN:
        return 0.0
    if not (w.startswith(k) or k.startswith(w)):
        return 0.0
    r = min(len(w), len(k)) / max(len(w), len(k))
    return r if r >= MIN_FUZZY_SCORE else 0.0


def _r3(x: float) -> float:
    return float(f"{x:.3f}")


class FlexRanker:
    def __init__(self, flex_json_path):
        j = json.loads(Path(flex_json_path).read_text(encoding="utf-8"))
        self.kw_vocab = j["kw_vocab"]

    def tf_vec(self, text: str) -> list[float]:
        q = query_tokens(text)
        out = []
        for k in self.kw_vocab:
            s = 0.0
            for w in q:
                s += _overlap(w, k)
            out.append(_r3(s))
        return out
