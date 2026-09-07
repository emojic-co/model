import json
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import typer

from files import GOLD_JSONL
from model.config import EMOJIS, MAX_TEXT_LEN, STYLES
from model.data import normalize

_HEX = re.compile(r"^#[0-9a-fA-F]{6}$")
_COLORS = {"red", "green", "blue", "dark", "bright"}


def test_gold_file_integrity():
    rows = [
        json.loads(x)
        for x in Path(GOLD_JSONL).read_text(encoding="utf-8").splitlines()
        if x.strip()
    ]
    assert len(rows) == 125, len(rows)
    counts = Counter(r["color"] for r in rows)
    assert set(counts) == _COLORS, counts
    assert all(v == 25 for v in counts.values()), counts
    evocab, svocab = set(EMOJIS), set(STYLES)
    for r in rows:
        nt = normalize(r["text"])
        assert nt and len(nt) <= MAX_TEXT_LEN, r["text"]
        toks = r["emojis"].split()
        assert toks, r
        bad = [e for e in toks if e not in evocab]
        assert not bad, bad
        assert r["styles"], r
        assert all(s in svocab for s in r["styles"]), r["styles"]
        assert _HEX.match(r["bg"][0]) and _HEX.match(r["bg"][1]), r
        assert _HEX.match(r["fg"]), r


_app = typer.Typer(
    add_completion=False,
    context_settings={"help_option_names": ["-h", "--help"]},
)


@_app.command()
def main() -> None:
    test_gold_file_integrity()
    print("ok")


if __name__ == "__main__":
    _app()
