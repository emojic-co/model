import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import typer

from model.config import MAX_TEXT_LEN
from model.data import normalize

_HEX = re.compile(r"^#[0-9a-fA-F]{6}$")
_COLORS = {"red", "green", "blue", "dark", "bright"}


def test_gold_rows_derived():
    from tools.report import GOLD_PER_COLOR, _gold_rows

    rows = _gold_rows()
    assert 0 < len(rows) <= len(_COLORS) * GOLD_PER_COLOR, len(rows)
    counts = Counter(r["color"] for r in rows)
    assert set(counts) == _COLORS, counts
    assert all(0 < v <= GOLD_PER_COLOR for v in counts.values()), counts
    assert rows == _gold_rows(), "sampling not deterministic"
    for r in rows:
        nt = normalize(r["text"])
        assert nt and len(nt) <= MAX_TEXT_LEN, r["text"]
        assert isinstance(r["styles"], list) and r["styles"], r
        assert _HEX.match(r["bg"][0]) and _HEX.match(r["bg"][1]), r
        assert _HEX.match(r["fg"]), r


def test_card_distance():
    from tools.report import _card_distance, _hex_to_offsets

    red = _hex_to_offsets("#ff0000") * 3
    assert _card_distance(list(red), list(red), "red") < 1e-6
    assert _card_distance(list(red), list(red), "dark") < 1e-6
    d = _card_distance(
        list(_hex_to_offsets("#ff0000") * 3),
        list(_hex_to_offsets("#00ff00") * 3),
        "green",
    )
    assert d > 0.2, d


def test_dark_ignores_chroma():
    from tools.report import _card_distance, _hex_to_offsets

    a = list(_hex_to_offsets("#000000") * 3)
    b = list(_hex_to_offsets("#0000ff") * 3)
    d_blue = _card_distance(a, b, "blue")
    d_dark = _card_distance(a, b, "dark")
    assert d_dark < d_blue, (d_dark, d_blue)
    assert d_dark > 0, d_dark


def test_linechart_series():
    from tools.report import _linechart

    svg = _linechart(
        [("1", 0.1), ("2", 0.5)],
        series=[("style", [0.2, 0.6], "lline2")],
        legend=("emoji", "style"),
    )
    assert 'class="lline2"' in svg
    assert ">emoji<" in svg and ">style<" in svg


_app = typer.Typer(
    add_completion=False,
    context_settings={"help_option_names": ["-h", "--help"]},
)


@_app.command()
def main() -> None:
    test_gold_rows_derived()
    test_card_distance()
    test_dark_ignores_chroma()
    test_linechart_series()
    print("ok")


if __name__ == "__main__":
    _app()
