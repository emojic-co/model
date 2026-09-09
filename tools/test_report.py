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
    if not rows:
        print("skip test_gold_rows_derived (no colour-tagged eval rows)")
        return
    assert len(rows) <= len(_COLORS) * GOLD_PER_COLOR, len(rows)
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


def test_linechart_three_way():
    from tools.report import _linechart

    svg = _linechart(
        [("1", 0.1), ("2", 0.5)],
        series=[
            ("FlexRank", [0.2, 0.4], "lline2"),
            ("Fusion", [0.3, 0.6], "lline3"),
        ],
        legend=("EmojiHead", "FlexRank", "Fusion"),
    )
    assert 'class="lline2"' in svg and 'class="lline3"' in svg
    assert ">EmojiHead<" in svg and ">FlexRank<" in svg and ">Fusion<" in svg


def test_emoji_html_three_way():
    from tools.report import EMOJI_KS, _emoji_html

    n = len(EMOJI_KS)
    d = {
        "eval": {
            "n": 100,
            "acc_at_k": [0.3 + 0.05 * i for i in range(n)],
            "flex_acc_at_k": [0.2 + 0.05 * i for i in range(n)],
            "fusion_acc_at_k": [0.4 + 0.05 * i for i in range(n)],
            "baseline": {"name": "overlap", "acc_at_k": [0.1 + 0.04 * i for i in range(n)]},
        },
        "keywords": {"n": 50, "acc_at_k": [0.2 + 0.05 * i for i in range(n)]},
    }
    h = _emoji_html(d)
    assert 'class="lline2"' in h and 'class="lline3"' in h
    assert ">EmojiHead<" in h and ">FlexRank<" in h and ">Fusion<" in h
    assert 'class="bline"' in h

    d["eval"]["flex_acc_at_k"] = None
    d["eval"]["fusion_acc_at_k"] = None
    h2 = _emoji_html(d)
    assert 'class="lline3"' not in h2
    assert "CLDR baseline (overlap)" in h2


def test_cldr_html_flex_series():
    from tools.report import EMOJI_KS, _cldr_html

    n = len(EMOJI_KS)
    d = {
        "n": 200,
        "acc_at_k": [0.3 + 0.04 * i for i in range(n)],
        "flex_acc_at_k": [0.9 + 0.005 * i for i in range(n)],
    }
    h = _cldr_html(d)
    assert 'class="lline2"' in h and ">FlexRank<" in h
    assert "FlexRank acc@1 0.90" in h
    assert 'class="lline2"' not in _cldr_html({**d, "flex_acc_at_k": None})


def test_cldr_flex_smoke():
    from files import FLEX_JSON
    from tools.report import EMOJI_KS, FLEX_SMOKE_MIN, _cldr_flex_smoke

    if not Path(FLEX_JSON).exists():
        print("skip test_cldr_flex_smoke (no flex.json)")
        return
    acc = _cldr_flex_smoke()
    assert acc is not None and len(acc) == len(EMOJI_KS), acc
    assert acc == sorted(acc), acc
    assert acc[0] >= FLEX_SMOKE_MIN, f"FlexRank acc@1 {acc[0]:.3f} < {FLEX_SMOKE_MIN}"


def test_flex_section_keys():
    from files import FLEX_JSON

    if not Path(FLEX_JSON).exists():
        print("skip test_flex_section_keys (no flex.json)")
        return
    from tools.report import _flex_logits, _flex_tensors

    rows = [
        {
            "flexsearch": [["\U0001f355", 3.0, 1.0, 1, 0, 2.0, 4, 5, 5, 5]],
            "flexq": {"tokens": 2, "matched": 1, "sum": 3.0, "max": 3.0, "cand": 1},
        },
        {"flexsearch": [], "flexq": {}},
    ]
    idx, raw, flexq = _flex_tensors(rows)
    assert idx.shape[0] == 2 and flexq.shape == (2, 5), (idx.shape, flexq.shape)
    from model.data import scatter_flex

    dense = scatter_flex(idx, raw)
    logits = _flex_logits(dense)
    assert logits.shape == dense.shape[:-1], logits.shape
    assert float(logits.min()) <= -1e8, float(logits.min())


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
    test_linechart_three_way()
    test_emoji_html_three_way()
    test_cldr_html_flex_series()
    test_cldr_flex_smoke()
    test_flex_section_keys()
    print("ok")


if __name__ == "__main__":
    _app()
