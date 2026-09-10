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
        print("skip test_gold_rows_derived (no data/colors.jsonl rows)")
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
            ("KWHead", [0.2, 0.4], "lline2"),
            ("Fusion", [0.3, 0.6], "lline3"),
        ],
        legend=("EmojiHead", "KWHead", "Fusion"),
    )
    assert 'class="lline2"' in svg and 'class="lline3"' in svg
    assert ">EmojiHead<" in svg and ">KWHead<" in svg and ">Fusion<" in svg


def test_emoji_html_three_way():
    from tools.report import EMOJI_KS, _emoji_html

    n = len(EMOJI_KS)
    d = {
        "eval": {
            "n": 100,
            "acc_at_k": [0.3 + 0.05 * i for i in range(n)],
            "kw_acc_at_k": [0.2 + 0.05 * i for i in range(n)],
            "fusion_acc_at_k": [0.4 + 0.05 * i for i in range(n)],
            "oracle_acc_at_k": [0.45 + 0.05 * i for i in range(n)],
            "baseline": {"name": "overlap", "acc_at_k": [0.1 + 0.04 * i for i in range(n)]},
        },
        "keywords": {"n": 50, "acc_at_k": [0.2 + 0.05 * i for i in range(n)]},
    }
    h = _emoji_html(d)
    assert 'class="lline2"' in h and 'class="lline3"' in h and 'class="lline4"' in h
    assert ">EmojiHead<" in h and ">KWHead<" in h and ">Fusion<" in h and ">Oracle<" in h
    assert 'class="bline"' in h

    d["eval"]["kw_acc_at_k"] = None
    d["eval"]["fusion_acc_at_k"] = None
    d["eval"]["oracle_acc_at_k"] = None
    h2 = _emoji_html(d)
    assert 'class="lline3"' not in h2 and 'class="lline4"' not in h2
    assert "CLDR baseline (overlap)" in h2


def test_kw_section_keys():
    from files import FLEX_JSON

    if not Path(FLEX_JSON).exists():
        print("skip test_kw_section_keys (no flex.json)")
        return
    from model.flexrank import FlexRanker

    r = FlexRanker(FLEX_JSON)
    v = r.tf_vec("pizza time with friends tonight")
    assert len(v) == len(r.kw_vocab) and len(v) > 500


def test_flex_keyword_candidates_and_section():
    from model.flexrank import query_tokens
    from tools.report import (
        EMOJIS,
        _flex_keyword_candidates,
        _section_keywords_flex,
    )

    cands = _flex_keyword_candidates()
    if not cands:
        print("skip test_flex_keyword_candidates_and_section (no data/ii.json)")
        return
    vocab = set(EMOJIS)
    for kw, tgt in cands:
        assert 3 <= len(kw) <= 6, kw
        assert query_tokens(kw) == [kw], kw
        assert tgt and all(e in vocab for e in tgt), (kw, tgt)
    assert list(cands) == sorted(cands), "candidates not sorted"

    assert _section_keywords_flex(None, None) == {}


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
    test_kw_section_keys()
    test_flex_keyword_candidates_and_section()
    print("ok")


if __name__ == "__main__":
    _app()
