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


def test_emoji_html_five_way():
    from tools.report import EMOJI_KS, _emoji_html

    n = len(EMOJI_KS)
    d = {
        "eval": {
            "n": 100,
            "acc_at_k": [0.3 + 0.05 * i for i in range(n)],
            "keywords_acc_at_k": [0.2 + 0.05 * i for i in range(n)],
            "fusion_gate_acc_at_k": [0.4 + 0.04 * i for i in range(n)],
            "fusion_gain_acc_at_k": [0.4 + 0.045 * i for i in range(n)],
            "fusion_mix_acc_at_k": [0.4 + 0.05 * i for i in range(n)],
            "baseline": {"name": "overlap", "acc_at_k": [0.1 + 0.04 * i for i in range(n)]},
        },
        "keywords": {"n": 50, "acc_at_k": [0.2 + 0.05 * i for i in range(n)]},
    }
    h = _emoji_html(d)
    assert 'class="lline2"' in h and 'class="lline3"' in h
    assert 'class="lline4"' in h and 'class="lline5"' in h
    assert ">EmojiHead<" in h and ">Keywords<" in h
    assert ">Fusion·Gate<" in h and ">Fusion·Mix<" in h
    assert 'class="bline"' in h

    d["eval"]["keywords_acc_at_k"] = None
    d["eval"]["fusion_gate_acc_at_k"] = None
    d["eval"]["fusion_gain_acc_at_k"] = None
    d["eval"]["fusion_mix_acc_at_k"] = None
    h2 = _emoji_html(d)
    assert 'class="lline3"' not in h2 and 'class="lline4"' not in h2
    assert "CLDR baseline (overlap)" in h2


def test_emoji_extra_acc_reads_precomputed_kw():
    import torch

    from model.data import record
    from tools.report import EMOJI_KS, EMOJIS, _emoji_extra_acc

    n = len(EMOJIS)
    rows = [
        record("pizza tonight", ["x"], ["Neutral"], ["#000", "#000", "#fff"], [[0, 0.9]]),
        record("a quiet walk", [], ["Neutral"], ["#000", "#000", "#fff"], []),
    ]
    tgt = torch.zeros(2, n)
    tgt[0, 0] = 1.0
    logit_m = torch.randn(2, n)
    out = _emoji_extra_acc(rows, tgt, logit_m)
    assert len(out["keywords"]) == len(EMOJI_KS)
    assert all(0.0 <= v <= 1.0 for v in out["keywords"])


def test_flex_keyword_candidates_and_section():
    from model.kwtokens import query_tokens
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


def test_keywords_flex_html():
    from tools.report import _keywords_flex_html

    empty = _keywords_flex_html({})
    assert "Model — Keyword vocab" in empty and "not available" in empty

    d = {
        "candidates": 3,
        "missed": 2,
        "ranked": [
            {
                "kw": "zzzz",
                "emojis": ["😀"],
                "top5": ["🍕", "🎉", "😀", "🐶", "🚀"],
                "rank": 42,
            },
            {"kw": "yyyy", "emojis": ["🎉", "🥳"], "top5": ["🍕", "🎉"], "rank": 11},
            {"kw": "aaaa", "emojis": ["🍕"], "top5": ["🍕"], "rank": 1},
        ],
    }
    h = _keywords_flex_html(d)
    assert "<h2>Model — Keyword vocab</h2>" in h
    assert "3 candidates" in h or "3</" in h or ">3<" in h
    assert "1 " in h and "missed" in h
    assert "<th>Keyword</th>" not in h and "<th>Rank</th>" not in h
    assert ">zzzz<" not in h and ">yyyy<" not in h
    assert "keywords_flex.ranked" in h


def test_section_keyword_exact_probe():
    from tools.report import _kw_proj, _section_keyword

    if not _kw_proj():
        print("skip test_section_keyword_exact_probe (no web/public/kwproj.json)")
        return
    d = _section_keyword(None, None)
    assert "exact" in d, d
    ex = d["exact"]
    assert 0 < ex["n"] <= ex["total"]
    assert len(ex["acc_at_k"]) == 10
    assert all(0.0 <= v <= 1.0 for v in ex["acc_at_k"])
    assert ex["acc_at_k"] == sorted(ex["acc_at_k"])


def test_keyword_html_renders():
    from tools.report import _keyword_html

    assert _keyword_html({}) == ""
    d = {
        "exact": {
            "n": 1500,
            "total": 4900,
            "acc_at_k": [0.9 + 0.005 * i for i in range(10)],
        },
        "fusion": {"n": 1500, "acc_at_k": [0.5 + 0.02 * i for i in range(10)]},
    }
    h = _keyword_html(d)
    assert "Keyword predictor — CLDR" in h
    assert "1500/4900" in h
    assert ">Exact kw<" in h and ">Fusion<" in h


def test_load_global_goals():
    from tools.report import _load_global_goals

    g = _load_global_goals()
    assert "emoji prediction" in g
    assert g["max text len"] == 42
    assert g["vocabulary"]["size"] == 700
    cov = g["vocabulary"]["coverage"]
    assert cov["country-flag"] == 1.0 and len(cov) == 101


def test_group_json_shape():
    from tools.report import _group_json

    grp = _group_json()
    if not grp:
        print("skip test_group_json_shape (no data/group.json)")
        return
    assert len(grp) == 101
    assert "face-smiling" in grp and "country-flag" in grp
    for members in grp.values():
        assert isinstance(members, list)
        assert all(isinstance(e, str) and e for e in members)
    assert len(grp["country-flag"]) > 200


def test_section_status_orders_and_grades():
    from tools.report import EMOJI_KS, _section_status

    n = len(EMOJI_KS)
    report = {
        "emoji": {
            "eval": {
                "acc_at_k": [0.5 + 0.03 * i for i in range(n)],
                "fusion_gain_acc_at_k": [0.6 + 0.03 * i for i in range(n)],
            }
        },
        "keyword": {
            "exact": {
                "n": 1500,
                "total": 4900,
                "acc_at_k": [0.96 + 0.002 * i for i in range(n)],
            },
            "fusion": {
                "n": 1500,
                "total": 4900,
                "acc_at_k": [0.7 + 0.02 * i for i in range(n)],
            },
        },
        "cldr": {"acc_at_k": [0.42 + 0.02 * i for i in range(n)]},
        "data": {"max_text_len": 32},
        "labels": {"emojis": 400},
        "cards": {
            "style_acc_at_k": [0.7 + 0.02 * i for i in range(n)],
            "per_color": {"all": {"pure_accuracy": 0.4}},
        },
    }
    st = _section_status(report)
    prios = [g["priority"] for g in st["goals"]]
    assert prios == sorted(prios)
    assert st["best_emoji_variant"] == "Fusion·Gain"
    by_goal = {g["goal"]: g for g in st["goals"]}
    assert by_goal["Exact keyword acc@1"]["priority"] == 1
    assert (
        by_goal["Exact keyword acc@1"]["current"]
        == report["keyword"]["fusion"]["acc_at_k"][0]
    )
    assert by_goal["Full-text emoji acc@1"]["priority"] == 2
    assert (
        by_goal["Full-text emoji acc@1"]["current"]
        == report["emoji"]["eval"]["fusion_gain_acc_at_k"][0]
    )
    assert by_goal["Fuzzy keyword acc@1"]["status"] == "na"
    assert by_goal["Color energy · global"]["priority"] == 4
    assert by_goal["Color energy · global"]["status"] == "na"
    assert by_goal["Style acc@1"]["priority"] == 5
    assert by_goal["Max text len"]["priority"] == 6
    assert by_goal["Max text len"]["status"] == "red"
    assert by_goal["Emoji vocab size"]["priority"] == 7
    assert by_goal["Emoji vocab size"]["status"] == "red"
    cov = by_goal["Vocab coverage (per Unicode group)"]
    assert cov["priority"] == 8
    if st["vocab_coverage"].get("measurable"):
        assert cov["status"] == "na"
        assert "deferred" in cov["note"]
    assert all(g["status"] in {"good", "amber", "red", "na"} for g in st["goals"])
    assert "vocab_coverage" in st


def test_vocab_coverage_shape():
    from tools.report import _vocab_coverage

    vc = _vocab_coverage()
    if not vc.get("measurable"):
        assert "reason" in vc
        return
    assert vc["groups_total"] == 101
    assert 0 <= vc["groups_passed"] <= vc["groups_measurable"] <= vc["groups_total"]
    assert 0.0 <= vc["score"] <= 1.0
    for v in vc["groups"].values():
        assert v["covered"] <= v["total"]
        if v["total"]:
            assert v["score"] is not None and 0.0 <= v["score"] <= 1.0
        else:
            assert v["score"] is None
    assert isinstance(vc["top_missing"], list)


def test_status_vocab_coverage_gate():
    from tools.report import EMOJI_KS, _section_status, _vocab_coverage

    vc = _vocab_coverage()
    if not vc.get("measurable"):
        return
    n = len(EMOJI_KS)
    hi = [0.999] * n
    report = {
        "emoji": {"eval": {"acc_at_k": hi, "fusion_gain_acc_at_k": hi}},
        "keyword": {"fusion": {"n": 1, "total": 1, "acc_at_k": hi}},
        "data": {"max_text_len": 42},
        "labels": {"emojis": 1000},
        "cards": {"style_acc_at_k": hi, "per_color": {}},
    }
    st = _section_status(report)
    by_goal = {g["goal"]: g for g in st["goals"]}
    cov = by_goal["Vocab coverage (per Unicode group)"]
    higher_open = any(
        g["status"] in ("red", "amber") for g in st["goals"] if g["priority"] <= 7
    )
    if higher_open:
        assert cov["status"] == "na" and "deferred" in cov["note"]
    else:
        assert cov["status"] in {"good", "red"}
        assert "deferred" not in cov["note"]


def test_status_html_colors_rows():
    from tools.report import _status_html

    status = {
        "goals": [
            {
                "goal": "G1",
                "priority": 1,
                "target": "≥ 0.90",
                "current": 0.5,
                "status": "red",
                "note": "hi",
            },
            {
                "goal": "G2",
                "priority": 2,
                "target": "≥ 700",
                "current": 800,
                "status": "good",
                "note": "",
            },
            {
                "goal": "G3",
                "priority": 3,
                "target": "high",
                "current": None,
                "status": "na",
                "note": "tbd",
            },
        ]
    }
    h = _status_html(status)
    assert 'class="sc-red"' in h and 'class="sc-good"' in h and 'class="sc-na"' in h
    assert "Goal status" in h
    assert "0.500" in h and "800" in h and "n/a" in h


def test_section_goals_compare_shape():
    import tools.report as rep
    from tools.report import EMOJI_KS, _section_goals

    n = len(EMOJI_KS)
    acc = [0.6 + 0.03 * i for i in range(n)]
    doc = {
        "meta": {"rationale": "step 1"},
        "goals": {
            "emoji prediction": {
                "exact keyword": {"acc@1": 0.95},
                "full text": {"acc@1": 0.70, "acc@5": 0.85},
            },
            "vocabulary": {"size": 700, "coverage": {"face-smiling": 0.9}},
        },
    }
    orig = rep._load_goals
    rep._load_goals = lambda: ("goal/x.yml", doc)
    try:
        report = {
            "emoji": {"eval": {"acc_at_k": acc, "fusion_gain_acc_at_k": acc}},
            "labels": {"emojis": 150},
            "status": {"vocab_coverage": {"groups": {"face-smiling": {"score": 0.5}}}},
        }
        g = _section_goals(report)
    finally:
        rep._load_goals = orig

    assert g["source_file"] == "goal/x.yml"
    t1 = g["compare"]["emoji prediction"]["full text"]["acc@1"]
    assert t1["target"] == 0.70 and t1["dir"] == "max"
    assert t1["actual"] == acc[0] and t1["met"] is False
    assert t1["delta"] == round(acc[0] - 0.70, 4)
    assert g["compare"]["vocabulary"]["size"]["actual"] == 150
    assert g["compare"]["vocabulary"]["size"]["met"] is False
    cov = g["compare"]["vocabulary"]["coverage"]["face-smiling"]
    assert cov["target"] == 0.9 and cov["actual"] == 0.5 and cov["met"] is False
    ex = g["compare"]["emoji prediction"]["exact keyword"]["acc@1"]
    assert ex["actual"] is None and ex["met"] is None
    assert g["summary"]["unmeasured"] == 1
    assert g["summary"]["unmet"] == 4


def test_goals_html_renders():
    from tools.report import _goals_html

    goals = {
        "source_file": "goal/2026-09-10-abc1234.yml",
        "meta": {"rationale": "prove keyword search"},
        "compare": {
            "emoji prediction": {
                "full text": {
                    "acc@1": {
                        "target": 0.7,
                        "actual": 0.651,
                        "met": False,
                        "dir": "max",
                        "delta": -0.049,
                    },
                },
                "exact keyword": {
                    "acc@1": {
                        "target": 0.95,
                        "actual": None,
                        "met": None,
                        "dir": "max",
                        "delta": None,
                    },
                },
            },
        },
        "summary": {"unmet": 1, "unmeasured": 1},
    }
    h = _goals_html(goals)
    assert "Goals for this iteration" in h
    assert "2026-09-10-abc1234.yml" in h
    assert "prove keyword search" in h
    assert 'class="sc-red"' in h and 'class="sc-na"' in h
    assert "emoji prediction.full text.acc@1" in h
    assert "emoji prediction.exact keyword.acc@1" in h
    assert "≥ 0.7" in h and "0.651" in h and "n/a" in h
    assert "1 unmet" in h and "1 unmeasured" in h


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
    test_emoji_html_five_way()
    test_emoji_extra_acc_reads_precomputed_kw()
    test_flex_keyword_candidates_and_section()
    test_keywords_flex_html()
    test_section_keyword_exact_probe()
    test_keyword_html_renders()
    test_load_global_goals()
    test_group_json_shape()
    test_section_status_orders_and_grades()
    test_vocab_coverage_shape()
    test_status_vocab_coverage_gate()
    test_status_html_colors_rows()
    test_section_goals_compare_shape()
    test_goals_html_renders()
    print("ok")


if __name__ == "__main__":
    _app()
