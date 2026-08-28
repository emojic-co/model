"""Behavioral test suite for the trained emojic model.

Three batteries, all run against the committed ``model.pt``:

1. Feelings -- feed each feeling's own name to the model and check the feeling
   head predicts that feeling.
2. Negations -- feed ``not <feeling>`` and check the feeling head does *not*
   predict ``<feeling>``; where an opposite is expected (``NEGATION_EXPECTED``),
   also check it predicts that opposite.
3. Emojis -- for each of the most frequent emojis in ``data.jsonl`` (see
   ``emoji_keywords.EMOJI_KEYWORDS``), feed a handful of strongly associated
   keywords and check the emoji head predicts that emoji (top-1 and top-3).

Writes a Markdown report to ``report/<MM-DD-HH:MM>.md`` and prints a summary.
``train.py`` calls :func:`run` at the end of every training run.

Standalone: ``uv run test_model.py``
"""

from __future__ import annotations

import collections
import datetime as dt
import json
from pathlib import Path

import torch

from config import MAX_TEXT_LEN
from data import (
    EMOJIS,
    FEELING,
    PAD_IDX,
    char2idx,
    normalize,
)
from emoji_keywords import EMOJI_KEYWORDS
from model import Model

MODEL_PT = Path("model.pt")
REPORT_DIR = Path("report")
DATA = Path("data.jsonl")
TOP_K = 3

# For "not <feeling>" prompts: the feeling we'd expect the model to fall back to.
# Only the confident opposites are listed; a feeling left out is only checked for
# *avoiding* its own negated label, not for landing on a specific alternative.
NEGATION_EXPECTED: dict[str, str] = {
    "Happy": "Sad",
    "Sad": "Happy",
    "Calm": "Anxious",
    "Anxious": "Calm",
    "Angry": "Calm",
    "Excited": "Neutral",
}


def load_model(path: Path = MODEL_PT) -> Model:
    model = Model()
    model.load_state_dict(torch.load(path, map_location="cpu"))
    model.eval()
    return model


def _encode(text: str) -> torch.Tensor:
    ids = [char2idx[c] for c in normalize(text)[:MAX_TEXT_LEN]]
    if not ids:
        ids = [PAD_IDX]
    return torch.tensor(ids, dtype=torch.long).unsqueeze(0)


@torch.no_grad()
def predict(model: Model, text: str) -> dict:
    """Return top emoji/feeling and the top-K emoji list for ``text``."""
    emoji_logits, feeling_logits = model(_encode(text))
    emoji_logits, feeling_logits = emoji_logits[0], feeling_logits[0]
    top_emoji_idx = torch.topk(emoji_logits, k=min(TOP_K, len(EMOJIS))).indices
    return {
        "emoji": EMOJIS[int(emoji_logits.argmax())],
        "feeling": FEELING[int(feeling_logits.argmax())],
        "emoji_topk": [EMOJIS[int(i)] for i in top_emoji_idx],
    }


def emoji_frequencies() -> collections.Counter:
    counts: collections.Counter = collections.Counter()
    if not DATA.exists():
        return counts
    with DATA.open(encoding="utf-8") as f:
        for line in f:
            counts[json.loads(line)["emoji"]] += 1
    return counts


def test_feelings(model: Model) -> list[dict]:
    rows = []
    for feeling in FEELING:
        got = predict(model, feeling)
        rows.append(
            {
                "feeling": feeling,
                "prompt": feeling.lower(),
                "predicted": got["feeling"],
                "pass": got["feeling"] == feeling,
            }
        )
    return rows


def test_negations(model: Model) -> list[dict]:
    rows = []
    for feeling in FEELING:
        prompt = f"not {feeling.lower()}"
        got = predict(model, prompt)["feeling"]
        expected = NEGATION_EXPECTED.get(feeling)
        avoided = got != feeling
        matched = expected is None or got == expected
        rows.append(
            {
                "feeling": feeling,
                "prompt": prompt,
                "predicted": got,
                "expected": expected,
                "avoided": avoided,
                "matched_expected": matched,
                "pass": avoided and matched,
            }
        )
    return rows


def test_emojis(model: Model) -> list[dict]:
    freqs = emoji_frequencies()
    ranks = {e: i + 1 for i, (e, _) in enumerate(freqs.most_common())}
    rows = []
    for emoji, keywords in EMOJI_KEYWORDS.items():
        hits, top3, cases = 0, 0, []
        for kw in keywords:
            got = predict(model, kw)
            hit = got["emoji"] == emoji
            in_top3 = emoji in got["emoji_topk"]
            hits += hit
            top3 += in_top3
            cases.append(
                {
                    "keyword": kw,
                    "predicted": got["emoji"],
                    "topk": got["emoji_topk"],
                    "pass": hit,
                    "top3": in_top3,
                }
            )
        rows.append(
            {
                "emoji": emoji,
                "rank": ranks.get(emoji),
                "count": freqs.get(emoji, 0),
                "n": len(keywords),
                "hits": hits,
                "top3": top3,
                "cases": cases,
            }
        )
    rows.sort(key=lambda r: (r["rank"] is None, r["rank"] or 0))
    return rows


def _pct(num: int, den: int) -> str:
    return f"{100 * num / den:.0f}%" if den else "n/a"


def build_report(
    feeling_rows: list[dict],
    negation_rows: list[dict],
    emoji_rows: list[dict],
) -> str:
    now = dt.datetime.now()
    f_pass = sum(r["pass"] for r in feeling_rows)
    n_avoided = sum(r["avoided"] for r in negation_rows)
    n_expected = [r for r in negation_rows if r["expected"] is not None]
    n_matched = sum(r["matched_expected"] for r in n_expected)
    e_hits = sum(r["hits"] for r in emoji_rows)
    e_top3 = sum(r["top3"] for r in emoji_rows)
    e_total = sum(r["n"] for r in emoji_rows)

    out: list[str] = []
    out.append(f"# Model test report — {now:%Y-%m-%d %H:%M}")
    out.append("")
    out.append(f"- Model: `{MODEL_PT}`")
    out.append(
        f"- Feelings: **{f_pass}/{len(feeling_rows)}** "
        f"name prompts predicted correctly ({_pct(f_pass, len(feeling_rows))})"
    )
    out.append(
        f"- Negations: **{n_avoided}/{len(negation_rows)}** `not <feeling>` "
        f"prompts avoided the negated feeling "
        f"({_pct(n_avoided, len(negation_rows))}); "
        f"**{n_matched}/{len(n_expected)}** hit the expected opposite"
    )
    out.append(
        f"- Emojis: **{e_hits}/{e_total}** keyword prompts top-1 "
        f"({_pct(e_hits, e_total)}), **{e_top3}/{e_total}** top-{TOP_K} "
        f"({_pct(e_top3, e_total)}), across {len(emoji_rows)} emojis"
    )
    out.append("")

    out.append("## Feelings")
    out.append("")
    out.append("| feeling | prompt | predicted | result |")
    out.append("| --- | --- | --- | --- |")
    for r in feeling_rows:
        mark = "✅" if r["pass"] else "❌"
        out.append(f"| {r['feeling']} | `{r['prompt']}` | {r['predicted']} | {mark} |")
    out.append("")

    out.append("## Negations")
    out.append("")
    out.append(
        "Pass = the feeling head does **not** return the negated feeling "
        "(and, where an opposite is expected, returns it)."
    )
    out.append("")
    out.append("| prompt | predicted | expected | avoided | result |")
    out.append("| --- | --- | --- | --- | --- |")
    for r in negation_rows:
        expected = r["expected"] or "—"
        avoided = "✅" if r["avoided"] else "❌"
        mark = "✅" if r["pass"] else "❌"
        out.append(
            f"| `{r['prompt']}` | {r['predicted']} | {expected} | {avoided} | {mark} |"
        )
    out.append("")

    out.append("## Emojis")
    out.append("")
    out.append("| emoji | data rank | count | top-1 | top-3 |")
    out.append("| --- | --- | --- | --- | --- |")
    for r in emoji_rows:
        rank = r["rank"] if r["rank"] is not None else "—"
        out.append(
            f"| {r['emoji']} | {rank} | {r['count']} | "
            f"{r['hits']}/{r['n']} ({_pct(r['hits'], r['n'])}) | "
            f"{r['top3']}/{r['n']} ({_pct(r['top3'], r['n'])}) |"
        )
    out.append("")

    out.append("### Per-keyword detail")
    out.append("")
    for r in emoji_rows:
        out.append(f"#### {r['emoji']}  (rank {r['rank'] or '—'})")
        out.append("")
        out.append("| keyword | predicted | top-3 | result |")
        out.append("| --- | --- | --- | --- |")
        for c in r["cases"]:
            if c["pass"]:
                mark = "✅"
            elif c["top3"]:
                mark = "~ top-3"
            else:
                mark = "❌"
            topk = " ".join(c["topk"])
            out.append(f"| `{c['keyword']}` | {c['predicted']} | {topk} | {mark} |")
        out.append("")

    return "\n".join(out)


def run(model: Model | None = None) -> Path:
    """Run both batteries, write the report, print a summary, return its path."""
    model = model or load_model()
    feeling_rows = test_feelings(model)
    negation_rows = test_negations(model)
    emoji_rows = test_emojis(model)

    report = build_report(feeling_rows, negation_rows, emoji_rows)
    REPORT_DIR.mkdir(exist_ok=True)
    path = REPORT_DIR / f"{dt.datetime.now():%m-%d-%H:%M}.md"
    path.write_text(report, encoding="utf-8")

    f_pass = sum(r["pass"] for r in feeling_rows)
    n_avoided = sum(r["avoided"] for r in negation_rows)
    e_hits = sum(r["hits"] for r in emoji_rows)
    e_total = sum(r["n"] for r in emoji_rows)
    print(
        f"feelings {f_pass}/{len(feeling_rows)} | "
        f"negations avoided {n_avoided}/{len(negation_rows)} | "
        f"emoji top-1 {e_hits}/{e_total} ({_pct(e_hits, e_total)}) | "
        f"report -> {path}"
    )
    return path


if __name__ == "__main__":
    run()
