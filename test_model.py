from __future__ import annotations

import datetime as dt
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
from model import Model

MODEL_PT = Path("model.pt")
REPORT_DIR = Path("report/model")

NEGATION_EXPECTED: dict[str, str] = {
    "Happy": "Sad",
    "Sad": "Happy",
    "Calm": "Anxious",
    "Anxious": "Calm",
    "Angry": "Calm",
}

EMOJI_CUES: list[tuple[str, str]] = [
    ("🎉", "party"),
    ("😢", "crying"),
    ("😡", "furious"),
    ("😴", "sleepy"),
    ("🔥", "fire"),
    ("☕", "coffee"),
    ("🤔", "thinking"),
    ("💔", "heartbroken"),
    ("🥳", "birthday"),
    ("😰", "nervous"),
    ("🚀", "rocket"),
    ("🙏", "grateful"),
    ("🥰", "adore"),
    ("😂", "hilarious"),
    ("🌧️", "rain"),
    ("🧘", "meditate"),
    ("😱", "terrified"),
    ("🙄", "whatever"),
    ("✨", "sparkle"),
    ("😔", "down"),
]


def load_model(path: Path = MODEL_PT) -> Model:
    model = Model()
    model.load_state_dict(torch.load(path, map_location="cpu"))
    model.eval()
    return model


def _encode(text: str) -> torch.Tensor:
    ids = [char2idx[c] for c in normalize(text)[:MAX_TEXT_LEN]]
    ids += [PAD_IDX] * (MAX_TEXT_LEN - len(ids))
    return torch.tensor(ids, dtype=torch.long).unsqueeze(0)


@torch.no_grad()
def predict(model: Model, text: str) -> dict:
    feeling_logits = model(_encode(text))[0]
    return {"feeling": FEELING[int(feeling_logits.argmax())]}


@torch.no_grad()
def predict_emojis(model: Model, text: str, k: int = 5) -> list[str]:
    _, q, emoji_embed = model(_encode(text))
    dists = torch.cdist(q, emoji_embed)[0]
    order = dists.argsort()[:k]
    return [EMOJIS[int(i)] for i in order]


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
    rows = []
    for emoji, word in EMOJI_CUES:
        top5 = predict_emojis(model, word, k=5)
        rows.append(
            {
                "word": word,
                "emoji": emoji,
                "predicted": top5[0],
                "top5": top5,
                "pass": top5[0] == emoji,
            }
        )
    return rows


def _pct(num: int, den: int) -> str:
    return f"{100 * num / den:.0f}%" if den else "n/a"


def _heading(feeling_rows, negation_rows, emoji_rows) -> str:
    f_pass = sum(r["pass"] for r in feeling_rows)
    n_pass = sum(r["pass"] for r in negation_rows)
    e_pass = sum(r["pass"] for r in emoji_rows)
    return (
        f"Feelings Accuracy {f_pass}/{len(feeling_rows)} | "
        f"Neg Feeling Score {n_pass}/{len(negation_rows)} | "
        f"Emojis Accuracy {e_pass}/{len(emoji_rows)}"
    )


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
    e_pass = sum(r["pass"] for r in emoji_rows)

    out: list[str] = []
    out.append(f"# Model test report — {now:%Y-%m-%d %H:%M}")
    out.append("")
    out.append(f"**{_heading(feeling_rows, negation_rows, emoji_rows)}**")
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
        f"- Emojis: **{e_pass}/{len(emoji_rows)}** cue words whose nearest "
        f"emoji is the paired one ({_pct(e_pass, len(emoji_rows))})"
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
    out.append(
        "20 cue words, one per emoji. Pass = the emoji head's nearest embedding "
        "is the paired emoji. Top 5 is nearest-first."
    )
    out.append("")
    out.append("| word | emoji | prediction | top 5 predictions (sorted) |")
    out.append("| --- | --- | --- | --- |")
    for r in emoji_rows:
        mark = "✅" if r["pass"] else "❌"
        top5 = " ".join(r["top5"])
        out.append(f"| {r['word']} | {r['emoji']} | {r['predicted']} {mark} | {top5} |")
    out.append("")

    return "\n".join(out)


def run(model: Model | None = None) -> Path:
    model = model or load_model()
    feeling_rows = test_feelings(model)
    negation_rows = test_negations(model)
    emoji_rows: list[dict] = []

    report = build_report(feeling_rows, negation_rows, emoji_rows)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    path = REPORT_DIR / f"{dt.datetime.now():%m-%d-%H:%M}.md"
    path.write_text(report, encoding="utf-8")

    print(f"{_heading(feeling_rows, negation_rows, emoji_rows)} | report -> {path}")
    return path


if __name__ == "__main__":
    run()
