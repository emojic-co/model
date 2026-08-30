import json
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from data import FEELING, normalize  # noqa: E402

EVAL_PATH = ROOT / "eval.jsonl"
REPORT_DIR = ROOT / "report" / "eval"
BUCKET = 5


def read(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def histogram(values, bucket=BUCKET):
    if not values:
        return ["_(no samples)_"]
    counts = Counter(v // bucket for v in values)
    hi = max(counts)
    peak = max(counts.values())
    lines = ["| range | count | |", "| --- | ---: | :-- |"]
    for b in range(hi + 1):
        n = counts.get(b, 0)
        bar = "#" * round(40 * n / peak) if n else ""
        lines.append(f"| {b * bucket}–{b * bucket + bucket - 1} | {n} | {bar} |")
    return lines


def summary(values):
    s = sorted(values)
    n = len(s)
    median = s[n // 2] if n % 2 else (s[n // 2 - 1] + s[n // 2]) / 2
    return f"min {s[0]} · median {median:g} · mean {sum(s) / n:.1f} · max {s[-1]}"


def build_report(rows):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M")
    out = [f"# eval.jsonl stats — {ts}", ""]
    out += [f"**{len(rows)} samples** from `eval.jsonl`", ""]

    fc = Counter(r["feeling"] for r in rows)
    out += ["## Feeling distribution", ""]
    out += ["| feeling | count | share |", "| --- | ---: | ---: |"]
    ordered = [f for f in FEELING if f in fc] + sorted(f for f in fc if f not in FEELING)
    for f in ordered:
        out.append(f"| {f} | {fc[f]} | {fc[f] / len(rows):.1%} |")
    out.append("")

    raw = [len(r["text"]) for r in rows]
    norm = [len(normalize(r["text"])) for r in rows]
    out += ["## Text length distribution", ""]
    out += [f"Raw `len(text)`: {summary(raw)}", ""]
    out += histogram(raw)
    out += ["", f"Normalized `len(normalize(text))`: {summary(norm)}", ""]
    out += histogram(norm)
    out.append("")

    ec = Counter(r["emoji"] for r in rows)
    ranked = ec.most_common()
    out += ["## Emoji distribution", "", f"{len(ec)} distinct emojis.", ""]
    out += ["### Top 10", "", "| emoji | count |", "| --- | ---: |"]
    for e, n in ranked[:10]:
        out.append(f"| {e} | {n} |")
    out += ["", "### Bottom 10", "", "| emoji | count |", "| --- | ---: |"]
    for e, n in reversed(ranked[-10:]):
        out.append(f"| {e} | {n} |")
    out.append("")

    return "\n".join(out)


def main():
    rows = read(EVAL_PATH)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    dest = REPORT_DIR / f"{datetime.now().strftime('%m-%d-%H:%M')}.md"
    dest.write_text(build_report(rows), encoding="utf-8")
    print(f"wrote {dest.relative_to(ROOT)} ({len(rows)} samples)")


if __name__ == "__main__":
    main()
