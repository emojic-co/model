import json
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

import torch

from config import EMOJIS
from data import TRAIN_PATH, normalize, read, text_to_tensor
from model import EmojiHead, TextEncoder

WORDS_PATH = Path("words.json")
REPORT_DIR = Path("report/test-emoji")
TOP_K = (1, 3, 5, 10)
SHOW_TOP = 5
FREQ_BUCKETS = 4


def _load(mod: torch.nn.Module, path: str) -> torch.nn.Module:
    mod.load_state_dict(torch.load(path, map_location="cpu"))
    mod.eval()
    return mod


def _emoji_counts() -> Counter:
    counts: Counter = Counter()
    for rec in read(TRAIN_PATH):
        for e in set(rec.emojis):
            counts[e] += 1
    return counts


def _evaluate(
    enc_path: str, emoji_path: str, words: list[dict], counts: Counter
) -> list[dict]:
    enc = _load(TextEncoder(), enc_path)
    head = _load(EmojiHead(), emoji_path)
    vocab = {e: i for i, e in enumerate(EMOJIS)}

    out = []
    with torch.no_grad():
        for entry in words:
            word = entry["word"]
            targets = [vocab[e] for e in entry["emojis"] if e in vocab]
            freqs = [counts.get(e, 0) for e in entry["emojis"] if e in vocab]
            emb = enc(text_to_tensor(normalize(word)).unsqueeze(0))
            order = head(emb).squeeze(0).argsort(descending=True).tolist()
            ranks = [order.index(t) + 1 for t in targets]
            out.append(
                {
                    "word": word,
                    "expected": entry["emojis"],
                    "length": len(word),
                    "freq_sum": sum(freqs),
                    "freq_avg": sum(freqs) / len(freqs) if freqs else 0,
                    "rank": min(ranks) if ranks else None,
                    "top": [EMOJIS[i] for i in order[:SHOW_TOP]],
                }
            )
    return out


def _acc(rows: list[dict]) -> tuple[int, dict, float]:
    scored = [r for r in rows if r["rank"] is not None]
    n = len(scored) or 1
    acc = {k: sum(r["rank"] <= k for r in scored) / n for k in TOP_K}
    mrr = sum(1.0 / r["rank"] for r in scored) / n
    return len(scored), acc, mrr


def _length_groups(results: list[dict]) -> list[tuple[str, list[dict]]]:
    by_len: dict[int, list[dict]] = {}
    for r in results:
        by_len.setdefault(r["length"], []).append(r)
    return [(str(k), by_len[k]) for k in sorted(by_len)]


def _freq_groups(results: list[dict]) -> list[tuple[str, list[dict]]]:
    ordered = sorted(results, key=lambda r: r["freq_avg"])
    n = len(ordered)
    groups = []
    for b in range(FREQ_BUCKETS):
        chunk = ordered[b * n // FREQ_BUCKETS : (b + 1) * n // FREQ_BUCKETS]
        if not chunk:
            continue
        lo = min(r["freq_avg"] for r in chunk)
        hi = max(r["freq_avg"] for r in chunk)
        groups.append((f"{lo:.0f}-{hi:.0f}", chunk))
    return groups


def _group_table(header: str, groups: list[tuple[str, list[dict]]]) -> list[str]:
    lines = [
        "",
        f"## {header}",
        "",
        "| group | n | acc@1 | acc@5 | acc@10 | mrr |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for label, rows in groups:
        scored_n, acc, mrr = _acc(rows)
        lines.append(
            f"| {label} | {scored_n} | {acc[1]:.0%} | {acc[5]:.0%} "
            f"| {acc[10]:.0%} | {mrr:.2f} |"
        )
    return lines


def _render(
    results: list[dict], enc_path: str, emoji_path: str, stamp: str
) -> str:
    scored_n, acc, mrr = _acc(results)
    lines = [
        f"# Emoji test - {stamp}",
        "",
        f"- model: `{enc_path}` + `{emoji_path}`",
        f"- words: {len(results)} ({scored_n} scored)",
        f"- MRR: {mrr:.3f}",
        "",
        "| metric | value |",
        "| --- | --- |",
    ]
    lines += [f"| acc@{k} | {acc[k]:.1%} |" for k in TOP_K]

    lines += _group_table("By word length (chars)", _length_groups(results))
    lines += _group_table(
        "By expected-emoji frequency in train.jsonl (avg)", _freq_groups(results)
    )

    lines += [
        "",
        "## Per word (grouped by length)",
        "",
        "| word | len | expected | freq | rank | hit@10 | top 5 |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for r in sorted(
        results, key=lambda x: (x["length"], x["rank"] is None, x["rank"] or 0, x["word"])
    ):
        rank = "-" if r["rank"] is None else str(r["rank"])
        hit = "-" if r["rank"] is None else ("y" if r["rank"] <= 10 else "n")
        lines.append(
            f"| {r['word']} | {r['length']} | {' '.join(r['expected'])} "
            f"| {r['freq_sum']} | {rank} | {hit} | {' '.join(r['top'])} |"
        )
    return "\n".join(lines) + "\n"


def test_emoji(
    enc_path: str = "enc.pt",
    emoji_path: str = "emoji.pt",
    words_path: str | Path = WORDS_PATH,
    write_report: bool = True,
) -> dict:
    words = json.loads(Path(words_path).read_text(encoding="utf-8"))
    results = _evaluate(enc_path, emoji_path, words, _emoji_counts())
    scored_n, acc, mrr = _acc(results)
    stamp = datetime.now().strftime("%m-%d-%H:%M")

    if write_report:
        REPORT_DIR.mkdir(parents=True, exist_ok=True)
        path = REPORT_DIR / f"{stamp}.md"
        path.write_text(_render(results, enc_path, emoji_path, stamp), encoding="utf-8")
        json_path = REPORT_DIR / f"{stamp}.json"
        json_path.write_text(
            json.dumps(
                {
                    "stamp": stamp,
                    "enc": enc_path,
                    "emoji": emoji_path,
                    "summary": {"acc": acc, "mrr": mrr, "n": scored_n},
                    "words": results,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"wrote {path}")
        print(f"wrote {json_path}")

    summary = "  ".join(f"acc@{k}={acc[k]:.0%}" for k in TOP_K)
    print(f"emoji test  {summary}  mrr={mrr:.3f}  (n={scored_n})")
    return {"acc": acc, "mrr": mrr, "n": scored_n}


if __name__ == "__main__":
    sys.exit(0 if test_emoji()["n"] else 1)
