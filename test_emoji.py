import json
import sys
from datetime import datetime
from pathlib import Path

import torch

from config import EMOJIS
from data import normalize, text_to_tensor
from model import EmojiHead, TextEncoder

WORDS_PATH = Path("words.json")
REPORT_DIR = Path("report/test-emoji")
TOP_K = (1, 3, 5, 10)
SHOW_TOP = 5


def _load(mod: torch.nn.Module, path: str) -> torch.nn.Module:
    mod.load_state_dict(torch.load(path, map_location="cpu"))
    mod.eval()
    return mod


def _evaluate(enc_path: str, emoji_path: str, words: list[dict]) -> list[dict]:
    enc = _load(TextEncoder(), enc_path)
    head = _load(EmojiHead(), emoji_path)
    vocab = {e: i for i, e in enumerate(EMOJIS)}

    out = []
    with torch.no_grad():
        for entry in words:
            word = entry["word"]
            targets = [vocab[e] for e in entry["emojis"] if e in vocab]
            emb = enc(text_to_tensor(normalize(word)).unsqueeze(0))
            order = head(emb).squeeze(0).argsort(descending=True).tolist()
            ranks = [order.index(t) + 1 for t in targets]
            out.append(
                {
                    "word": word,
                    "expected": entry["emojis"],
                    "unknown": [e for e in entry["emojis"] if e not in vocab],
                    "rank": min(ranks) if ranks else None,
                    "top": [EMOJIS[i] for i in order[:SHOW_TOP]],
                }
            )
    return out


def _metrics(results: list[dict]) -> tuple[dict, float, int]:
    scored = [r for r in results if r["rank"] is not None]
    n = len(scored) or 1
    acc = {k: sum(r["rank"] <= k for r in scored) / n for k in TOP_K}
    mrr = sum(1.0 / r["rank"] for r in scored) / n
    return acc, mrr, len(scored)


def _render(
    results: list[dict],
    acc: dict,
    mrr: float,
    scored_n: int,
    enc_path: str,
    emoji_path: str,
    stamp: str,
) -> str:
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
    lines += [
        "",
        "| word | expected | rank | hit@10 | top 5 |",
        "| --- | --- | --- | --- | --- |",
    ]
    for r in sorted(results, key=lambda x: (x["rank"] is None, x["rank"] or 0, x["word"])):
        rank = "-" if r["rank"] is None else str(r["rank"])
        hit = "-" if r["rank"] is None else ("y" if r["rank"] <= 10 else "n")
        lines.append(
            f"| {r['word']} | {' '.join(r['expected'])} | {rank} | {hit} "
            f"| {' '.join(r['top'])} |"
        )
    return "\n".join(lines) + "\n"


def test_emoji(
    enc_path: str = "enc.pt",
    emoji_path: str = "emoji.pt",
    words_path: str | Path = WORDS_PATH,
    write_report: bool = True,
) -> dict:
    words = json.loads(Path(words_path).read_text(encoding="utf-8"))
    results = _evaluate(enc_path, emoji_path, words)
    acc, mrr, scored_n = _metrics(results)
    stamp = datetime.now().strftime("%m-%d-%H:%M")

    if write_report:
        REPORT_DIR.mkdir(parents=True, exist_ok=True)
        path = REPORT_DIR / f"{stamp}.md"
        path.write_text(
            _render(results, acc, mrr, scored_n, enc_path, emoji_path, stamp),
            encoding="utf-8",
        )
        print(f"wrote {path}")

    summary = "  ".join(f"acc@{k}={acc[k]:.0%}" for k in TOP_K)
    print(f"emoji test  {summary}  mrr={mrr:.3f}  (n={scored_n})")
    return {"acc": acc, "mrr": mrr, "n": scored_n}


if __name__ == "__main__":
    sys.exit(0 if test_emoji()["n"] else 1)
