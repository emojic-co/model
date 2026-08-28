"""Scan data.jsonl for texts that exceed MAX_TEXT_LEN after normalize().

Default: report counts. With --prune: drop those rows from data.jsonl in place.
"""

import json
import sys

from config import MAX_TEXT_LEN
from data import normalize


def scan(path: str = "data.jsonl", prune: bool = False):
    kept = []
    dropped = 0
    total = 0
    worst = 0

    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            total += 1
            n = len(normalize(json.loads(line)["text"]))
            worst = max(worst, n)
            if n > MAX_TEXT_LEN:
                dropped += 1
            else:
                kept.append(line)

    print(f"MAX_TEXT_LEN             = {MAX_TEXT_LEN}")
    print(f"samples                  = {total}")
    print(f"normalized > MAX_TEXT_LEN = {dropped} ({dropped / total:.1%})")
    print(f"longest normalized text   = {worst} chars")

    if prune:
        with open(path, "w", encoding="utf-8") as f:
            f.writelines(kept)
        print(f"pruned -> {len(kept)} rows remain in {path}")


if __name__ == "__main__":
    scan(prune="--prune" in sys.argv)
