import json
import random
import sys
from pathlib import Path

import typer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from files import COLORS_JSONL
from model.config import MAX_TEXT_LEN, SEED
from model.data import normalize

CARD_COLORS = ("red", "green", "blue", "dark", "bright")
PER_COLOR = 300

_app = typer.Typer(
    add_completion=False,
    context_settings={"help_option_names": ["-h", "--help"]},
)


@_app.command()
def main(archive: str, per_color: int = PER_COLOR, out: str = COLORS_JSONL) -> None:
    src = Path(archive)
    pools: dict[str, list] = {c: [] for c in CARD_COLORS}
    seen: dict[str, set] = {c: set() for c in CARD_COLORS}
    for line in src.open(encoding="utf-8"):
        r = json.loads(line)
        c = r.get("color")
        if c not in pools:
            continue
        if not (isinstance(r.get("bg"), list) and len(r["bg"]) == 2 and r.get("fg")):
            continue
        nt = normalize(r["text"])
        if not nt or len(nt) > MAX_TEXT_LEN or nt in seen[c]:
            continue
        seen[c].add(nt)
        pools[c].append(
            {
                "text": r["text"],
                "emojis": r["emojis"],
                "styles": r["styles"],
                "bg": r["bg"],
                "fg": r["fg"],
                "color": c,
            }
        )

    rng = random.Random(SEED)
    rows = []
    for c in CARD_COLORS:
        pool = sorted(pools[c], key=lambda r: normalize(r["text"]))
        pick = sorted(
            rng.sample(pool, min(per_color, len(pool))),
            key=lambda r: normalize(r["text"]),
        )
        rows.extend(pick)
        print(f"{c}: pool {len(pool)} -> kept {len(pick)}")

    dst = Path(out)
    with dst.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"wrote {len(rows)} rows to {dst}")


if __name__ == "__main__":
    _app()
