"""For every emoji without a Noto animation, the single closest emoji that has an original Noto animation smaller than
MAX_EXAMPLE_BYTES (the median original), by cosine similarity of the EmojiHead embeddings. Emojis whose best match
is below min_sim get no entry, and the animate prompt then has no example. Feeds tools/animate/example.ts."""

import csv
import json

import torch
import torch.nn.functional as F
import typer

from files import ANIMATION_CSV, ANIMATION_NEIGHBORS_JSON, EMOJI_PT, NOTO_LOTTIE_INDEX_JSON
from model.config import EMOJIS
from model.model import EmojiHead
from model.runmeta import load_pt

_app = typer.Typer(add_completion=False)

MAX_EXAMPLE_BYTES = 60_700  # median size of the original Noto animations; keep in sync with tools/animate/example.ts


def bare(e: str) -> str:
    return e.replace("️", "")


@_app.command()
def main(min_sim: float = 0.3) -> None:
    sd, _ = load_pt(EMOJI_PT)
    head = EmojiHead()
    head.load_state_dict(sd)
    emb = F.normalize(head.embed.weight.detach(), dim=1)
    pos = {bare(e): i for i, e in enumerate(EMOJIS)}

    # animation.csv marks the emojis Noto animates; the rest had none, and the index holds Noto's own plus ours.
    no_anim = [r["emoji"] for r in csv.DictReader(open(ANIMATION_CSV)) if not r["noto"]]
    no_anim_bare = {bare(e) for e in no_anim}
    index: dict[str, str] = json.load(open(NOTO_LOTTIE_INDEX_JSON))

    def small(e: str) -> bool:
        return (NOTO_LOTTIE_INDEX_JSON.parent / f"{index[e]}.json").stat().st_size < MAX_EXAMPLE_BYTES

    original = [e for e in index if bare(e) not in no_anim_bare and bare(e) in pos and small(e)]
    rows = torch.tensor([pos[bare(e)] for e in original])

    out: dict[str, str] = {}
    for e in no_anim:
        if bare(e) not in pos:
            continue
        sim = emb[rows] @ emb[pos[bare(e)]]
        best = int(sim.argmax())
        if sim[best] >= min_sim:  # a weak match (e.g. 🦺 -> ⭐) misleads more than it helps
            out[e] = original[best]
    ANIMATION_NEIGHBORS_JSON.write_text(json.dumps(out, ensure_ascii=False, indent=0) + "\n")
    print(f"{len(out)}/{len(no_anim)} emojis (similarity >= {min_sim}) -> nearest of {len(original)} small originally animated, wrote {ANIMATION_NEIGHBORS_JSON}")


if __name__ == "__main__":
    _app()
