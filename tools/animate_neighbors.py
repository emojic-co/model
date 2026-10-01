"""For every emoji without a Noto animation, the closest emojis that DO have an original Noto animation, by cosine
similarity of the EmojiHead embeddings. Feeds the example in the animate prompt (tools/animate/example.ts)."""

import csv
import json

import torch
import torch.nn.functional as F
import typer

from files import ANIMATION_NEIGHBORS_JSON, EMOJI_ANIMATION_CSV, EMOJI_PT, NOTO_LOTTIE_INDEX_JSON
from model.config import EMOJIS
from model.model import EmojiHead
from model.runmeta import load_pt

_app = typer.Typer(add_completion=False)


def bare(e: str) -> str:
    return e.replace("️", "")


@_app.command()
def main(k: int = 3) -> None:
    sd, _ = load_pt(EMOJI_PT)
    head = EmojiHead()
    head.load_state_dict(sd)
    emb = F.normalize(head.embed.weight.detach(), dim=1)
    pos = {bare(e): i for i, e in enumerate(EMOJIS)}

    # emoji_animation.csv lists the emojis that had no Noto animation; the index holds Noto's own plus ours.
    no_anim = [r["emoji"] for r in csv.DictReader(open(EMOJI_ANIMATION_CSV))]
    no_anim_bare = {bare(e) for e in no_anim}
    original = [e for e in json.load(open(NOTO_LOTTIE_INDEX_JSON)) if bare(e) not in no_anim_bare and bare(e) in pos]
    rows = torch.tensor([pos[bare(e)] for e in original])

    out: dict[str, list[str]] = {}
    for e in no_anim:
        if bare(e) not in pos:
            continue
        sim = emb[rows] @ emb[pos[bare(e)]]
        out[e] = [original[i] for i in sim.topk(k).indices.tolist()]
    ANIMATION_NEIGHBORS_JSON.write_text(json.dumps(out, ensure_ascii=False, indent=0) + "\n")
    print(f"{len(out)}/{len(no_anim)} emojis -> {k} nearest of {len(original)} originally animated, wrote {ANIMATION_NEIGHBORS_JSON}")


if __name__ == "__main__":
    _app()
