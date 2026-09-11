import json
import math
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import torch
import typer

from files import CLDR_JSONL, KWPROJ_JSON
from model.config import EMOJIS
from model.data import normalize as norm_text
from model.data import text_to_tensor
from model.kwtokens import query_tokens
from model.model import EmojiEmbedding, EmojiHead, FusionHead, TextEncoder
from model.runmeta import load_pt

CLDR_MIN_KEYWORD_LEN = 3
KW_PRIMARY_BONUS = 0.15


def _load(mod, path):
    sd, _ = load_pt(path)
    mod.load_state_dict(sd)
    mod.eval()
    return mod


def _cldr_keywords() -> dict:
    words: dict = {}
    with open(CLDR_JSONL, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            word = str(d.get("text", ""))
            if len(word) < CLDR_MIN_KEYWORD_LEN or not any(c.isalpha() for c in word):
                continue
            targets = words.setdefault(word, [])
            for e in str(d.get("emojis", "")).split():
                if e not in targets:
                    targets.append(e)
    return words


def _kw_proj() -> dict:
    return json.loads(Path(KWPROJ_JSON).read_text(encoding="utf-8")).get("proj", {})


def _kw_exact_dense(text: str, proj: dict) -> torch.Tensor:
    vec = torch.zeros(len(EMOJIS))
    for word in query_tokens(text):
        pl = proj.get(word)
        if not pl:
            continue
        base = 1.0 / math.log2(1 + len(pl))
        for j, e in enumerate(pl):
            v = base + (KW_PRIMARY_BONUS if j == 0 else 0.0)
            if v > vec[e]:
                vec[e] = v
    return vec


def _rank(scores: torch.Tensor, ids: list[int]) -> int:
    order = scores.argsort(dim=-1, descending=True).tolist()
    return min(order.index(j) + 1 for j in ids)


def main(pt: Path = typer.Option(Path("pt"), "--pt"), top: int = 30) -> None:
    words = _cldr_keywords()
    proj = _kw_proj()
    if not words:
        print(f"no rows in {CLDR_JSONL}")
        raise typer.Exit(1)
    if not proj:
        print(f"no {KWPROJ_JSON} (run bun run regen)")
        raise typer.Exit(1)

    vocab = {e: i for i, e in enumerate(EMOJIS)}
    rows = [(w, [vocab[e] for e in exp if e in vocab]) for w, exp in words.items()]
    rows = [(w, ids) for w, ids in rows if ids]
    if not rows:
        print("no cldr rows target an in-vocab emoji")
        raise typer.Exit(1)

    enc = _load(TextEncoder(), pt / "enc.pt")
    emoji_head = _load(EmojiHead(), pt / "emoji.pt")
    emb = _load(EmojiEmbedding(), pt / "emoji_embed.pt")
    fusion = _load(FusionHead(), pt / "fusion.pt")

    with torch.no_grad():
        texts = torch.stack([text_to_tensor(norm_text(w)) for w, _ in rows])
        text_emb = enc(texts)
        logit_m = emb.score(emoji_head(text_emb))
        kw_dense = torch.stack([_kw_exact_dense(w, proj) for w, _ in rows])
        fused = fusion(text_emb.detach(), logit_m.detach(), kw_dense)

    regressions = []
    improvements = []
    ties = 0
    for i, (word, ids) in enumerate(rows):
        kw_rank = _rank(kw_dense[i], ids)
        fusion_rank = _rank(fused[i], ids)
        kw_strength = kw_dense[i].max().item()
        row = (word, ids, kw_rank, fusion_rank, kw_strength)
        if fusion_rank > kw_rank:
            regressions.append(row)
        elif fusion_rank < kw_rank:
            improvements.append(row)
        else:
            ties += 1

    n = len(rows)
    print(
        f"rows: {n}  fusion worse: {len(regressions)}  "
        f"fusion better: {len(improvements)}  tied: {ties}"
    )

    kw_acc1 = sum(1 for i, (_, ids) in enumerate(rows) if _rank(kw_dense[i], ids) == 1)
    fusion_acc1 = sum(1 for i, (_, ids) in enumerate(rows) if _rank(fused[i], ids) == 1)
    print(f"acc@1  kw-exact: {kw_acc1 / n:.4f}  fusion: {fusion_acc1 / n:.4f}")

    emoji_counts: Counter = Counter()
    for _word, ids, _kw_rank, _fusion_rank, _kw_strength in regressions:
        for i in ids:
            emoji_counts[EMOJIS[i]] += 1
    print()
    print("top target emojis among fusion's regressed rows:")
    for e, c in emoji_counts.most_common(15):
        print(f"  {e}  {c}")

    strengths = sorted(r[4] for r in regressions)
    if strengths:
        print()
        print(
            f"regressed rows' kw-strength: min={strengths[0]:.3f} "
            f"median={strengths[len(strengths) // 2]:.3f} max={strengths[-1]:.3f}"
        )

    print()
    print(f"worst regressions (kw rank -> fusion rank), top {top}:")
    regressions.sort(key=lambda r: r[3] - r[2], reverse=True)
    for word, ids, kw_rank, fusion_rank, kw_strength in regressions[:top]:
        tgt = "/".join(EMOJIS[i] for i in ids)
        print(
            f"  {word!r:20} target={tgt}  kw_rank={kw_rank}  "
            f"fusion_rank={fusion_rank}  kw_strength={kw_strength:.3f}"
        )


if __name__ == "__main__":
    typer.run(main)
