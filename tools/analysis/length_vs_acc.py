import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import torch
import typer

from files import EVAL_JSONL
from model.config import ENCODER_DILATION, ENCODER_KERNEL_SIZE, MAX_TEXT_LEN
from model.data import EMOJIS, read, text_to_tensor
from model.model import EmojiEmbedding, EmojiHead, TextEncoder
from model.runmeta import load_pt

EMOJI_KS = [1, 5, 10]


def _load(mod: torch.nn.Module, path: str) -> torch.nn.Module:
    sd, _ = load_pt(path)
    mod.load_state_dict(sd)
    mod.eval()
    return mod


def _acc_at_k(logits: torch.Tensor, target: torch.Tensor, k: int) -> torch.Tensor:
    k = min(k, logits.size(-1))
    top = logits.topk(k, dim=-1).indices
    return target.gather(1, top).amax(dim=-1)


def receptive_field() -> int:
    return 1 + (ENCODER_KERNEL_SIZE - 1) * sum(ENCODER_DILATION)


def _bucket(length: int, rf: int) -> str:
    lo = rf // 2
    if length <= lo:
        return f"1-{lo}"
    if length <= rf:
        return f"{lo + 1}-{rf}"
    return f"{rf + 1}-{MAX_TEXT_LEN}"


def main(pt: Path = typer.Option(Path("pt"), "--pt")) -> None:
    enc = _load(TextEncoder(), str(pt / "enc.pt"))
    head = _load(EmojiHead(), str(pt / "emoji.pt"))
    embed = _load(EmojiEmbedding(), str(pt / "emoji_embed.pt"))

    rf = receptive_field()
    vocab = {e: i for i, e in enumerate(EMOJIS)}
    rows = [r for r in read(EVAL_JSONL) if r.emojis]

    buckets: dict[str, list[int]] = {}
    for i, r in enumerate(rows):
        buckets.setdefault(_bucket(len(r.text), rf), []).append(i)

    texts = torch.stack([text_to_tensor(r.text) for r in rows])
    tgt = torch.zeros(len(rows), len(EMOJIS))
    for i, r in enumerate(rows):
        for e in r.emojis:
            tgt[i, vocab[e]] = 1.0

    with torch.no_grad():
        logits = embed.score(head(enc(texts)))

    print(f"RECEPTIVE_FIELD={rf}  MAX_TEXT_LEN={MAX_TEXT_LEN}  eval rows={len(rows)}")
    print(f"{'bucket':<10} {'n':>5} " + " ".join(f"acc@{k:<3}" for k in EMOJI_KS))
    for name in sorted(buckets, key=lambda b: int(b.split("-")[0])):
        idx = torch.tensor(buckets[name])
        blog, btgt = logits[idx], tgt[idx]
        accs = [_acc_at_k(blog, btgt, k).mean().item() for k in EMOJI_KS]
        print(f"{name:<10} {len(idx):>5} " + " ".join(f"{a:.4f}" for a in accs))


_app = typer.Typer(
    add_completion=False,
    context_settings={"help_option_names": ["-h", "--help"]},
)


@_app.command()
def cli(pt: Path = typer.Option(Path("pt"), "--pt")) -> None:
    """Bucket data/eval.jsonl by text length and report EmojiHead Acc@k per bucket."""
    main(pt)


if __name__ == "__main__":
    _app()
