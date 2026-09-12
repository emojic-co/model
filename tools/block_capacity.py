import re
import sys
from itertools import accumulate
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import typer
from torch import nn

from files import EMOJI_PT, ENC_PT, EVAL_JSONL, KEYWORDS_JSONL, TERMS_JSONL
from model.data import PAD_IDX, VOCAB_SIZE, read, text_to_tensor
from model.model import TextEncoderBlock
from model.runmeta import load_pt

SOURCES = [
    ("keywords", KEYWORDS_JSONL),
    ("terms", TERMS_JSONL),
    ("eval", EVAL_JSONL),
]

_LIST_RE = re.compile(r"\[[^\]]*\]")


def _to_ints(s: str) -> list[int]:
    return [int(x) for x in s.strip("[]").split(",")]


def _parse_encoder_config(meta: dict) -> tuple[int, list[int], list[int]]:
    line = next(c for c in meta["config"] if c.startswith("ENCODER:"))
    rest = line.split("ENCODER:")[1]
    channels_str, dilation_str = _LIST_RE.findall(rest)
    char_embed_size = int(_LIST_RE.sub("", rest).split()[0])
    return char_embed_size, _to_ints(channels_str), _to_ints(dilation_str)


class _Encoder(nn.Module):
    def __init__(self, char_embed_size: int, channels: list[int], dilation: list[int]):
        super().__init__()
        self.char_embed = nn.Embedding(VOCAB_SIZE, char_embed_size, padding_idx=PAD_IDX)
        io = zip([char_embed_size, *channels[:-1]], channels, dilation, strict=True)
        self.blocks = nn.ModuleList(
            [TextEncoderBlock(i=i, o=o, dilation=d) for i, o, d in io])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.char_embed(x).transpose(1, 2)
        n = (x != PAD_IDX).sum(dim=1, keepdim=True)
        pooled = []
        for block in self.blocks:
            out = block(out)
            keep = torch.arange(out.shape[-1], device=x.device) < n
            masked = out.masked_fill(~keep.unsqueeze(1), float("-inf"))
            pooled.append(torch.max(masked, dim=-1).values)
        return torch.cat(pooled, dim=-1)


def block_bounds(channels: list[int]) -> list[tuple[int, int]]:
    ends = list(accumulate(channels))
    starts = [0, *ends[:-1]]
    return list(zip(starts, ends, strict=True))


def main() -> None:
    enc_sd, enc_meta = load_pt(ENC_PT)
    emoji_sd, _ = load_pt(EMOJI_PT)

    char_embed_size, channels, dilation = _parse_encoder_config(enc_meta)
    enc = _Encoder(char_embed_size, channels, dilation)
    enc.load_state_dict(enc_sd)
    enc.eval()

    weight = emoji_sd["net.1.weight"]
    bounds = block_bounds(channels)

    rows: list[tuple] = []
    with torch.no_grad():
        for source, path in SOURCES:
            rec = next(iter(read(path)))
            emb = enc(text_to_tensor(rec.text).unsqueeze(0)).squeeze(0)
            q = weight @ emb
            q_norm = q.norm().item()

            for i, ((start, end), d) in enumerate(zip(bounds, dilation, strict=True)):
                w_slice = weight[:, start:end]
                a_slice = emb[start:end]
                col_norms = w_slice.norm(dim=0)
                contrib_norm = (w_slice @ a_slice).norm().item()

                rows.append((
                    source,
                    rec.text,
                    i,
                    f"{start}:{end}",
                    d,
                    col_norms.mean().item(),
                    col_norms.max().item(),
                    a_slice.norm().item(),
                    contrib_norm,
                    100 * contrib_norm / q_norm if q_norm else 0.0,
                ))

    header = (
        f"{'source':<9} {'text':<24} {'blk':>3} {'range':>9} {'dil':>3} "
        f"{'w_norm_mean':>11} {'w_norm_max':>10} {'act_norm':>9} "
        f"{'contrib_norm':>12} {'contrib_%':>9}"
    )
    print(header)
    print("-" * len(header))
    for (
        source, text, i, rng, d,
        w_mean, w_max, a_norm, contrib_norm, contrib_pct,
    ) in rows:
        print(
            f"{source:<9} {text[:24]:<24} {i:>3} {rng:>9} {d:>3} "
            f"{w_mean:>11.4f} {w_max:>10.4f} {a_norm:>9.4f} "
            f"{contrib_norm:>12.4f} {contrib_pct:>8.1f}%"
        )


_app = typer.Typer(
    add_completion=False,
    context_settings={"help_option_names": ["-h", "--help"]},
)


@_app.command()
def cli() -> None:
    """Print per-block EmojiHead weight column-norms and per-sample contribution."""
    main()


if __name__ == "__main__":
    _app()
