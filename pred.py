import json
import sys
from pathlib import Path

import torch
import typer

from config import MAX_TEXT_LEN, SEED
from data import EMOJIS, STYLES, normalize, text_to_tensor
from model import ColorGen, EmojiHead, StyleHead, TextEncoder
from runmeta import load_pt


def rgb_to_hex(rgb: torch.Tensor) -> list[str]:
    assert rgb.shape == (9,), "Input tensor must be of shape (9,)"

    ints = (rgb + 127.5).clamp(0, 255).to(torch.int32).cpu().tolist()

    def f2h(val: int) -> str:
        return f"{val:02x}"

    return [f"#{f2h(ints[i])}{f2h(ints[i + 1])}{f2h(ints[i + 2])}" for i in range(0, 9, 3)]


def _load(mod: torch.nn.Module, path: str) -> torch.nn.Module:
    sd, meta = load_pt(path)
    mod.load_state_dict(sd)
    mod._pt_meta = meta  # type: ignore
    mod.eval()
    return mod


def top_labels(
    logits: torch.Tensor,
    names: list[str],
    thresh: float = 0.5,
    min_k: int = 1,
    max_k: int = 6,
) -> list[str]:
    probs = logits.sigmoid().squeeze(0)
    order = probs.argsort(descending=True).tolist()
    picked = [i for i in order if probs[i] >= thresh][:max_k]
    if len(picked) < min_k:
        picked = order[:min_k]
    return [names[i] for i in picked]


def read_texts(lines: list[str]) -> list[str]:
    texts = []
    for line in lines:
        text = normalize(line)[:MAX_TEXT_LEN]
        if text:
            texts.append(text)
    return texts


def predict(
    texts: list[str],
    enc_path: str = "enc.pt",
    gen_path: str = "gen.pt",
    style_path: str = "style.pt",
    emoji_path: str = "emoji.pt",
) -> list[dict]:
    torch.manual_seed(SEED)

    enc = _load(TextEncoder(), enc_path)
    gen = _load(ColorGen(), gen_path)
    style = _load(StyleHead(), style_path)
    emoji = _load(EmojiHead(), emoji_path)

    records = []
    with torch.no_grad():
        for text in texts:
            text_tensor = text_to_tensor(text).unsqueeze(0)
            emb = enc(text_tensor)

            styles = top_labels(style(emb), STYLES, min_k=1, max_k=3)
            emojis = top_labels(emoji(emb), EMOJIS, min_k=1, max_k=1)

            colors = gen(emb).squeeze(0)
            hexes = rgb_to_hex(colors)

            records.append(
                {
                    "text": text,
                    "emojis": " ".join(emojis),
                    "styles": styles,
                    "bg": hexes[:2],
                    "fg": hexes[2],
                }
            )
    return records


_app = typer.Typer(
    add_completion=False,
    context_settings={"help_option_names": ["-h", "--help"]},
)


@_app.command()
def main(
    file: Path | None = typer.Argument(
        None, help="Read texts from this file, one per line (defaults to stdin)."
    ),
    output: Path | None = typer.Option(
        None, "-o", "--output", help="Write predictions here instead of stdout."
    ),
) -> None:
    """Run the inference graph over texts (one per line) from a file or stdin."""
    if file:
        lines = file.read_text(encoding="utf-8").splitlines()
    else:
        lines = sys.stdin.read().splitlines()
    records = predict(read_texts(lines))
    out_lines = [json.dumps(rec, ensure_ascii=False) for rec in records]

    if output:
        output.write_text("".join(line + "\n" for line in out_lines), encoding="utf-8")
    else:
        for line in out_lines:
            print(line)


if __name__ == "__main__":
    _app()
