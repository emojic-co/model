import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import typer

from model.config import MAX_TEXT_LEN, SEED
from model.data import EMOJIS, STYLES, normalize, text_to_tensor
from model.model import ColorGen, EmojiHead, StyleHead, TextEncoder
from model.runmeta import load_pt


def rgb_to_hex(rgb: torch.Tensor) -> list[str]:
    assert rgb.shape == (9,), "Input tensor must be of shape (9,)"

    ints = (rgb + 127.5).clamp(0, 255).to(torch.int32).cpu().tolist()

    def f2h(val: int) -> str:
        return f"{val:02x}"

    return [f"#{f2h(ints[i])}{f2h(ints[i + 1])}{f2h(ints[i + 2])}" for i in range(0, 9, 3)]


def _load(mod: torch.nn.Module, path: str | Path) -> torch.nn.Module:
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
        line = line.strip()
        if not line:
            continue
        text = normalize(json.loads(line)["text"])[:MAX_TEXT_LEN]
        if text:
            texts.append(text)
    return texts


def predict(
    texts: list[str],
    pt_dir: Path,
) -> list[dict]:
    torch.manual_seed(SEED)

    enc = _load(TextEncoder(), pt_dir / "enc.pt")
    gen = _load(ColorGen(), pt_dir / "gen.pt")
    style = _load(StyleHead(), pt_dir / "style.pt")
    emoji = _load(EmojiHead(), pt_dir / "emoji.pt")

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
    file: Path = typer.Argument(
        ...,
        help="Read data/data.jsonl-schema rows from this file, one per line.",
    ),
    pt: Path = typer.Option(
        ..., "--pt", help="Folder containing enc.pt/style.pt/emoji.pt/gen.pt."
    ),
    output: Path | None = typer.Option(
        None, "-o", "--output", help="Write predictions here instead of stdout."
    ),
) -> None:
    """Run the inference graph over the `text` field of jsonl rows from a file."""
    lines = file.read_text(encoding="utf-8").splitlines()
    records = predict(read_texts(lines), pt)
    out_lines = [json.dumps(rec, ensure_ascii=False) for rec in records]

    if output:
        output.write_text("".join(line + "\n" for line in out_lines), encoding="utf-8")
    else:
        for line in out_lines:
            print(line)


if __name__ == "__main__":
    _app()
