import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import typer
from tqdm import tqdm

from files import FLEX_JSON
from model.config import MAX_TEXT_LEN, SEED
from model.data import EMOJIS, FLEX_N, STYLES, normalize, text_to_tensor
from model.flexrank import FlexRanker
from model.model import (
    ColorGen,
    EmojiEmbedding,
    EmojiHead,
    FusionHead,
    KWHead,
    StyleHead,
    TextEncoder,
)
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
    emoji_embed = _load(EmojiEmbedding(), pt_dir / "emoji_embed.pt")
    emoji = _load(EmojiHead(), pt_dir / "emoji.pt")

    fusion = kw = ranker = None
    if (
        (pt_dir / "fusion.pt").exists()
        and (pt_dir / "kw.pt").exists()
        and Path(FLEX_JSON).exists()
    ):
        fusion = _load(FusionHead(), pt_dir / "fusion.pt")
        kw = _load(KWHead(), pt_dir / "kw.pt")
        ranker = FlexRanker(FLEX_JSON)
    else:
        print(
            "kw.pt / fusion.pt / flex.json missing -- skipping fusion_top_labels",
            file=sys.stderr,
        )

    records = []
    with torch.no_grad():
        for text in tqdm(texts, desc="predicting"):
            text_tensor = text_to_tensor(text).unsqueeze(0)
            emb = enc(text_tensor)

            styles = top_labels(style(emb), STYLES, min_k=1, max_k=3)
            q_txt = emoji(emb)
            emoji_logits = emoji_embed.score(q_txt)
            emojis = top_labels(emoji_logits, EMOJIS, min_k=1, max_k=1)

            colors = gen(emb).squeeze(0)
            hexes = rgb_to_hex(colors)

            record = {
                "text": text,
                "emojis": " ".join(emojis),
                "styles": styles,
                "bg": hexes[:2],
                "fg": hexes[2],
            }
            if fusion is not None:
                tf = torch.zeros(1, FLEX_N)
                for i, v in enumerate(ranker.tf_vec(text)):
                    tf[0, i] = v
                q_kw = kw(tf)
                a = fusion(emb, tf).unsqueeze(-1)
                fusion_logits = emoji_embed.score(a * q_txt + (1 - a) * q_kw)
                record["fusion_top_labels"] = top_labels(
                    fusion_logits, EMOJIS, min_k=1, max_k=1
                )
            records.append(record)
    return records


_app = typer.Typer(
    add_completion=False,
    context_settings={"help_option_names": ["-h", "--help"]},
)


@_app.command()
def main(
    file: Path | None = typer.Argument(
        None,
        help="Read data/data.jsonl-schema rows from this file, one per line "
        "(defaults to stdin).",
    ),
    pt: Path = typer.Option(
        ..., "--pt", help="Folder containing enc.pt/style.pt/emoji.pt/gen.pt."
    ),
    output: Path | None = typer.Option(
        None, "-o", "--output", help="Write predictions here instead of stdout."
    ),
) -> None:
    """Run the inference graph over the `text` field of jsonl rows from a file or stdin."""
    if file:
        lines = file.read_text(encoding="utf-8").splitlines()
    else:
        lines = sys.stdin.read().splitlines()
    records = predict(read_texts(lines), pt)
    out_lines = [json.dumps(rec, ensure_ascii=False) for rec in records]

    if output:
        output.write_text("".join(line + "\n" for line in out_lines), encoding="utf-8")
    else:
        for line in out_lines:
            print(line)


if __name__ == "__main__":
    _app()
