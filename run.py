import json

import torch

from config import SEED
from data import EMOJIS, EVAL_PATH, STYLES, read, text_to_tensor
from model import ColorGen, EmojiHead, StyleHead, TextEncoder


def sample(n=200):
    records = list(read(EVAL_PATH))
    return records[:n]


def rgb_to_hex(rgb: torch.Tensor) -> list[str]:
    assert rgb.shape == (9,), "Input tensor must be of shape (9,)"

    ints = (rgb + 127.5).clamp(0, 255).to(torch.int32).cpu().tolist()

    def f2h(val: int) -> str:
        return f"{val:02x}"

    return [f"#{f2h(ints[i])}{f2h(ints[i + 1])}{f2h(ints[i + 2])}" for i in range(0, 9, 3)]


def _load(mod: torch.nn.Module, path: str) -> torch.nn.Module:
    mod.load_state_dict(torch.load(path, map_location="cpu"))
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


def predict(
    enc_path: str = "enc.pt",
    gen_path: str = "gen.pt",
    style_path: str = "style.pt",
    emoji_path: str = "emoji.pt",
):
    torch.manual_seed(SEED)

    enc = _load(TextEncoder(), enc_path)
    gen = _load(ColorGen(), gen_path)
    style = _load(StyleHead(), style_path)
    emoji = _load(EmojiHead(), emoji_path)

    records = sample()

    with open("pred.jsonl", "w", encoding="utf-8") as f:
        with torch.no_grad():
            for rec in records:
                text_tensor = text_to_tensor(rec.text).unsqueeze(0)
                emb = enc(text_tensor)

                styles = top_labels(style(emb), STYLES, min_k=1, max_k=3)
                emojis = top_labels(emoji(emb), EMOJIS, min_k=1, max_k=1)

                colors = gen(emb).squeeze(0)
                hexes = rgb_to_hex(colors)

                out_record = {
                    "text": rec.text,
                    "emojis": " ".join(emojis),
                    "styles": styles,
                    "bg": hexes[:2],
                    "fg": hexes[2],
                }

                f.write(json.dumps(out_record, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    predict()
