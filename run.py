import json

import torch

from config import SEED
from data import EMOJIS, EVAL_PATH, FEELINGS, read, text_to_tensor
from model import ColorGen, EmojiHead, FeelingHead, TextEncoder


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


def predict(
    enc_path: str = "enc.pt",
    gen_path: str = "gen.pt",
    feels_path: str = "feels.pt",
    emoji_path: str = "emoji.pt",
):
    torch.manual_seed(SEED)

    enc = _load(TextEncoder(), enc_path)
    gen = _load(ColorGen(), gen_path)
    feels = _load(FeelingHead(), feels_path)
    emoji = _load(EmojiHead(), emoji_path)

    records = sample()

    with open("pred.jsonl", "w", encoding="utf-8") as f:
        with torch.no_grad():
            for rec in records:
                text_tensor = text_to_tensor(rec.text).unsqueeze(0)
                emb = enc(text_tensor)

                feeling = FEELINGS[int(feels(emb).argmax(dim=-1).item())]

                q, emoji_vec = emoji(emb)
                emoji_logits = q @ emoji_vec.t()
                emoji_pred = EMOJIS[int(emoji_logits.argmax(dim=-1).item())]

                colors = gen(emb).squeeze(0)
                hexes = rgb_to_hex(colors)

                out_record = {
                    "text": rec.text,
                    "emoji": emoji_pred,
                    "feeling": feeling,
                    "bg": hexes[:2],
                    "fg": hexes[2],
                }

                f.write(json.dumps(out_record, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    predict()
