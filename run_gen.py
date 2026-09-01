import json
import random

import torch

from data import EVAL_PATH, EmojiDataset, read
from model import ColorGen


def sample(n=20):
    records = list(read(EVAL_PATH))
    return random.sample(records, min(n, len(records)))


def rgb_to_hex(rgb: torch.Tensor) -> list[str]:
    assert rgb.shape == (9,), "Input tensor must be of shape (9,)"

    # Shift from [-127.5, 127.5] back to [0, 255] and clamp bounds
    ints = (rgb + 127.5).clamp(0, 255).to(torch.int32).cpu().tolist()

    def f2h(val: int) -> str:
        return f"{val:02x}"

    return [
        f"#{f2h(ints[i])}{f2h(ints[i+1])}{f2h(ints[i+2])}"
        for i in range(0, 9, 3)
    ]


def predict(weights_path: str = "gen.pt"):
    gen = ColorGen()
    gen.load_state_dict(torch.load(weights_path, map_location="cpu"))
    gen.eval()

    records = sample(20)
    ds = EmojiDataset(records)

    with open("pred.jsonl", "w", encoding="utf-8") as f:
        with torch.no_grad():
            for i, rec in enumerate(records):
                # ds[i] returns (text, emoji, feeling, colors)
                # We need text (index 0) with a batch dimension: (1, MAX_TEXT_LEN)
                text_tensor = ds[i][0].unsqueeze(0)

                # Generate palette output -> shape (9,)
                output = gen(text_tensor).squeeze(0)
                generated_hexes = rgb_to_hex(output)

                # Reconstruct output matching the original JSONL schema ('bg', 'fg')
                out_record = {
                    "text": rec.text,
                    "emoji": rec.emoji,
                    "feeling": rec.feeling,
                    "bg": generated_hexes[:2],
                    "fg": generated_hexes[2],
                }

                f.write(json.dumps(out_record, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    predict()
