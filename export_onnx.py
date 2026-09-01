import json
import warnings
from datetime import UTC, datetime
from pathlib import Path

import torch
from torch import nn
from torch.nn.functional import normalize

from config import EMOJIS, FEELINGS, MAX_TEXT_LEN, SEED, TEXT_EMBED_SIZE, Z_WEIGHT
from data import CHARS, PAD_IDX
from model import ColorGen, EmojiHead, FeelingHead, TextEncoder

WEB_PUBLIC = Path("web/public")
ONNX_OPSET = 18
COLOR_SAMPLES = 5

CONST_Z = normalize(
    torch.randn(
        COLOR_SAMPLES,
        TEXT_EMBED_SIZE,
        generator=torch.Generator().manual_seed(SEED),
    ),
    dim=-1,
)


def _load(mod: nn.Module, path: str) -> nn.Module:
    mod.load_state_dict(torch.load(path, map_location="cpu"))
    mod.eval()
    return mod


def _strip_spectral_norm(mod: nn.Module) -> None:
    for m in mod.modules():
        try:
            nn.utils.remove_spectral_norm(m)
        except (ValueError, RuntimeError):
            pass


class ExportWrapper(nn.Module):
    def __init__(
        self,
        enc: nn.Module,
        feels: nn.Module,
        emoji: nn.Module,
        gen: nn.Module,
    ) -> None:
        super().__init__()
        self.enc = enc
        self.feels = feels
        self.emoji = emoji
        self.gen = gen
        self.register_buffer("z", CONST_Z)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        emb = self.enc(x)
        feeling_logits = self.feels(emb)
        q, emoji_vec = self.emoji(emb)
        emoji_logits = q @ emoji_vec.t()
        seed = (1 - Z_WEIGHT) * emb + Z_WEIGHT * self.z
        color = torch.tanh(self.gen.net(seed)) * 127.5 + 127.5
        return feeling_logits, emoji_logits, color


def export_onnx(wrapper: nn.Module, dst: Path) -> None:
    dummy = torch.zeros(1, MAX_TEXT_LEN, dtype=torch.long)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        torch.onnx.export(
            wrapper,
            (dummy,),
            str(dst),
            input_names=["input"],
            output_names=["feeling_logits", "emoji_logits", "color"],
            opset_version=ONNX_OPSET,
            dynamo=False,
            dynamic_axes={
                "input": {0: "batch"},
                "feeling_logits": {0: "batch"},
                "emoji_logits": {0: "batch"},
            },
        )


def export_web(wrapper: nn.Module) -> None:
    WEB_PUBLIC.mkdir(parents=True, exist_ok=True)
    export_onnx(wrapper, WEB_PUBLIC / "model.onnx")
    meta = {
        "chars": CHARS,
        "pad_idx": PAD_IDX,
        "max_text_len": MAX_TEXT_LEN,
        "emojis": EMOJIS,
        "feelings": FEELINGS,
        "exported_at": datetime.now(UTC).isoformat(timespec="minutes"),
    }
    (WEB_PUBLIC / "meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (WEB_PUBLIC / "config.json").write_text(
        json.dumps({"max_text_len": MAX_TEXT_LEN}, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    enc = _load(TextEncoder(), "enc.pt")
    feels = _load(FeelingHead(), "feels.pt")
    emoji = _load(EmojiHead(), "emoji.pt")
    gen = _load(ColorGen(), "gen.pt")

    if feels.net.weight.shape[0] != len(FEELINGS):
        raise SystemExit(
            f"feels.pt has {feels.net.weight.shape[0]} feelings, "
            f"labels.json has {len(FEELINGS)} -- retrain or restore labels.json"
        )
    if emoji.embed.weight.shape[0] != len(EMOJIS):
        raise SystemExit(
            f"emoji.pt has {emoji.embed.weight.shape[0]} emojis, "
            f"labels.json has {len(EMOJIS)} -- retrain or restore labels.json"
        )

    _strip_spectral_norm(enc)

    wrapper = ExportWrapper(enc, feels, emoji, gen).eval()
    export_web(wrapper)
    print(f"wrote {WEB_PUBLIC}/model.onnx + meta.json + config.json")
