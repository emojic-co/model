"""Build the standalone GitHub Pages site into `docs/`.

Single command: `uv run build_web.py`. Takes the trained checkpoint (`model.pt`)
plus the static sources in `site/` and produces a fully self-contained page that
runs inference in the browser via onnxruntime-web -- no backend.

Outputs (all under `docs/`):

    model.onnx   exported from model.pt (browser runs this)
    meta.json    CHARS / MAX_TEXT_LEN / EMOJIS / feelings / feeling palette,
                 so the JS never hardcodes anything main.py owns
    index.html   copied from site/
    app.js       copied from site/  (onnxruntime-web inference)
    style.css    copied from web/style.css  (single source of truth for styling)
    vendor/      copied from vendor/ort/  (vendored onnxruntime-web, wasm backend)
    .nojekyll    so GitHub Pages serves vendor/ and .mjs files untouched

CI (.github/workflows/deploy-pages.yml) runs exactly this and uploads `docs/` as
the Pages artifact; `docs/` itself is gitignored.
"""

import json
import shutil
from pathlib import Path

import torch
from torch import nn

from config import MAX_TEXT_LEN
from main import (
    CHARS,
    EMOJIS,
    PAD_IDX,
    Model,
    feeling_colors,
)
from main import feeling as FEELINGS

ROOT = Path(__file__).parent
OUT = ROOT / "docs"
MODEL_PATH = ROOT / "model.pt"
OPSET = 18


class ExportWrapper(nn.Module):
    """Inference-only forward that ONNX can trace.

    The trained model uses `pack_padded_sequence`, which does not export. Here we
    run the LSTM over the full padded sequence and gather the top-layer hidden
    state at the last real character. For right-padded input the LSTM output at
    step t depends only on steps 0..t, so `out[b, len_b - 1]` is exactly the
    packed `h_n[-1]` the classifier reads during training.
    """

    def __init__(self, m: Model):
        super().__init__()
        self.m = m

    def forward(self, x: torch.Tensor):
        mask = x != PAD_IDX
        lengths = mask.sum(dim=1).clamp(min=1)

        emb = self.m.embedding(x)
        conv_out = self.m.relu(self.m.conv(emb.permute(0, 2, 1)))
        conv_out = conv_out * mask.unsqueeze(1).to(conv_out.dtype)

        out, _ = self.m.rnn(conv_out.permute(0, 2, 1))
        idx = (lengths - 1).view(-1, 1, 1).expand(-1, 1, out.size(2))
        last = out.gather(1, idx).squeeze(1)  # dropout is identity in eval mode

        return self.m.emoji(last), self.m.feeling(last)


def export_onnx(dst: Path) -> None:
    if not MODEL_PATH.is_file():
        raise SystemExit("model.pt not found -- train first with `uv run main.py`")

    model = Model()
    model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu", weights_only=True))
    model.eval()

    dummy = torch.zeros(1, MAX_TEXT_LEN, dtype=torch.long)
    torch.onnx.export(
        ExportWrapper(model).eval(),
        dummy,
        str(dst),
        input_names=["input"],
        output_names=["emoji_logits", "feeling_logits"],
        opset_version=OPSET,
        external_data=False,  # keep weights inside the single .onnx file
        dynamic_axes={
            "input": {0: "batch"},
            "emoji_logits": {0: "batch"},
            "feeling_logits": {0: "batch"},
        },
    )


def write_meta(dst: Path) -> None:
    meta = {
        "chars": CHARS,
        "pad_idx": PAD_IDX,
        "max_text_len": MAX_TEXT_LEN,
        "emojis": EMOJIS,
        "feelings": FEELINGS,
        "feeling_palette": {name: feeling_colors(name) for name in FEELINGS},
    }
    dst.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    if OUT.exists():
        shutil.rmtree(OUT)
    (OUT / "vendor").mkdir(parents=True)

    export_onnx(OUT / "model.onnx")
    write_meta(OUT / "meta.json")

    shutil.copy2(ROOT / "site" / "index.html", OUT / "index.html")
    shutil.copy2(ROOT / "site" / "app.js", OUT / "app.js")
    shutil.copy2(ROOT / "web" / "style.css", OUT / "style.css")

    for f in sorted((ROOT / "vendor" / "ort").iterdir()):
        if f.suffix in {".js", ".mjs", ".wasm"}:
            shutil.copy2(f, OUT / "vendor" / f.name)

    (OUT / ".nojekyll").write_text("", encoding="utf-8")

    print(f"Built {OUT.relative_to(ROOT)}/ :")
    for f in sorted(OUT.rglob("*")):
        if f.is_file():
            print(f"  {f.relative_to(OUT)}  ({f.stat().st_size:,} B)")


if __name__ == "__main__":
    main()
