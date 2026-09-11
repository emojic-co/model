import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import onnx
import torch

from model.config import EMOJIS, MAX_TEXT_LEN
from model.data import FLEX_RAW_DIM, FLEXQ_DIM
from model.export_onnx import ExportWrapper, _strip_spectral_norm, export_onnx
from model.model import ColorGen, EmojiHead, StyleHead, TextEncoder

V = len(EMOJIS)


def _wrapper():
    torch.manual_seed(0)
    enc = TextEncoder().eval()
    _strip_spectral_norm(enc)
    return ExportWrapper(
        enc,
        StyleHead().eval(),
        EmojiHead().eval(),
        ColorGen().eval(),
    ).eval()


def _dim(vi, axis):
    d = vi.type.tensor_type.shape.dim[axis]
    return d.dim_param or d.dim_value


def test_onnx_io_shapes():
    with tempfile.TemporaryDirectory() as d:
        dst = Path(d) / "model.onnx"
        export_onnx(_wrapper(), dst)
        m = onnx.load(str(dst))
        onnx.checker.check_model(m)

        ins = {vi.name: vi for vi in m.graph.input}
        outs = {vi.name for vi in m.graph.output}
        assert set(ins) == {"input", "flex", "flex_q"}, set(ins)
        assert outs == {"style_logits", "emoji_logits", "color"}, outs

        assert _dim(ins["input"], 1) == MAX_TEXT_LEN
        assert _dim(ins["flex"], 1) == V
        assert _dim(ins["flex"], 2) == FLEX_RAW_DIM
        assert _dim(ins["flex_q"], 1) == FLEXQ_DIM
        for name in ("input", "flex", "flex_q"):
            assert _dim(ins[name], 0) == "batch"


if __name__ == "__main__":
    test_onnx_io_shapes()
    print("ok")
