import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import typer

from model.data import AB_RANGE
from model.export_onnx import ExportWrapper
from model.model import ColorGen, EmojiHead, StyleHead, TextEncoder


def test_export_wrapper_color_output_is_raw_unit_range():
    wrapper = ExportWrapper(
        TextEncoder(), StyleHead(), EmojiHead(), ColorGen()
    ).eval()
    x = torch.zeros(1, 42, dtype=torch.long)
    x[0, 0] = 1  # at least one non-PAD token; an all-PAD row NaNs the encoder's pool
    with torch.no_grad():
        _, _, color = wrapper(x)
    assert color.shape == (5, 9)
    assert torch.all(color >= -1.0 - 1e-6) and torch.all(color <= 1.0 + 1e-6)
    # would be ~[0, 255] under the old RGB export transform -- pin that it's not
    assert float(color.abs().max()) <= 1.0 + 1e-6


def test_export_web_meta_has_color_ab_range():
    import model.export_onnx as export_onnx

    tmp_path = Path(tempfile.mkdtemp(prefix="emojic-export-onnx-"))
    orig_web_public = export_onnx.WEB_PUBLIC
    orig_android_assets = export_onnx.ANDROID_ASSETS_DIR
    export_onnx.WEB_PUBLIC = tmp_path / "web"
    export_onnx.ANDROID_ASSETS_DIR = tmp_path / "android"
    try:
        wrapper = ExportWrapper(
            TextEncoder(), StyleHead(), EmojiHead(), ColorGen()
        ).eval()
        export_onnx.export_web(wrapper)

        meta = json.loads((tmp_path / "web" / "meta.json").read_text())
        assert meta["color_ab_range"] == AB_RANGE
    finally:
        export_onnx.WEB_PUBLIC = orig_web_public
        export_onnx.ANDROID_ASSETS_DIR = orig_android_assets


_app = typer.Typer(
    add_completion=False,
    context_settings={"help_option_names": ["-h", "--help"]},
)


@_app.command()
def main() -> None:
    """Run the export_onnx.py OKLAB-unit export assertion checks."""
    test_export_wrapper_color_output_is_raw_unit_range()
    test_export_web_meta_has_color_ab_range()
    print("ok")


if __name__ == "__main__":
    _app()
