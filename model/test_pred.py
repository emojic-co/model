import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import typer

from model.pred import predict, read_rows


def test_read_rows_normalizes_and_drops_empty():
    rows = read_rows(['{"text": "  Hello   WORLD  "}', '{"text": ""}'])
    assert rows == [{"text": "hello world"}]


def test_predict_emits_three_valid_hex_colors():
    from files import PtFile
    from model.model import ColorGen, EmojiHead, StyleHead, TextEncoder
    from model.runmeta import save_pt

    pt_dir = Path(tempfile.mkdtemp(prefix="emojic-pred-"))
    save_pt(TextEncoder().state_dict(), PtFile.ENC.in_dir(pt_dir), stage="test")
    save_pt(ColorGen().state_dict(), PtFile.GEN.in_dir(pt_dir), stage="test")
    save_pt(StyleHead().state_dict(), PtFile.STYLE.in_dir(pt_dir), stage="test")
    save_pt(EmojiHead().state_dict(), PtFile.EMOJI.in_dir(pt_dir), stage="test")

    records = predict([{"text": "hello world"}], pt_dir)
    assert len(records) == 1
    r = records[0]
    assert len(r["bg"]) == 2
    for h in [*r["bg"], r["fg"]]:
        assert h.startswith("#") and len(h) == 7


_app = typer.Typer(
    add_completion=False,
    context_settings={"help_option_names": ["-h", "--help"]},
)


@_app.command()
def main() -> None:
    """Run the pred.py assertion checks."""
    test_read_rows_normalizes_and_drops_empty()
    test_predict_emits_three_valid_hex_colors()
    print("ok")


if __name__ == "__main__":
    _app()
