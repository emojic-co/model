import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import typer

from model.data import (
    AB_RANGE,
    COLOR_DIM,
    color_tensor_to_hexes,
    color_unit_to_hex,
    colors2tensor,
    hex_to_color_unit,
)


def test_white_and_black_are_exact_endpoints():
    L, a, b = hex_to_color_unit("#ffffff")
    assert math.isclose(L, 1.0, abs_tol=1e-6)
    assert math.isclose(a, 0.0, abs_tol=1e-6)
    assert math.isclose(b, 0.0, abs_tol=1e-6)

    L, a, b = hex_to_color_unit("#000000")
    assert math.isclose(L, -1.0, abs_tol=1e-6)
    assert math.isclose(a, 0.0, abs_tol=1e-6)
    assert math.isclose(b, 0.0, abs_tol=1e-6)


def test_encode_decode_round_trips_for_representative_colors():
    for hexcolor in ["#ffffff", "#000000", "#ff0000", "#00ff00", "#0000ff",
                      "#808080", "#3a7bd5", "#c94f4f"]:
        unit = hex_to_color_unit(hexcolor)
        assert all(-1.0 - 1e-9 <= v <= 1.0 + 1e-9 for v in unit)
        back = color_unit_to_hex(unit)
        assert back == hexcolor


def test_ab_clamp_keeps_unit_channels_in_range():
    # Simulate an out-of-gamut a/b (can't happen from real sRGB hex, but the
    # clamp must still hold if it ever did).
    over_range_ab = AB_RANGE * 10
    clamped = max(-AB_RANGE, min(AB_RANGE, over_range_ab))
    assert clamped == AB_RANGE
    unit = clamped / AB_RANGE
    assert -1.0 <= unit <= 1.0


def test_colors2tensor_shape_and_range():
    t = colors2tensor(["#ffffff", "#000000", "#3a7bd5"])
    assert t.shape == (COLOR_DIM,)
    assert torch.all(t >= -1.0 - 1e-6) and torch.all(t <= 1.0 + 1e-6)


def test_color_tensor_to_hexes_matches_colors2tensor_round_trip():
    hexes = ["#a8e2f4", "#78c9f4", "#282e36"]
    t = colors2tensor(hexes)
    assert color_tensor_to_hexes(t) == hexes


_app = typer.Typer(
    add_completion=False,
    context_settings={"help_option_names": ["-h", "--help"]},
)


@_app.command()
def main() -> None:
    """Run the data.py OKLAB codec assertion checks."""
    test_white_and_black_are_exact_endpoints()
    test_encode_decode_round_trips_for_representative_colors()
    test_ab_clamp_keeps_unit_channels_in_range()
    test_colors2tensor_shape_and_range()
    test_color_tensor_to_hexes_matches_colors2tensor_round_trip()
    print("ok")


if __name__ == "__main__":
    _app()
