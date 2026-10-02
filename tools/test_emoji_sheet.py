import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tools.emoji_sheet import GRID_SIZE, frame_indices, resolve, sheet


def test_frame_indices_cover_animation():
    idx = frame_indices(101)
    assert len(idx) == GRID_SIZE**2 and idx[0] == 0 and idx[-1] < 101 and idx == sorted(idx)
    assert len(frame_indices(10, 64)) == 64  # short animation repeats frames


def test_resolve_forms():
    assert resolve("x/a.json") == Path("x/a.json")
    assert resolve("1f195").name == "1f195.json"
    assert resolve("🆕").name == "1f195.json"
    assert resolve("🌡").name == "1f321_fe0f.json"


def test_sheet_size_background_and_grid():
    im = sheet(resolve("1f195"), cell=32, grid=3, pad=4)
    assert im.mode == "RGB" and im.size == (3 * 40, 3 * 40)
    assert im.getpixel((2, 2)) == (255, 255, 255)  # padding is white
    assert im.getpixel((0, 0)) != (255, 255, 255)  # grid line
    assert im.getpixel((40, 20)) != (255, 255, 255)  # interior grid line
    assert im.getpixel((20, 20)) != (255, 255, 255)  # frame drawn
    assert sheet(resolve("1f195"), cell=32).size == (GRID_SIZE * 48, GRID_SIZE * 48)
