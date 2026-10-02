"""Animated emoji (Lottie) -> one PNG sprite sheet of N x N evenly spaced frames."""

import sys
from pathlib import Path
from typing import Annotated

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import typer
from PIL import Image, ImageDraw
from rlottie_python import LottieAnimation

from files import NOTO_LOTTIE_DIR

GRID_SIZE = 5
PAD = 8
GRID_LINE = "#e0e0e0"

_app = typer.Typer(add_completion=False)


def resolve(src: str) -> Path:
    """An emoji, hex codepoints (`1f195`, `1f321_fe0f`) or a path to a Lottie json."""
    if Path(src).suffix == ".json":
        return Path(src)
    if "_" not in src and all(c in "0123456789abcdefABCDEF" for c in src):
        cps = [src.lower()]
    else:
        cps = [f"{ord(c):x}" for c in src]
    path = NOTO_LOTTIE_DIR / f"{'_'.join(cps)}.json"
    if not path.exists() and len(cps) == 1:  # Noto keeps the VS16 on text-default emojis
        path = NOTO_LOTTIE_DIR / f"{cps[0]}_fe0f.json"
    if not path.exists() and cps[-1] == "fe0f":
        path = NOTO_LOTTIE_DIR / f"{'_'.join(cps[:-1])}.json"
    return path


def frame_indices(total: int, n: int = GRID_SIZE**2) -> list[int]:
    """n frames evenly spaced over [0, total); repeats if the animation is shorter."""
    return [k * total // n for k in range(n)]


def sheet(
    lottie: Path, cell: int = 128, grid: int = GRID_SIZE, pad: int = PAD
) -> Image.Image:
    """White sheet; frames sit in (cell + 2 * pad) slots split by light grid lines."""
    anim = LottieAnimation.from_file(str(lottie))
    slot = cell + 2 * pad
    out = Image.new("RGB", (grid * slot, grid * slot), "white")
    total = anim.lottie_animation_get_totalframe()
    for k, f in enumerate(frame_indices(total, grid * grid)):
        im = anim.render_pillow_frame(frame_num=f, width=cell, height=cell)
        out.paste(im, (k % grid * slot + pad, k // grid * slot + pad), im)
    draw = ImageDraw.Draw(out)
    for i in range(grid + 1):
        x = min(i * slot, grid * slot - 1)
        draw.line([(x, 0), (x, grid * slot)], fill=GRID_LINE)
        draw.line([(0, x), (grid * slot, x)], fill=GRID_LINE)
    return out


@_app.command()
def main(
    src: Annotated[
        str, typer.Argument(help="emoji, hex codepoints (1f600), or path to .json")
    ],
    out: Annotated[Path | None, typer.Option("-o", help="output png")] = None,
    cell: Annotated[int, typer.Option(help="tile size in px")] = 128,
    grid_size: Annotated[int, typer.Option(help="tiles per side")] = GRID_SIZE,
    pad: Annotated[int, typer.Option(help="padding around each tile in px")] = PAD,
) -> None:
    path = resolve(src)
    if not path.exists():
        raise typer.BadParameter(f"no Lottie file at {path}")
    out = out or Path(f"{path.stem}_sheet.png")
    sheet(path, cell, grid_size, pad).save(out)
    typer.echo(out)


if __name__ == "__main__":
    _app()
