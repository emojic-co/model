"""N x N sprite sheet (e.g. from Gemini or `emoji-sheet`) -> .webm animation."""

import subprocess
import sys
from pathlib import Path
from typing import Annotated

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import typer
from PIL import Image, ImageDraw

GRID_SIZE = 5
INSET = 4  # px shaved off each cell edge to drop the grid lines
BG_THRESH = 24  # per-channel distance from the corner colour still counted as background
MATTE = (255, 0, 255)

_app = typer.Typer(add_completion=False)


def split(
    sheet: Image.Image, grid: int = GRID_SIZE, inset: int = INSET
) -> list[Image.Image]:
    """Row-major cells of an evenly divided sheet, each inset by `inset` px."""
    w, h = sheet.size
    cw, ch = w / grid, h / grid
    return [
        sheet.crop(
            (
                round(c * cw) + inset,
                round(r * ch) + inset,
                round((c + 1) * cw) - inset,
                round((r + 1) * ch) - inset,
            )
        )
        for r in range(grid)
        for c in range(grid)
    ]


def cut_background(tile: Image.Image) -> Image.Image:
    """RGBA tile with the background (flood-filled from its corners) made transparent."""
    probe = tile.convert("RGB")
    w, h = probe.size
    for xy in [(0, 0), (w - 1, 0), (0, h - 1), (w - 1, h - 1)]:
        ImageDraw.floodfill(probe, xy, MATTE, thresh=BG_THRESH)
    bg = (np.asarray(probe) == MATTE).all(axis=2)
    out = tile.convert("RGBA")
    out.putalpha(Image.fromarray(np.where(bg, 0, 255).astype("uint8")))
    return out


def encode(frames: list[Image.Image], out: Path, fps: int, alpha: bool) -> None:
    w, h = frames[0].size
    pix = "yuva420p" if alpha else "yuv420p"
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-f", "rawvideo", "-pix_fmt", "rgba", "-s", f"{w}x{h}", "-r", str(fps), "-i", "-",
        "-c:v", "libvpx-vp9", "-pix_fmt", pix, "-b:v", "0", "-crf", "20",
        "-auto-alt-ref", "0", str(out),
    ]  # fmt: skip
    proc = subprocess.run(cmd, input=b"".join(f.tobytes() for f in frames))
    if proc.returncode:
        raise typer.Exit(proc.returncode)


@_app.command()
def main(
    sheet: Annotated[Path, typer.Argument(help="sprite sheet image")],
    out: Annotated[Path | None, typer.Option("-o", help="output webm")] = None,
    grid_size: Annotated[int, typer.Option(help="tiles per side")] = GRID_SIZE,
    fps: Annotated[int, typer.Option(help="frames per second")] = 20,
    inset: Annotated[int, typer.Option(help="px trimmed from each tile edge")] = INSET,
    size: Annotated[
        int | None, typer.Option(help="output px per side (default: tile size)")
    ] = None,
    opaque: Annotated[bool, typer.Option(help="keep the white background")] = False,
    skip: Annotated[int, typer.Option(help="drop this many trailing frames")] = 0,
    pingpong: Annotated[bool, typer.Option(help="play forward then backward")] = False,
    loops: Annotated[int, typer.Option(help="repeat the clip this many times")] = 1,
) -> None:
    tiles = split(Image.open(sheet).convert("RGB"), grid_size, inset)
    tiles = tiles[: len(tiles) - skip] if skip else tiles
    side = (size or min(tiles[0].size)) // 2 * 2  # yuv420 needs even dimensions
    frames = [
        (t if opaque else cut_background(t))
        .convert("RGBA")
        .resize((side, side), Image.LANCZOS)
        for t in tiles
    ]
    if pingpong:  # skip the turnaround frames so they aren't shown twice
        frames = frames + frames[-2:0:-1]
    frames = frames * loops
    out = out or sheet.with_suffix(".webm")
    encode(frames, out, fps, alpha=not opaque)
    typer.echo(out)


if __name__ == "__main__":
    _app()
