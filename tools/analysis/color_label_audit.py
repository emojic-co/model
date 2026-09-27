import colorsys
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import torch
import typer

from tools.report import PURE_HEX, _hex_to_rgb, _valid_color_rows

HUE_COLORS = ("red", "green", "blue")
LIGHTNESS_COLORS = ("dark", "bright")
CHROMA_FLOOR = 0.02
L_MID = 0.5


def _bg_hls(row: dict) -> tuple[float, float, float]:
    """Mean of the two bg swatches' RGB, as (hue_deg, lightness, saturation)."""
    bg1 = torch.tensor(_hex_to_rgb(row["bg"][0]), dtype=torch.float32)
    bg2 = torch.tensor(_hex_to_rgb(row["bg"][1]), dtype=torch.float32)
    r, g, b = ((torch.stack([bg1, bg2]).mean(dim=0) + 1.0) / 2.0).tolist()
    h, lightness, s = colorsys.rgb_to_hls(r, g, b)
    return h * 360.0, lightness, s


def _hue_deg(hx: str) -> float:
    r, g, b = ((torch.tensor(_hex_to_rgb(hx)) + 1.0) / 2.0).tolist()
    h, _, _ = colorsys.rgb_to_hls(r, g, b)
    return h * 360.0


def _hue_dist(a: float, b: float) -> float:
    return abs((a - b + 180) % 360 - 180)


def _audit(rows: list[dict]) -> list[dict]:
    hue_ref = {c: _hue_deg(PURE_HEX[c]) for c in HUE_COLORS}
    out = []
    for r in rows:
        label = r.get("color")
        hue, light_l, chroma = _bg_hls(r)

        if label in HUE_COLORS:
            if chroma < CHROMA_FLOOR:
                nearest, note = "(desaturated)", "no hue signal"
            else:
                nearest = min(HUE_COLORS, key=lambda c: _hue_dist(hue, hue_ref[c]))
                note = "wrong hue family" if nearest != label else ""
        elif label in LIGHTNESS_COLORS:
            nearest = "dark" if light_l < L_MID else "bright"
            note = "wrong lightness" if nearest != label else ""
        else:
            continue

        out.append(
            {
                "text": r["text"],
                "bg": r["bg"],
                "fg": r["fg"],
                "label": label,
                "nearest": nearest,
                "chroma": chroma,
                "L": light_l,
                "suspicious": bool(note),
                "note": note,
            }
        )
    return out


def main(top: int = typer.Option(12, "--top", help="rows to print per color")) -> None:
    rows = _valid_color_rows()
    if not rows:
        print("no rows in data/colors.jsonl")
        raise typer.Exit(1)

    audited = _audit(rows)

    print(f"{'color':<8} {'n':>5} {'suspicious':>10} {'frac':>6} {'desaturated':>11}")
    for c in (*HUE_COLORS, *LIGHTNESS_COLORS):
        rs = [a for a in audited if a["label"] == c]
        n = len(rs) or 1
        suspicious = [a for a in rs if a["suspicious"]]
        desat = [a for a in rs if a["nearest"] == "(desaturated)"]
        print(
            f"{c:<8} {len(rs):>5} {len(suspicious):>10} "
            f"{len(suspicious) / n:>6.2f} {len(desat):>11}"
        )

    print()
    print(
        "Rows whose background hue family (red/green/blue) or lightness "
        "(dark/bright) disagrees with the row's own color label — sorted "
        "chroma/lightness-distance worst first within each color. '(desaturated)' "
        f"means chroma < {CHROMA_FLOOR} — too gray for a hue call, reported "
        "separately, not counted as a hue mismatch."
    )
    for c in (*HUE_COLORS, *LIGHTNESS_COLORS):
        rs = [a for a in audited if a["label"] == c and a["suspicious"]]
        rs.sort(key=lambda a: -a["chroma"] if c in HUE_COLORS else -abs(a["L"] - L_MID))
        if not rs:
            continue
        print(f"\n=== {c} ({rs[0]['note']}) ===")
        for a in rs[:top]:
            print(
                f"  labeled {a['label']:<6} looks like {a['nearest']:<8} "
                f"L={a['L']:.2f} chroma={a['chroma']:.3f} "
                f"bg={a['bg']} fg={a['fg']!r} text={a['text']!r}"
            )


_app = typer.Typer(
    add_completion=False,
    context_settings={"help_option_names": ["-h", "--help"]},
)


@_app.command()
def cli(top: int = typer.Option(12, "--top", help="rows to print per color")) -> None:
    """Flag data/colors.jsonl rows whose bg hue family (red/green/blue) or
    lightness (dark/bright) disagrees with the row's own color label — a
    label-noise audit for the color GAN gold set."""
    main(top)


if __name__ == "__main__":
    _app()
