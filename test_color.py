import html
import json
import sys
from datetime import datetime
from pathlib import Path

import torch
from torch.nn.functional import normalize

from config import SEED, TEXT_EMBED_SIZE
from data import EVAL_PATH, hex2rgb, read, text_to_tensor
from model import ColorGen, TextEncoder

REPORT_DIR = Path("report/test-color")
MIN_ROWS = 15
SHOW_EX = 6
COLOR_SAMPLES = 5

ANCHORS = {
    "red": "#c0392b",
    "orange": "#e07a3f",
    "amber": "#d9a441",
    "yellow": "#e8d44d",
    "olive": "#8a8a3f",
    "green": "#4c9a52",
    "teal": "#3f9a90",
    "blue": "#4a6fd1",
    "navy": "#22304a",
    "purple": "#7a5aa8",
    "pink": "#d98cc0",
    "brown": "#6b4a35",
    "cream": "#f1e3c3",
    "black": "#1c1c1c",
    "white": "#f2f2f2",
    "gray": "#8a8a8a",
}

CONST_Z = normalize(
    torch.randn(
        COLOR_SAMPLES,
        TEXT_EMBED_SIZE,
        generator=torch.Generator().manual_seed(SEED),
    ),
    dim=-1,
)


def _load(mod: torch.nn.Module, path: str) -> torch.nn.Module:
    mod.load_state_dict(torch.load(path, map_location="cpu"))
    mod.eval()
    return mod


def _srgb_to_linear(c: float) -> float:
    return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4


def _oklab(rgb: tuple[float, float, float]) -> tuple[float, float, float]:
    r, g, b = (_srgb_to_linear(v / 255.0) for v in rgb)
    lm = 0.4122214708 * r + 0.5363325363 * g + 0.0514459929 * b
    mm = 0.2119034982 * r + 0.6806995451 * g + 0.1073969566 * b
    sm = 0.0883024619 * r + 0.2817188376 * g + 0.6299787005 * b
    l_ = lm ** (1 / 3)
    m_ = mm ** (1 / 3)
    s_ = sm ** (1 / 3)
    return (
        0.2104542553 * l_ + 0.7936177850 * m_ - 0.0040720468 * s_,
        1.9779984951 * l_ - 2.4285922050 * m_ + 0.4505937099 * s_,
        0.0259040371 * l_ + 0.7827717662 * m_ - 0.8086757660 * s_,
    )


ANCHOR_LAB = {name: _oklab(hex2rgb(hx)) for name, hx in ANCHORS.items()}


def _delta_e(a: tuple[float, float, float], b: tuple[float, float, float]) -> float:
    return sum((x - y) ** 2 for x, y in zip(a, b, strict=True)) ** 0.5


def _bucket(rgb: tuple[float, float, float]) -> str:
    lab = _oklab(rgb)
    return min(ANCHOR_LAB, key=lambda name: _delta_e(lab, ANCHOR_LAB[name]))


def _rgb_to_hex(rgb: tuple[float, float, float]) -> str:
    return "#" + "".join(f"{int(max(0, min(255, round(v)))):02x}" for v in rgb)


def _target_bg(colors: list[str]) -> tuple[float, float, float]:
    a, b = hex2rgb(colors[0]), hex2rgb(colors[1])
    return tuple((x + y) / 2 for x, y in zip(a, b, strict=True))  # type: ignore


def _gen_bgs(gen: ColorGen, emb: torch.Tensor) -> list[tuple[float, float, float]]:
    cond = emb.expand(COLOR_SAMPLES, -1)
    out = gen(cond, CONST_Z) + 127.5
    bgs = (out[:, 0:3] + out[:, 3:6]) / 2
    return [tuple(row.tolist()) for row in bgs]


def _evaluate(enc_path: str, gen_path: str, records: list) -> list[dict]:
    enc = _load(TextEncoder(), enc_path)
    gen = _load(ColorGen(), gen_path)

    rows = []
    with torch.no_grad():
        for rec in records:
            target = _target_bg(rec.colors)
            tgt_bucket = _bucket(target)
            emb = enc(text_to_tensor(rec.text).unsqueeze(0))
            gen_bgs = _gen_bgs(gen, emb)  # type: ignore
            gen_buckets = [_bucket(c) for c in gen_bgs]
            deltas = [_delta_e(_oklab(target), _oklab(c)) for c in gen_bgs]
            rows.append(
                {
                    "text": rec.text,
                    "target": target,
                    "target_hex": _rgb_to_hex(target),
                    "bucket": tgt_bucket,
                    "gen_hex": [_rgb_to_hex(c) for c in gen_bgs],
                    "gen_buckets": gen_buckets,
                    "hit": tgt_bucket in gen_buckets,
                    "sample_hits": sum(b == tgt_bucket for b in gen_buckets),
                    "min_delta_e": min(deltas),
                }
            )
    return rows


def _color_stats(rows: list[dict]) -> dict:
    n = len(rows)
    return {
        "n": n,
        "accuracy": sum(r["hit"] for r in rows) / n,
        "per_sample_hit": sum(r["sample_hits"] for r in rows) / (n * COLOR_SAMPLES),
        "mean_delta_e": sum(r["min_delta_e"] for r in rows) / n,
    }


def _collect(rows: list[dict]) -> list[dict]:
    by_bucket: dict[str, list[dict]] = {}
    for r in rows:
        by_bucket.setdefault(r["bucket"], []).append(r)
    colors = []
    for name, group in by_bucket.items():
        if len(group) < MIN_ROWS:
            continue
        stats = _color_stats(group)
        colors.append(
            {
                "name": name,
                "anchor_hex": ANCHORS[name],
                "rows": group,
                **stats,
            }
        )
    colors.sort(key=lambda c: c["accuracy"])
    return colors


def _swatch(hx: str, label: str = "") -> str:
    tip = f' title="{html.escape(label)}"' if label else ""
    return f'<span class="sw" style="background:{hx}"{tip}></span>'


def _render_html(colors: list[dict], summary: dict, meta: dict) -> str:
    out = [
        "<!doctype html><meta charset='utf-8'>",
        f"<title>Color test - {meta['stamp']}</title>",
        "<style>",
        "body{font:14px/1.5 system-ui,sans-serif;margin:2rem;max-width:960px}",
        "table{border-collapse:collapse;margin:1rem 0}",
        "th,td{border:1px solid #ccc;padding:4px 8px;text-align:left}",
        ".sw{display:inline-block;width:22px;height:22px;border:1px solid #0003;",
        "vertical-align:middle;border-radius:3px;margin-right:2px}",
        ".ex{margin:.3rem 0}",
        ".miss{color:#b00}",
        "h3{margin:1.4rem 0 .3rem}",
        "</style>",
        f"<h1>Color test &mdash; {meta['stamp']}</h1>",
        f"<p>model: <code>{meta['enc']}</code> + <code>{meta['gen']}</code><br>",
        f"colors: {summary['n_colors']} &nbsp; texts: {summary['n_texts']} &nbsp; ",
        f"{COLOR_SAMPLES} fixed noise vectors/text<br>",
        f"macro accuracy: {summary['accuracy']:.1%} &nbsp; ",
        f"mean &Delta;E: {summary['mean_delta_e']:.3f}</p>",
        "<table><tr><th>color</th><th>anchor</th><th>n</th><th>accuracy</th>",
        "<th>per-sample hit</th><th>mean &Delta;E</th></tr>",
    ]
    for c in colors:
        out.append(
            f"<tr><td>{_swatch(c['anchor_hex'], c['name'])}{c['name']}</td>"
            f"<td><code>{c['anchor_hex']}</code></td><td>{c['n']}</td>"
            f"<td>{c['accuracy']:.1%}</td><td>{c['per_sample_hit']:.1%}</td>"
            f"<td>{c['mean_delta_e']:.3f}</td></tr>"
        )
    out.append("</table>")

    for c in colors:
        out.append(
            f"<h3>{_swatch(c['anchor_hex'])}{c['name']} &mdash; {c['accuracy']:.1%}</h3>"
        )
        worst = sorted(c["rows"], key=lambda r: (r["hit"], -r["min_delta_e"]))
        for r in worst[:SHOW_EX]:
            cls = "" if r["hit"] else " miss"
            gens = "".join(
                _swatch(hx, b) for hx, b in zip(r["gen_hex"], r["gen_buckets"], strict=True)
            )
            out.append(
                f"<div class='ex{cls}'>{_swatch(r['target_hex'], r['bucket'])}"
                f"&nbsp; {gens} &nbsp; "
                f"{'hit' if r['hit'] else 'miss'} &nbsp; "
                f"<span>{html.escape(r['text'])}</span></div>"
            )
    return "\n".join(out) + "\n"


def test_color(
    enc_path: str = "enc.pt",
    gen_path: str = "gen.pt",
    eval_path: str = EVAL_PATH,
    write_report: bool = True,
) -> dict:
    records = list(read(eval_path))
    rows = _evaluate(enc_path, gen_path, records)
    colors = _collect(rows)
    stamp = datetime.now().strftime("%m-%d-%H:%M")

    scored = [r for c in colors for r in c["rows"]]
    summary = {
        "n_colors": len(colors),
        "n_texts": len(scored),
        "accuracy": (sum(c["accuracy"] for c in colors) / len(colors) if colors else 0.0),
        "mean_delta_e": (
            sum(c["mean_delta_e"] for c in colors) / len(colors) if colors else 0.0
        ),
    }
    meta = {"stamp": stamp, "enc": enc_path, "gen": gen_path}

    if write_report:
        REPORT_DIR.mkdir(parents=True, exist_ok=True)
        html_path = REPORT_DIR / f"{stamp}.html"
        html_path.write_text(_render_html(colors, summary, meta), encoding="utf-8")
        json_path = REPORT_DIR / f"{stamp}.json"
        json_path.write_text(
            json.dumps(
                {
                    **meta,
                    "summary": summary,
                    "colors": [
                        {
                            "name": c["name"],
                            "anchor_hex": c["anchor_hex"],
                            "n": c["n"],
                            "accuracy": c["accuracy"],
                            "per_sample_hit": c["per_sample_hit"],
                            "mean_delta_e": c["mean_delta_e"],
                        }
                        for c in colors
                    ],
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"wrote {html_path}")
        print(f"wrote {json_path}")

    line = "  ".join(f"{c['name']}={c['accuracy']:.0%}" for c in colors)
    print(
        f"color test  macro={summary['accuracy']:.0%}  dE={summary['mean_delta_e']:.3f}")
    print(f"  {line}")
    return {"summary": summary, "colors": colors}


if __name__ == "__main__":
    sys.exit(0 if test_color()["summary"]["n_colors"] else 1)
