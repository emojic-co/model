import html
import json
import sys
from datetime import datetime
from pathlib import Path

import torch
import typer
from torch.nn.functional import normalize

from config import (
    ENERGY_KEYWORD_MAX_TEXTS,
    ENERGY_KEYWORD_MIN_TEXTS,
    ENERGY_KEYWORDS_PATH,
    ENERGY_Z_SAMPLES,
    SEED,
    TEXT_EMBED_SIZE,
)
from data import (
    TRAIN_PATH,
    hex2rgb,
    load_energy_keywords,
    read,
    text_to_tensor,
)
from model import COLOR_SCALE, ColorGen, TextEncoder, rgb_to_oklab
from runmeta import load_pt, model_slug, run_meta, stamp_lines, write_meta_yml

REPORT_DIR = Path("report/test-color")
SHOW_EX = 12

Z_BANK = normalize(
    torch.randn(
        ENERGY_Z_SAMPLES,
        TEXT_EMBED_SIZE,
        generator=torch.Generator().manual_seed(SEED),
    ),
    dim=-1,
)


def _load(mod: torch.nn.Module, path: str) -> torch.nn.Module:
    sd, meta = load_pt(path)
    mod.load_state_dict(sd)
    mod._pt_meta = meta
    mod.eval()
    return mod


def energy_distance(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    mode = "donot_use_mm_for_euclid_dist"
    xy = torch.cdist(x, y, compute_mode=mode).mean()
    xx = torch.cdist(x, x, compute_mode=mode).mean()
    yy = torch.cdist(y, y, compute_mode=mode).mean()
    return (2 * xy - xx - yy).clamp(min=0.0).sqrt()


def _split_energy(pts: torch.Tensor) -> torch.Tensor:
    m = pts.size(0)
    half = m // 2
    perm = torch.randperm(m, generator=torch.Generator().manual_seed(SEED))
    return energy_distance(pts[perm[:half]], pts[perm[half : 2 * half]])


def _rgb_to_hex(rgb) -> str:
    return "#" + "".join(f"{int(max(0, min(255, round(v)))):02x}" for v in rgb)


def _palette_hex(flat) -> list[str]:
    vals = list(flat)
    return [_rgb_to_hex(vals[i : i + 3]) for i in range(0, 9, 3)]


def _keyword_groups(train_path: str) -> dict[str, list]:
    kws = [k.lower() for k in load_energy_keywords(ENERGY_KEYWORDS_PATH)]
    hits: dict[str, list] = {k: [] for k in kws}
    for r in read(train_path):
        for k in kws:
            if k in r.text:
                hits[k].append(r)

    g = torch.Generator().manual_seed(SEED)
    out: dict[str, list] = {}
    for k in kws:
        rows = hits[k]
        if len(rows) < ENERGY_KEYWORD_MIN_TEXTS:
            print(f"keyword {k!r}: {len(rows)} < {ENERGY_KEYWORD_MIN_TEXTS} matches, skip")
            continue
        if len(rows) > ENERGY_KEYWORD_MAX_TEXTS:
            idx = torch.randperm(len(rows), generator=g)[:ENERGY_KEYWORD_MAX_TEXTS].tolist()
            rows = [rows[i] for i in idx]
        out[k] = rows
    return out


def _evaluate(
    enc_path: str, gen_path: str, groups: dict[str, list]
) -> tuple[list[dict], dict]:
    enc = _load(TextEncoder(), enc_path)
    gen = _load(ColorGen(), gen_path)

    out = []
    with torch.no_grad():
        for kw, recs in groups.items():
            text_ids = torch.stack([text_to_tensor(r.text) for r in recs])
            real_rgb = torch.stack(
                [
                    torch.tensor(
                        [c for h in r.colors for c in hex2rgb(h)], dtype=torch.float32
                    )
                    - 127.5
                    for r in recs
                ]
            )
            n = text_ids.size(0)
            rep = text_ids.repeat_interleave(ENERGY_Z_SAMPLES, dim=0)
            z = Z_BANK.repeat(n, 1)
            fake_rgb = gen(enc(rep), z)

            real = rgb_to_oklab(real_rgb)
            fake = rgb_to_oklab(fake_rgb)
            gen_ed = energy_distance(real, fake).item()
            ref_ed = _split_energy(real).item()

            fake_disp = (fake_rgb + COLOR_SCALE).view(n, ENERGY_Z_SAMPLES, 9)

            g = torch.Generator().manual_seed(SEED)
            ridx = torch.randperm(n, generator=g)[:SHOW_EX].tolist()
            zidx = torch.randint(ENERGY_Z_SAMPLES, (len(ridx),), generator=g).tolist()
            ex = [
                {
                    "text": recs[i].text,
                    "real_hex": list(recs[i].colors),
                    "gen_hex": _palette_hex(fake_disp[i, s].tolist()),
                }
                for i, s in zip(ridx, zidx, strict=True)
            ]

            out.append(
                {
                    "keyword": kw,
                    "n": n,
                    "energy_gen": gen_ed,
                    "energy_ref": ref_ed,
                    "gap": gen_ed - ref_ed,
                    "ex": ex,
                }
            )
    out.sort(key=lambda c: -c["gap"])
    return out, {enc_path: enc._pt_meta, gen_path: gen._pt_meta}


def _swatch(hx: str, label: str = "") -> str:
    tip = f' title="{html.escape(label)}"' if label else ""
    return f'<span class="sw" style="background:{hx}"{tip}></span>'


def _palette(hexes: list[str]) -> str:
    return "".join(_swatch(h) for h in hexes)


def _render_html(groups: list[dict], summary: dict, meta: dict) -> str:
    out = [
        "<!doctype html><meta charset='utf-8'>",
        f"<title>Color test - {meta['stamp']}</title>",
        "<style>",
        "body{font:14px/1.5 system-ui,sans-serif;margin:2rem;max-width:1200px}",
        "table{border-collapse:collapse;margin:1rem 0}",
        "th,td{border:1px solid #ccc;padding:4px 8px;text-align:left}",
        ".sw{display:inline-block;width:22px;height:22px;border:1px solid #0003;",
        "vertical-align:middle;border-radius:3px;margin-right:2px}",
        ".cols{display:grid;grid-template-columns:1fr 1fr;gap:0 2rem}",
        "h3{margin:1.4rem 0 .3rem}",
        "</style>",
        f"<h1>Color test &mdash; {meta['stamp']}</h1>",
        "<p>" + "<br>".join(html.escape(ln) for ln in meta["stamp_lines"]) + "<br>",
        f"keywords: {summary['n_keywords']} &nbsp; texts: {summary['n_texts']} &nbsp; ",
        f"{ENERGY_Z_SAMPLES} fixed noise vectors/text<br>",
        f"macro energy gen&harr;real: {summary['energy_gen']:.3f} &nbsp; ",
        f"macro energy real&harr;real: {summary['energy_ref']:.3f} &nbsp; ",
        f"macro gap: {summary['gap']:+.3f}</p>",
        "<table><tr><th>keyword</th><th>n</th><th>energy gen&harr;real</th>",
        "<th>energy real&harr;real</th><th>gap</th></tr>",
    ]
    for c in groups:
        out.append(
            f"<tr><td>{c['keyword']}</td><td>{c['n']}</td>"
            f"<td>{c['energy_gen']:.3f}</td><td>{c['energy_ref']:.3f}</td>"
            f"<td>{c['gap']:+.3f}</td></tr>"
        )
    out.append("</table>")

    out.append("<div class='cols'>")
    for c in groups:
        out.append("<div>")
        out.append(
            f"<h3>{c['keyword']} &mdash; gen {c['energy_gen']:.3f} / "
            f"ref {c['energy_ref']:.3f} / gap {c['gap']:+.3f}</h3>"
        )
        out.append("<table><tr><th>real</th><th>gen</th><th>text</th></tr>")
        for e in c["ex"]:
            out.append(
                f"<tr><td>{_palette(e['real_hex'])}</td>"
                f"<td>{_palette(e['gen_hex'])}</td>"
                f"<td>{html.escape(e['text'])}</td></tr>"
            )
        out.append("</table>")
        out.append("</div>")
    out.append("</div>")
    return "\n".join(out) + "\n"


def test_color(
    enc_path: str = "enc.pt",
    gen_path: str = "gen.pt",
    train_path: str = TRAIN_PATH,
    write_report: bool = True,
) -> dict:
    groups, metas = _evaluate(enc_path, gen_path, _keyword_groups(train_path))
    probe_meta = run_meta()
    stamp = datetime.now().strftime("%y-%m-%d-%H-%M")
    enc_meta = metas.get(enc_path)
    gen_meta = metas.get(gen_path)

    n = len(groups)
    summary = {
        "n_keywords": n,
        "n_texts": sum(c["n"] for c in groups),
        "energy_gen": (sum(c["energy_gen"] for c in groups) / n if n else 0.0),
        "energy_ref": (sum(c["energy_ref"] for c in groups) / n if n else 0.0),
        "gap": (sum(c["gap"] for c in groups) / n if n else 0.0),
    }
    meta = {
        "stamp": stamp,
        "enc": enc_path,
        "gen": gen_path,
        "stamp_lines": stamp_lines(enc_meta, enc_path, probe_meta),
    }

    if write_report:
        out_dir = REPORT_DIR / f"{stamp}-{model_slug(enc_meta)}"
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "report.html").write_text(
            _render_html(groups, summary, meta), encoding="utf-8"
        )
        (out_dir / "report.json").write_text(
            json.dumps(
                {
                    "stamp": stamp,
                    "enc": enc_path,
                    "gen": gen_path,
                    "summary": summary,
                    "keywords": [
                        {
                            "keyword": c["keyword"],
                            "n": c["n"],
                            "energy_gen": c["energy_gen"],
                            "energy_ref": c["energy_ref"],
                            "gap": c["gap"],
                        }
                        for c in groups
                    ],
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        warnings = []
        if enc_meta and gen_meta and enc_meta.get("sha") != gen_meta.get("sha"):
            warnings.append(f"{enc_path} and {gen_path} were saved from different commits")
        if (enc_meta is None) != (gen_meta is None):
            warnings.append(
                f"{enc_path} and {gen_path}: one carries embedded "
                "metadata, the other is legacy"
            )
        doc = {
            "report_type": "test-color",
            "generated": probe_meta["generated"],
            "probe_commit": probe_meta["sha"],
            "probe_dirty": probe_meta["dirty"],
        }
        if warnings:
            doc["warnings"] = warnings
        doc["models"] = {enc_path: enc_meta, gen_path: gen_meta}
        doc["summary"] = summary
        write_meta_yml(out_dir, doc)
        print(f"wrote {out_dir}/")

    line = "  ".join(f"{c['keyword']}={c['gap']:+.3f}" for c in groups)
    print(
        f"color test  gen={summary['energy_gen']:.3f}  "
        f"ref={summary['energy_ref']:.3f}  gap={summary['gap']:+.3f}"
    )
    print(f"  {line}")
    return {"summary": summary, "keywords": groups}


_app = typer.Typer(
    add_completion=False,
    context_settings={"help_option_names": ["-h", "--help"]},
)


@_app.command()
def main() -> None:
    """Score enc.pt + gen.pt on color-word groups; write report/test-color/<ts>-<sha>/."""
    sys.exit(0 if test_color()["summary"]["n_keywords"] else 1)


if __name__ == "__main__":
    _app()
