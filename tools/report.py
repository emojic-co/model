import html
import json
import sys
from collections import Counter
from datetime import datetime
from functools import cache
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import typer

from files import DATA_JSONL, KEYWORDS_JSON
from model.config import EMOJIS, STYLES
from model.data import EVAL_PATH, TRAIN_PATH, read, text_to_tensor
from model.data import normalize as norm_text
from model.model import EmojiHead, TextEncoder
from model.runmeta import load_pt, run_meta

DATA_PATH = DATA_JSONL
KEYWORDS_PATH = KEYWORDS_JSON

EMOJI_KS = list(range(1, 11))
KEYWORD_MISS_K = 5
KEYWORD_TOP = 5


def _ts() -> str:
    return datetime.now().strftime("%y-%m-%d-%H-%M")


@cache
def _rows(path: str) -> tuple:
    p = Path(path)
    if not p.exists():
        return ()
    out = []
    with p.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return tuple(out)


def _load(mod, path):
    sd, _ = load_pt(path)
    try:
        mod.load_state_dict(sd)
    except RuntimeError as exc:
        return None, str(exc).splitlines()[0]
    mod.eval()
    return mod, None


def _acc_at_k(logits, target, k):
    k = min(k, logits.size(-1))
    top = logits.topk(k, dim=-1).indices
    return target.gather(1, top).amax(dim=-1)


def _provenance(pt: Path):
    enc_pt, emoji_pt = str(pt / "enc.pt"), str(pt / "emoji.pt")
    rm = run_meta()
    paths = [enc_pt, emoji_pt]
    metas = {p: (load_pt(p)[1] if Path(p).exists() else None) for p in paths}
    present = {p: m for p, m in metas.items() if m}
    missing = [p for p in paths if not Path(p).exists()]
    legacy = [p for p in paths if Path(p).exists() and metas[p] is None]
    shas = {m.get("sha") for m in present.values()}
    enc_meta = metas.get(enc_pt)
    model_sha = enc_meta.get("sha") if enc_meta else "nometa"
    train_now = rm.get("train_sha")
    model_train = enc_meta.get("train_sha") if enc_meta else None

    issues = []
    if len(shas) > 1:
        issues.append(
            "model .pt files saved from different commits: "
            + ", ".join(sorted(s or "?" for s in shas))
        )
    if legacy:
        issues.append("legacy .pt without embedded metadata: " + ", ".join(legacy))
    if enc_meta and train_now and model_train and train_now != model_train:
        issues.append(
            f"model trained on train.jsonl {model_train} but current is {train_now} "
            "- retrain before trusting Emojis"
        )
    for p in missing:
        issues.append(f"{p} missing")

    return {
        "ts": _ts(),
        "report_code": rm.get("sha"),
        "report_dirty": rm.get("dirty"),
        "model_sha": model_sha,
        "model_train_sha": model_train,
        "current_train_sha": train_now,
        "models": metas,
        "consistent": not issues,
        "issues": issues,
    }


def _length_distribution():
    lens = Counter()
    for d in _rows(DATA_PATH):
        lens[len(norm_text(str(d.get("text", ""))))] += 1
    return sorted(lens.items(), key=lambda kv: kv[0], reverse=True)


def _section_data():
    return {
        "records": {
            "data": len(_rows(DATA_PATH)),
            "train": len(_rows(TRAIN_PATH)),
            "eval": len(_rows(EVAL_PATH)),
        },
        "length_distribution": _length_distribution(),
    }


def _section_labels():
    return {"styles": len(STYLES), "emojis": len(EMOJIS)}


def _keyword_probe(enc, head):
    words = json.loads(Path(KEYWORDS_PATH).read_text(encoding="utf-8"))
    vocab = {e: i for i, e in enumerate(EMOJIS)}
    rows = []
    with torch.no_grad():
        for word, exp in words.items():
            ids = [vocab[e] for e in exp if e in vocab]
            emb = enc(text_to_tensor(norm_text(word)).unsqueeze(0))
            order = head(emb).squeeze(0).argsort(descending=True).tolist()
            rank = min((order.index(i) + 1 for i in ids), default=None)
            rows.append(
                {
                    "keyword": word,
                    "expected": exp,
                    "rank": rank,
                    "top5": [EMOJIS[j] for j in order[:KEYWORD_TOP]],
                }
            )
    scored = [r["rank"] for r in rows if r["rank"] is not None]
    n = len(scored) or 1
    misses = sorted(
        (r for r in rows if r["rank"] is not None and r["rank"] > KEYWORD_MISS_K),
        key=lambda r: r["rank"],
        reverse=True,
    )
    return {
        "n": len(scored),
        "acc_at_k": [sum(r <= k for r in scored) / n for k in EMOJI_KS],
        "misses": misses,
    }


def _section_emoji(enc, head, eval_records):
    if enc is None or head is None:
        return {}
    d = {}
    rows = [r for r in eval_records if r.emojis]
    if rows:
        vocab = {e: i for i, e in enumerate(EMOJIS)}
        texts = torch.stack([text_to_tensor(r.text) for r in rows])
        tgt = torch.zeros(len(rows), len(EMOJIS))
        for i, r in enumerate(rows):
            for e in r.emojis:
                tgt[i, vocab[e]] = 1.0
        with torch.no_grad():
            logits = head(enc(texts))
        d["eval"] = {
            "n": len(rows),
            "acc_at_k": [_acc_at_k(logits, tgt, k).mean().item() for k in EMOJI_KS],
        }
    d["keywords"] = _keyword_probe(enc, head)
    return d


def build_report(pt: Path, only: str = "", out: str = "report") -> Path:
    want = {s.strip() for s in only.split(",") if s.strip()} or {
        "data",
        "labels",
        "emoji",
    }
    enc_pt, emoji_pt = pt / "enc.pt", pt / "emoji.pt"
    prov = _provenance(pt)

    enc = emoji_head = None
    if "emoji" in want and enc_pt.exists():
        enc, err = _load(TextEncoder(), enc_pt)
        if err:
            prov["issues"].append(f"{enc_pt} could not load: {err}")
        if enc is not None and emoji_pt.exists():
            emoji_head, err = _load(EmojiHead(), emoji_pt)
            if err:
                prov["issues"].append(f"{emoji_pt} could not load: {err}")
    prov["consistent"] = not prov["issues"]

    eval_records = list(read(EVAL_PATH)) if "emoji" in want else []

    report = {
        "generated": datetime.now().isoformat(timespec="seconds"),
        "provenance": prov,
    }
    if "data" in want:
        report["data"] = _section_data()
    if "labels" in want:
        report["labels"] = _section_labels()
    if "emoji" in want:
        report["emoji"] = _section_emoji(enc, emoji_head, eval_records)

    out_dir = Path(out) / f"{prov['ts']}-{prov['model_sha']}"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (out_dir / "report.html").write_text(_render_html(report), encoding="utf-8")
    print(f"{out_dir}/report.html")
    return out_dir


# ------------------------------- rendering -------------------------------

_STYLE = """
:root{--ink:#1b1f24;--dim:#656b73;--line:#e2e5e9;--panel:#f5f6f8;--accent:#4b32d6;
--good-bg:#e6f6ec;--good-bd:#b2dec1;--good-fg:#157a3f;--warn-bg:#fdf2e2;
--warn-bd:#efd4a2;--warn-fg:#a25c07}
*{box-sizing:border-box}
body{font:17px/1.6 -apple-system,BlinkMacSystemFont,"Segoe UI",system-ui,sans-serif;
color:var(--ink);margin:0;background:#fff}
.wrap{max-width:920px;margin:0 auto;padding:44px 30px 130px}
h1{font-size:28px;margin:0 0 4px}
h2{font-size:24px;margin:64px 0 6px;padding-bottom:8px;border-bottom:2px solid var(--ink)}
h3{font-size:15px;text-transform:uppercase;letter-spacing:.04em;color:var(--dim);
margin:34px 0 12px}
.sub{color:var(--dim);font-size:15px}
.note{color:var(--dim);font-size:14.5px;margin:8px 0 0}
code{background:var(--panel);padding:1px 6px;border-radius:4px;font-size:15px}
.banner{border-radius:10px;padding:13px 17px;margin:20px 0 0;font-size:15.5px;
background:var(--good-bg);border:1px solid var(--good-bd)}
.banner.amber{background:var(--warn-bg);border-color:var(--warn-bd)}
.banner ul{margin:6px 0 0;padding-left:20px}
.counts{display:grid;grid-template-columns:repeat(3,1fr);gap:18px;margin:18px 0}
.counts.n2{grid-template-columns:repeat(2,1fr)}
.count{border:1px solid var(--line);border-radius:12px;padding:20px 22px}
.count .v{font-size:38px;font-weight:650;font-variant-numeric:tabular-nums}
.count .k{font-size:14px;color:var(--dim);margin-top:3px}
.chart{display:flex;align-items:flex-end;gap:5px;height:210px;
border-bottom:2px solid var(--ink);margin:26px 0 0}
.chart .bar{flex:1;background:var(--accent);border-radius:3px 3px 0 0;
position:relative;min-height:2px}
.chart .bar .num{position:absolute;bottom:100%;left:-4px;right:-4px;text-align:center;
font-size:11px;color:var(--dim);margin-bottom:3px;font-variant-numeric:tabular-nums}
.chart .bar .lbl{position:absolute;top:100%;left:50%;transform:translateX(-50%);
margin-top:8px;font-size:15px;white-space:nowrap}
.chart.rot{margin-bottom:62px}
.chart.rot .bar .lbl{left:auto;right:50%;transform:rotate(-40deg);
transform-origin:top right;font-size:12.5px;margin-top:9px}
.chart-cap{font-size:14px;color:var(--dim);margin:10px 0 0}
.linechart{display:block;width:100%;height:auto;margin:20px 0 0}
.linechart .gline{stroke:var(--line);stroke-width:1}
.linechart .gtext{font-size:11px;fill:var(--dim)}
.linechart .lline{fill:none;stroke:var(--accent);stroke-width:2.5}
.linechart .dot{fill:var(--accent)}
.linechart .vtext{font-size:11px;fill:var(--dim);font-variant-numeric:tabular-nums}
.linechart .xtext{font-size:12px;fill:var(--ink)}
table{border-collapse:collapse;width:100%;margin:16px 0;font-size:16px}
th,td{border-bottom:1px solid var(--line);padding:10px 12px;text-align:left}
th{font-size:13px;color:var(--dim);text-transform:uppercase;letter-spacing:.03em}
td.n,th.n{text-align:right;font-variant-numeric:tabular-nums}
tr:last-child td{border-bottom:none}
"""


def _esc(x) -> str:
    return html.escape(str(x))


def _fnum(n) -> str:
    return f"{n:,}"


def _bars(items, maxv, rotated=False) -> str:
    out = []
    for row in items:
        if row is None:
            out.append('<div class="gap"></div>')
            continue
        label, value = row[0], row[1]
        color = row[2] if len(row) > 2 else None
        h = (100 * value / maxv) if maxv else 0.0
        style = f"height:{max(h, 0.05):.2f}%"
        if color:
            style += f";background:{color}"
        num = "" if rotated else f'<span class="num">{_fnum(value)}</span>'
        out.append(
            f'<div class="bar" style="{style}">{num}'
            f'<span class="lbl">{_esc(label)}</span></div>'
        )
    cls = "chart rot" if rotated else "chart"
    return f'<div class="{cls}">{"".join(out)}</div>'


def _linechart(points, y_max=1.0) -> str:
    w, h, pad_l, pad_r, pad_t, pad_b = 760, 220, 34, 10, 22, 30
    plot_w, plot_h = w - pad_l - pad_r, h - pad_t - pad_b
    n = len(points)

    def px(i):
        return pad_l + (plot_w * i / (n - 1) if n > 1 else plot_w / 2)

    def py(v):
        return pad_t + plot_h * (1 - (v / y_max if y_max else 0.0))

    coords = [(px(i), py(v)) for i, (_, v) in enumerate(points)]
    grid = "".join(
        f'<line x1="{pad_l}" y1="{py(g * y_max):.1f}" x2="{w - pad_r}" '
        f'y2="{py(g * y_max):.1f}" class="gline"/>'
        f'<text x="{pad_l - 8}" y="{py(g * y_max) + 4:.1f}" class="gtext" '
        f'text-anchor="end">{g * y_max:.2f}</text>'
        for g in (0.0, 0.25, 0.5, 0.75, 1.0)
    )
    poly = " ".join(f"{cx:.1f},{cy:.1f}" for cx, cy in coords)
    dots = "".join(
        f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="3.5" class="dot"/>'
        f'<text x="{cx:.1f}" y="{cy - 9:.1f}" class="vtext" '
        f'text-anchor="middle">{v:.2f}</text>'
        f'<text x="{cx:.1f}" y="{h - 8}" class="xtext" '
        f'text-anchor="middle">{_esc(lbl)}</text>'
        for (lbl, v), (cx, cy) in zip(points, coords, strict=True)
    )
    return (
        f'<svg viewBox="0 0 {w} {h}" class="linechart">'
        f"{grid}"
        f'<polyline points="{poly}" class="lline"/>'
        f"{dots}</svg>"
    )


def _header_html(report) -> str:
    p = report["provenance"]
    train8 = (p["current_train_sha"] or "?")[:8]
    line = (
        f"{p['ts']} · model <code>{_esc(p['model_sha'])}</code> · "
        f"train.jsonl <code>{_esc(train8)}</code> · "
        f"report code <code>{_esc(p['report_code'])}</code>"
    )
    if p["consistent"]:
        banner = (
            f'<div class="banner">All model <code>.pt</code> from commit '
            f"<code>{_esc(p['model_sha'])}</code>, trained on the "
            f"<code>train.jsonl</code> on disk. Metrics are current.</div>"
        )
    else:
        lis = "".join(f"<li>{_esc(x)}</li>" for x in p["issues"])
        banner = (
            f'<div class="banner amber"><strong>Check provenance.</strong>'
            f"<ul>{lis}</ul></div>"
        )
    return f'<h1>emojic model report</h1><div class="sub">{line}</div>{banner}'


def _data_html(d) -> str:
    r = d["records"]
    cells = "".join(
        f'<div class="count"><div class="v">{_fnum(r[k])}</div>'
        f'<div class="k">records · {k}.jsonl</div></div>'
        for k in ("data", "train", "eval")
    )
    dist = d["length_distribution"]
    maxv = max((v for _, v in dist), default=1)
    bars = _bars([[str(length), count] for length, count in dist], maxv, rotated=True)
    return (
        f'<h2>Data</h2><div class="counts">{cells}</div>'
        "<h3>Text length distribution — data.jsonl (normalized, longest first)</h3>"
        f"{bars}"
    )


def _labels_html(d) -> str:
    cells = (
        f'<div class="count"><div class="v">{_fnum(d["styles"])}</div>'
        '<div class="k">styles</div></div>'
        f'<div class="count"><div class="v">{_fnum(d["emojis"])}</div>'
        '<div class="k">emojis</div></div>'
    )
    return f'<h2>Labels</h2><div class="counts n2">{cells}</div>'


def _emoji_html(d) -> str:
    if not d:
        return '<h2>Model — Emojis</h2><p class="note">enc.pt / emoji.pt not available.</p>'
    out = ["<h2>Model — Emojis</h2>"]
    if "eval" in d:
        e = d["eval"]
        points = list(zip((str(k) for k in EMOJI_KS), e["acc_at_k"], strict=True))
        out.append(f"<h3>Performance on eval.jsonl ({e['n']} rows)</h3>{_linechart(points)}")
    kw = d.get("keywords")
    if kw:
        points = list(zip((str(k) for k in EMOJI_KS), kw["acc_at_k"], strict=True))
        out.append(
            f"<h3>Performance on keywords.json ({kw['n']} words)</h3>{_linechart(points)}"
        )
        rows = "".join(
            f"<tr><td>{_esc(m['keyword'])}</td>"
            f"<td>{_esc(' '.join(m['expected']))}</td>"
            f'<td class="n">{m["rank"]}</td>'
            f"<td>{_esc(' '.join(m['top5']))}</td></tr>"
            for m in kw["misses"]
        )
        out.append(
            f"<h3>Missed keywords — target not in top {KEYWORD_MISS_K} "
            f"({len(kw['misses'])} of {kw['n']})</h3>"
            '<table><tr><th>Keyword</th><th>Expected</th><th class="n">Rank</th>'
            f"<th>Top {KEYWORD_TOP} predicted</th></tr>{rows}</table>"
        )
    return "".join(out)


def _render_html(report) -> str:
    body = [_header_html(report)]
    if "data" in report:
        body.append(_data_html(report["data"]))
    if "labels" in report:
        body.append(_labels_html(report["labels"]))
    if "emoji" in report:
        body.append(_emoji_html(report["emoji"]))
    return (
        '<!doctype html><meta charset="utf-8">'
        f"<title>emojic report — {report['provenance']['ts']}</title>"
        f'<style>{_STYLE}</style><div class="wrap">{"".join(body)}</div>\n'
    )


_app = typer.Typer(
    add_completion=False,
    context_settings={"help_option_names": ["-h", "--help"]},
)


@_app.command()
def main(
    pt: Path = typer.Option(..., "--pt", help="Folder containing enc.pt/emoji.pt."),
    only: str = "",
    out: str = "report",
) -> None:
    """Evaluate enc/emoji .pt + data files; write report/<ts>-<sha>/."""
    build_report(pt, only=only, out=out)


if __name__ == "__main__":
    _app()
