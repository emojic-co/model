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
from torch.nn.functional import normalize

from config import (
    EMOJIS,
    ENERGY_KEYWORD_MAX_TEXTS,
    ENERGY_Z_SAMPLES,
    SEED,
    STYLES,
    TEXT_EMBED_SIZE,
)
from data import (
    EVAL_PATH,
    TRAIN_PATH,
    colors2tensor,
    hex2rgb,
    read,
    text_to_tensor,
)
from data import (
    normalize as norm_text,
)
from model import ColorGen, EmojiHead, StyleHead, TextEncoder, rgb_to_oklab
from runmeta import load_pt, run_meta

DATA_PATH = "data.jsonl"
WORDS_PATH = "words.json"

Z = ENERGY_Z_SAMPLES
ENERGY_MAX = ENERGY_KEYWORD_MAX_TEXTS
COLOR_DELTA_E = 0.15
BASELINE_TEXTS = 200

CARD_COLORS = {
    "red": "#c0392b",
    "orange": "#e07a3f",
    "yellow": "#e8d44d",
    "green": "#4c9a52",
    "blue": "#4a6fd1",
    "purple": "#7a5aa8",
}
BG_ANCHOR_NAMES = ("red", "yellow", "green", "blue", "purple")

THRESHOLDS = {
    "emoji.acc@1": (0.50, 0.30),
    "emoji.acc@5": (0.70, 0.50),
    "emoji.acc@10": (0.80, 0.60),
    "emoji.MRR@10": (0.50, 0.35),
    "style.acc@1": (0.60, 0.45),
    "style.acc@5": (0.85, 0.70),
    "style.MRR@5": (0.60, 0.45),
    "style.mAP@5": (0.60, 0.45),
}
ENERGY_BANDS = {"gen": (0.15, 0.25), "gap": (0.03, 0.10)}


def _seeded():
    return torch.Generator().manual_seed(SEED)


Z_BANK = normalize(torch.randn(Z, TEXT_EMBED_SIZE, generator=_seeded()), dim=-1)


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


def _emoji_lists(rows: tuple) -> list:
    out = []
    for d in rows:
        e = d.get("emojis", "")
        if isinstance(e, list):
            out.append([t for t in e if t])
        else:
            out.append([t for t in str(e).split() if t])
    return out


def _load(mod, path):
    sd, _ = load_pt(path)
    try:
        mod.load_state_dict(sd)
    except RuntimeError as exc:
        return None, str(exc).splitlines()[0]
    mod.eval()
    return mod, None


def _subsample(items, cap):
    if len(items) <= cap:
        return list(items)
    idx = torch.randperm(len(items), generator=_seeded())[:cap].tolist()
    return [items[i] for i in idx]


def _hex_lab(hx: str) -> torch.Tensor:
    rgb = torch.tensor(hex2rgb(hx), dtype=torch.float32) - 127.5
    return rgb_to_oklab(rgb)


def _acc_at_k(logits, target, k):
    k = min(k, logits.size(-1))
    top = logits.topk(k, dim=-1).indices
    return target.gather(1, top).amax(dim=-1)


def _mrr_at_k(logits, target, k):
    k = min(k, logits.size(-1))
    top = logits.topk(k, dim=-1).indices
    rel = target.gather(1, top)
    ranks = torch.arange(1, k + 1, device=logits.device)
    return (rel / ranks).amax(dim=-1)


def _ap_at_k(logits, target, k):
    k = min(k, logits.size(-1))
    top = logits.topk(k, dim=-1).indices
    rel = target.gather(1, top)
    ranks = torch.arange(1, k + 1, device=logits.device)
    prec = rel.cumsum(dim=-1) / ranks
    denom = target.sum(dim=-1).clamp(max=k).clamp(min=1.0)
    return (prec * rel).sum(dim=-1) / denom


def _energy(x, y):
    m = "donot_use_mm_for_euclid_dist"
    xy = torch.cdist(x, y, compute_mode=m).mean()
    xx = torch.cdist(x, x, compute_mode=m).mean()
    yy = torch.cdist(y, y, compute_mode=m).mean()
    return (2 * xy - xx - yy).clamp(min=0.0).sqrt()


def _split_energy(pts):
    n = pts.size(0)
    half = n // 2
    perm = torch.randperm(n, generator=_seeded())
    return _energy(pts[perm[:half]], pts[perm[half : 2 * half]])


def _bg_mean_lab(rgb9):
    return rgb_to_oklab(rgb9).reshape(-1, 3, 3)[:, :2, :].mean(1)


def _provenance():
    rm = run_meta()
    paths = ["enc.pt", "style.pt", "emoji.pt", "gen.pt", "tst.pt"]
    metas = {p: (load_pt(p)[1] if Path(p).exists() else None) for p in paths}
    present = {p: m for p, m in metas.items() if m}
    missing = [p for p in paths if not Path(p).exists()]
    legacy = [p for p in paths if Path(p).exists() and metas[p] is None]
    shas = {m.get("sha") for m in present.values()}
    enc_meta = metas.get("enc.pt")
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
            "- retrain before trusting Emojis / Styles / Colors"
        )
    for p in ("enc.pt", "style.pt", "emoji.pt", "gen.pt"):
        if p in missing:
            skipped = " - Colors section skipped" if p == "gen.pt" else ""
            issues.append(f"{p} missing{skipped}")

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


def _section_data():
    return {
        "records": {
            "data": len(_rows(DATA_PATH)),
            "train": len(_rows(TRAIN_PATH)),
            "eval": len(_rows(EVAL_PATH)),
        }
    }


def _keyword_probe(enc, head, limit=10):
    words = json.loads(Path(WORDS_PATH).read_text(encoding="utf-8"))
    vocab = {e: i for i, e in enumerate(EMOJIS)}
    rows = []
    with torch.no_grad():
        for w in words:
            word = w["word"]
            exp = w.get("emojis", [])
            ids = [vocab[e] for e in exp if e in vocab]
            emb = enc(text_to_tensor(norm_text(word)).unsqueeze(0))
            order = head(emb).squeeze(0).argsort(descending=True).tolist()
            rank = min((order.index(i) + 1 for i in ids), default=None)
            rows.append(
                {
                    "keyword": word,
                    "expected": exp,
                    "rank": rank,
                    "top3": [EMOJIS[j] for j in order[:3]],
                }
            )
    scored = [r["rank"] for r in rows if r["rank"] is not None]
    n = len(scored) or 1
    rows.sort(key=lambda r: (r["rank"] is None, r["rank"] or 0), reverse=True)
    return {
        "n": len(scored),
        "acc@1": sum(r <= 1 for r in scored) / n,
        "acc@3": sum(r <= 3 for r in scored) / n,
        "acc@5": sum(r <= 5 for r in scored) / n,
        "acc@10": sum(r <= 10 for r in scored) / n,
        "MRR": sum(1.0 / r for r in scored) / n,
        "worst": limit,
        "words": rows,
    }


def _section_emoji(enc, head, eval_records):
    d = {
        "distinct": {
            name: len({t for lst in _emoji_lists(_rows(path)) for t in lst})
            for name, path in (
                ("data", DATA_PATH),
                ("train", TRAIN_PATH),
                ("eval", EVAL_PATH),
            )
        }
    }
    df = Counter()
    for lst in _emoji_lists(_rows(TRAIN_PATH)):
        for t in set(lst):
            df[t] += 1
    d["top"] = [[e, n] for e, n in df.most_common(10)]
    d["bottom"] = [[e, n] for e, n in sorted(df.items(), key=lambda kv: (kv[1], kv[0]))[:10]]
    d["thin"] = sum(1 for e in EMOJIS if df.get(e, 0) < 20)

    if enc is None or head is None:
        return d

    rows = [r for r in eval_records if r.emojis]
    if not rows:
        return d
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
        "acc@1": _acc_at_k(logits, tgt, 1).mean().item(),
        "acc@5": _acc_at_k(logits, tgt, 5).mean().item(),
        "acc@10": _acc_at_k(logits, tgt, 10).mean().item(),
        "MRR@10": _mrr_at_k(logits, tgt, 10).mean().item(),
    }
    d["keywords"] = _keyword_probe(enc, head)
    return d


def _section_style(enc, head, train_records, eval_records):
    c = Counter()
    for r in train_records:
        for s in r.styles:
            c[s] += 1
    d = {
        "distribution": [
            [s, c.get(s, 0)] for s in sorted(c, key=lambda k: c[k], reverse=True)
        ]
    }
    if enc is None or head is None or not eval_records:
        return d
    idx = {s: i for i, s in enumerate(STYLES)}
    texts = torch.stack([text_to_tensor(r.text) for r in eval_records])
    tgt = torch.zeros(len(eval_records), len(STYLES))
    for i, r in enumerate(eval_records):
        for s in r.styles:
            tgt[i, idx[s]] = 1.0
    with torch.no_grad():
        logits = head(enc(texts))
    d["eval"] = {
        "n": len(eval_records),
        "acc@1": _acc_at_k(logits, tgt, 1).mean().item(),
        "acc@5": _acc_at_k(logits, tgt, 5).mean().item(),
        "MRR@5": _mrr_at_k(logits, tgt, 5).mean().item(),
        "mAP@5": _ap_at_k(logits, tgt, 5).mean().item(),
    }
    return d


def _color_card(name, hx, enc, gen, eval_records, train_records):
    a = _hex_lab(hx)
    with torch.no_grad():
        cond = enc(text_to_tensor(norm_text(name)).unsqueeze(0)).repeat(Z, 1)
        bg = _bg_mean_lab(gen(cond, Z_BANK))
    hit = (torch.linalg.vector_norm(bg - a, dim=-1) < COLOR_DELTA_E).float().mean().item()

    base_rows = _subsample(eval_records, BASELINE_TEXTS)
    with torch.no_grad():
        bcond = enc(torch.stack([text_to_tensor(r.text) for r in base_rows]))
        bbg = _bg_mean_lab(
            gen(bcond.repeat_interleave(Z, 0), Z_BANK.repeat(len(base_rows), 1))
        )
    base = (torch.linalg.vector_norm(bbg - a, dim=-1) < COLOR_DELTA_E).float().mean().item()
    lift = hit - base

    grp = _subsample([r for r in train_records if name in r.text], ENERGY_MAX)
    egap = None
    if len(grp) >= 32:
        greal = rgb_to_oklab(torch.stack([colors2tensor(r.colors) for r in grp]))
        with torch.no_grad():
            gcond = enc(torch.stack([text_to_tensor(r.text) for r in grp]))
            gfake = rgb_to_oklab(
                gen(gcond.repeat_interleave(Z, 0), Z_BANK.repeat(len(grp), 1))
            )
        egap = (_energy(greal, gfake) - _split_energy(greal)).item()

    verdict = "text-driven" if lift >= 0.25 else "weak" if lift >= 0.10 else "random"
    return {
        "name": name,
        "anchor": hx,
        "hit_rate": hit,
        "baseline": base,
        "lift": lift,
        "energy_gap": egap,
        "verdict": verdict,
    }


def _section_color(enc, gen, train_records, eval_records):
    anchors = torch.stack([_hex_lab(CARD_COLORS[n]) for n in BG_ANCHOR_NAMES])
    bg = (
        torch.tensor(
            [[hex2rgb(h) for h in r.colors[:2]] for r in train_records],
            dtype=torch.float32,
        )
        - 127.5
    )
    mean = rgb_to_oklab(bg).mean(1)
    dist, arg = torch.cdist(mean, anchors).min(dim=-1)
    names = [
        BG_ANCHOR_NAMES[j] if dist[i] < COLOR_DELTA_E else "other"
        for i, j in enumerate(arg.tolist())
    ]
    counts = Counter(names)
    ordered = sorted(
        ((k, v) for k, v in counts.items() if k != "other"),
        key=lambda kv: kv[1],
        reverse=True,
    )
    ordered.append(("other", counts.get("other", 0)))
    d = {"bg_distribution": [[k, v] for k, v in ordered]}

    if enc is None or gen is None:
        return d

    ev = _subsample(eval_records, ENERGY_MAX)
    real = rgb_to_oklab(torch.stack([colors2tensor(r.colors) for r in ev]))
    with torch.no_grad():
        cond = enc(torch.stack([text_to_tensor(r.text) for r in ev]))
        fake = rgb_to_oklab(gen(cond.repeat_interleave(Z, 0), Z_BANK.repeat(len(ev), 1)))
    ge = _energy(real, fake).item()
    rf = _split_energy(real).item()
    d["energy"] = {"gen": ge, "ref": rf, "gap": ge - rf}
    d["cards"] = [
        _color_card(n, CARD_COLORS[n], enc, gen, eval_records, train_records)
        for n in CARD_COLORS
    ]
    return d


def build_report(only: str = "", out: str = "report") -> Path:
    want = {s.strip() for s in only.split(",") if s.strip()} or {
        "data",
        "emoji",
        "style",
        "color",
    }
    prov = _provenance()

    enc = emoji_head = style_head = gen = None
    if want & {"emoji", "style", "color"} and Path("enc.pt").exists():
        enc, err = _load(TextEncoder(), "enc.pt")
        if err:
            prov["issues"].append(f"enc.pt could not load: {err}")
    if "emoji" in want and enc is not None and Path("emoji.pt").exists():
        emoji_head, err = _load(EmojiHead(), "emoji.pt")
        if err:
            prov["issues"].append(f"emoji.pt could not load: {err}")
    if "style" in want and enc is not None and Path("style.pt").exists():
        style_head, err = _load(StyleHead(), "style.pt")
        if err:
            prov["issues"].append(f"style.pt could not load: {err}")
    if "color" in want and enc is not None and Path("gen.pt").exists():
        gen, err = _load(ColorGen(), "gen.pt")
        if err:
            prov["issues"].append(f"gen.pt could not load: {err}")
    prov["consistent"] = not prov["issues"]

    train_records = list(read(TRAIN_PATH)) if want & {"style", "color"} else []
    eval_records = list(read(EVAL_PATH)) if want & {"emoji", "style", "color"} else []

    report = {
        "generated": datetime.now().isoformat(timespec="seconds"),
        "provenance": prov,
    }
    if "data" in want:
        report["data"] = _section_data()
    if "emoji" in want:
        report["emoji"] = _section_emoji(enc, emoji_head, eval_records)
    if "style" in want:
        report["style"] = _section_style(enc, style_head, train_records, eval_records)
    if "color" in want:
        report["color"] = _section_color(enc, gen, train_records, eval_records)

    out_dir = Path(out) / f"{prov['ts']}-{prov['model_sha']}"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (out_dir / "report.html").write_text(_render_html(report), encoding="utf-8")
    print(f"wrote {out_dir}/report.html")
    return out_dir


# ------------------------------- rendering -------------------------------

_STYLE = """
:root{--ink:#1b1f24;--dim:#656b73;--line:#e2e5e9;--panel:#f5f6f8;--accent:#4b32d6;
--good-bg:#e6f6ec;--good-bd:#b2dec1;--good-fg:#157a3f;--warn-bg:#fdf2e2;
--warn-bd:#efd4a2;--warn-fg:#a25c07;--bad-bg:#fbe8e8;--bad-bd:#eeb8b8;--bad-fg:#b12222}
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
.count{border:1px solid var(--line);border-radius:12px;padding:20px 22px}
.count .v{font-size:38px;font-weight:650;font-variant-numeric:tabular-nums}
.count .k{font-size:14px;color:var(--dim);margin-top:3px}
.metrics{display:grid;grid-template-columns:repeat(4,1fr);gap:18px;margin:20px 0}
.metrics.n3{grid-template-columns:repeat(3,1fr)}
.metric{border-radius:16px;padding:24px 22px;border:1px solid}
.metric .v{font-size:50px;font-weight:700;line-height:1;font-variant-numeric:tabular-nums}
.metric .k{font-size:15px;margin-top:10px;font-weight:650}
.metric .h{font-size:13px;opacity:.8;margin-top:3px}
.metric.good{background:var(--good-bg);border-color:var(--good-bd);color:var(--good-fg)}
.metric.warn{background:var(--warn-bg);border-color:var(--warn-bd);color:var(--warn-fg)}
.metric.bad{background:var(--bad-bg);border-color:var(--bad-bd);color:var(--bad-fg)}
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
.chart .gap{flex:0 0 26px;background:none;min-height:0}
.chart-cap{font-size:14px;color:var(--dim);margin:10px 0 0}
.colorgrid{display:grid;grid-template-columns:repeat(3,1fr);gap:18px;margin:20px 0}
.colorcard{border-radius:16px;padding:22px 20px;min-height:160px;display:flex;
flex-direction:column;justify-content:space-between}
.colorcard .name{font-size:22px;font-weight:700;text-transform:capitalize}
.colorcard .verdict{font-size:16px;font-weight:650;margin-top:5px}
.colorcard .nums{font-size:13.5px;opacity:.93;margin-top:14px;line-height:1.55}
.colorcard .badge{display:inline-block;background:rgba(255,255,255,.24);
border-radius:999px;padding:3px 11px;font-size:13.5px;font-weight:700}
.colorcard.dark-text{color:#16181d}
.colorcard.dark-text .badge{background:rgba(0,0,0,.13)}
.colorcard.light-text{color:#fff}
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


def _mcard(label, value, cls, hint="") -> str:
    h = f'<div class="h">{_esc(hint)}</div>' if hint else ""
    return (
        f'<div class="metric {cls}"><div class="v">{value}</div>'
        f'<div class="k">{_esc(label)}</div>{h}</div>'
    )


def _cls(key, val) -> str:
    good, warn = THRESHOLDS[key]
    return "good" if val >= good else "warn" if val >= warn else "bad"


def _ecls(key, val) -> str:
    good, warn = ENERGY_BANDS[key]
    return "good" if val <= good else "warn" if val <= warn else "bad"


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


def _light_text(hx) -> bool:
    r, g, b = hex2rgb(hx)
    return (0.2126 * r + 0.7152 * g + 0.0722 * b) < 150


def _card_html(c) -> str:
    tcls = "light-text" if _light_text(c["anchor"]) else "dark-text"
    sym = {
        "text-driven": "✓ text-driven",
        "weak": "~ weak",
        "random": "✗ ≈ random",
    }[c["verdict"]]
    eg = "n/a" if c["energy_gap"] is None else f"{c['energy_gap']:+.2f}"
    return (
        f'<div class="colorcard {tcls}" style="background:{c["anchor"]}">'
        f'<div><div class="name">{_esc(c["name"])}</div>'
        f'<div class="verdict">{sym}</div></div>'
        f'<div class="nums">energy gap {eg} · hit {c["hit_rate"]:.2f} '
        f"vs {c['baseline']:.2f} random<br>"
        f'<span class="badge">lift {c["lift"]:+.2f}</span></div></div>'
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
    return f'<h2>Data</h2><div class="counts">{cells}</div>'


def _emoji_html(d) -> str:
    out = ["<h2>Emojis</h2>", "<h3>Distinct emojis</h3>", '<div class="counts">']
    for k in ("data", "train", "eval"):
        out.append(
            f'<div class="count"><div class="v">{_fnum(d["distinct"][k])}</div>'
            f'<div class="k">in {k}.jsonl</div></div>'
        )
    out.append("</div>")

    out.append("<h3>Frequency in train.jsonl — top 10 &amp; bottom 10 (shared scale)</h3>")
    maxv = d["top"][0][1] if d["top"] else 1
    items = list(d["top"]) + [None] + list(d["bottom"])
    out.append(_bars(items, maxv))
    out.append(
        f'<p class="chart-cap">Bottom-10 counts are '
        f"{d['bottom'][0][1] if d['bottom'] else 0}–"
        f"{d['bottom'][-1][1] if d['bottom'] else 0} rows each — slivers at this "
        f"scale. {d['thin']} / {len(EMOJIS)} leaderboard emojis have &lt;20 rows.</p>"
    )

    if "eval" in d:
        e = d["eval"]
        cards = (
            _mcard("acc@1", f"{e['acc@1']:.2f}", _cls("emoji.acc@1", e["acc@1"]))
            + _mcard("acc@5", f"{e['acc@5']:.2f}", _cls("emoji.acc@5", e["acc@5"]))
            + _mcard("acc@10", f"{e['acc@10']:.2f}", _cls("emoji.acc@10", e["acc@10"]))
            + _mcard("MRR@10", f"{e['MRR@10']:.2f}", _cls("emoji.MRR@10", e["MRR@10"]))
        )
        out.append(
            f"<h3>Retrieval performance — eval.jsonl ({e['n']} rows)</h3>"
            f'<div class="metrics">{cards}</div>'
        )
    if "keywords" in d:
        kw = d["keywords"]
        rows = "".join(
            f"<tr><td>{_esc(k['keyword'])}</td><td>{_esc(' '.join(k['expected']))}</td>"
            f'<td class="n">{"—" if k["rank"] is None else k["rank"]}</td>'
            f"<td>{_esc(' '.join(k['top3']))}</td></tr>"
            for k in kw["words"][: kw["worst"]]
        )
        out.append(
            f"<h3>Keyword probe — words.json ({kw['n']} words)</h3>"
            f'<p class="note">acc@1 {kw["acc@1"]:.2f} · acc@5 {kw["acc@5"]:.2f} · '
            f"acc@10 {kw['acc@10']:.2f} · MRR {kw['MRR']:.2f}</p>"
            "<h3>Worst 10 keywords</h3>"
            '<table><tr><th>Keyword</th><th>Expected</th><th class="n">Rank</th>'
            f"<th>Top 3 predicted</th></tr>{rows}</table>"
        )
    return "".join(out)


def _style_html(d) -> str:
    out = ["<h2>Styles</h2>", "<h3>Distribution in train.jsonl — all labels</h3>"]
    dist = d["distribution"]
    maxv = dist[0][1] if dist else 1
    out.append(_bars(dist, maxv, rotated=True))
    if dist and dist[-1][1]:
        out.append(
            f'<p class="chart-cap">Most / least common = '
            f"{dist[0][1] / dist[-1][1]:.0f}×.</p>"
        )
    if "eval" in d:
        e = d["eval"]
        cards = (
            _mcard("acc@1", f"{e['acc@1']:.2f}", _cls("style.acc@1", e["acc@1"]))
            + _mcard("acc@5", f"{e['acc@5']:.2f}", _cls("style.acc@5", e["acc@5"]))
            + _mcard("MRR@5", f"{e['MRR@5']:.2f}", _cls("style.MRR@5", e["MRR@5"]))
            + _mcard("mAP@5", f"{e['mAP@5']:.2f}", _cls("style.mAP@5", e["mAP@5"]))
        )
        out.append(
            f"<h3>Retrieval performance — eval.jsonl ({e['n']} rows)</h3>"
            f'<div class="metrics">{cards}</div>'
        )
    return "".join(out)


def _color_html(d) -> str:
    out = ["<h2>Colors</h2>", "<h3>Background colour distribution in train.jsonl</h3>"]
    out.append(
        '<p class="note">Each record\'s two <code>bg</code> stops averaged in OKLab, '
        f"assigned to the nearest anchor within ΔE &lt; {COLOR_DELTA_E:.2f}, else "
        "<em>other</em>. Sorted by count; <em>other</em> last.</p>"
    )
    dist = d["bg_distribution"]
    maxv = max((v for _, v in dist), default=1)
    palette = {**CARD_COLORS, "other": "#9aa0a8"}
    out.append(_bars([[k, v, palette.get(k, "#9aa0a8")] for k, v in dist], maxv))

    if "energy" in d:
        e = d["energy"]
        cards = (
            _mcard("gen ↔ real", f"{e['gen']:.3f}", _ecls("gen", e["gen"]))
            + _mcard("real ↔ real (floor)", f"{e['ref']:.3f}", "good")
            + _mcard("gap", f"{e['gap']:+.3f}", _ecls("gap", e["gap"]), "0 = perfect")
        )
        out.append(
            "<h3>Palette realism — OKLab energy distance</h3>"
            f'<div class="metrics n3">{cards}</div>'
        )
    if "cards" in d:
        out.append("<h3>Per colour — driven by the text, or random?</h3>")
        out.append(
            '<p class="note">Feed the colour name into the model; measure how close '
            "generated backgrounds land to that colour vs. a random-text baseline.</p>"
        )
        out.append('<div class="colorgrid">')
        out.append("".join(_card_html(c) for c in d["cards"]))
        out.append("</div>")
    return "".join(out)


def _render_html(report) -> str:
    body = [_header_html(report)]
    if "data" in report:
        body.append(_data_html(report["data"]))
    if "emoji" in report:
        body.append(_emoji_html(report["emoji"]))
    if "style" in report:
        body.append(_style_html(report["style"]))
    if "color" in report:
        body.append(_color_html(report["color"]))
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
def main(only: str = "", out: str = "report") -> None:
    """Evaluate enc/style/emoji/gen .pt + data files; write report/<ts>-<sha>/."""
    build_report(only=only, out=out)


if __name__ == "__main__":
    _app()
