from model.runmeta import load_pt, run_meta
from model.pred import predict as _predict
from model.model import (
    ColorGen,
    EmojiHead,
    StyleHead,
    TextEncoder,
)
from model.kwtokens import word_count
from model.export_onnx import CONST_Z
from model.data import normalize as norm_text
from model.data import EVAL_PATH, TRAIN_PATH, read, text_to_tensor
from model.config import (
    EMOJIS,
    ENCODER_CHANNELS,
    ENCODER_DILATION,
    ENCODER_KERNEL_SIZE,
    HIDDEN_SIZE_GEN,
    MAX_TEXT_LEN,
    SEED,
    STYLES,
)
from model.color import COLOR_SHIFT, energy_distance, rgb_to_oklab
from files import (
    CLDR_BASELINE_JSON,
    COLORS_JSONL,
    DATA_JSONL,
    GOALS_YML,
    GROUP_JSON,
    II_JSON,
    KEYWORDS_JSONL,
    REPORT_DIR,
    TERMS_JSONL,
    PtFile,
)
from torch import nn
import yaml
import typer
import torch
import html
import json
import random
import re
import subprocess
import sys
from collections import Counter
from datetime import datetime
from functools import cache
from itertools import accumulate
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


DATA_PATH = DATA_JSONL
REPO_ROOT = Path(__file__).resolve().parent.parent

EMOJI_KS = list(range(1, 11))
ACC_K_INDEX = {"acc@1": 0, "acc@5": 4, "acc@10": 9}
CARD_PURE_THRESHOLD_RGB = 0.251
CARD_PURE_THRESHOLD_L = 0.6
CARD_COLORS = ("red", "green", "blue", "dark", "bright")
GOLD_PER_COLOR = 25
KEYWORD_FAILS_MAX_ROWS = 100
EVAL_SAMPLE_N = 40
SENS_TEXT_N = 40
SENS_SWATCH_N = 10
COLOR_KEYWORD_SAMPLE_N = 300
COLOR_KEYWORD_EXAMPLE_N = 10
LANG_PAIR_TOPK = 5
LANG_PAIR_AGREE_THRESH = 0.2
LANG_PAIR_SWATCH_N = 10
LANG_PAIRS = (
    ("green", "ירוק"),
    ("blue", "כחול"),
    ("yellow", "צהוב"),
    ("purple", "סגול"),
    ("black", "שחור"),
    ("white", "לבן"),
    ("pink", "ורוד"),
    ("brown", "חום"),
    ("happy", "שמח"),
    ("angry", "כועס"),
    ("tired", "עייף"),
    ("dog", "כלב"),
    ("coffee", "קפה"),
    ("pizza", "פיצה"),
    ("cake", "עוגה"),
    ("star", "כוכב"),
    ("water", "מים"),
    ("rain", "גשם"),
    ("night", "לילה"),
    ("morning", "בוקר"),
    ("beach", "חוף"),
    ("snow", "שלג"),
    ("wind", "רוח"),
    ("king", "מלך"),
    ("queen", "מלכה"),
    ("friend", "חבר"),
    ("book", "ספר"),
    ("phone", "טלפון"),
    ("car", "מכונית"),
    ("house", "בית"),
    ("gift", "מתנה"),
    ("guitar", "גיטרה"),
    ("bicycle", "אופניים"),
    ("camera", "מצלמה"),
    ("key", "מפתח"),
    ("clock", "שעון"),
    ("umbrella", "מטריה"),
    ("birthday", "יום הולדת"),
)
EVAL_SAMPLE_PT_FILES = (
    PtFile.ENC,
    PtFile.STYLE,
    PtFile.EMOJI,
    PtFile.GEN,
)


def _ts() -> str:
    return datetime.now().strftime("%y-%m-%d-%H-%M")


@cache
def _rows(path: Path) -> tuple:
    if not path.exists():
        return ()
    out = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return tuple(out)


def _load(mod: nn.Module, path: Path):
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
    enc_pt, emoji_pt = PtFile.ENC.in_dir(pt), PtFile.EMOJI.in_dir(pt)
    style_pt, gen_pt = PtFile.STYLE.in_dir(pt), PtFile.GEN.in_dir(pt)
    rm = run_meta()
    paths = [enc_pt, emoji_pt, style_pt, gen_pt]
    metas = {str(p): (load_pt(p)[1] if p.exists() else None) for p in paths}
    present = {p: m for p, m in metas.items() if m}
    missing = [p for p in paths if not p.exists()]
    legacy = [p for p in paths if p.exists() and metas[str(p)] is None]
    shas = {m.get("sha") for m in present.values()}
    enc_meta = metas.get(str(enc_pt))
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
        issues.append(
            "legacy .pt without embedded metadata: "
            + ", ".join(str(p) for p in legacy)
        )
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


def _length_distribution(path: Path):
    lens = Counter()
    for d in _rows(path):
        lens[len(norm_text(str(d.get("text", ""))))] += 1
    return sorted(lens.items(), key=lambda kv: kv[0], reverse=True)


def _section_data():
    return {
        "records": {
            "data": len(_rows(DATA_PATH)),
            "train": len(_rows(TRAIN_PATH)),
            "eval": len(_rows(EVAL_PATH)),
        },
        "length_distribution": _length_distribution(TRAIN_PATH),
        "keywords_length_distribution": _length_distribution(KEYWORDS_JSONL),
        "terms_length_distribution": _length_distribution(TERMS_JSONL),
        "max_text_len": MAX_TEXT_LEN,
    }


def _section_labels():
    return {"styles": len(STYLES), "emojis": len(EMOJIS)}


def _probe(words, enc, head):
    vocab = {e: i for i, e in enumerate(EMOJIS)}
    scored = []
    with torch.no_grad():
        for word, exp in words.items():
            ids = [vocab[e] for e in exp if e in vocab]
            if not ids:
                continue
            emb = enc(text_to_tensor(norm_text(word)).unsqueeze(0))
            logits = head(emb)
            order = logits.squeeze(0).argsort(descending=True).tolist()
            scored.append(min(order.index(i) + 1 for i in ids))
    n = len(scored) or 1
    return {
        "n": len(scored),
        "total": len(words),
        "acc_at_k": [sum(r <= k for r in scored) / n for k in EMOJI_KS],
    }


@cache
def _ii_json() -> dict:
    if not II_JSON.exists():
        return {}
    return json.loads(II_JSON.read_text(encoding="utf-8"))


@cache
def _flex_keyword_candidates() -> tuple:
    from model.kwtokens import query_tokens

    vocab = set(EMOJIS)
    out = []
    for kw, emojis in _ii_json().items():
        if not 3 <= len(kw) <= 6:
            continue
        if query_tokens(kw) != [kw]:
            continue
        tgt = [e for e in emojis if e in vocab]
        if not tgt:
            continue
        out.append((kw, tgt))
    return tuple(sorted(out))


def _section_keywords_flex(enc, emoji_head) -> dict:
    cands = _flex_keyword_candidates()
    if enc is None or emoji_head is None or not cands:
        return {}
    idx = {e: i for i, e in enumerate(EMOJIS)}
    with torch.no_grad():
        texts = torch.stack([text_to_tensor(norm_text(kw)) for kw, _ in cands])
        order = emoji_head(enc(texts)).argsort(dim=-1, descending=True)
    ranked = []
    for row, (kw, tgt) in enumerate(cands):
        pos = order[row].tolist()
        rank = min(pos.index(idx[e]) + 1 for e in tgt)
        ranked.append(
            {
                "kw": kw,
                "emojis": tgt,
                "top5": [EMOJIS[i] for i in pos[:5]],
                "rank": rank,
            }
        )
    ranked.sort(key=lambda r: r["rank"], reverse=True)
    return {
        "candidates": len(ranked),
        "missed": sum(1 for r in ranked if r["rank"] > 10),
        "ranked": ranked,
    }


def _keywords_rows() -> tuple:
    vocab = set(EMOJIS)
    out = []
    for d in _rows(KEYWORDS_JSONL):
        word = str(d.get("text", ""))
        emojis = [e for e in str(d.get("emojis", "")).split() if e in vocab]
        if not word or not emojis or word_count(word) != 1:
            continue
        out.append((word, emojis, str(d.get("src", ""))))
    return tuple(out)


def _section_keyword_fails(enc, emoji_head) -> dict:
    rows = _keywords_rows()
    if enc is None or emoji_head is None or not rows:
        return {}
    idx = {e: i for i, e in enumerate(EMOJIS)}
    with torch.no_grad():
        texts = torch.stack([text_to_tensor(norm_text(w)) for w, _, _ in rows])
        order = emoji_head(enc(texts)).argsort(dim=-1, descending=True)
    fails = []
    for row, (word, exp, src) in enumerate(rows):
        pos = order[row].tolist()
        ids = [idx[e] for e in exp]
        rank = min(pos.index(i) + 1 for i in ids)
        if rank == 1:
            continue
        fails.append(
            {
                "kw": word,
                "src": src,
                "target": " ".join(exp),
                "predicted": " ".join(EMOJIS[i] for i in pos[:10]),
                "rank": rank,
            }
        )
    fails.sort(key=lambda r: r["rank"], reverse=True)
    return {"n": len(rows), "failed": len(fails), "rows": fails}


def _words_from(path: Path) -> dict[str, list[str]]:
    vocab = set(EMOJIS)
    words: dict[str, list[str]] = {}
    for d in _rows(path):
        text = str(d.get("text", ""))
        if not text or len(norm_text(text)) > MAX_TEXT_LEN:
            continue
        targets = words.setdefault(text, [])
        for e in str(d.get("emojis", "")).split():
            if e in vocab and e not in targets:
                targets.append(e)
    return {w: es for w, es in words.items() if es}


def _section_keyword_probe(enc, head) -> dict:
    words = _words_from(KEYWORDS_JSONL)
    if enc is None or head is None or not words:
        return {}
    return dict(_probe(words, enc, head))


def _section_term_probe(enc, head) -> dict:
    words = _words_from(TERMS_JSONL)
    if enc is None or head is None or not words:
        return {}
    return dict(_probe(words, enc, head))


_BLOCK_CAPACITY_SOURCES = [
    ("keywords", KEYWORDS_JSONL),
    ("terms", TERMS_JSONL),
    ("eval", EVAL_PATH),
]


def _section_style_dist(enc, style_head, eval_records) -> dict:
    rows = [r for r in eval_records if r.styles]
    if enc is None or style_head is None or not rows:
        return {}
    texts = torch.stack([text_to_tensor(r.text) for r in rows])
    with torch.no_grad():
        logits = style_head(enc(texts))
    pred_counts = Counter(STYLES[i] for i in logits.argmax(dim=-1).tolist())
    gt_counts = Counter(s for r in rows for s in r.styles)
    n = len(rows)
    dist = [
        {"style": s, "gt": gt_counts.get(
            s, 0) / n, "pred": pred_counts.get(s, 0) / n}
        for s in STYLES
    ]
    dist.sort(key=lambda d: d["gt"], reverse=True)
    return {"n": n, "dist": dist}


def _section_block_capacity(enc) -> dict:
    if enc is None:
        return {}
    ends = list(accumulate(ENCODER_CHANNELS))
    bounds = list(zip([0, *ends[:-1]], ends, strict=True))
    weight = enc.proj[1].weight
    rows = []
    with torch.no_grad():
        for source, path in _BLOCK_CAPACITY_SOURCES:
            texts = [r.text for r in read(path)]
            if not texts:
                continue
            pooled = enc.pool(torch.stack([text_to_tensor(t) for t in texts]))
            q_norm = (pooled @ weight.t()).norm(dim=-1).clamp_min(1e-12)
            for i, ((start, end), d) in enumerate(
                zip(bounds, ENCODER_DILATION, strict=True)
            ):
                w_slice = weight[:, start:end]
                a_slice = pooled[:, start:end]
                col_norms = w_slice.norm(dim=0)
                contrib_norm = (a_slice @ w_slice.t()).norm(dim=-1)
                rows.append(
                    {
                        "source": source,
                        "n": len(texts),
                        "block": i,
                        "range": f"{start}:{end}",
                        "dilation": d,
                        "w_norm_mean": col_norms.mean().item(),
                        "w_norm_max": col_norms.max().item(),
                        "act_rms": (a_slice.norm(dim=-1) / (end - start) ** 0.5)
                        .mean()
                        .item(),
                        "contrib_norm": contrib_norm.mean().item(),
                        "contrib_pct": (100 * contrib_norm / q_norm).mean().item(),
                    }
                )
    return {"rows": rows} if rows else {}


def _section_channel_rank(enc) -> dict:
    if enc is None:
        return {}
    ends = list(accumulate(ENCODER_CHANNELS))
    bounds = list(zip([0, *ends[:-1]], ends, strict=True))
    rows = []
    with torch.no_grad():
        for source, path in _BLOCK_CAPACITY_SOURCES:
            texts = [r.text for r in read(path)]
            if not texts:
                continue
            emb = enc.pool(torch.stack([text_to_tensor(t) for t in texts]))
            for i, (start, end) in enumerate(bounds):
                channels = end - start
                a = emb[:, start:end]
                a = a - a.mean(dim=0, keepdim=True)
                var = torch.linalg.svdvals(a).pow(2)
                total = var.sum()
                if total <= 0:
                    eff_rank, rank95 = 0.0, 0
                else:
                    eff_rank = (total.pow(2) / var.pow(2).sum()).item()
                    cumvar = torch.cumsum(var, dim=0) / total
                    rank95 = int(torch.searchsorted(cumvar, 0.95).item()) + 1
                rows.append(
                    {
                        "source": source,
                        "n": len(texts),
                        "block": i,
                        "channels": channels,
                        "eff_rank": eff_rank,
                        "eff_rank_pct": 100 * eff_rank / channels,
                        "rank95": rank95,
                        "rank95_pct": 100 * rank95 / channels,
                    }
                )
    return {"rows": rows} if rows else {}


@cache
def _cldr_baseline():
    if not CLDR_BASELINE_JSON.exists():
        return None
    try:
        data = json.loads(CLDR_BASELINE_JSON.read_text(encoding="utf-8"))
        methods = data["methods"]
    except (json.JSONDecodeError, KeyError, OSError):
        return None
    if not methods:
        return None
    name = max(
        methods,
        key=lambda m: (methods[m]["acc_at_k"][-1], methods[m]["mrr"]),
    )
    return {"name": name, "acc_at_k": methods[name]["acc_at_k"]}


def _strip_variation(e: str) -> str:
    return e.replace("\ufe0f", "")


@cache
def _group_json() -> dict:
    if not GROUP_JSON.exists():
        return {}
    return json.loads(GROUP_JSON.read_text(encoding="utf-8"))


@cache
def _load_global_goals() -> dict:
    if not GOALS_YML.exists():
        return {}
    return (yaml.safe_load(GOALS_YML.read_text(encoding="utf-8")) or {}).get("goals") or {}


def _coverage_targets() -> dict:
    return ((_load_global_goals().get("vocabulary") or {}).get("coverage")) or {}


def _vocab_coverage(targets=None) -> dict:
    groups = _group_json()
    if not groups:
        return {
            "measurable": False,
            "reason": f"{GROUP_JSON} absent (run bun run build-groups)",
        }
    targets = _coverage_targets() if targets is None else targets
    vocab_norm = {_strip_variation(e) for e in EMOJIS}
    out_groups = {}
    measurable = passed = 0
    missing = []
    for name, members in groups.items():
        total = len(members)
        tgt = targets.get(name)
        covered = sum(1 for e in members if _strip_variation(e) in vocab_norm)
        missing.extend(e for e in members if _strip_variation(e) not in vocab_norm)
        if not total:
            out_groups[name] = {
                "score": None,
                "covered": 0,
                "total": 0,
                "target": tgt,
                "passed": None,
            }
            continue
        score = covered / total
        ok = None if tgt is None else score >= tgt
        out_groups[name] = {
            "score": score,
            "covered": covered,
            "total": total,
            "target": tgt,
            "passed": ok,
        }
        if ok is not None:
            measurable += 1
            passed += bool(ok)
    return {
        "measurable": True,
        "groups": out_groups,
        "groups_total": len(groups),
        "groups_measurable": measurable,
        "groups_passed": passed,
        "score": passed / measurable if measurable else 0.0,
        "passed": measurable > 0 and passed == measurable,
        "top_missing": missing[:30],
    }


def _meets(cur, target, direction) -> bool:
    if target is None:
        return False
    return cur <= target if direction == "min" else cur >= target


def _grade(cur, target, direction) -> str:
    if cur is None or target is None:
        return "na"
    return "good" if _meets(cur, target, direction) else "red"


def _best_emoji_acc(emoji_eval) -> tuple:
    acc = emoji_eval.get("acc_at_k")
    return ("EmojiHead", acc) if acc else (None, None)


def _dig(node, *keys):
    for k in keys:
        if isinstance(node, dict):
            node = node.get(k)
        elif isinstance(node, (list, tuple)) and isinstance(k, int):
            node = node[k] if -len(node) <= k < len(node) else None
        else:
            return None
    return node


def _acc_rows(
    goals,
    priority,
    label,
    targets,
    values,
    note,
    direction="max",
):
    for name, idx in ACC_K_INDEX.items():
        tgt = (targets or {}).get(name)
        cur = values[idx] if values and len(values) > idx else None
        goals.append(
            {
                "goal": f"{label} {name}",
                "priority": priority,
                "dir": direction,
                "target": "—" if tgt is None else f"≥ {tgt:.2f}",
                "current": cur,
                "status": _grade(cur, tgt, direction),
                "note": note,
            }
        )


def _section_status(report) -> dict:
    emoji_eval = (report.get("emoji") or {}).get("eval") or {}
    keyword = report.get("keyword") or {}
    cards = report.get("cards") or {}
    data = report.get("data") or {}
    g = _load_global_goals()
    ep = g.get("emoji prediction") or {}
    sp = g.get("style prediction") or {}
    vocab_g = g.get("vocabulary") or {}

    best_name, best = _best_emoji_acc(emoji_eval)
    term = report.get("term") or {}
    goals = []

    _acc_rows(
        goals,
        1,
        "Keyword emoji",
        ep.get("keyword"),
        keyword.get("acc_at_k"),
        ""
        if keyword.get("acc_at_k")
        else "unmeasured — needs data/keywords.jsonl and enc.pt/emoji.pt",
    )
    _acc_rows(
        goals,
        2,
        "Term emoji",
        ep.get("term"),
        term.get("acc_at_k"),
        ""
        if term.get("acc_at_k")
        else "unmeasured — needs data/terms.jsonl and enc.pt/emoji.pt",
    )
    _acc_rows(
        goals,
        3,
        "Full-text emoji",
        ep.get("full text"),
        best or [],
        f"best variant: {best_name}" if best_name else "emoji.eval not evaluated this run",
    )

    _acc_rows(
        goals,
        5,
        "Style",
        sp.get("full text"),
        cards.get("style_acc_at_k"),
        "cards off this run" if not cards.get("style_acc_at_k") else "",
    )

    mtl = g.get("max text len")
    mtl_cur = data.get("max_text_len")
    goals.append(
        {
            "goal": "Max text len",
            "priority": 6,
            "dir": "max",
            "target": "—" if mtl is None else f"≥ {mtl}",
            "current": mtl_cur,
            "status": _grade(mtl_cur, mtl, "max"),
            "note": f"config.MAX_TEXT_LEN = {mtl_cur}" if mtl_cur is not None else "",
        }
    )

    size_tgt = vocab_g.get("size")
    vocab_size = len(EMOJIS)
    goals.append(
        {
            "goal": "Emoji vocab size",
            "priority": 7,
            "dir": "max",
            "target": "—" if size_tgt is None else f"≥ {size_tgt}",
            "current": vocab_size,
            "status": _grade(vocab_size, size_tgt, "max"),
            "note": "",
        }
    )

    higher_open = any(x["status"] == "red" for x in goals if x["priority"] <= 7)
    vc = _vocab_coverage()
    if not vc.get("measurable"):
        goals.append(
            {
                "goal": "Vocab coverage (per Unicode group)",
                "priority": 8,
                "dir": "max",
                "target": "all groups ≥ target",
                "current": None,
                "status": "na",
                "note": f"not measurable — {vc.get('reason', '')}",
            }
        )
    else:
        frac = f"{vc['groups_passed']}/{vc['groups_measurable']} groups meet target"
        worst = sorted(
            (v["score"] - v["target"], k)
            for k, v in vc["groups"].items()
            if v["score"] is not None and v["target"] is not None
        )[:4]
        weak = " · weakest: " + \
            ", ".join(f"{k} {d:+.2f}" for d, k in worst) if worst else ""
        if higher_open:
            status, note = "na", "deferred — lower priority than open goals above · " + frac
        else:
            status = "good" if vc["passed"] else "red"
            note = frac + weak
        goals.append(
            {
                "goal": "Vocab coverage (per Unicode group)",
                "priority": 8,
                "dir": "max",
                "target": "all groups ≥ target",
                "current": vc["groups_passed"],
                "status": status,
                "note": note,
            }
        )

    goals.sort(key=lambda x: x["priority"])
    return {
        "best_emoji_variant": best_name,
        "summary": dict(Counter(x["status"] for x in goals)),
        "goals": goals,
        "vocab_coverage": vc,
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
            enc_emb = enc(texts)
            logits = head(enc_emb)
        d["eval"] = {
            "n": len(rows),
            "acc_at_k": [_acc_at_k(logits, tgt, k).mean().item() for k in EMOJI_KS],
            "baseline": _cldr_baseline(),
        }
    return d


def _section_length_acc(enc, head, eval_records) -> dict:
    if enc is None or head is None:
        return {}
    rows = [r for r in eval_records if r.emojis]
    if not rows:
        return {}
    rf = 1 + (ENCODER_KERNEL_SIZE - 1) * sum(ENCODER_DILATION)
    lo = rf // 2

    def bucket(n):
        if n <= lo:
            return f"1-{lo}"
        if n <= rf:
            return f"{lo + 1}-{rf}"
        return f"{rf + 1}-{MAX_TEXT_LEN}"

    vocab = {e: i for i, e in enumerate(EMOJIS)}
    texts = torch.stack([text_to_tensor(r.text) for r in rows])
    tgt = torch.zeros(len(rows), len(EMOJIS))
    for i, r in enumerate(rows):
        for e in r.emojis:
            tgt[i, vocab[e]] = 1.0
    with torch.no_grad():
        logits = head(enc(texts))

    groups: dict[str, list[int]] = {}
    for i, r in enumerate(rows):
        groups.setdefault(bucket(len(r.text)), []).append(i)

    buckets = []
    for name in sorted(groups, key=lambda b: int(b.split("-")[0])):
        idx = torch.tensor(groups[name])
        blog, btgt = logits[idx], tgt[idx]
        buckets.append(
            {
                "bucket": name,
                "n": len(idx),
                "acc_at_k": [_acc_at_k(blog, btgt, k).mean().item() for k in EMOJI_KS],
            }
        )
    return {"receptive_field": rf, "max_text_len": MAX_TEXT_LEN, "buckets": buckets}


def _valid_color_rows() -> list[dict]:
    return [
        r
        for r in _rows(COLORS_JSONL)
        if isinstance(r.get("bg"), list)
        and len(r["bg"]) == 2
        and r.get("fg")
        and 0 < len(norm_text(str(r.get("text", "")))) <= MAX_TEXT_LEN
    ]


def _gold_rows():
    rows = _valid_color_rows()
    if not rows:
        return []
    rng = random.Random(SEED)
    out = []
    for color in CARD_COLORS:
        pool = sorted(
            (r for r in rows if r.get("color") == color),
            key=lambda r: norm_text(r["text"]),
        )
        for r in rng.sample(pool, min(GOLD_PER_COLOR, len(pool))):
            out.append({**r, "color": color})
    return out


def _section_cards(enc, style_head, emoji_head, gen, gold_rows):
    if None in (enc, style_head, emoji_head, gen) or not gold_rows:
        return {}
    rows = list(gold_rows)
    ids = torch.stack([text_to_tensor(norm_text(r["text"])) for r in rows])
    evocab = {e: i for i, e in enumerate(EMOJIS)}
    svocab = {s: i for i, s in enumerate(STYLES)}
    etgt = torch.zeros(len(rows), len(EMOJIS))
    stgt = torch.zeros(len(rows), len(STYLES))
    for i, r in enumerate(rows):
        for e in str(r["emojis"]).split():
            if e in evocab:
                etgt[i, evocab[e]] = 1.0
        for s in r["styles"]:
            if s in svocab:
                stgt[i, svocab[s]] = 1.0
    with torch.no_grad():
        emb = enc(ids)
        elog = emoji_head(emb)
        slog = style_head(emb)
        cond = emb[:, None, :].expand(-1, CONST_Z.shape[0], -
                                      1).reshape(-1, emb.shape[-1])
        z = CONST_Z[None, :, :].expand(
            len(rows), -1, -1).reshape(-1, CONST_Z.shape[-1])
        palettes = gen(cond, z).reshape(len(rows), CONST_Z.shape[0], 9)
    emoji_acc = [_acc_at_k(elog, etgt, k).mean().item() for k in EMOJI_KS]
    style_acc = [_acc_at_k(slog, stgt, k).mean().item() for k in EMOJI_KS]
    out_rows = []
    for i, r in enumerate(rows):
        gold9 = (
            _hex_to_offsets(r["bg"][0])
            + _hex_to_offsets(r["bg"][1])
            + _hex_to_offsets(r["fg"])
        )
        dp = min(
            _pure_distance(palettes[i, k].tolist(), r["color"])
            for k in range(palettes.shape[1])
        )
        flat = palettes[i, 0].tolist()
        gen_l, gen_chroma = _l_chroma(flat)
        gold_l, gold_chroma = _l_chroma(gold9)
        out_rows.append(
            {
                "color": r["color"],
                "text": r["text"],
                "emoji": EMOJIS[int(elog[i].argmax())],
                "style": STYLES[int(slog[i].argmax())],
                "bg1": _offsets_to_hex(flat[0:3]),
                "bg2": _offsets_to_hex(flat[3:6]),
                "text_color": _offsets_to_hex(flat[6:9]),
                "gt_emoji": " ".join(str(r["emojis"]).split()),
                "gt_style": " · ".join(r["styles"]),
                "gt_bg1": r["bg"][0],
                "gt_bg2": r["bg"][1],
                "gt_text_color": r["fg"],
                "dP": dp,
                "hit_pure": dp < _pure_threshold(r["color"]),
                "gen_l": gen_l,
                "gold_l": gold_l,
                "gen_chroma": gen_chroma,
                "gold_chroma": gold_chroma,
            }
        )

    def _stats(idxs):
        rs = [out_rows[i] for i in idxs]
        n = len(rs) or 1
        return {
            "pure_accuracy": sum(x["hit_pure"] for x in rs) / n,
            "pure_mean_distance": sum(x["dP"] for x in rs) / n,
            "l_bias": sum(x["gen_l"] - x["gold_l"] for x in rs) / n,
            "chroma_bias": sum(x["gen_chroma"] - x["gold_chroma"] for x in rs) / n,
        }

    per_color = {
        c: _stats([i for i, r in enumerate(out_rows) if r["color"] == c])
        for c in CARD_COLORS
    }
    per_color["all"] = _stats(list(range(len(out_rows))))
    return {
        "n": len(rows),
        "pure_threshold_rgb": CARD_PURE_THRESHOLD_RGB,
        "pure_threshold_l": CARD_PURE_THRESHOLD_L,
        "emoji_acc_at_k": emoji_acc,
        "style_acc_at_k": style_acc,
        "per_color": per_color,
        "rows": out_rows,
    }


def _section_gen_sensitivity(enc, gen, gold_rows) -> dict:
    if enc is None or gen is None or not gold_rows:
        return {}
    rng = random.Random(SEED)
    texts = sorted({norm_text(r["text"]) for r in gold_rows})
    texts = rng.sample(texts, min(SENS_TEXT_N, len(texts)))
    m = len(texts)
    k = CONST_Z.shape[0]
    if m < 2:
        return {}
    ids = torch.stack([text_to_tensor(t) for t in texts])
    with torch.no_grad():
        emb = enc(ids)
        cond = emb[:, None, :].expand(-1, k, -1).reshape(-1, emb.shape[-1])
        z = CONST_Z[None, :, :].expand(m, -1, -1).reshape(-1, CONST_Z.shape[-1])
        palettes = gen(cond, z).reshape(m, k, 9)
    pts = rgb_to_oklab(palettes)

    mode = "donot_use_mm_for_euclid_dist"
    mask_k = (~torch.eye(k, dtype=torch.bool)).reshape(-1)
    same_text = torch.cdist(pts, pts, compute_mode=mode).reshape(m, k * k)
    noise_spread = same_text[:, mask_k].mean().item()

    pts_by_z = pts.transpose(0, 1)
    mask_m = (~torch.eye(m, dtype=torch.bool)).reshape(-1)
    diff_text = torch.cdist(pts_by_z, pts_by_z, compute_mode=mode).reshape(k, m * m)
    text_spread = diff_text[:, mask_m].mean().item()

    grand_mean = pts.mean(dim=(0, 1))
    text_means = pts.mean(dim=1)
    z_means = pts.mean(dim=0)
    ss_total = ((pts - grand_mean) ** 2).sum().item()
    ss_text = (k * (text_means - grand_mean) ** 2).sum().item()
    ss_noise = (m * (z_means - grand_mean) ** 2).sum().item()

    swatches = []
    for i in range(min(SENS_SWATCH_N, m)):
        cards = []
        for kk in range(k):
            flat = palettes[i, kk].tolist()
            cards.append(
                {
                    "bg1": _offsets_to_hex(flat[0:3]),
                    "bg2": _offsets_to_hex(flat[3:6]),
                }
            )
        swatches.append({"text": texts[i], "cards": cards})

    return {
        "n_texts": m,
        "n_z": k,
        "noise_spread": noise_spread,
        "text_spread": text_spread,
        "ratio": noise_spread / text_spread if text_spread else None,
        "text_variance_share": ss_text / ss_total if ss_total else None,
        "noise_variance_share": ss_noise / ss_total if ss_total else None,
        "swatches": swatches,
    }


def _section_color_keywords(enc, gen) -> dict:
    cg_goals = _load_global_goals().get("color generator") or {}
    targets = cg_goals.get("energy distance") or {}
    if enc is None or gen is None or not targets:
        return {}
    rows = _rows(DATA_PATH)
    out = []
    for keyword in sorted(targets):
        matched = [
            r
            for r in rows
            if r.get("colors")
            and 0 < len(norm_text(str(r.get("text", "")))) <= MAX_TEXT_LEN
            and keyword.lower() in str(r["text"]).lower()
        ]
        rng = random.Random(f"{SEED}:color_keyword:{keyword}")
        sample = rng.sample(matched, min(COLOR_KEYWORD_SAMPLE_N, len(matched)))
        target = targets[keyword]
        if not sample:
            out.append(
                {
                    "keyword": keyword,
                    "n": 0,
                    "target": target,
                    "value": None,
                    "status": "na",
                    "examples": [],
                }
            )
            continue
        ids = torch.stack([text_to_tensor(norm_text(r["text"])) for r in sample])
        gt9 = torch.tensor(
            [
                _hex_to_offsets(r["colors"][0]["bg"][0])
                + _hex_to_offsets(r["colors"][0]["bg"][1])
                + _hex_to_offsets(r["colors"][0]["fg"])
                for r in sample
            ],
            dtype=torch.float32,
        )
        with torch.no_grad():
            fake9 = gen(enc(ids))
        value = energy_distance(rgb_to_oklab(fake9), rgb_to_oklab(gt9)).item()
        examples = []
        for i in range(min(COLOR_KEYWORD_EXAMPLE_N, len(sample))):
            flat = fake9[i].tolist()
            examples.append(
                {
                    "text": sample[i]["text"],
                    "gt_bg1": sample[i]["colors"][0]["bg"][0],
                    "gt_bg2": sample[i]["colors"][0]["bg"][1],
                    "gt_fg": sample[i]["colors"][0]["fg"],
                    "pred_bg1": _offsets_to_hex(flat[0:3]),
                    "pred_bg2": _offsets_to_hex(flat[3:6]),
                    "pred_fg": _offsets_to_hex(flat[6:9]),
                }
            )
        out.append(
            {
                "keyword": keyword,
                "n": len(sample),
                "target": target,
                "value": value,
                "status": _grade(value, target, "min"),
                "examples": examples,
            }
        )
    return {"rows": out, "sample_n": COLOR_KEYWORD_SAMPLE_N}


def _section_lang_consistency(enc, emoji_head, style_head, gen) -> dict:
    if None in (enc, emoji_head, style_head, gen):
        return {}
    en_ids = torch.stack(
        [text_to_tensor(norm_text(en)[:MAX_TEXT_LEN]) for en, _ in LANG_PAIRS]
    )
    he_ids = torch.stack(
        [text_to_tensor(norm_text(he)[:MAX_TEXT_LEN]) for _, he in LANG_PAIRS]
    )
    with torch.no_grad():
        enc_en, enc_he = enc(en_ids), enc(he_ids)
        emoji_en = emoji_head(enc_en)
        emoji_he = emoji_head(enc_he)
        style_en, style_he = style_head(enc_en), style_head(enc_he)
        zeros = torch.zeros(
            *enc_en.shape[:-1], HIDDEN_SIZE_GEN,
            device=enc_en.device, dtype=enc_en.dtype
        )
        color_en, color_he = gen(enc_en, zeros), gen(enc_he, zeros)
    ok_en, ok_he = rgb_to_oklab(color_en), rgb_to_oklab(color_he)

    rows = []
    for i, (en, he) in enumerate(LANG_PAIRS):
        top_en = set(emoji_en[i].topk(LANG_PAIR_TOPK).indices.tolist())
        top_he = set(emoji_he[i].topk(LANG_PAIR_TOPK).indices.tolist())
        jaccard = len(top_en & top_he) / len(top_en | top_he)
        s_en, s_he = style_en[i].argmax().item(), style_he[i].argmax().item()
        flat_en, flat_he = color_en[i].tolist(), color_he[i].tolist()
        rows.append(
            {
                "en": en,
                "he": he,
                "emb_cos": nn.functional.cosine_similarity(
                    enc_en[i], enc_he[i], dim=0
                ).item(),
                "emoji_jaccard": jaccard,
                "emoji_top_en": [EMOJIS[j] for j in top_en],
                "emoji_top_he": [EMOJIS[j] for j in top_he],
                "style_en": STYLES[s_en],
                "style_he": STYLES[s_he],
                "style_match": s_en == s_he,
                "color_dist": (ok_en[i] - ok_he[i]).norm().item(),
                "en_bg1": _offsets_to_hex(flat_en[0:3]),
                "en_bg2": _offsets_to_hex(flat_en[3:6]),
                "en_fg": _offsets_to_hex(flat_en[6:9]),
                "he_bg1": _offsets_to_hex(flat_he[0:3]),
                "he_bg2": _offsets_to_hex(flat_he[3:6]),
                "he_fg": _offsets_to_hex(flat_he[6:9]),
            }
        )
    rows.sort(key=lambda r: -r["color_dist"])

    n = len(rows)
    agree = [r for r in rows if r["emoji_jaccard"] >= LANG_PAIR_AGREE_THRESH]
    disagree = [r for r in rows if r["emoji_jaccard"] < LANG_PAIR_AGREE_THRESH]
    return {
        "n": n,
        "topk": LANG_PAIR_TOPK,
        "agree_thresh": LANG_PAIR_AGREE_THRESH,
        "n_agree": len(agree),
        "emb_cos_mean": sum(r["emb_cos"] for r in rows) / n,
        "emoji_jaccard_mean": sum(r["emoji_jaccard"] for r in rows) / n,
        "style_match_rate": sum(r["style_match"] for r in rows) / n,
        "color_dist_mean": sum(r["color_dist"] for r in rows) / n,
        "color_dist_mean_agree": (
            sum(r["color_dist"] for r in agree) / len(agree) if agree else None
        ),
        "color_dist_mean_disagree": (
            sum(r["color_dist"] for r in disagree) /
            len(disagree) if disagree else None
        ),
        "rows": rows,
    }


def _section_eval_samples(pt: Path, eval_records) -> dict:
    pool = [r for r in eval_records if r.emojis]
    if not pool or not all((pt / n).exists() for n in EVAL_SAMPLE_PT_FILES):
        return {}
    rng = random.Random(SEED)
    rows = rng.sample(pool, min(EVAL_SAMPLE_N, len(pool)))
    try:
        preds = _predict([{"text": r.text} for r in rows], pt)
    except RuntimeError:
        return {}
    out_rows = []
    for r, p in zip(rows, preds, strict=True):
        out_rows.append(
            {
                "text": r.text,
                "gt_emoji": r.emojis[0],
                "gt_feeling": r.styles[0],
                "gt_bg1": r.colors[0][0],
                "gt_bg2": r.colors[0][1],
                "gt_text_color": r.colors[0][2],
                "emoji": p["emojis"].split()[0],
                "feeling": p["styles"][0],
                "bg1": p["bg"][0],
                "bg2": p["bg"][1],
                "text_color": p["fg"],
            }
        )
    return {"n": len(out_rows), "rows": out_rows}


def build_report(pt: Path, only: str = "", out: Path = REPORT_DIR) -> Path:
    want = {s.strip() for s in only.split(",") if s.strip()} or {
        "data",
        "labels",
        "emoji",
        "keywords_flex",
        "keyword_fails",
        "keyword",
        "cards",
        "gen_sensitivity",
        "color_keywords",
        "style_dist",
        "block_capacity",
        "channel_rank",
        "length_acc",
        "eval_samples",
        "lang_consistency",
    }
    enc_pt, emoji_pt = PtFile.ENC.in_dir(pt), PtFile.EMOJI.in_dir(pt)
    style_pt, gen_pt = PtFile.STYLE.in_dir(pt), PtFile.GEN.in_dir(pt)
    prov = _provenance(pt)

    enc = emoji_head = style_head = gen = None
    need_enc = bool(
        {
            "emoji",
            "keyword",
            "cards",
            "gen_sensitivity",
            "color_keywords",
            "keywords_flex",
            "keyword_fails",
            "style_dist",
            "block_capacity",
            "channel_rank",
            "length_acc",
            "lang_consistency",
        }
        & want
    )
    if need_enc and enc_pt.exists():
        enc, err = _load(TextEncoder(), enc_pt)
        if err:
            prov["issues"].append(f"{enc_pt} could not load: {err}")
    if enc is not None and emoji_pt.exists():
        emoji_head, err = _load(EmojiHead(), emoji_pt)
        if err:
            prov["issues"].append(f"{emoji_pt} could not load: {err}")
    if enc is not None and {"cards", "style_dist", "lang_consistency"} & want:
        if style_pt.exists():
            style_head, err = _load(StyleHead(), style_pt)
            if err:
                prov["issues"].append(f"{style_pt} could not load: {err}")
    gen_wanted = {"cards", "gen_sensitivity",
                  "color_keywords", "lang_consistency"} & want
    if enc is not None and gen_wanted and gen_pt.exists():
        gen, err = _load(ColorGen(), gen_pt)
        if err:
            prov["issues"].append(f"{gen_pt} could not load: {err}")
    prov["consistent"] = not prov["issues"]

    eval_records = (
        list(read(EVAL_PATH))
        if {"emoji", "style_dist", "length_acc", "eval_samples"} & want
        else []
    )
    gold_rows = _gold_rows() if {"cards", "gen_sensitivity"} & want else ()

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
    if "keywords_flex" in want:
        report["keywords_flex"] = _section_keywords_flex(enc, emoji_head)
    if "keyword_fails" in want:
        report["keyword_fails"] = _section_keyword_fails(enc, emoji_head)
    if "keyword" in want:
        report["keyword"] = _section_keyword_probe(enc, emoji_head)
        report["term"] = _section_term_probe(enc, emoji_head)
    if "cards" in want:
        report["cards"] = _section_cards(
            enc, style_head, emoji_head, gen, gold_rows)
    if "gen_sensitivity" in want:
        report["gen_sensitivity"] = _section_gen_sensitivity(enc, gen, gold_rows)
    if "color_keywords" in want:
        report["color_keywords"] = _section_color_keywords(enc, gen)
    if "style_dist" in want:
        report["style_dist"] = _section_style_dist(enc, style_head, eval_records)
    if "block_capacity" in want:
        report["block_capacity"] = _section_block_capacity(enc)
    if "channel_rank" in want:
        report["channel_rank"] = _section_channel_rank(enc)
    if "length_acc" in want:
        report["length_acc"] = _section_length_acc(enc, emoji_head, eval_records)
    if "eval_samples" in want:
        report["eval_samples"] = _section_eval_samples(pt, eval_records)
    if "lang_consistency" in want:
        report["lang_consistency"] = _section_lang_consistency(
            enc, emoji_head, style_head, gen
        )

    report["status"] = _section_status(report)

    out_dir = out / f"{prov['ts']}-{prov['model_sha']}"
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
--warn-bd:#efd4a2;--warn-fg:#a25c07;--bad-bg:#fdeaea;--bad-bd:#f0b6b6;
--bad-fg:#b3261e}
*{box-sizing:border-box}
body{font:17px/1.6 -apple-system,BlinkMacSystemFont,"Segoe UI",system-ui,sans-serif;
color:var(--ink);margin:0;background:#fff}
.page{max-width:1180px;margin:0 auto;display:flex;align-items:flex-start;gap:28px}
.wrap{max-width:920px;flex:1 1 auto;min-width:0;padding:44px 30px 130px}
.toc{flex:0 0 200px;position:sticky;top:0;max-height:100vh;overflow-y:auto;
padding:44px 4px 30px 30px;font-size:13.5px}
.toc-h{color:var(--dim);text-transform:uppercase;letter-spacing:.04em;
font-size:12px;margin-bottom:8px}
.toc ul{list-style:none;margin:0;padding:0;border-left:2px solid var(--line)}
.toc li{margin:0}
.toc a{display:block;color:var(--dim);text-decoration:none;padding:5px 0 5px 12px;
margin-left:-2px;border-left:2px solid transparent}
.toc a:hover{color:var(--ink);border-left-color:var(--accent)}
@media (max-width:980px){.toc{display:none}}
h1{font-size:28px;margin:0 0 4px}
h2{font-size:24px;margin:64px 0 6px;padding-bottom:8px;border-bottom:2px solid var(--ink);
scroll-margin-top:16px}
h3{font-size:15px;text-transform:uppercase;letter-spacing:.04em;color:var(--dim);
margin:34px 0 12px}
h4{font-size:13px;letter-spacing:.02em;color:var(--dim);margin:18px 0 8px}
.sub{color:var(--dim);font-size:15px}
.note{color:var(--dim);font-size:14.5px;margin:8px 0 0}
code{background:var(--panel);padding:1px 6px;border-radius:4px;font-size:15px}
.banner{border-radius:10px;padding:13px 17px;margin:20px 0 0;font-size:15.5px;
background:var(--good-bg);border:1px solid var(--good-bd)}
.banner.amber{background:var(--warn-bg);border-color:var(--warn-bd)}
.banner ul{margin:6px 0 0;padding-left:20px}
.status-h{margin-top:34px}
.scorecard td:first-child{font-weight:600}
.scorecard .gnote{font-weight:400;font-size:13px;color:var(--dim);margin-top:3px}
.scorecard tr.sc-good{background:var(--good-bg)}
.scorecard tr.sc-red{background:var(--bad-bg)}
.scorecard tr.sc-na{background:var(--panel);color:var(--dim)}
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
.linechart .lline2{fill:none;stroke:#e07b00;stroke-width:2.5}
.linechart .lline3{fill:none;stroke:#0a9c8b;stroke-width:2.5}
.linechart .lline4{fill:none;stroke:#8b5cf6;stroke-width:2.5}
.linechart .lline5{fill:none;stroke:#d6336c;stroke-width:2.5}
.linechart .bline{fill:none;stroke:var(--dim);stroke-width:2;stroke-dasharray:5 4}
.linechart .btext{font-size:11px;fill:var(--dim);font-variant-numeric:tabular-nums}
.linechart .dot{fill:var(--accent)}
.linechart .vtext{font-size:11px;fill:var(--dim);font-variant-numeric:tabular-nums}
.linechart .xtext{font-size:12px;fill:var(--ink)}
table{border-collapse:collapse;width:100%;margin:16px 0;font-size:16px;
border:1px solid var(--line)}
th,td{border-bottom:1px solid var(--line);padding:10px 12px;text-align:left}
th{font-size:13px;color:var(--ink);font-weight:700;text-transform:uppercase;
letter-spacing:.03em;background:var(--panel);border-bottom:2px solid var(--ink)}
td.n,th.n{text-align:right;font-variant-numeric:tabular-nums}
tr:last-child td{border-bottom:none}
table:not(.scorecard) tr:nth-child(even){background:var(--panel)}
table:not(.scorecard) tr:hover{background:var(--good-bg)}
.blk-dot{display:inline-block;width:9px;height:9px;border-radius:50%;
margin-right:7px;vertical-align:middle}
.cards-grid{display:grid;grid-template-columns:repeat(5,1fr);gap:10px;margin:18px 0 0}
.sample-cards-grid{display:grid;grid-template-columns:repeat(4,1fr);gap:10px;margin:18px 0 0}
.sample-card{width:100%;aspect-ratio:1/1;border:1px solid var(--line);border-radius:12px}
.mini{aspect-ratio:1/1;border-radius:12px;padding:12px 10px;display:flex;
flex-direction:column;justify-content:center;align-items:center;text-align:center;
overflow:hidden}
.mini .em{font-size:24px;line-height:1}
.mini .tx{font-size:12px;font-weight:600;margin-top:6px;overflow-wrap:anywhere;
line-height:1.3}
.mini .st{font-size:9px;letter-spacing:.06em;text-transform:uppercase;margin-top:6px;
opacity:.75}
.sens-grid{display:flex;flex-direction:column;gap:8px;margin:18px 0 0}
.sens-row{display:flex;align-items:center;gap:10px}
.sens-label{flex:0 0 180px;font-size:13px;overflow:hidden;text-overflow:ellipsis;
white-space:nowrap}
.sens-swatches{display:flex;gap:6px;flex:1 1 auto}
.sens-swatch{width:44px;height:44px;border-radius:8px;border:1px solid var(--line)}
.kw-grid{display:grid;grid-template-columns:repeat(4,1fr);gap:10px;margin:14px 0 0}
.kw-grid .grp{grid-column:span 2;font-size:12px;letter-spacing:.04em;
text-transform:uppercase;color:var(--dim)}
"""


_BLOCK_COLORS = ["#4b32d6", "#e07b00", "#0a9c8b", "#d6336c", "#8b5cf6"]


def _block_color(i: int) -> str:
    return _BLOCK_COLORS[i % len(_BLOCK_COLORS)]


def _block_bg(i: int, amount: float = 0.8) -> str:
    hx = _block_color(i).lstrip("#")
    mixed = (
        round(int(hx[j: j + 2], 16) + (255 - int(hx[j: j + 2], 16)) * amount)
        for j in (0, 2, 4)
    )
    return "#" + "".join(f"{v:02x}" for v in mixed)


def _esc(x) -> str:
    return html.escape(str(x))


def _fnum(n) -> str:
    return f"{n:,}"


def _hex_to_offsets(hx: str) -> list[float]:
    hx = hx.lstrip("#")
    return [int(hx[i: i + 2], 16) - COLOR_SHIFT for i in (0, 2, 4)]


def _offsets_to_hex(vals) -> str:
    ints = [max(0, min(255, round(v + COLOR_SHIFT))) for v in vals]
    return "#" + "".join(f"{v:02x}" for v in ints)


def _l_chroma(vals9) -> tuple[float, float]:
    lab = rgb_to_oklab(torch.tensor(vals9, dtype=torch.float32)).reshape(3, 3)
    return lab[:, 0].mean().item(), lab[:, 1:3].norm(dim=-1).mean().item()


PURE_HEX = {
    "red": "#ff0000",
    "green": "#00ff00",
    "blue": "#0000ff",
    "dark": "#000000",
    "bright": "#ffffff",
}


def _pure_threshold(color: str) -> float:
    if color in ("dark", "bright"):
        return CARD_PURE_THRESHOLD_L
    return CARD_PURE_THRESHOLD_RGB


def _pure_distance(pred9, color: str) -> float:
    p = rgb_to_oklab(torch.tensor(pred9, dtype=torch.float32)).reshape(3, 3)
    bg = p[:2].mean(dim=0)
    pure = rgb_to_oklab(torch.tensor(
        _hex_to_offsets(PURE_HEX[color]), dtype=torch.float32))
    if color in ("dark", "bright"):
        return (bg[0] - pure[0]).abs().item()
    return (bg - pure).norm().item()


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


def _linechart(points, y_max=1.0, baseline=None, legend=None, series=None) -> str:
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
    base = ""
    if baseline is not None:
        bcoords = [(px(i), py(v)) for i, v in enumerate(baseline)]
        bpoly = " ".join(f"{cx:.1f},{cy:.1f}" for cx, cy in bcoords)
        bx, by = bcoords[-1]
        base = (
            f'<polyline points="{bpoly}" class="bline"/>'
            f'<text x="{bx:.1f}" y="{by - 7:.1f}" class="btext" '
            f'text-anchor="end">{baseline[-1]:.2f}</text>'
        )
    extra = ""
    for _name, vals, cls in series or ():
        scoords = [(px(i), py(v)) for i, v in enumerate(vals)]
        spoly = " ".join(f"{cx:.1f},{cy:.1f}" for cx, cy in scoords)
        sdots = "".join(
            f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="3" class="dot"/>' for cx, cy in scoords
        )
        extra += f'<polyline points="{spoly}" class="{cls}"/>{sdots}'
    leg = ""
    if legend and series:
        parts = [
            f'<line x1="{pad_l}" y1="14" x2="{pad_l + 20}" y2="14" class="lline"/>'
            f'<text x="{pad_l + 26}" y="18" class="gtext">{_esc(legend[0])}</text>'
        ]
        lx = pad_l + 120
        for (_name, _vals, cls), lbl in zip(series, legend[1:], strict=False):
            parts.append(
                f'<line x1="{lx}" y1="14" x2="{lx + 20}" y2="14" class="{cls}"/>'
                f'<text x="{lx + 26}" y="18" class="gtext">{_esc(lbl)}</text>'
            )
            lx += 120
        leg = "".join(parts)
    elif legend:
        leg = (
            f'<line x1="{pad_l}" y1="14" x2="{pad_l + 20}" y2="14" class="lline"/>'
            f'<text x="{pad_l + 26}" y="18" class="gtext">{_esc(legend[0])}</text>'
            f'<line x1="{pad_l + 130}" y1="14" x2="{pad_l + 150}" y2="14" class="bline"/>'
            f'<text x="{pad_l + 156}" y="18" class="gtext">{_esc(legend[1])}</text>'
        )
    return (
        f'<svg viewBox="0 0 {w} {h}" class="linechart">'
        f"{grid}{base}"
        f'<polyline points="{poly}" class="lline"/>'
        f"{dots}{extra}{leg}</svg>"
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


_STATUS_WORD = {
    "good": "✅ on target",
    "red": "🔴 below target",
    "na": "⚪ not measured",
}


def _status_html(status) -> str:
    rows = []
    for g in status["goals"]:
        cur = g["current"]
        if cur is None:
            cur_txt = "n/a"
        elif isinstance(cur, float):
            cur_txt = f"{cur:.3f}"
        else:
            cur_txt = _fnum(cur)
        note = f'<div class="gnote">{_esc(g["note"])}</div>' if g.get(
            "note") else ""
        rows.append(
            f'<tr class="sc-{g["status"]}"><td>{_esc(g["goal"])}{note}</td>'
            f'<td class="n">{_esc(g["target"])}</td>'
            f'<td class="n">{cur_txt}</td>'
            f"<td>{_STATUS_WORD[g['status']]}</td></tr>"
        )
    out = [
        '<h2 class="status-h">Goal status</h2>'
        '<table class="scorecard"><tr><th>Goal — priority order</th>'
        '<th class="n">Target</th>'
        '<th class="n">Current value</th><th>Status</th></tr>' +
        "".join(rows) + "</table>"
    ]
    vc = status.get("vocab_coverage") or {}
    if vc.get("measurable"):
        grp_rows = []
        for name, v in vc["groups"].items():
            sc = "—" if v["score"] is None else f"{v['score']:.2f}"
            tg = "—" if v["target"] is None else f"{v['target']:.2f}"
            klass = (
                "sc-na" if v["passed"] is None else "sc-good" if v["passed"] else "sc-red"
            )
            grp_rows.append(
                f'<tr class="{klass}"><td>{_esc(name)}</td>'
                f'<td class="n">{v["covered"]}/{v["total"]}</td>'
                f'<td class="n">{sc}</td><td class="n">{tg}</td></tr>'
            )
        out.append(
            "<details><summary>Per-group vocab coverage — "
            f"{vc['groups_passed']}/{vc['groups_measurable']} groups meet target"
            "</summary>"
            '<table class="scorecard"><tr><th>Unicode group</th>'
            '<th class="n">In vocab</th><th class="n">Score</th>'
            '<th class="n">Target</th></tr>' +
            "".join(grp_rows) + "</table></details>"
        )
    return "".join(out)


def _acc_chart_html(title, source, d) -> str:
    if not d or not d.get("acc_at_k"):
        return (
            f"<h2>{_esc(title)}</h2>"
            f'<p class="note">unavailable — needs enc.pt / emoji.pt and '
            f"a non-empty {_esc(source)}.</p>"
        )
    points = list(zip((str(k) for k in EMOJI_KS), d["acc_at_k"], strict=True))
    return (
        f"<h2>{_esc(title)}</h2>"
        f"<h3>{d['n']}/{d['total']} in-vocab rows — {_esc(source)}</h3>"
        f"{_linechart(points)}"
    )


def _data_html(d) -> str:
    r = d["records"]
    cells = "".join(
        f'<div class="count"><div class="v">{_fnum(r[k])}</div>'
        f'<div class="k">records · {k}.jsonl</div></div>'
        for k in ("data", "train", "eval")
    )
    dist = d["length_distribution"]
    maxv = max((v for _, v in dist), default=1)
    bars = _bars([[str(length), count]
                 for length, count in dist], maxv, rotated=True)
    kw_dist = d["keywords_length_distribution"]
    kw_maxv = max((v for _, v in kw_dist), default=1)
    kw_bars = _bars(
        [[str(length), count] for length, count in kw_dist], kw_maxv, rotated=True
    )
    terms_dist = d["terms_length_distribution"]
    terms_maxv = max((v for _, v in terms_dist), default=1)
    terms_bars = _bars(
        [[str(length), count] for length, count in terms_dist], terms_maxv, rotated=True
    )
    mtl = d.get("max_text_len")
    mtl_html = f'<p class="note">MAX_TEXT_LEN = {mtl}</p>' if mtl is not None else ""
    return (
        f'<h2>Data</h2><div class="counts">{cells}</div>{mtl_html}'
        "<h3>Text length distribution — train.jsonl (normalized, longest first)</h3>"
        f"{bars}"
        "<h3>Text length distribution — keywords.jsonl (normalized, longest first)</h3>"
        f"{kw_bars}"
        "<h3>Text length distribution — terms.jsonl (normalized, longest first)</h3>"
        f"{terms_bars}"
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
        bl = e.get("baseline")
        legend = ("model", f"CLDR baseline ({bl['name']})") if bl else None
        chart = _linechart(
            points,
            baseline=bl["acc_at_k"] if bl else None,
            legend=legend,
        )
        note = (
            ""
            if bl
            else '<p class="note">CLDR baseline unavailable — run '
            "<code>bun run regen</code> to write "
            "<code>data/cldr-baseline.json</code>.</p>"
        )
        out.append(
            f"<h3>Performance on eval.jsonl ({e['n']} rows)</h3>{chart}{note}")
    return "".join(out)


def _keywords_flex_html(d) -> str:
    if not d:
        return (
            "<h2>Model — Keyword vocab</h2>"
            '<p class="note">enc.pt / emoji.pt not available.</p>'
        )
    covered = d["candidates"] - d["missed"]
    rate = covered / d["candidates"] if d["candidates"] else 0.0
    return (
        "<h2>Model — Keyword vocab</h2>"
        f'<p class="note">{_fnum(d["candidates"])} candidates &middot; '
        f"{_fnum(covered)} reach rank &le; 10 ({rate:.1%}) &middot; "
        f"{_fnum(d['missed'])} missed (rank &gt; 10)</p>"
        '<p class="note">Per-keyword miss list is in <code>report.json</code> '
        "(<code>keywords_flex.ranked</code>).</p>"
    )


def _keyword_fails_html(d) -> str:
    if not d:
        return (
            "<h2>Failed keywords — data/keywords.jsonl</h2>"
            '<p class="note">enc.pt / emoji.pt not available, or '
            "data/keywords.jsonl is missing/empty.</p>"
        )
    rows = d["rows"][:KEYWORD_FAILS_MAX_ROWS]
    trows = "".join(
        f"<tr><td>{_esc(r['kw'])}</td><td>{_esc(r['src'])}</td>"
        f"<td>{_esc(r['target'])}</td><td>{_esc(r['predicted'])}</td>"
        f'<td class="n">{r["rank"]}</td></tr>'
        for r in rows
    )
    note = (
        f"Showing worst {len(rows)} of {d['failed']}."
        if d["failed"] > len(rows)
        else ""
    )
    return (
        "<h2>Failed keywords — data/keywords.jsonl</h2>"
        f'<p class="note">{d["failed"]}/{d["n"]} keywords where the model\'s top-1 '
        f"prediction misses the target. {note}</p>"
        "<table><tr><th>Keyword</th><th>Src</th><th>Target</th>"
        '<th>Predicted</th><th class="n">Rank</th></tr>' + trows + "</table>"
    )


def _cards_html(d) -> str:
    if not d:
        return (
            "<h2>Cards</h2>"
            '<p class="note">Unavailable — needs enc.pt / style.pt / emoji.pt / '
            "gen.pt and a non-empty data/colors.jsonl.</p>"
        )
    out = [
        "<h2>Cards</h2>",
        '<p class="note">End-to-end test of the shipped inference graph on '
        f"{d['n']} gold rows sampled from eval.jsonl by their annotated colour tag "
        "(up to 25 drawn at random per colour).</p>",
    ]
    ek = list(zip((str(k) for k in EMOJI_KS), d["emoji_acc_at_k"], strict=True))
    chart = _linechart(
        ek,
        series=[("style", d["style_acc_at_k"], "lline2")],
        legend=("emoji", "style"),
    )
    out.append(f"<h3>Emoji &amp; style acc@k — gold set</h3>{chart}")
    pc = d["per_color"]

    def _fsigned(v) -> str:
        return f"{v:+.3f}" if v is not None else "–"

    trows = "".join(
        f"<tr><td>{_esc(c)}</td>"
        f'<td class="n">{pc[c]["pure_mean_distance"]:.3f}</td>'
        f'<td class="n">{pc[c]["pure_accuracy"]:.2f}</td>'
        f'<td class="n">{_fsigned(pc[c]["l_bias"])}</td>'
        f'<td class="n">{_fsigned(pc[c]["chroma_bias"])}</td></tr>'
        for c in (*CARD_COLORS, "all")
    )
    out.append(
        "<h3>Colour distance — d(model, pure)</h3>"
        '<p class="note">Pure acc: dP &lt; '
        f"{d['pure_threshold_rgb']:.3f} (r/g/b), &lt; {d['pure_threshold_l']:.2f} "
        "(|&Delta;L|, dark/bright) between the model's single top output and the "
        "category's pure hue. &Delta;L/&Delta;chroma: mean Oklab lightness/chroma "
        "of the model's single top output minus gold (from data/colors.jsonl), "
        "signed — positive means the model runs lighter/more saturated than "
        "gold.</p>"
        "<table><tr><th>Color</th>"
        '<th class="n">Pure dist</th><th class="n">Pure acc</th>'
        '<th class="n">&Delta;L</th><th class="n">&Delta;chroma</th></tr>'
        f"{trows}</table>"
    )
    by_color = {}
    for r in d["rows"]:
        by_color.setdefault(r["color"], []).append(r)

    def _grids(bg1, bg2, tc, em, st):
        for c in CARD_COLORS:
            cards = "".join(
                '<div class="mini" style="background:linear-gradient(135deg,'
                f'{_esc(r[bg1])},{_esc(r[bg2])});color:{_esc(r[tc])}">'
                f'<span class="em">{_esc(r[em])}</span>'
                f'<span class="tx">{_esc(r["text"])}</span>'
                f'<span class="st">{_esc(r[st])}</span></div>'
                for r in by_color.get(c, [])
            )
            out.append(f'<h3>{_esc(c)}</h3><div class="cards-grid">{cards}</div>')

    out.append("<h3>Model output</h3>")
    _grids("bg1", "bg2", "text_color", "emoji", "style")
    out.append("<h3>Ground truth — gold set (eval.jsonl annotations)</h3>")
    _grids("gt_bg1", "gt_bg2", "gt_text_color", "gt_emoji", "gt_style")
    return "".join(out)


def _gen_sensitivity_html(d) -> str:
    if not d:
        return (
            "<h2>Model — Generator sensitivity (text vs. noise)</h2>"
            '<p class="note">Unavailable — needs enc.pt / gen.pt and a '
            "non-empty data/colors.jsonl.</p>"
        )
    ratio = d["ratio"]
    if ratio is None:
        verdict, banner_cls = "not enough spread to judge.", "amber"
    elif ratio > 1.5:
        verdict = (
            "the generator moves far more when only the noise seed changes than "
            "when only the text changes — output is noise-dominated and text "
            "conditioning looks weak."
        )
        banner_cls = "amber"
    elif ratio < 0.8:
        verdict = (
            "text changes move the output at least as much as the noise seed "
            "does — conditioning looks intact."
        )
        banner_cls = ""
    else:
        verdict = "noise and text move the output by comparable amounts."
        banner_cls = "amber"
    out = [
        "<h2>Model — Generator sensitivity (text vs. noise)</h2>",
        '<p class="note">Takes '
        f"{d['n_texts']} texts sampled from the gold set and runs every one "
        f"through all {d['n_z']} fixed noise seeds (CONST_Z — the same seeds "
        "the shipped web app renders as its card variants), then compares two "
        "things: how far the output moves when only the noise seed changes "
        "(same text), against how far it moves when only the text changes "
        "(same seed). Distances are mean pairwise Oklab distance over the "
        "full 9-dim card vector, matching model/color.py:energy_distance's "
        "unit.</p>",
        "<table><tr><th>Metric</th><th class=\"n\">Value</th></tr>"
        "<tr><td>Mean distance — same text, different noise seed</td>"
        f'<td class="n">{d["noise_spread"]:.3f}</td></tr>'
        "<tr><td>Mean distance — different text, same noise seed</td>"
        f'<td class="n">{d["text_spread"]:.3f}</td></tr>',
    ]
    if ratio is not None:
        out.append(
            "<tr><td>Ratio (noise-driven / text-driven)</td>"
            f'<td class="n">{ratio:.2f}&times;</td></tr>'
        )
    if d["text_variance_share"] is not None:
        out.append(
            "<tr><td>Output variance explained by text identity</td>"
            f'<td class="n">{d["text_variance_share"]:.1%}</td></tr>'
            "<tr><td>Output variance explained by noise seed</td>"
            f'<td class="n">{d["noise_variance_share"]:.1%}</td></tr>'
        )
    out.append("</table>")
    out.append(f'<div class="banner {banner_cls}">{_esc(verdict)}</div>')
    rows_html = "".join(
        f'<div class="sens-row"><div class="sens-label">{_esc(s["text"])}</div>'
        '<div class="sens-swatches">'
        + "".join(
            '<div class="sens-swatch" style="background:linear-gradient(135deg,'
            f'{_esc(c["bg1"])},{_esc(c["bg2"])})"></div>'
            for c in s["cards"]
        )
        + "</div></div>"
        for s in d["swatches"]
    )
    out.append(
        f"<h3>Same text, {d['n_z']} noise seeds each</h3>"
        f'<div class="sens-grid">{rows_html}</div>'
    )
    return "".join(out)


def _color_keywords_html(d) -> str:
    if not d or not d.get("rows"):
        return (
            "<h2>Model — Generator energy distance by keyword</h2>"
            '<p class="note">Unavailable — needs enc.pt / gen.pt and '
            "goals.yml color generator.energy distance targets.</p>"
        )

    def _vtxt(r) -> str:
        return "–" if r["value"] is None else f"{r['value']:.3f}"

    trows = "".join(
        f'<tr class="sc-{r["status"]}"><td>{_esc(r["keyword"])}</td>'
        f'<td class="n">{r["n"]}</td>'
        f'<td class="n">{_vtxt(r)}</td>'
        f'<td class="n">&le; {r["target"]:.3f}</td>'
        f"<td>{_STATUS_WORD[r['status']]}</td></tr>"
        for r in d["rows"]
    )
    return (
        "<h2>Model — Generator energy distance by keyword</h2>"
        '<p class="note">For each keyword under goals.yml\'s '
        "<code>color generator.energy distance</code>, up to "
        f"{d['sample_n']} rows of data/data.jsonl whose text contains that "
        "keyword are sampled (deterministic, seeded), one ground-truth colour "
        "kept per row, and one generated palette drawn per row with fresh "
        "random generator noise. Energy distance is model/color.py:"
        "energy_distance between the two sets, in the same Oklab 9-dim card "
        "vector unit used elsewhere in this report.</p>"
        '<table class="scorecard"><tr><th>Keyword</th><th class="n">N sampled</th>'
        '<th class="n">Energy distance</th><th class="n">Target</th>'
        "<th>Status</th></tr>" + trows + "</table>"
        + _color_keyword_examples_html(d["rows"])
    )


def _kw_cell(kind: str, ex: dict) -> str:
    bg1, bg2, fg = ex[f"{kind}_bg1"], ex[f"{kind}_bg2"], ex[f"{kind}_fg"]
    return (
        '<div class="mini" style="background:linear-gradient(135deg,'
        f'{_esc(bg1)},{_esc(bg2)});color:{_esc(fg)}">'
        f'<span class="tx">{_esc(ex["text"])}</span></div>'
    )


def _color_keyword_examples_html(rows) -> str:
    out = []
    for r in rows:
        examples = r.get("examples")
        if not examples:
            continue
        cells = []
        for i in range(0, len(examples), 2):
            pair = examples[i: i + 2]
            cells += [_kw_cell("gt", ex) for ex in pair]
            cells += [_kw_cell("pred", ex) for ex in pair]
        out.append(
            f"<h3>{_esc(r['keyword'])}</h3>"
            '<div class="kw-grid"><div class="grp">Ground truth</div>'
            '<div class="grp">Model prediction</div>' + "".join(cells) + "</div>"
        )
    return "".join(out)


def _lang_consistency_html(d) -> str:
    if not d or not d.get("rows"):
        return (
            "<h2>Cross-lingual consistency (en vs he)</h2>"
            '<p class="note">Unavailable — needs enc.pt, emoji.pt, '
            "style.pt and gen.pt.</p>"
        )

    def _pct(v) -> str:
        return "–" if v is None else f"{v:.1%}"

    def _num(v) -> str:
        return "–" if v is None else f"{v:.3f}"

    def _row_html(r) -> str:
        style_cell = (
            "="
            if r["style_match"]
            else f"{_esc(r['style_en'])} / {_esc(r['style_he'])}"
        )
        flag = (
            "⚠"
            if r["emoji_jaccard"] >= d["agree_thresh"]
            and r["color_dist"] > d["color_dist_mean"]
            else ""
        )
        return (
            f"<tr><td>{_esc(r['en'])}</td><td>{_esc(r['he'])}</td>"
            f'<td class="n">{r["emb_cos"]:.3f}</td>'
            f'<td class="n">{r["emoji_jaccard"]:.2f}</td>'
            f"<td>{style_cell}</td>"
            f'<td class="n">{r["color_dist"]:.3f}</td>'
            f"<td>{flag}</td></tr>"
        )

    trows = "".join(_row_html(r) for r in d["rows"])
    swatch_rows = []
    for r in d["rows"][:LANG_PAIR_SWATCH_N]:
        en_cell = _kw_cell(
            "en",
            {
                "en_bg1": r["en_bg1"],
                "en_bg2": r["en_bg2"],
                "en_fg": r["en_fg"],
                "text": r["en"],
            },
        )
        he_cell = _kw_cell(
            "he",
            {
                "he_bg1": r["he_bg1"],
                "he_bg2": r["he_bg2"],
                "he_fg": r["he_fg"],
                "text": r["he"],
            },
        )
        swatch_rows.append((r["en"], en_cell, he_cell))
    swatch_html = "".join(
        f"<h3>{_esc(en)}</h3>"
        '<div class="kw-grid"><div class="grp">English</div>'
        f'<div class="grp">Hebrew</div>{en_cell}{he_cell}</div>'
        for en, en_cell, he_cell in swatch_rows
    )

    return (
        "<h2>Cross-lingual consistency (en vs he)</h2>"
        '<p class="note">For each of '
        f"{d['n']} curated en/he word pairs (tools/report.py:LANG_PAIRS, "
        "each word verified present in data/keywords.jsonl / data/terms.jsonl "
        "for both languages), the English and Hebrew text are encoded "
        "separately and compared: cosine similarity of the raw TextEncoder "
        f"embedding, Jaccard overlap of the top-{d['topk']} EmojiHead "
        "predictions, StyleHead top-1 agreement, and ColorGen output distance "
        "(Oklab, zero noise vector so only text conditioning is compared, "
        "same 9-dim card unit as elsewhere in this report). Rows sorted by "
        "color distance, worst first. ⚠ flags a pair whose emoji "
        f"predictions agree (Jaccard &ge; {d['agree_thresh']}) but whose "
        "color distance is still above the mean — an emoji/color split like "
        "the one that motivated this section.</p>"
        '<p class="note">Mean encoder cosine ' + _num(d["emb_cos_mean"])
        + f", mean emoji Jaccard@{d['topk']} " + _num(d["emoji_jaccard_mean"])
        + ", style top-1 agreement " + _pct(d["style_match_rate"])
        + ", mean color distance " + _num(d["color_dist_mean"])
        + f" (agreeing-emoji pairs, n={d['n_agree']}: "
        + _num(d["color_dist_mean_agree"]) + "; disagreeing-emoji pairs: "
        + _num(d["color_dist_mean_disagree"]) + ").</p>"
        '<table><tr><th>en</th><th>he</th>'
        '<th class="n">Embed cos</th>'
        f'<th class="n">Emoji Jaccard@{d["topk"]}</th>'
        '<th>Style (en/he)</th><th class="n">Color dist</th><th></th></tr>'
        + trows + "</table>" + swatch_html
    )


def _style_dist_html(d) -> str:
    if not d or not d.get("dist"):
        return (
            "<h2>Model — Style prediction distribution</h2>"
            '<p class="note">Unavailable — needs enc.pt / style.pt and a '
            "non-empty data/eval.jsonl.</p>"
        )
    trows = "".join(
        f"<tr><td>{_esc(r['style'])}</td>"
        f'<td class="n">{r["gt"]:.1%}</td>'
        f'<td class="n">{r["pred"]:.1%}</td>'
        f'<td class="n">{r["pred"] - r["gt"]:+.1%}</td></tr>'
        for r in d["dist"]
    )
    return (
        "<h2>Model — Style prediction distribution</h2>"
        f'<p class="note">Top-1 StyleHead prediction distribution vs. ground '
        f"truth style label distribution across {d['n']} rows of "
        "data/eval.jsonl. GT is multi-label per row so its column can sum "
        "above 100%; predicted is single-label (argmax) and always sums to "
        "100%.</p>"
        '<table><tr><th>Style</th><th class="n">GT %</th>'
        '<th class="n">Predicted %</th><th class="n">&Delta;</th></tr>'
        + trows
        + "</table>"
    )


def _block_capacity_independent_table(rows) -> str:
    by_block = {}
    for r in rows:
        by_block.setdefault(r["block"], r)
    trows = "".join(
        f'<tr style="background:{_block_bg(r["block"])}">'
        f'<td class="n"><span class="blk-dot" '
        f'style="background:{_block_color(r["block"])}"></span>{r["block"]}</td>'
        f'<td class="n">{_esc(r["range"])}</td>'
        f'<td class="n">{r["dilation"]}</td>'
        f'<td class="n">{r["w_norm_mean"]:.3f}</td>'
        f'<td class="n">{r["w_norm_max"]:.3f}</td></tr>'
        for r in by_block.values()
    )
    return (
        '<table><tr><th class="n">Block</th><th class="n">Channels</th>'
        '<th class="n">Dilation</th><th class="n">W col-norm mean</th>'
        '<th class="n">W col-norm max</th></tr>' + trows + "</table>"
    )


def _block_capacity_dependent_table(rows) -> str:
    trows = "".join(
        f'<tr style="background:{_block_bg(r["block"])}">'
        f'<td>{_esc(r["source"])}</td><td class="n">{r["n"]}</td>'
        f'<td class="n"><span class="blk-dot" '
        f'style="background:{_block_color(r["block"])}"></span>{r["block"]}</td>'
        f'<td class="n">{r["act_rms"]:.3f}</td>'
        f'<td class="n">{r["contrib_norm"]:.3f}</td>'
        f'<td class="n">{r["contrib_pct"]:.1f}%</td></tr>'
        for r in rows
    )
    return (
        "<table><tr><th>Source</th><th class=\"n\">N</th><th class=\"n\">Block</th>"
        '<th class="n">Activation RMS/ch</th><th class="n">Contribution norm</th>'
        '<th class="n">Contribution %</th></tr>' + trows + "</table>"
    )


def _block_capacity_html(d) -> str:
    if not d or not d.get("rows"):
        return (
            "<h2>Model — Encoder block capacity</h2>"
            '<p class="note">Unavailable — needs enc.pt and data from '
            "data/keywords.jsonl, data/terms.jsonl, data/eval.jsonl.</p>"
        )
    rows = d["rows"]
    return (
        "<h2>Model — Encoder block capacity</h2>"
        '<p class="note">Per-block <code>proj</code> weight column-norms and '
        "each block's actual contribution to the encoder's projected "
        "embedding, averaged over every sample in keywords/terms/eval — see "
        "<code>tools/block_capacity.py</code>. Activation RMS/ch is the "
        "per-block activation norm divided by sqrt(channel count), so it's "
        "comparable across blocks with different channel widths. Each "
        "encoder block keeps the same color across every row/source below.</p>"
        "<h3>Input-independent</h3>"
        + _block_capacity_independent_table(rows)
        + "<h3>Input-dependent</h3>"
        + _block_capacity_dependent_table(rows)
    )


def _channel_rank_html(d) -> str:
    if not d or not d.get("rows"):
        return (
            "<h2>Model — Encoder channel utilization</h2>"
            '<p class="note">Unavailable — needs enc.pt and data from '
            "data/keywords.jsonl, data/terms.jsonl, data/eval.jsonl.</p>"
        )
    trows = "".join(
        f'<tr style="background:{_block_bg(r["block"])}">'
        f'<td>{_esc(r["source"])}</td><td class="n">{r["n"]}</td>'
        f'<td class="n"><span class="blk-dot" '
        f'style="background:{_block_color(r["block"])}"></span>{r["block"]}</td>'
        f'<td class="n">{r["channels"]}</td>'
        f'<td class="n">{r["eff_rank"]:.1f}</td>'
        f'<td class="n">{r["eff_rank_pct"]:.1f}%</td>'
        f'<td class="n">{r["rank95"]}</td>'
        f'<td class="n">{r["rank95_pct"]:.1f}%</td></tr>'
        for r in d["rows"]
    )
    return (
        "<h2>Model — Encoder channel utilization</h2>"
        '<p class="note">Per-block effective dimensionality of the encoder\'s '
        "own activations (head-independent), computed by SVD of each "
        "block's centered per-sample activations over every sample in "
        "keywords/terms/eval. Effective rank is the participation ratio "
        "(&Sigma;&lambda;)&sup2;/&Sigma;&lambda;&sup2; over that block's "
        "activation-covariance eigenvalues &lambda; — an occupancy count "
        "robust to long tails, vs. Rank@95%, the raw number of components "
        "needed to explain 95% of variance. A block whose ratio sits well "
        "below 100% is spending channels that carry mostly redundant "
        "signal (over-provisioned); a block near 100% has no spare "
        "capacity left for encoding more (under-provisioned). See "
        "<code>tools/block_capacity.py</code>.</p>"
        "<table><tr><th>Source</th><th class=\"n\">N</th><th class=\"n\">Block</th>"
        '<th class="n">Channels</th><th class="n">Effective rank</th>'
        '<th class="n">Eff. rank %</th><th class="n">Rank@95%</th>'
        '<th class="n">Rank@95% %</th></tr>' + trows + "</table>"
    )


def _length_acc_html(d) -> str:
    if not d or not d.get("buckets"):
        return (
            "<h2>Model — Accuracy vs. text length</h2>"
            '<p class="note">Unavailable — needs enc.pt / emoji.pt '
            "and data/eval.jsonl.</p>"
        )
    rf, mtl = d["receptive_field"], d["max_text_len"]
    trows = "".join(
        f'<tr><td>{_esc(b["bucket"])}</td><td class="n">{b["n"]}</td>'
        + "".join(
            f'<td class="n">{a:.3f}</td>'
            for a in (b["acc_at_k"][0], b["acc_at_k"][4], b["acc_at_k"][9])
        )
        + "</tr>"
        for b in d["buckets"]
    )
    return (
        "<h2>Model — Accuracy vs. text length</h2>"
        f'<p class="note">EmojiHead Acc@k on data/eval.jsonl, bucketed by char '
        f"length around the encoder's receptive field ({rf} of {mtl} "
        "MAX_TEXT_LEN chars) — see "
        "<code>tools/analysis/length_vs_acc.py</code>.</p>"
        '<table><tr><th>Length bucket</th><th class="n">N</th>'
        '<th class="n">Acc@1</th><th class="n">Acc@5</th>'
        '<th class="n">Acc@10</th></tr>' + trows + "</table>"
    )


def _render_cards(cards: list[dict]) -> list[str]:
    proc = subprocess.run(
        ["bun", "run", "tools/data/report-cards.ts"],
        input=json.dumps(cards),
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
        check=True,
    )
    return json.loads(proc.stdout)


def _eval_samples_html(d) -> str:
    if not d or not d.get("rows"):
        return (
            "<h2>Model — Sample cards (eval.jsonl)</h2>"
            '<p class="note">Unavailable — needs enc.pt / style.pt / emoji.pt / '
            "gen.pt and data/eval.jsonl.</p>"
        )
    rows = d["rows"]
    gt_cards = [
        {
            "text": r["text"],
            "emoji": r["gt_emoji"],
            "feeling": r["gt_feeling"],
            "colors": {
                "bg1": r["gt_bg1"],
                "bg2": r["gt_bg2"],
                "text_color": r["gt_text_color"],
            },
        }
        for r in rows
    ]
    mo_cards = [
        {
            "text": r["text"],
            "emoji": r["emoji"],
            "feeling": r["feeling"],
            "colors": {"bg1": r["bg1"], "bg2": r["bg2"], "text_color": r["text_color"]},
        }
        for r in rows
    ]
    docs = _render_cards(gt_cards + mo_cards)
    gt_docs, mo_docs = docs[: len(rows)], docs[len(rows):]
    cells = []
    for i in range(0, len(rows) - 1, 2):
        cells += [gt_docs[i], gt_docs[i + 1], mo_docs[i], mo_docs[i + 1]]
    grid = "".join(
        f'<iframe class="sample-card" loading="lazy" srcdoc="{_esc(doc)}"></iframe>'
        for doc in cells
    )
    return (
        "<h2>Model — Sample cards (eval.jsonl)</h2>"
        f'<p class="note">{d["n"]} random records from data/eval.jsonl (seeded), '
        "rendered with the webapp's own card markup/CSS — ground truth (left pair) "
        "vs. live model output (right pair) per row.</p>"
        f'<div class="sample-cards-grid">{grid}</div>'
    )


_H2_RE = re.compile(r"<h2([^>]*)>(.*?)</h2>")


def _slugify(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-") or "section"


def _inject_toc_ids(html_body: str) -> tuple[str, list[tuple[str, str]]]:
    seen: dict[str, int] = {}
    items: list[tuple[str, str]] = []

    def repl(m: re.Match) -> str:
        attrs, inner = m.group(1), m.group(2)
        title = html.unescape(re.sub(r"<[^>]+>", "", inner))
        slug = _slugify(title)
        n = seen.get(slug, 0)
        seen[slug] = n + 1
        if n:
            slug = f"{slug}-{n}"
        items.append((slug, title))
        return f'<h2{attrs} id="{slug}">{inner}</h2>'

    return _H2_RE.sub(repl, html_body), items


def _toc_html(items: list[tuple[str, str]]) -> str:
    if not items:
        return ""
    lis = "".join(
        f'<li><a href="#{slug}">{_esc(title)}</a></li>' for slug, title in items)
    return f'<nav class="toc"><div class="toc-h">On this page</div><ul>{lis}</ul></nav>'


def _render_html(report) -> str:
    body = [_header_html(report)]
    if "status" in report:
        body.append(_status_html(report["status"]))
    if "style_dist" in report:
        body.append(_style_dist_html(report["style_dist"]))
    if "block_capacity" in report:
        body.append(_block_capacity_html(report["block_capacity"]))
    if "channel_rank" in report:
        body.append(_channel_rank_html(report["channel_rank"]))
    if "eval_samples" in report:
        body.append(_eval_samples_html(report["eval_samples"]))
    if "length_acc" in report:
        body.append(_length_acc_html(report["length_acc"]))
    if "data" in report:
        body.append(_data_html(report["data"]))
    if "labels" in report:
        body.append(_labels_html(report["labels"]))
    if "emoji" in report:
        body.append(_emoji_html(report["emoji"]))
    if "keywords_flex" in report:
        body.append(_keywords_flex_html(report["keywords_flex"]))
    if "keyword_fails" in report:
        body.append(_keyword_fails_html(report["keyword_fails"]))
    if "keyword" in report:
        body.append(
            _acc_chart_html(
                "Model — Keyword accuracy", "data/keywords.jsonl", report["keyword"]
            )
        )
    if "term" in report:
        body.append(
            _acc_chart_html("Model — Term accuracy",
                            "data/terms.jsonl", report["term"])
        )
    if "cards" in report:
        body.append(_cards_html(report["cards"]))
    if "gen_sensitivity" in report:
        body.append(_gen_sensitivity_html(report["gen_sensitivity"]))
    if "color_keywords" in report:
        body.append(_color_keywords_html(report["color_keywords"]))
    if "lang_consistency" in report:
        body.append(_lang_consistency_html(report["lang_consistency"]))
    html_body, toc_items = _inject_toc_ids("".join(body))
    return (
        '<!doctype html><meta charset="utf-8">'
        f"<title>emojic report — {report['provenance']['ts']}</title>"
        f"<style>{_STYLE}</style>"
        f'<div class="page">{_toc_html(toc_items)}'
        f'<div class="wrap">{html_body}</div></div>\n'
    )


_app = typer.Typer(
    add_completion=False,
    context_settings={"help_option_names": ["-h", "--help"]},
)


@_app.command()
def main(
    pt: Path = typer.Option(..., "--pt", help="Folder containing enc.pt/emoji.pt."),
    only: str = "",
    out: Path = REPORT_DIR,
) -> None:
    """Evaluate enc/emoji .pt + data files; write report/<ts>-<sha>/."""
    build_report(pt, only=only, out=out)


if __name__ == "__main__":
    _app()
