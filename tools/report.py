import html
import json
import math
import random
import re
import subprocess
import sys
import tempfile
from collections import Counter
from datetime import datetime
from functools import cache
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import typer
import yaml
from torch.nn.functional import normalize as _l2norm

from files import (
    CLDR_BASELINE_JSON,
    CLDR_JSONL,
    COLORS_JSONL,
    DATA_JSONL,
    EMOJI_EMBED_PT,
    GOAL_DIR,
    GOALS_YML,
    GROUP_JSON,
    II_JSON,
    KWPROJ_JSON,
)
from model.color import COLOR_SHIFT, rgb_to_oklab
from model.config import EMOJIS, MAX_TEXT_LEN, SEED, STYLES, Z_WEIGHT
from model.data import EVAL_PATH, TRAIN_PATH, _row_kw, read, text_to_tensor
from model.data import normalize as norm_text
from model.export_onnx import CONST_Z
from model.model import (
    ColorGen,
    EmojiEmbedding,
    EmojiHead,
    StyleHead,
    TextEncoder,
)
from model.runmeta import load_pt, run_meta

DATA_PATH = DATA_JSONL
REPO_ROOT = Path(__file__).resolve().parent.parent
KW_SEARCH_TIMEOUT = 120

EMOJI_KS = list(range(1, 11))
ACC_K_INDEX = {"acc@1": 0, "acc@5": 4, "acc@10": 9}
CLDR_MIN_KEYWORD_LEN = 3
CARD_DIST_THRESHOLD = 0.05
CARD_PURE_THRESHOLD_RGB = 0.251
CARD_PURE_THRESHOLD_L = 0.6
CARD_COLORS = ("red", "green", "blue", "dark", "bright")
GOLD_PER_COLOR = 25
KW_PRIMARY_BONUS = 0.15


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


@cache
def _emoji_embed():
    if not Path(EMOJI_EMBED_PT).exists():
        return None
    m, err = _load(EmojiEmbedding(), EMOJI_EMBED_PT)
    return None if err else m


def _emoji_extra_acc(records, tgt):
    kw_dense = torch.stack([_row_kw(r) for r in records])
    return {"keywords": [_acc_at_k(kw_dense, tgt, k).mean().item() for k in EMOJI_KS]}


def _provenance(pt: Path):
    enc_pt, emoji_pt = str(pt / "enc.pt"), str(pt / "emoji.pt")
    style_pt, gen_pt = str(pt / "style.pt"), str(pt / "gen.pt")
    emoji_embed_pt = str(pt / "emoji_embed.pt")
    rm = run_meta()
    paths = [enc_pt, emoji_pt, emoji_embed_pt, style_pt, gen_pt]
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
    for d in _rows(TRAIN_PATH):
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
        "max_text_len": MAX_TEXT_LEN,
    }


def _section_labels():
    return {"styles": len(STYLES), "emojis": len(EMOJIS)}


def _probe(words, enc, head):
    emb_tbl = _emoji_embed()
    if emb_tbl is None:
        return {"n": 0, "total": len(words), "acc_at_k": [0.0] * len(EMOJI_KS)}
    vocab = {e: i for i, e in enumerate(EMOJIS)}
    scored = []
    with torch.no_grad():
        for word, exp in words.items():
            ids = [vocab[e] for e in exp if e in vocab]
            if not ids:
                continue
            emb = enc(text_to_tensor(norm_text(word)).unsqueeze(0))
            logits = emb_tbl.score(head(emb))
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
    p = Path(II_JSON)
    if not p.exists():
        return {}
    return json.loads(p.read_text(encoding="utf-8"))


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
    emb = _emoji_embed()
    cands = _flex_keyword_candidates()
    if enc is None or emoji_head is None or emb is None or not cands:
        return {}
    idx = {e: i for i, e in enumerate(EMOJIS)}
    with torch.no_grad():
        texts = torch.stack([text_to_tensor(norm_text(kw)) for kw, _ in cands])
        order = emb.score(emoji_head(enc(texts))).argsort(dim=-1, descending=True)
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


def _cldr_keywords() -> dict:
    words = {}
    for d in _rows(str(CLDR_JSONL)):
        word = str(d.get("text", ""))
        if len(word) < CLDR_MIN_KEYWORD_LEN or not re.search(r"[a-zA-Z]", word):
            continue
        targets = words.setdefault(word, [])
        for e in str(d.get("emojis", "")).split():
            if e not in targets:
                targets.append(e)
    return words


def _cldr_probe(enc, head):
    return _probe(_cldr_keywords(), enc, head)


@cache
def _kw_proj() -> dict:
    p = Path(KWPROJ_JSON)
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8")).get("proj", {})
    except (json.JSONDecodeError, OSError):
        return {}


def _kw_exact_dense(text: str) -> torch.Tensor:
    from model.kwtokens import query_tokens

    proj = _kw_proj()
    vec = torch.zeros(len(EMOJIS))
    for word in query_tokens(text):
        pl = proj.get(word)
        if not pl:
            continue
        base = 1.0 / math.log2(1 + len(pl))
        for j, e in enumerate(pl):
            v = base + (KW_PRIMARY_BONUS if j == 0 else 0.0)
            if v > vec[e]:
                vec[e] = v
    return vec


def _rank_acc(scores, id_lists, total) -> dict:
    order = scores.argsort(dim=-1, descending=True)
    ranks = []
    for i, ids in enumerate(id_lists):
        pos = order[i].tolist()
        ranks.append(min(pos.index(j) + 1 for j in ids))
    n = len(ranks) or 1
    return {
        "n": len(ranks),
        "total": total,
        "acc_at_k": [sum(r <= k for r in ranks) / n for k in EMOJI_KS],
    }


def _kw_search_rows(rows: list) -> dict | None:
    if not rows:
        return None
    lines = [
        json.dumps({"text": word, "emojis": " ".join(EMOJIS[i] for i in ids)})
        for word, ids in rows
    ]
    tmp = tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False, encoding="utf-8")
    try:
        tmp.write("\n".join(lines) + "\n")
        tmp.close()
        proc = subprocess.run(
            ["bun", "run", "tools/analysis/kw-search.ts", tmp.name, "--json"],
            capture_output=True,
            text=True,
            cwd=REPO_ROOT,
            timeout=KW_SEARCH_TIMEOUT,
        )
        if proc.returncode != 0:
            return None
        return json.loads(proc.stdout)
    except (OSError, subprocess.TimeoutExpired, json.JSONDecodeError):
        return None
    finally:
        Path(tmp.name).unlink(missing_ok=True)


def _section_keyword(enc, head) -> dict:
    words = _cldr_keywords()
    if not words or not _kw_proj():
        return {}
    vocab = {e: i for i, e in enumerate(EMOJIS)}
    rows = []
    for word, exp in words.items():
        ids = [vocab[e] for e in exp if e in vocab]
        if ids:
            rows.append((word, ids))
    if not rows:
        return {}
    total = len(words)
    id_lists = [ids for _, ids in rows]
    kw_dense = torch.stack([_kw_exact_dense(word) for word, _ in rows])
    out = {"exact": _rank_acc(kw_dense, id_lists, total)}

    emb = _emoji_embed()
    if enc is not None and head is not None and emb is not None:
        with torch.no_grad():
            texts = torch.stack([text_to_tensor(norm_text(w)) for w, _ in rows])
            text_emb = enc(texts)
            logit_m = emb.score(head(text_emb))
        out["model"] = _rank_acc(logit_m, id_lists, total)

    search = _kw_search_rows(rows)
    if search is not None:
        by_text = {r["text"]: r["kw"] for r in search.get("rows", [])}
        fuzzy_dense = torch.stack([_row_kw({"kw": by_text.get(w, [])}) for w, _ in rows])
        out["fuzzy"] = _rank_acc(fuzzy_dense, id_lists, total)
    return out


@cache
def _cldr_baseline():
    p = Path(CLDR_BASELINE_JSON)
    if not p.exists():
        return None
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
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
    p = Path(GROUP_JSON)
    if not p.exists():
        return {}
    return json.loads(p.read_text(encoding="utf-8"))


@cache
def _load_global_goals() -> dict:
    p = Path(GOALS_YML)
    if not p.exists():
        return {}
    return (yaml.safe_load(p.read_text(encoding="utf-8")) or {}).get("goals") or {}


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


def _grade_merged(cur, iter_target, global_target, direction) -> str:
    if cur is None or global_target is None:
        return "na"
    if _meets(cur, global_target, direction):
        return "good"
    if _meets(cur, iter_target, direction):
        return "amber"
    return "red"


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


def _set_nested(root: dict, path: list, value) -> None:
    for k in path[:-1]:
        root = root.setdefault(k, {})
    root[path[-1]] = value


def _flatten_leaves(node, prefix="") -> dict:
    out = {}
    for k, v in (node or {}).items():
        dotted = f"{prefix}.{k}" if prefix else k
        if isinstance(v, dict):
            out.update(_flatten_leaves(v, dotted))
        else:
            out[dotted] = v
    return out


def _iter_goal_leaves() -> dict:
    loaded = _load_goals()
    if loaded is None:
        return {}
    _path, doc = loaded
    return _flatten_leaves(doc.get("goals") or {})


@cache
def _load_goals():
    d = Path(GOAL_DIR)
    if not d.is_dir():
        return None
    files = sorted([*d.glob("*.yml"), *d.glob("*.yaml")])
    if not files:
        return None
    path = files[-1]
    return str(path), (yaml.safe_load(path.read_text(encoding="utf-8")) or {})


def _goal_specs() -> dict:
    def full_text(i):
        return lambda r: _dig(_best_emoji_acc(_dig(r, "emoji", "eval") or {})[1] or [], i)

    def exact_kw(i):
        return lambda r: _dig(r, "keyword", "exact", "acc_at_k", i)

    def fuzzy_kw(i):
        return lambda r: _dig(r, "keyword", "fuzzy", "acc_at_k", i)

    def style(i):
        return lambda r: _dig(r, "cards", "style_acc_at_k", i)

    def color(c):
        return lambda r: _dig(r, "cards", "per_color", c, "gt_mean_distance")

    specs = {
        "max text len": ("max", lambda r: _dig(r, "data", "max_text_len")),
        "vocabulary.size": ("max", lambda r: _dig(r, "labels", "emojis")),
        "color generator.energy distance.global": (
            "min",
            lambda r: _dig(r, "cards", "energy"),
        ),
    }
    for name, idx in ACC_K_INDEX.items():
        specs[f"emoji prediction.exact keyword.{name}"] = ("max", exact_kw(idx))
        specs[f"emoji prediction.fuzzy keyword.{name}"] = ("max", fuzzy_kw(idx))
        specs[f"emoji prediction.full text.{name}"] = ("max", full_text(idx))
        specs[f"style prediction.full text.{name}"] = ("max", style(idx))
    for c in CARD_COLORS:
        specs[f"color generator.energy distance.{c}"] = ("min", color(c))
    return specs


_COVERAGE_PREFIX = "vocabulary.coverage."


def _goal_spec_for(dotted, specs):
    spec = specs.get(dotted)
    if spec is not None:
        return spec
    if dotted.startswith(_COVERAGE_PREFIX):
        grp = dotted[len(_COVERAGE_PREFIX) :]
        return (
            "max",
            lambda r, grp=grp: _dig(r, "status", "vocab_coverage", "groups", grp, "score"),
        )
    return ("max", lambda r: None)


def _section_goals(report) -> dict | None:
    loaded = _load_goals()
    if loaded is None:
        return None
    path, doc = loaded
    tree = doc.get("goals") or {}
    specs = _goal_specs()
    compare: dict = {}
    counts = Counter()

    def walk(node, prefix):
        for k, v in node.items():
            dotted = f"{prefix}.{k}" if prefix else k
            if isinstance(v, dict):
                walk(v, dotted)
                continue
            direction, fn = _goal_spec_for(dotted, specs)
            actual = fn(report)
            if actual is None:
                met, delta = None, None
                counts["unmeasured"] += 1
            else:
                met = actual >= v if direction == "max" else actual <= v
                delta = round(actual - v, 4)
                counts["met" if met else "unmet"] += 1
            _set_nested(
                compare,
                dotted.split("."),
                {
                    "target": v,
                    "actual": actual,
                    "met": met,
                    "dir": direction,
                    "delta": delta,
                },
            )

    walk(tree, "")
    return {
        "source_file": path,
        "meta": doc.get("meta", {}),
        "compare": compare,
        "summary": dict(counts),
    }


def _acc_rows(
    goals,
    priority,
    label,
    dotted_prefix,
    targets,
    iter_leaves,
    values,
    note,
    direction="max",
):
    for name, idx in ACC_K_INDEX.items():
        tgt = (targets or {}).get(name)
        iter_tgt = iter_leaves.get(f"{dotted_prefix}.{name}")
        cur = values[idx] if values and len(values) > idx else None
        goals.append(
            {
                "goal": f"{label} {name}",
                "priority": priority,
                "dir": direction,
                "target": "—" if tgt is None else f"≥ {tgt:.2f}",
                "iter_target": "—" if iter_tgt is None else f"≥ {iter_tgt:.2f}",
                "current": cur,
                "status": _grade_merged(cur, iter_tgt, tgt, direction),
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
    energy_tgt = (g.get("color generator") or {}).get("energy distance") or {}
    vocab_g = g.get("vocabulary") or {}
    iter_leaves = _iter_goal_leaves()

    best_name, best = _best_emoji_acc(emoji_eval)
    kw_exact = keyword.get("exact") or {}
    kw_fuzzy = keyword.get("fuzzy") or {}
    goals = []

    _acc_rows(
        goals,
        1,
        "Exact keyword",
        "emoji prediction.exact keyword",
        ep.get("exact keyword"),
        iter_leaves,
        kw_exact.get("acc_at_k"),
        "standalone kw-search on cldr.jsonl keywords (exact-matched kw vector)",
    )
    _acc_rows(
        goals,
        2,
        "Full-text emoji",
        "emoji prediction.full text",
        ep.get("full text"),
        iter_leaves,
        best or [],
        f"best variant: {best_name}" if best_name else "emoji.eval not evaluated this run",
    )
    _acc_rows(
        goals,
        3,
        "Fuzzy keyword",
        "emoji prediction.fuzzy keyword",
        ep.get("fuzzy keyword"),
        iter_leaves,
        kw_fuzzy.get("acc_at_k"),
        "standalone kw-search on fuzzy-matched kw vector"
        if kw_fuzzy.get("acc_at_k")
        else "unmeasured — needs bun + web/public/kwproj.json (tools/analysis/kw-search.ts)",
    )

    energy_global = cards.get("energy")
    iter_energy_global = iter_leaves.get("color generator.energy distance.global")
    goals.append(
        {
            "goal": "Color energy · global",
            "priority": 4,
            "dir": "min",
            "target": "—"
            if energy_tgt.get("global") is None
            else f"≤ {energy_tgt['global']:.2f}",
            "iter_target": "—"
            if iter_energy_global is None
            else f"≤ {iter_energy_global:.2f}",
            "current": energy_global,
            "status": _grade_merged(
                energy_global, iter_energy_global, energy_tgt.get("global"), "min"
            ),
            "note": "cards.energy — not wired yet" if energy_global is None else "",
        }
    )
    for c in CARD_COLORS:
        cur = _dig(cards, "per_color", c, "gt_mean_distance")
        iter_tgt = iter_leaves.get(f"color generator.energy distance.{c}")
        goals.append(
            {
                "goal": f"Color energy · {c}",
                "priority": 4,
                "dir": "min",
                "target": "—" if energy_tgt.get(c) is None else f"≤ {energy_tgt[c]:.2f}",
                "iter_target": "—" if iter_tgt is None else f"≤ {iter_tgt:.2f}",
                "current": cur,
                "status": _grade_merged(cur, iter_tgt, energy_tgt.get(c), "min"),
                "note": "cards off this run" if cur is None else "",
            }
        )

    _acc_rows(
        goals,
        5,
        "Style",
        "style prediction.full text",
        sp.get("full text"),
        iter_leaves,
        cards.get("style_acc_at_k"),
        "cards off this run" if not cards.get("style_acc_at_k") else "",
    )

    mtl = g.get("max text len")
    mtl_iter = iter_leaves.get("max text len")
    mtl_cur = data.get("max_text_len")
    goals.append(
        {
            "goal": "Max text len",
            "priority": 6,
            "dir": "max",
            "target": "—" if mtl is None else f"≥ {mtl}",
            "iter_target": "—" if mtl_iter is None else f"≥ {mtl_iter}",
            "current": mtl_cur,
            "status": _grade_merged(mtl_cur, mtl_iter, mtl, "max"),
            "note": f"config.MAX_TEXT_LEN = {mtl_cur}" if mtl_cur is not None else "",
        }
    )

    size_tgt = vocab_g.get("size")
    size_iter = iter_leaves.get("vocabulary.size")
    vocab_size = len(EMOJIS)
    goals.append(
        {
            "goal": "Emoji vocab size",
            "priority": 7,
            "dir": "max",
            "target": "—" if size_tgt is None else f"≥ {size_tgt}",
            "iter_target": "—" if size_iter is None else f"≥ {size_iter}",
            "current": vocab_size,
            "status": _grade_merged(vocab_size, size_iter, size_tgt, "max"),
            "note": "",
        }
    )

    higher_open = any(x["status"] in ("red", "amber") for x in goals if x["priority"] <= 7)
    vc = _vocab_coverage()
    cov_iter_targets = {
        k[len(_COVERAGE_PREFIX) :]: v
        for k, v in iter_leaves.items()
        if k.startswith(_COVERAGE_PREFIX)
    }
    vc_iter = _vocab_coverage(cov_iter_targets) if cov_iter_targets else None
    if not vc.get("measurable"):
        goals.append(
            {
                "goal": "Vocab coverage (per Unicode group)",
                "priority": 8,
                "dir": "max",
                "target": "all groups ≥ target",
                "iter_target": "—",
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
        weak = " · weakest: " + ", ".join(f"{k} {d:+.2f}" for d, k in worst) if worst else ""
        iter_txt = (
            "—"
            if vc_iter is None
            else f"{vc_iter['groups_passed']}/{vc_iter['groups_measurable']}"
            " (this iteration)"
        )
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
                "iter_target": iter_txt,
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
    emb = _emoji_embed()
    if enc is None or head is None or emb is None:
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
            q_txt = head(enc_emb)
            logits = emb.score(q_txt)
        extra = _emoji_extra_acc(rows, tgt)
        d["eval"] = {
            "n": len(rows),
            "acc_at_k": [_acc_at_k(logits, tgt, k).mean().item() for k in EMOJI_KS],
            "keywords_acc_at_k": extra.get("keywords"),
            "baseline": _cldr_baseline(),
        }
    return d


def _section_cldr(enc, head):
    emb = _emoji_embed()
    if enc is None or head is None or emb is None:
        return {}
    return dict(_cldr_probe(enc, head))


def _gold_rows():
    rows = [
        r
        for r in _rows(COLORS_JSONL)
        if isinstance(r.get("bg"), list)
        and len(r["bg"]) == 2
        and r.get("fg")
        and 0 < len(norm_text(str(r.get("text", "")))) <= MAX_TEXT_LEN
    ]
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
    emb_tbl = _emoji_embed()
    if None in (enc, style_head, emoji_head, gen, emb_tbl) or not gold_rows:
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
        elog = emb_tbl.score(emoji_head(emb))
        slog = style_head(emb)
        seed = (1 - Z_WEIGHT) * _l2norm(emb)[:, None, :] + Z_WEIGHT * CONST_Z[None, :, :]
        raw = gen.net(seed.reshape(-1, seed.shape[-1]))
        palettes = (torch.tanh(raw) * 127.5).reshape(len(rows), CONST_Z.shape[0], 9)
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
        df = min(
            _card_distance(palettes[i, k].tolist(), gold9, r["color"])
            for k in range(palettes.shape[1])
        )
        flat = palettes[i, 0].tolist()
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
                "dF": df,
                "hit_pure": dp < _pure_threshold(r["color"]),
                "hit": df < CARD_DIST_THRESHOLD,
            }
        )

    def _stats(rs):
        n = len(rs) or 1
        return {
            "pure_accuracy": sum(x["hit_pure"] for x in rs) / n,
            "pure_mean_distance": sum(x["dP"] for x in rs) / n,
            "gt_accuracy": sum(x["dF"] < CARD_DIST_THRESHOLD for x in rs) / n,
            "gt_mean_distance": sum(x["dF"] for x in rs) / n,
        }

    per_color = {c: _stats([x for x in out_rows if x["color"] == c]) for c in CARD_COLORS}
    per_color["all"] = _stats(out_rows)
    return {
        "n": len(rows),
        "threshold": CARD_DIST_THRESHOLD,
        "pure_threshold_rgb": CARD_PURE_THRESHOLD_RGB,
        "pure_threshold_l": CARD_PURE_THRESHOLD_L,
        "emoji_acc_at_k": emoji_acc,
        "style_acc_at_k": style_acc,
        "per_color": per_color,
        "rows": out_rows,
    }


def build_report(pt: Path, only: str = "", out: str = "report") -> Path:
    want = {s.strip() for s in only.split(",") if s.strip()} or {
        "data",
        "labels",
        "emoji",
        "keywords_flex",
        "cldr",
        "cards",
    }
    enc_pt, emoji_pt = pt / "enc.pt", pt / "emoji.pt"
    style_pt, gen_pt = pt / "style.pt", pt / "gen.pt"
    prov = _provenance(pt)

    enc = emoji_head = style_head = gen = None
    need_enc = bool({"emoji", "cldr", "cards", "keywords_flex"} & want)
    if need_enc and enc_pt.exists():
        enc, err = _load(TextEncoder(), enc_pt)
        if err:
            prov["issues"].append(f"{enc_pt} could not load: {err}")
    if enc is not None and emoji_pt.exists():
        emoji_head, err = _load(EmojiHead(), emoji_pt)
        if err:
            prov["issues"].append(f"{emoji_pt} could not load: {err}")
    if "cards" in want and enc is not None:
        if style_pt.exists():
            style_head, err = _load(StyleHead(), style_pt)
            if err:
                prov["issues"].append(f"{style_pt} could not load: {err}")
        if gen_pt.exists():
            gen, err = _load(ColorGen(), gen_pt)
            if err:
                prov["issues"].append(f"{gen_pt} could not load: {err}")
    prov["consistent"] = not prov["issues"]

    eval_records = list(read(EVAL_PATH)) if "emoji" in want else []
    gold_rows = _gold_rows() if "cards" in want else ()

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
    if "cldr" in want:
        report["cldr"] = _section_cldr(enc, emoji_head)
        report["keyword"] = _section_keyword(enc, emoji_head)
    if "cards" in want:
        report["cards"] = _section_cards(enc, style_head, emoji_head, gen, gold_rows)

    report["status"] = _section_status(report)
    goals = _section_goals(report)
    if goals is not None:
        report["goals"] = goals

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
--warn-bd:#efd4a2;--warn-fg:#a25c07;--bad-bg:#fdeaea;--bad-bd:#f0b6b6;
--bad-fg:#b3261e}
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
.status-h{margin-top:34px}
.scorecard td:first-child{font-weight:600}
.scorecard .gnote{font-weight:400;font-size:13px;color:var(--dim);margin-top:3px}
.scorecard tr.sc-good{background:var(--good-bg)}
.scorecard tr.sc-amber{background:var(--warn-bg)}
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
table{border-collapse:collapse;width:100%;margin:16px 0;font-size:16px}
th,td{border-bottom:1px solid var(--line);padding:10px 12px;text-align:left}
th{font-size:13px;color:var(--dim);text-transform:uppercase;letter-spacing:.03em}
td.n,th.n{text-align:right;font-variant-numeric:tabular-nums}
tr:last-child td{border-bottom:none}
.cards-grid{display:grid;grid-template-columns:repeat(5,1fr);gap:10px;margin:18px 0 0}
.mini{aspect-ratio:1/1;border-radius:12px;padding:12px 10px;display:flex;
flex-direction:column;justify-content:center;align-items:center;text-align:center;
overflow:hidden}
.mini .em{font-size:24px;line-height:1}
.mini .tx{font-size:12px;font-weight:600;margin-top:6px;overflow-wrap:anywhere;
line-height:1.3}
.mini .st{font-size:9px;letter-spacing:.06em;text-transform:uppercase;margin-top:6px;
opacity:.75}
"""


def _esc(x) -> str:
    return html.escape(str(x))


def _fnum(n) -> str:
    return f"{n:,}"


def _hex_to_offsets(hx: str) -> list[float]:
    hx = hx.lstrip("#")
    return [int(hx[i : i + 2], 16) - COLOR_SHIFT for i in (0, 2, 4)]


def _offsets_to_hex(vals) -> str:
    ints = [max(0, min(255, round(v + COLOR_SHIFT))) for v in vals]
    return "#" + "".join(f"{v:02x}" for v in ints)


def _card_distance(pred9, gold9, color: str) -> float:
    p = rgb_to_oklab(torch.tensor(pred9, dtype=torch.float32)).reshape(3, 3)
    g = rgb_to_oklab(torch.tensor(gold9, dtype=torch.float32)).reshape(3, 3)
    if color in ("dark", "bright"):
        d = (p[:, 0] - g[:, 0]).abs()
    else:
        d = (p - g).norm(dim=-1)
    return d.mean().item()


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
    pure = rgb_to_oklab(torch.tensor(_hex_to_offsets(PURE_HEX[color]), dtype=torch.float32))
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
    "amber": "🟡 close",
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
        note = f'<div class="gnote">{_esc(g["note"])}</div>' if g.get("note") else ""
        rows.append(
            f'<tr class="sc-{g["status"]}"><td>{_esc(g["goal"])}{note}</td>'
            f'<td class="n">{_esc(g["target"])}</td>'
            f'<td class="n">{_esc(g.get("iter_target", "—"))}</td>'
            f'<td class="n">{cur_txt}</td>'
            f"<td>{_STATUS_WORD[g['status']]}</td></tr>"
        )
    out = [
        '<h2 class="status-h">Goal status</h2>'
        '<table class="scorecard"><tr><th>Goal — priority order</th>'
        '<th class="n">Global target</th><th class="n">Current target</th>'
        '<th class="n">Current value</th><th>Status</th></tr>' + "".join(rows) + "</table>"
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
            '<th class="n">Target</th></tr>' + "".join(grp_rows) + "</table></details>"
        )
    return "".join(out)


_GOAL_WORD = {
    True: "✅ met",
    False: "🔴 miss",
    None: "⚪ unmeasured",
}


def _flatten_compare(node, prefix=""):
    for k, v in node.items():
        dotted = f"{prefix}.{k}" if prefix else k
        if isinstance(v, dict) and "target" in v and "dir" in v:
            yield dotted, v
        elif isinstance(v, dict):
            yield from _flatten_compare(v, dotted)


def _goals_html(goals) -> str:
    rows = []
    for path, leaf in _flatten_compare(goals["compare"]):
        met = leaf["met"]
        klass = {True: "sc-good", False: "sc-red", None: "sc-na"}[met]
        op = "≥" if leaf["dir"] == "max" else "≤"
        actual = leaf["actual"]
        if actual is None:
            act_txt = "n/a"
        elif isinstance(actual, float):
            act_txt = f"{actual:.3f}"
        else:
            act_txt = _fnum(actual)
        delta = leaf["delta"]
        delta_txt = "" if delta is None else f" ({delta:+g})"
        rows.append(
            f'<tr class="{klass}"><td>{_esc(path)}</td>'
            f'<td class="n">{op} {_esc(leaf["target"])}</td>'
            f'<td class="n">{act_txt}{delta_txt}</td>'
            f"<td>{_GOAL_WORD[met]}</td></tr>"
        )
    s = goals["summary"]
    tally = " · ".join(f"{s[k]} {k}" for k in ("met", "unmet", "unmeasured") if s.get(k))
    meta = goals.get("meta") or {}
    rationale = meta.get("rationale")
    note = f'<div class="note">{_esc(rationale.strip())}</div>' if rationale else ""
    return (
        '<h2 class="status-h">Goals for this iteration — per-leaf detail '
        f'<span class="sub">{_esc(Path(goals["source_file"]).name)}</span></h2>'
        f"{note}"
        '<table class="scorecard"><tr><th>Goal</th>'
        '<th class="n">Target</th><th class="n">Actual</th><th>Result</th></tr>'
        + "".join(rows)
        + "</table>"
        f'<div class="note">{_esc(tally)}</div>'
    )


def _word_acc_table_html(title, model_vals, search_vals, missing_note) -> str:
    if not (model_vals or search_vals):
        return f'<h2>{_esc(title)}</h2><p class="note">{_esc(missing_note)}</p>'

    def cell(vals, idx):
        return "—" if not vals else f"{vals[idx]:.3f}"

    rows = "".join(
        f"<tr><td>{_esc(k)}</td>"
        f'<td class="n">{cell(model_vals, idx)}</td>'
        f'<td class="n">{cell(search_vals, idx)}</td></tr>'
        for k, idx in ACC_K_INDEX.items()
    )
    return (
        f"<h2>{_esc(title)}</h2>"
        '<table><tr><th></th><th class="n">EmojiHead</th>'
        '<th class="n">Search</th></tr>'
        f"{rows}</table>"
    )


def _exact_word_accuracy_html(report) -> str:
    kw = report.get("keyword") or {}
    return _word_acc_table_html(
        "Exact Word Accuracy",
        (kw.get("model") or {}).get("acc_at_k"),
        (kw.get("exact") or {}).get("acc_at_k"),
        "keyword predictor unavailable — needs enc.pt / emoji.pt / emoji_embed.pt "
        "and web/public/kwproj.json.",
    )


def _full_text_accuracy_html(report) -> str:
    e = (report.get("emoji") or {}).get("eval") or {}
    return _word_acc_table_html(
        "Full Text Accuracy",
        e.get("acc_at_k"),
        e.get("keywords_acc_at_k"),
        "emoji eval unavailable — needs enc.pt / emoji.pt / emoji_embed.pt.",
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
    bars = _bars([[str(length), count] for length, count in dist], maxv, rotated=True)
    mtl = d.get("max_text_len")
    mtl_html = f'<p class="note">MAX_TEXT_LEN = {mtl}</p>' if mtl is not None else ""
    return (
        f'<h2>Data</h2><div class="counts">{cells}</div>{mtl_html}'
        "<h3>Text length distribution — train.jsonl (normalized, longest first)</h3>"
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
        bl = e.get("baseline")
        series = [("Keywords", e["keywords_acc_at_k"], "lline2")] if e.get(
            "keywords_acc_at_k"
        ) else []
        if series:
            legend = ("EmojiHead", *(name for name, _, _ in series))
        elif bl:
            legend = ("model", f"CLDR baseline ({bl['name']})")
        else:
            legend = None
        chart = _linechart(
            points,
            baseline=bl["acc_at_k"] if bl else None,
            legend=legend,
            series=series or None,
        )
        note = (
            ""
            if bl
            else '<p class="note">CLDR baseline unavailable — run '
            "<code>bun run regen</code> to write "
            "<code>data/cldr-baseline.json</code>.</p>"
        )
        out.append(f"<h3>Performance on eval.jsonl ({e['n']} rows)</h3>{chart}{note}")
    return "".join(out)


def _keywords_flex_html(d) -> str:
    if not d:
        return (
            "<h2>Model — Keyword vocab</h2>"
            '<p class="note">enc.pt / emoji.pt / emoji_embed.pt not available.</p>'
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


def _cldr_html(d) -> str:
    if not d:
        return '<h2>CLDR</h2><p class="note">enc.pt / emoji.pt not available.</p>'
    points = list(zip((str(k) for k in EMOJI_KS), d["acc_at_k"], strict=True))
    return (
        "<h2>CLDR</h2>"
        f"<h3>Model-only EmojiHead on cldr.jsonl ({d['n']} words)</h3>"
        f"{_linechart(points)}"
    )


def _keyword_html(d) -> str:
    ex = (d or {}).get("exact")
    if not ex:
        return ""
    points = list(zip((str(k) for k in EMOJI_KS), ex["acc_at_k"], strict=True))
    series = []
    if d.get("model"):
        series.append(("EmojiHead", d["model"]["acc_at_k"], "lline3"))
    if d.get("fuzzy"):
        series.append(("Standalone search (fuzzy)", d["fuzzy"]["acc_at_k"], "lline4"))
    legend = ("Exact kw", *(name for name, _, _ in series)) if series else None
    return (
        "<h2>Keyword predictor — CLDR</h2>"
        f"<h3>Non-learned kw predictor on cldr.jsonl "
        f"({ex['n']}/{ex['total']} in-vocab keywords)</h3>"
        f"{_linechart(points, legend=legend, series=series or None)}"
    )


def _cards_html(d) -> str:
    if not d:
        return (
            "<h2>Cards</h2>"
            '<p class="note">Unavailable — needs enc.pt / style.pt / emoji.pt / '
            "emoji_embed.pt / gen.pt and a non-empty data/colors.jsonl.</p>"
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
    trows = "".join(
        f"<tr><td>{_esc(c)}</td>"
        f'<td class="n">{pc[c]["pure_mean_distance"]:.3f}</td>'
        f'<td class="n">{pc[c]["pure_accuracy"]:.2f}</td>'
        f'<td class="n">{pc[c]["gt_mean_distance"]:.3f}</td>'
        f'<td class="n">{pc[c]["gt_accuracy"]:.2f}</td></tr>'
        for c in (*CARD_COLORS, "all")
    )
    out.append(
        "<h3>Colour distance — d(model, pure) then d(model, GT)</h3>"
        '<p class="note">Pure acc: dP &lt; '
        f"{d['pure_threshold_rgb']:.3f} (r/g/b), &lt; {d['pure_threshold_l']:.2f} "
        f"(|&Delta;L|, dark/bright). GT acc: dF &lt; {d['threshold']:.2f}.</p>"
        "<table><tr><th>Color</th>"
        '<th class="n">Pure dist</th><th class="n">Pure acc</th>'
        '<th class="n">GT dist</th><th class="n">GT acc</th></tr>'
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


def _render_html(report) -> str:
    body = [_header_html(report)]
    if "status" in report:
        body.append(_status_html(report["status"]))
    if "goals" in report:
        body.append(_goals_html(report["goals"]))
    if "keyword" in report:
        body.append(_exact_word_accuracy_html(report))
    if "emoji" in report:
        body.append(_full_text_accuracy_html(report))
    if "data" in report:
        body.append(_data_html(report["data"]))
    if "labels" in report:
        body.append(_labels_html(report["labels"]))
    if "emoji" in report:
        body.append(_emoji_html(report["emoji"]))
    if "keywords_flex" in report:
        body.append(_keywords_flex_html(report["keywords_flex"]))
    if "keyword" in report:
        body.append(_keyword_html(report["keyword"]))
    if "cldr" in report:
        body.append(_cldr_html(report["cldr"]))
    if "cards" in report:
        body.append(_cards_html(report["cards"]))
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
