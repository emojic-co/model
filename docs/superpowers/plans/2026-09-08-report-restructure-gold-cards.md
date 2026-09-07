# Report restructure + gold-standard cards Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reshape `tools/report.py` to drop the per-failure tables, keep keyword/CLDR retrieval as acc@k charts only, and add an end-to-end "Cards" section that runs the shipped inference graph against a hand-authored gold-standard dataset; also delete `upsample --report` / `--cldr`.

**Architecture:** A new committed `data/gold.jsonl` (125 rows, 25 per colour) drives a new `_section_cards` in `tools/report.py`. Cards reuses the exact colour math from `model/export_onnx.py` (the 5 constant-noise palettes `CONST_Z`, the `(1-Z_WEIGHT)*l2norm(emb)+Z_WEIGHT*z` mix, `tanh*127.5` in **offset** space) plus `model/color.py:rgb_to_oklab` for a per-colour distance `dF`. `report.py` gains non-fatal `style.pt`/`gen.pt` loads. The keyword/CLDR probes lose their miss-table machinery. The Bun `upsample.ts` loses two report-coupled modes and `tools/data/report.ts` is deleted.

**Tech Stack:** Python 3.13, PyTorch (CPU), Typer, `uv`; Bun/TypeScript for the data toolchain; plain-assert test scripts (no pytest), `bun test` for TS.

**Spec:** `docs/superpowers/specs/2026-09-08-report-restructure-gold-cards-design.md`

## Global Constraints

- **No comments or docstrings** in source. Do not add any. Keep existing `# type: ignore` / `# noqa` / shebangs and the pre-existing Typer command docstrings; do not add new ones.
- Package management is `uv` only — never `pip install`. Bun toolchain uses `bun`.
- `MAX_TEXT_LEN = 42`; `model/data.py:normalize` lowercases, collapses whitespace, collapses any 3+ char run to 2, drops non-vocab chars.
- OKLab: `model/color.py:rgb_to_oklab` expects sRGB **byte offsets** (value − 127.5, roughly [-127.5, 127.5]); it adds 127.5 and divides by 255 internally. Its output L channel is index 0 of the last dim.
- The shipped colour graph (must stay byte-identical): `seed = (1 - Z_WEIGHT) * l2norm(emb) + Z_WEIGHT * z` with `z` one of the 5 rows of `model.export_onnx.CONST_Z`; palette `= tanh(gen.net(seed)) * 127.5` in offset space (the web/ONNX path then adds 127.5).
- `EMOJIS` is a dynamic vocab from `data/labels.json` (currently 750); `STYLES` is the fixed 21.
- Run all Python from the repo root; all tool scripts assume repo root as CWD.
- Verify Python with `uv run ruff check .` and `uv run ruff format --check .` — do not run training.

---

## File Structure

- **Modify** `files.py` — add `GOLD_JSONL`.
- **Modify** `files.ts` — add `GOLD_JSONL`.
- **Create** `data/gold.jsonl` — 125 committed gold rows (25 per colour). Schema = `data/data.jsonl` (`text`, `emojis` space-string, `styles` list, `bg` two hex, `fg` one hex) plus `color` ∈ {red,green,blue,dark,bright}.
- **Create** `tools/test_report.py` — plain-assert checks (gold-file integrity, `_card_distance`, `_linechart` series). Typer app like `model/test_runmeta.py`.
- **Modify** `tools/report.py` — the bulk: simplify `_probe`, drop miss tables + `missed_texts`, extend `_linechart`, add `_section_cards` + `_cards_html`, add `style.pt`/`gen.pt` loads, extend `_provenance`, `_STYLE` additions, CLI help.
- **Modify** `tools/data/upsample.ts` — remove `--report` and `--cldr` modes and their helpers.
- **Delete** `tools/data/report.ts` — only `upsample.ts` imports it.
- **Modify** `tools/data/upsample.test.ts` — drop `failingEmojis` / `missedCldrKeywords` cases.
- **Modify** `CLAUDE.md` — update the `upsample.ts` paragraph; add `tools/test_report.py` to the quick-verify list.

---

## Task 1: `GOLD_JSONL` constant + author `data/gold.jsonl`

**Files:**
- Modify: `files.py` (after `PRED_JSONL` line)
- Modify: `files.ts` (after `PRED_JSONL` line)
- Create: `data/gold.jsonl`
- Create: `tools/test_report.py`

**Interfaces:**
- Produces: `from files import GOLD_JSONL` → `"data/gold.jsonl"`; `import { GOLD_JSONL } from "./files.ts"`.
- Produces: `data/gold.jsonl` — 125 lines, exactly 25 with each `color` value, every `emojis` token in the current `data/labels.json` vocab, every `styles` entry one of the 21, `bg`/`fg` valid `#rrggbb`, `normalize(text)` non-empty and ≤ 42 chars.
- Produces: `tools/test_report.py:test_gold_file_integrity()`.

- [ ] **Step 1: Add the path constant**

`files.py` — add under `PRED_JSONL = f"{DATA_DIR}/pred.jsonl"`:

```python
GOLD_JSONL = f"{DATA_DIR}/gold.jsonl"
```

`files.ts` — add under `export const PRED_JSONL = ...`:

```typescript
export const GOLD_JSONL = `${DATA_DIR}/gold.jsonl`
```

- [ ] **Step 2: Author `data/gold.jsonl`**

One compact JSON object per line, key order `text, emojis, styles, bg, fg, color`. 25 rows per colour, in colour blocks (red, then green, then blue, then dark, then bright). Rules:

- `text`: a short (roughly 12–40 char) phrase with a **strong, unambiguous** tie to the colour — a canonical object, scene, or statement. Vary voice/mood so the style predictions have something to bite on. Must survive `normalize` to a non-empty string ≤ 42 chars (ASCII, ordinary punctuation only).
- `emojis`: 1–3 space-separated emojis, **every one present in `data/labels.json`**, that a careful annotator would pick for that text. Prefer specific over generic.
- `styles`: 1–2 labels from exactly this set — `Joyful, Excited, Hopeful, Serene, Tender, Playful, Whimsical, Awed, Earnest, Determined, Proud, Wistful, Melancholy, Anxious, Tense, Furious, Irritated, Disgusted, Startled, Sarcastic, Deadpan`.
- `bg`: two `#rrggbb` stops forming a gradient that plainly reads as the colour. `fg`: one `#rrggbb` with good contrast on that gradient.
- Double-check every row against these rules a second time before moving on.

Seed rows (use verbatim, then continue each block to 25 in the same spirit):

```
{"text":"fresh strawberries in a white bowl","emojis":"🍓","styles":["Joyful"],"bg":["#ff4d4d","#c81d1d"],"fg":"#fff0f0","color":"red"}
{"text":"stop the car right now","emojis":"🛑 🚗","styles":["Tense","Startled"],"bg":["#e23b3b","#a10f0f"],"fg":"#ffffff","color":"red"}
{"text":"her crimson lipstick","emojis":"💄 💋","styles":["Playful"],"bg":["#d21f3c","#7a0b1e"],"fg":"#ffe9ee","color":"red"}
{"text":"a ripe tomato from the vine","emojis":"🍅","styles":["Earnest"],"bg":["#ff5544","#b81c1c"],"fg":"#fff4f2","color":"red"}
{"text":"i am so furious right now","emojis":"😡 🔥","styles":["Furious"],"bg":["#c0140f","#5e0606"],"fg":"#ffdede","color":"red"}
{"text":"moss on the forest floor","emojis":"🌿","styles":["Serene"],"bg":["#3fa34d","#1e6b2e"],"fg":"#f0fff2","color":"green"}
{"text":"a crisp green apple","emojis":"🍏 🍎","styles":["Playful"],"bg":["#5bd15b","#2e9e2e"],"fg":"#f4fff4","color":"green"}
{"text":"fresh basil from the garden","emojis":"🌱 🌿","styles":["Hopeful"],"bg":["#4caf50","#256d29"],"fg":"#f2fff3","color":"green"}
{"text":"the traffic light finally turned","emojis":"🚦","styles":["Determined"],"bg":["#43c463","#1c7a3a"],"fg":"#ffffff","color":"green"}
{"text":"a four leaf clover for luck","emojis":"🍀","styles":["Hopeful","Whimsical"],"bg":["#3faa4a","#186b28"],"fg":"#effff0","color":"green"}
{"text":"waves rolling over the reef","emojis":"🌊","styles":["Awed"],"bg":["#2b6fe0","#123f8a"],"fg":"#eef4ff","color":"blue"}
{"text":"a clear cloudless sky","emojis":"☀️ 🌤️","styles":["Serene","Hopeful"],"bg":["#3d8bff","#1857c4"],"fg":"#f0f6ff","color":"blue"}
{"text":"cold water from the tap","emojis":"💧 🧊","styles":["Deadpan"],"bg":["#2f80ed","#123f8a"],"fg":"#eef4ff","color":"blue"}
{"text":"blueberries by the handful","emojis":"🫐","styles":["Joyful"],"bg":["#3a6fd8","#1b3f8f"],"fg":"#eef3ff","color":"blue"}
{"text":"denim jeans on the line","emojis":"👖","styles":["Earnest"],"bg":["#4a72c0","#22407e"],"fg":"#f2f5ff","color":"blue"}
{"text":"the power went out at midnight","emojis":"🌙 🕯️","styles":["Anxious"],"bg":["#1c1f26","#05060a"],"fg":"#c7ccd6","color":"dark"}
{"text":"a moonless forest path","emojis":"🌲 🌑","styles":["Wistful","Melancholy"],"bg":["#161a22","#04050a"],"fg":"#b8bec9","color":"dark"}
{"text":"black coffee before dawn","emojis":"☕","styles":["Deadpan"],"bg":["#20232b","#080a0f"],"fg":"#cfd3db","color":"dark"}
{"text":"shadows filled the cellar","emojis":"🕳️","styles":["Tense"],"bg":["#181b21","#050609"],"fg":"#c2c7d0","color":"dark"}
{"text":"a starless night at sea","emojis":"⚓ 🌌","styles":["Melancholy"],"bg":["#13161d","#03040a"],"fg":"#b6bcc7","color":"dark"}
{"text":"sunrise flooding the kitchen","emojis":"🌅","styles":["Hopeful","Joyful"],"bg":["#fff3b0","#ffd24d"],"fg":"#5a4200","color":"bright"}
{"text":"fireworks lighting the sky","emojis":"🎆 ✨","styles":["Excited"],"bg":["#fff6c2","#ffe066","#ffffff"][:2] ,"fg":"#5c4a00","color":"bright"}
{"text":"noon sun on white sand","emojis":"🏖️ ☀️","styles":["Serene"],"bg":["#fff7cc","#ffe25a"],"fg":"#5c4600","color":"bright"}
{"text":"a bright idea just hit me","emojis":"💡","styles":["Excited","Hopeful"],"bg":["#fff5b0","#ffdb3d"],"fg":"#4d3c00","color":"bright"}
{"text":"stadium floodlights came on","emojis":"🏟️ 🔦","styles":["Awed"],"bg":["#fff8d6","#ffe368"],"fg":"#574400","color":"bright"}
```

Note: fix the malformed `"🎆 ✨"` seed row when transcribing — write `"bg":["#fff6c2","#ffe066"]` (two stops, no slice expression). It appears mangled above on purpose so you re-check it.

- [ ] **Step 3: Write `tools/test_report.py` with the integrity check**

```python
import json
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import typer

from files import GOLD_JSONL
from model.config import EMOJIS, MAX_TEXT_LEN, STYLES
from model.data import normalize

_HEX = re.compile(r"^#[0-9a-fA-F]{6}$")
_COLORS = {"red", "green", "blue", "dark", "bright"}


def test_gold_file_integrity():
    rows = [
        json.loads(x)
        for x in Path(GOLD_JSONL).read_text(encoding="utf-8").splitlines()
        if x.strip()
    ]
    assert len(rows) == 125, len(rows)
    counts = Counter(r["color"] for r in rows)
    assert set(counts) == _COLORS, counts
    assert all(v == 25 for v in counts.values()), counts
    evocab, svocab = set(EMOJIS), set(STYLES)
    for r in rows:
        nt = normalize(r["text"])
        assert nt and len(nt) <= MAX_TEXT_LEN, r["text"]
        toks = r["emojis"].split()
        assert toks, r
        bad = [e for e in toks if e not in evocab]
        assert not bad, bad
        assert r["styles"], r
        assert all(s in svocab for s in r["styles"]), r["styles"]
        assert _HEX.match(r["bg"][0]) and _HEX.match(r["bg"][1]), r
        assert _HEX.match(r["fg"]), r


_app = typer.Typer(
    add_completion=False,
    context_settings={"help_option_names": ["-h", "--help"]},
)


@_app.command()
def main() -> None:
    test_gold_file_integrity()
    print("ok")


if __name__ == "__main__":
    _app()
```

- [ ] **Step 4: Run the integrity check**

Run: `uv run python tools/test_report.py`
Expected: prints `ok`. If it asserts, fix the offending gold rows (wrong count per colour, out-of-vocab emoji, bad style label, bad hex, over-length text) and re-run until `ok`.

- [ ] **Step 5: Lint**

Run: `uv run ruff check tools/test_report.py && uv run ruff format --check tools/test_report.py files.py`
Expected: clean (run `uv run ruff format tools/test_report.py` if needed).

- [ ] **Step 6: Commit**

```bash
git add files.py files.ts data/gold.jsonl tools/test_report.py
git commit -m "feat(report): add data/gold.jsonl gold-standard set + GOLD_JSONL

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01JpvX2Tgv7PeX1RJpdqop6c"
```

- [ ] **Step 7: STOP for user review**

Tell the user `data/gold.jsonl` is authored and passes the integrity check, and ask them to review the file before proceeding to Task 2. Do not continue until they approve (they may ask for text/annotation edits — apply, re-run Step 4, amend the commit).

---

## Task 2: OKLab card-distance helper + unit tests

**Files:**
- Modify: `tools/report.py` — add helpers near the other module-level helpers (after `_fnum`, before `_bars`).
- Modify: `tools/test_report.py` — add two tests.

**Interfaces:**
- Consumes: `model.color.rgb_to_oklab`, `model.color.COLOR_SHIFT`.
- Produces: `_hex_to_offsets(hx: str) -> list[float]` (3 values, `int - COLOR_SHIFT`); `_offsets_to_hex(vals: list[float]) -> str` (`#rrggbb`, round+clamp `v + COLOR_SHIFT`); `_card_distance(pred9: list[float], gold9: list[float], color: str) -> float` — mean over 3 slots of the OKLab slot distance; slot distance is `‖Δ(L,a,b)‖` for red/green/blue, `|ΔL|` for dark/bright.

- [ ] **Step 1: Write the failing tests**

Add to `tools/test_report.py` (and add `test_card_distance` + `test_dark_ignores_chroma` to `main`):

```python
def test_card_distance():
    from tools.report import _card_distance, _hex_to_offsets

    red = _hex_to_offsets("#ff0000") * 3
    assert _card_distance(list(red), list(red), "red") < 1e-6
    assert _card_distance(list(red), list(red), "dark") < 1e-6
    d = _card_distance(
        list(_hex_to_offsets("#ff0000") * 3),
        list(_hex_to_offsets("#00ff00") * 3),
        "green",
    )
    assert d > 0.2, d


def test_dark_ignores_chroma():
    from tools.report import _card_distance, _hex_to_offsets

    a = list(_hex_to_offsets("#000000") * 3)
    b = list(_hex_to_offsets("#0000ff") * 3)
    d_blue = _card_distance(a, b, "blue")
    d_dark = _card_distance(a, b, "dark")
    assert d_dark < d_blue, (d_dark, d_blue)
    assert d_dark > 0, d_dark
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run python tools/test_report.py`
Expected: FAIL — `ImportError: cannot import name '_card_distance'` (or `AttributeError`).

- [ ] **Step 3: Implement the helpers in `tools/report.py`**

Add the import with the other model imports near the top:

```python
from model.color import COLOR_SHIFT, rgb_to_oklab
```

Add the helpers (after `_fnum`):

```python
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
```

- [ ] **Step 4: Run to verify it passes**

Run: `uv run python tools/test_report.py`
Expected: prints `ok`.

- [ ] **Step 5: Lint**

Run: `uv run ruff check tools/report.py tools/test_report.py && uv run ruff format --check tools/report.py tools/test_report.py`
Expected: clean.

- [ ] **Step 6: Commit**

```bash
git add tools/report.py tools/test_report.py
git commit -m "feat(report): OKLab card-distance helpers (dF)

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01JpvX2Tgv7PeX1RJpdqop6c"
```

---

## Task 3: Simplify `_probe`; remove the miss tables and missed-texts

**Files:**
- Modify: `tools/report.py` — `_probe`, `_keyword_probe`, `_cldr_probe`, `_section_emoji`, `_section_cldr`, `_emoji_html`, `_cldr_html`; delete `_emoji_row_index`; drop constants and now-unused imports.

**Interfaces:**
- Produces: `_probe(words, enc, head) -> {"n": int, "total": int, "acc_at_k": list[float]}` (length-10, k=1..10). Only keywords with ≥1 in-vocab target contribute to `n` / `acc_at_k`.
- Produces: `report["emoji"]["keywords"]` and `report["cldr"]` both have shape `{n, total, acc_at_k}`; `report["emoji"]` no longer has `missed_texts`.

- [ ] **Step 1: Replace `_probe` and its callers**

Replace `_probe` (currently lines ~144–195) with:

```python
def _probe(words, enc, head):
    vocab = {e: i for i, e in enumerate(EMOJIS)}
    scored = []
    with torch.no_grad():
        for word, exp in words.items():
            ids = [vocab[e] for e in exp if e in vocab]
            if not ids:
                continue
            emb = enc(text_to_tensor(norm_text(word)).unsqueeze(0))
            order = head(emb).squeeze(0).argsort(descending=True).tolist()
            scored.append(min(order.index(i) + 1 for i in ids))
    n = len(scored) or 1
    return {
        "n": len(scored),
        "total": len(words),
        "acc_at_k": [sum(r <= k for r in scored) / n for k in EMOJI_KS],
    }
```

In `_cldr_probe`, change the call from `_probe(words, enc, head, all_targets=True)` to `_probe(words, enc, head)`. `_keyword_probe` is unchanged.

Delete `_emoji_row_index` entirely (only the old `_probe` miss path used it).

- [ ] **Step 2: Drop `missed_texts` from `_section_emoji`**

In `_section_emoji`, keep the `d["eval"]` block (the acc@k chart data and `baseline`) and the final `d["keywords"] = _keyword_probe(enc, head)`. Delete everything computing `order`, `rank_of`, `missed`, `sample`, and `d["missed_texts"]`.

- [ ] **Step 3: Drop the removed constants and imports**

Remove module constants `KEYWORD_MISS_K`, `KEYWORD_TOP`, `MISSED_TEXT_N`, `MISSED_TEXT_TOP`. Keep `EMOJI_KS` and `CLDR_MIN_KEYWORD_LEN`. Remove `import random`. Change `from model.config import EMOJIS, SEED, STYLES` to `from model.config import EMOJIS, STYLES` (SEED becomes unused). Leave `import re` (still used by `_cldr_probe`).

- [ ] **Step 4: Strip the tables from `_emoji_html`**

Keep the `<h2>Model — Emojis</h2>` header, the `if "eval" in d:` chart block, and the `kw = d.get("keywords")` chart block. Delete: the `mt = d.get("missed_texts")` block and its `<table>`; the `kw` `misses` `<table>` block (keep only `out.append(f"<h3>Performance on keywords.json ({kw['n']} words)</h3>{_linechart(points)}")`).

- [ ] **Step 5: Strip the table from `_cldr_html`**

Keep the `<h2>CLDR</h2>` header and the `_linechart(points)` block. Delete the `rows = "".join(...)` and the `out.append(f"<h3>All misses ...")` `<table>` block. Result:

```python
def _cldr_html(d) -> str:
    if not d:
        return '<h2>CLDR</h2><p class="note">enc.pt / emoji.pt not available.</p>'
    points = list(zip((str(k) for k in EMOJI_KS), d["acc_at_k"], strict=True))
    return (
        "<h2>CLDR</h2>"
        f"<h3>Performance on cldr.jsonl ({d['n']} words)</h3>{_linechart(points)}"
    )
```

- [ ] **Step 6: Lint**

Run: `uv run ruff check tools/report.py && uv run ruff format --check tools/report.py`
Expected: clean. Fix any unused-import / unused-name warnings it flags.

- [ ] **Step 7: Run the report end to end**

Run: `uv run python tools/report.py --pt pt`
Expected: prints a `report/<ts>-<sha>/report.html` path, no traceback. Open `report.json` from that folder and confirm: `emoji.keywords` is `{n, total, acc_at_k}` with no `misses`; there is no `emoji.missed_texts`; `cldr` is `{n, total, acc_at_k}` with no `misses`. Open `report.html` and confirm the "Missed texts", "Missed keywords", and "All misses" tables are gone while the three acc@k line charts remain.

- [ ] **Step 8: Commit**

```bash
git add tools/report.py
git commit -m "refactor(report): drop miss tables; keyword/CLDR probes are acc@k only

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01JpvX2Tgv7PeX1RJpdqop6c"
```

---

## Task 4: `_linechart` multi-series support

**Files:**
- Modify: `tools/report.py` — `_linechart` signature + body; add one colour to `_STYLE`.
- Modify: `tools/test_report.py` — add `test_linechart_series`.

**Interfaces:**
- Produces: `_linechart(points, y_max=1.0, baseline=None, legend=None, series=None)`. `series` is a list of `(name: str, values: list[float], css_class: str)`; each is drawn as a polyline (+ small dots) on the same axes. When `series` and `legend` are both given, `legend` is `(main_name, *series_names)` and a swatch row is drawn. Existing single-line / baseline callers pass no `series` and render identically.

- [ ] **Step 1: Write the failing test**

Add to `tools/test_report.py` (and to `main`):

```python
def test_linechart_series():
    from tools.report import _linechart

    svg = _linechart(
        [("1", 0.1), ("2", 0.5)],
        series=[("style", [0.2, 0.6], "lline2")],
        legend=("emoji", "style"),
    )
    assert 'class="lline2"' in svg
    assert ">emoji<" in svg and ">style<" in svg
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run python tools/test_report.py`
Expected: FAIL — `_linechart() got an unexpected keyword argument 'series'`.

- [ ] **Step 3: Implement**

In `_STYLE`, add after the `.linechart .lline{...}` rule:

```css
.linechart .lline2{fill:none;stroke:#e07b00;stroke-width:2.5}
```

Change `def _linechart(points, y_max=1.0, baseline=None, legend=None) -> str:` to add `series=None`. After the `dots = "".join(...)` block, add:

```python
    extra = ""
    for _name, vals, cls in series or ():
        scoords = [(px(i), py(v)) for i, v in enumerate(vals)]
        spoly = " ".join(f"{cx:.1f},{cy:.1f}" for cx, cy in scoords)
        sdots = "".join(
            f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="3" class="dot"/>'
            for cx, cy in scoords
        )
        extra += f'<polyline points="{spoly}" class="{cls}"/>{sdots}'
```

Replace the `leg = ""` / `if legend:` block with:

```python
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
```

In the returned SVG string, insert `{extra}` right before `{leg}`:

```python
    return (
        f'<svg viewBox="0 0 {w} {h}" class="linechart">'
        f"{grid}{base}"
        f'<polyline points="{poly}" class="lline"/>'
        f"{dots}{extra}{leg}</svg>"
    )
```

- [ ] **Step 4: Run to verify it passes**

Run: `uv run python tools/test_report.py`
Expected: prints `ok`.

- [ ] **Step 5: Regression-check existing charts**

Run: `uv run python tools/report.py --pt pt`
Expected: no traceback; open `report.html` and confirm the eval-acc chart still shows the dashed CLDR-baseline line and its "model" / "CLDR baseline" legend unchanged.

- [ ] **Step 6: Lint + commit**

```bash
uv run ruff check tools/report.py tools/test_report.py && uv run ruff format --check tools/report.py tools/test_report.py
git add tools/report.py tools/test_report.py
git commit -m "feat(report): _linechart multi-series support

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01JpvX2Tgv7PeX1RJpdqop6c"
```

---

## Task 5: `_section_cards` + `style.pt`/`gen.pt` loads + provenance

**Files:**
- Modify: `tools/report.py` — imports; `CARD_DIST_THRESHOLD` constant; `_provenance`; `build_report` model loading + `want` set + `report` dict; new `_section_cards`.

**Interfaces:**
- Consumes: `_hex_to_offsets`, `_card_distance`, `_acc_at_k`, `_rows`, `EMOJI_KS`; `model.export_onnx.CONST_Z`; `model.config.Z_WEIGHT`; `model.model.{ColorGen, StyleHead}`; `files.GOLD_JSONL`.
- Produces: `_section_cards(enc, style_head, emoji_head, gen, gold_rows) -> dict` — `{}` when any model is `None` or `gold_rows` empty, else:
  `{"n": int, "threshold": float, "emoji_acc_at_k": list[float](10), "style_acc_at_k": list[float](10), "per_color": {c: {"accuracy": float, "mean_distance": float} for c in [red,green,blue,dark,bright,"all"]}, "rows": [{"color","text","emoji","style","bg1","bg2","text_color","dF","hit"}]}`.
- Produces: `report["cards"]`; `"cards"` in the default `want` set.

- [ ] **Step 1: Add imports and the threshold constant**

Add to the model imports near the top of `tools/report.py`:

```python
from model.config import EMOJIS, STYLES, Z_WEIGHT
from model.export_onnx import CONST_Z
from model.model import ColorGen, EmojiHead, StyleHead, TextEncoder
from torch.nn.functional import normalize as _l2norm
```

(Merge with the existing `from model.model import EmojiHead, TextEncoder` and `from model.config import ...` lines — do not duplicate.)

Add near the other module constants:

```python
CARD_DIST_THRESHOLD = 0.10
CARD_COLORS = ("red", "green", "blue", "dark", "bright")
```

- [ ] **Step 2: Extend `_provenance`**

At the top of `_provenance`, change:

```python
    enc_pt, emoji_pt = str(pt / "enc.pt"), str(pt / "emoji.pt")
    rm = run_meta()
    paths = [enc_pt, emoji_pt]
```

to:

```python
    enc_pt, emoji_pt = str(pt / "enc.pt"), str(pt / "emoji.pt")
    style_pt, gen_pt = str(pt / "style.pt"), str(pt / "gen.pt")
    rm = run_meta()
    paths = [enc_pt, emoji_pt, style_pt, gen_pt]
```

Leave the rest of `_provenance` unchanged (`enc_meta` still keys on `enc_pt`; `model_sha` still from `enc`).

- [ ] **Step 3: Write `_section_cards`**

Add above `build_report`:

```python
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
                "dF": df,
                "hit": df < CARD_DIST_THRESHOLD,
            }
        )
    per_color = {}
    for c in CARD_COLORS:
        ds = [x["dF"] for x in out_rows if x["color"] == c]
        per_color[c] = {
            "accuracy": sum(d < CARD_DIST_THRESHOLD for d in ds) / (len(ds) or 1),
            "mean_distance": sum(ds) / (len(ds) or 1),
        }
    alld = [x["dF"] for x in out_rows]
    per_color["all"] = {
        "accuracy": sum(d < CARD_DIST_THRESHOLD for d in alld) / (len(alld) or 1),
        "mean_distance": sum(alld) / (len(alld) or 1),
    }
    return {
        "n": len(rows),
        "threshold": CARD_DIST_THRESHOLD,
        "emoji_acc_at_k": emoji_acc,
        "style_acc_at_k": style_acc,
        "per_color": per_color,
        "rows": out_rows,
    }
```

- [ ] **Step 4: Wire into `build_report`**

Change the default `want` set to include `"cards"`:

```python
    want = {s.strip() for s in only.split(",") if s.strip()} or {
        "data",
        "labels",
        "emoji",
        "cldr",
        "cards",
    }
```

Replace the model-loading block with:

```python
    enc_pt, emoji_pt = pt / "enc.pt", pt / "emoji.pt"
    style_pt, gen_pt = pt / "style.pt", pt / "gen.pt"
    prov = _provenance(pt)

    enc = emoji_head = style_head = gen = None
    need_enc = bool({"emoji", "cldr", "cards"} & want)
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
    gold_rows = _rows(str(GOLD_JSONL)) if "cards" in want else ()
```

Add the section to the `report` dict assembly, after the `cldr` block:

```python
    if "cards" in want:
        report["cards"] = _section_cards(enc, style_head, emoji_head, gen, gold_rows)
```

Add `GOLD_JSONL` to the `from files import ...` line.

- [ ] **Step 5: Lint**

Run: `uv run ruff check tools/report.py && uv run ruff format --check tools/report.py`
Expected: clean.

- [ ] **Step 6: Run the report; inspect the JSON**

Run: `uv run python tools/report.py --pt pt`
Expected: no traceback. In the new `report.json`, confirm a `cards` object with `n: 125`, `emoji_acc_at_k` / `style_acc_at_k` each length 10 and non-decreasing, `per_color` with the six keys, and 125 `rows` each having `bg1`/`bg2`/`text_color` hex and a numeric `dF`. If `pt/gen.pt` or `pt/style.pt` is absent, `cards` is `{}` and the provenance banner lists them missing — that is expected; note it and continue.

- [ ] **Step 7: Commit**

```bash
git add tools/report.py
git commit -m "feat(report): _section_cards end-to-end gold-set evaluation

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01JpvX2Tgv7PeX1RJpdqop6c"
```

---

## Task 6: `_cards_html` rendering + wiring + threshold tuning

**Files:**
- Modify: `tools/report.py` — `_STYLE` (grid CSS); new `_cards_html`; `_render_html`.

**Interfaces:**
- Consumes: `_section_cards`'s return dict; `_linechart(..., series=..., legend=...)`; `EMOJI_KS`.
- Produces: `_cards_html(d: dict) -> str`; `_render_html` appends the cards section when `"cards" in report`.

- [ ] **Step 1: Add grid CSS to `_STYLE`**

Append to the `_STYLE` string (before the closing `"""`):

```css
.cards-grid{display:grid;grid-template-columns:repeat(5,1fr);gap:10px;margin:18px 0 0}
.mini{aspect-ratio:4/3;border-radius:12px;padding:12px 10px;display:flex;
flex-direction:column;justify-content:center;align-items:center;text-align:center;
overflow:hidden}
.mini .em{font-size:24px;line-height:1}
.mini .tx{font-size:12px;font-weight:600;margin-top:6px;overflow-wrap:anywhere;
line-height:1.3}
.mini .st{font-size:9px;letter-spacing:.06em;text-transform:uppercase;margin-top:6px;
opacity:.75}
```

- [ ] **Step 2: Write `_cards_html`**

Add after `_cldr_html`:

```python
def _cards_html(d) -> str:
    if not d:
        return (
            "<h2>Cards</h2>"
            '<p class="note">enc.pt / style.pt / emoji.pt / gen.pt not all available.</p>'
        )
    out = [
        "<h2>Cards</h2>",
        '<p class="note">End-to-end test of the shipped inference graph on '
        f"data/gold.jsonl ({d['n']} rows).</p>",
    ]
    by_color = {}
    for r in d["rows"]:
        by_color.setdefault(r["color"], []).append(r)
    for c in CARD_COLORS:
        cards = "".join(
            '<div class="mini" style="background:linear-gradient(135deg,'
            f'{_esc(r["bg1"])},{_esc(r["bg2"])});color:{_esc(r["text_color"])}">'
            f'<span class="em">{_esc(r["emoji"])}</span>'
            f'<span class="tx">{_esc(r["text"])}</span>'
            f'<span class="st">{_esc(r["style"])}</span></div>'
            for r in by_color.get(c, [])
        )
        out.append(f'<h3>{_esc(c)}</h3><div class="cards-grid">{cards}</div>')
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
        f'<td class="n">{pc[c]["accuracy"]:.2f}</td>'
        f'<td class="n">{pc[c]["mean_distance"]:.3f}</td></tr>'
        for c in (*CARD_COLORS, "all")
    )
    out.append(
        f'<h3>Accuracy (dF &lt; {d["threshold"]:.2f}) &amp; mean distance</h3>'
        '<table><tr><th>Color</th><th class="n">Accuracy</th>'
        '<th class="n">Mean distance</th></tr>'
        f"{trows}</table>"
    )
    return "".join(out)
```

- [ ] **Step 3: Wire into `_render_html`**

After the `if "cldr" in report: body.append(_cldr_html(report["cldr"]))` line, add:

```python
    if "cards" in report:
        body.append(_cards_html(report["cards"]))
```

- [ ] **Step 4: Lint**

Run: `uv run ruff check tools/report.py && uv run ruff format --check tools/report.py`
Expected: clean.

- [ ] **Step 5: Render and eyeball**

Run: `uv run python tools/report.py --pt pt`
Expected: open `report/<ts>-<sha>/report.html` in a browser. Confirm:
- Five `<h3>` colour blocks, each a 5×5 grid of 25 mini-cards with a 135° gradient, a large emoji, the gold text, and an uppercase style caption.
- One acc@k line chart with two lines (emoji + style) and an "emoji / style" legend.
- The accuracy / mean-distance table with rows red, green, blue, dark, bright, all.

- [ ] **Step 6: Tune `CARD_DIST_THRESHOLD`**

Look at the `mean_distance` column and the spread of `dF` in `report.json` `cards.rows`. Pick a `CARD_DIST_THRESHOLD` that makes "accuracy" discriminating (not all 0.00, not all 1.00) — a value near the median `dF` of the red/green/blue rows is a reasonable anchor. Edit the constant, re-run, and settle on one value. Leave it at that value (no code comment).

- [ ] **Step 7: Run the full plain-assert suite**

Run: `uv run python tools/test_report.py && uv run python model/test_runmeta.py && uv run python model/test_train_cli.py`
Expected: each prints `ok` (or its success line).

- [ ] **Step 8: Commit**

```bash
git add tools/report.py
git commit -m "feat(report): Cards section HTML — 5x5 grids, acc@k chart, dF table

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01JpvX2Tgv7PeX1RJpdqop6c"
```

---

## Task 7: Delete `upsample --report` and `--cldr`

**Files:**
- Modify: `tools/data/upsample.ts`
- Delete: `tools/data/report.ts`
- Modify: `tools/data/upsample.test.ts`

**Interfaces:**
- Produces: `upsample.ts` with no `--report` / `--cldr` / `--max-pair-freq` options, no `latestReport` / `Report` / `Miss` import, no `failingEmojis` / `missedCldrKeywords` exports, no `FAIL_RANK` / `MAX_PAIR_FREQ` / `CLDR_PER` / `CLDR_KEYWORDS` constants, no `genCldrPrompt` / `genCldrBatch`.
- Standalone modes after this: `--negation`, `--single-emoji`, `--colors`, `--keywords`.

- [ ] **Step 1: Remove the import and constants**

In `tools/data/upsample.ts`:
- Delete line: `import { type Miss, type Report, latestReport } from "./report.ts"`.
- Delete constants `FAIL_RANK`, `MAX_PAIR_FREQ`, `CLDR_PER`, `CLDR_KEYWORDS`. Keep `COLORS`, `COLOR_PER`, `COLOR_BATCH`, `KEYWORDS_PER`, `batchSizes`, `colorBatchPlan`.

- [ ] **Step 2: Remove the helper functions**

Delete `export function failingEmojis(...)` and `export function missedCldrKeywords(...)` in full. Grep for other callers first — `grep -n "failingEmojis\|missedCldrKeywords" tools/` — there must be none outside `upsample.test.ts` (handled in Step 6).

- [ ] **Step 3: Remove the prompt/generation helpers**

Delete `function genCldrPrompt(...)` and `async function genCldrBatch(...)`. Keep `cleanLines`, `genColorPrompt`, `genColorBatch`, `pickVoice`.

- [ ] **Step 4: Remove the CLI options and validation**

- Delete the `.option("--report", ...)` and `.option("--cldr", ...)` lines.
- Delete the `.option("--max-pair-freq <n>", ...)` line.
- In the `--per` option help string, drop the `${CLDR_PER} with --cldr, ` clause.
- In the `--count` option help string, drop the ` / missed keywords for --cldr (default ${CLDR_KEYWORDS})` clause.
- Delete `const maxPairFreq = Number(options.maxPairFreq ?? MAX_PAIR_FREQ)`.
- Delete the `const cldr = Boolean(options.cldr)` line.
- In `const standalone = negation || singleEmoji || cldr || colors || kw`, drop `cldr ||`.
- In the `standalone ? ... : NEG_COUNT` count-default chain and the `?? (singleEmoji ? SINGLE_EMOJI_COUNT : cldr ? CLDR_KEYWORDS : NEG_COUNT)` expression, drop the `cldr ? CLDR_KEYWORDS :` branch.
- In the per-default chain around line 374 (`? CLDR_PER`), drop the `--cldr` branch.
- Delete the `if (options.maxPairFreq != null && !options.report && !cldr) { console.warn(...) }` block and the `if (!(maxPairFreq >= 0)) { ... process.exit(1) }` block.
- In the `--rare cannot be combined ...` guard, drop `|| options.report`.
- In the `if (standalone && (options.report || only || ...))` warn and its message string, drop `options.report ||` and `--report / `.
- In the `standaloneName` ternary chain, drop the `cldr ? "cldr" :` branch.

- [ ] **Step 5: Remove the mode branches and their references**

- Delete `let cldrKws: { keyword: string; targets: string[] }[] = []`.
- Delete the entire `} else if (cldr) {` selection block (the `latestReport()` / `report.cldr?.misses` / `missedCldrKeywords` block).
- Delete the entire `} else if (options.report) {` selection block (the `report.emoji?.keywords` / `failingEmojis` block).
- Delete the `--dry` branch that prints `would generate ... ${cldrKws.length} keywords x ${per}` (the cldr dry line).
- Delete the generation branch that calls `genCldrBatch(...)` (around the `genBar.start(cldrKws.length, 0)` block).
- Delete the `if (cldr) console.log(\`keywords : ${cldrKws.length}\`)` summary line.
- Grep the file once more for `cldr`, `cldrKws`, `options.report`, `maxPairFreq`, `Miss`, `Report` and remove any straggler references. Leave the unrelated `--colors` code intact.

- [ ] **Step 6: Trim `upsample.test.ts`**

- In the top `import { ... } from "./upsample.ts"`, remove `failingEmojis,` and `missedCldrKeywords,`.
- Delete the three tests: `test("failingEmojis collects deduped targets ...")`, `test("failingEmojis skips keywords whose pair_freq ...")`, `test("missedCldrKeywords dedupes keywords ...")`.

- [ ] **Step 7: Delete `tools/data/report.ts`**

```bash
git rm tools/data/report.ts
```

Confirm nothing else imports it: `grep -rn "data/report\|from \"./report\"\|from \"../report\"" tools/ web/ | grep -v node_modules` → no hits.

- [ ] **Step 8: Verify**

Run:
```bash
bun test tools/data/upsample.test.ts
bun run tools/data/upsample.ts --help
```
Expected: tests pass; `--help` prints usage with **no** `--report`, `--cldr`, or `--max-pair-freq` and does not throw (a throw here means a leftover value reference to a deleted symbol). Also run `bun test tools/data/` to confirm no other suite broke.

- [ ] **Step 9: Commit**

```bash
git add tools/data/upsample.ts tools/data/upsample.test.ts
git commit -m "chore(upsample): drop --report and --cldr report-coupled modes

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01JpvX2Tgv7PeX1RJpdqop6c"
```

---

## Task 8: Docs — `CLAUDE.md`

**Files:**
- Modify: `CLAUDE.md`

- [ ] **Step 1: Update the `upsample.ts` paragraph**

In the `tools/data/upsample.ts` bullet, delete the clause describing `--report` ("`--report` targets emoji failing the latest report's `data/keywords.json` keyword probe (reads the newest `report/<ts>-<sha>/report.json`'s `emoji.keywords.words`).") and the `--cldr` description sentence. Adjust any adjacent "mutually exclusive with `--report`/`--cldr`/…" lists so they no longer name those flags. Leave `--rare`, `--negation`, `--single-emoji`, `--colors`, `--keywords` text intact.

- [ ] **Step 2: Update the `tools/report.py` bullet**

Reword the report bullet so it matches the new shape: the **Model → Emojis** section is the eval `acc@k` line chart plus the `data/keywords.json` `acc@k` line chart (no "Missed texts" table, no missed-keyword table); the **CLDR** section is just the `acc@k` line chart (no missed-keyword table); add a **Cards** section — an end-to-end test that runs the shipped inference graph on `data/gold.jsonl` (125 hand-authored rows, 25 per colour red/green/blue/dark/bright), rendering a 5×5 mini-card grid per colour, an emoji+style `acc@k` line chart, and a per-colour table of threshold accuracy (`dF < CARD_DIST_THRESHOLD`) and mean OKLab distance `dF`. Note `report.py` now also loads `pt/style.pt` and `pt/gen.pt` (non-fatal if missing). Keep the note that this is an interim report shape.

- [ ] **Step 3: Update the quick-verify list**

In "Environment & commands" → "Quick verify", add `uv run python tools/test_report.py` alongside `model/test_runmeta.py` and `model/test_train_cli.py`.

- [ ] **Step 4: Mention `data/gold.jsonl` in the `data/` structure bullet**

In the "Project structure" `data/` bullet, add `gold.jsonl` to the list of committed files (committed, hand-authored gold-standard set for the report's Cards section).

- [ ] **Step 5: Commit**

```bash
git add CLAUDE.md
git commit -m "docs: CLAUDE.md for report restructure + gold cards; drop upsample --report/--cldr

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01JpvX2Tgv7PeX1RJpdqop6c"
```

---

## Final verification

- [ ] `uv run ruff check . && uv run ruff format --check .` — clean.
- [ ] `uv run python tools/test_report.py` — `ok`.
- [ ] `uv run python model/test_runmeta.py` — `ok`.
- [ ] `uv run python model/test_train_cli.py` — success line.
- [ ] `uv run python tools/report.py --pt pt` — renders; Cards section present (or a clean "not all available" note if `pt/` lacks `gen.pt`/`style.pt`).
- [ ] `bun test tools/data/` — all pass.
- [ ] `bun run tools/data/upsample.ts --help` — no `--report` / `--cldr` / `--max-pair-freq`, no throw.
- [ ] `git grep -n "missedCldrKeywords\|failingEmojis\|latestReport\|data/report\.ts\|MISSED_TEXT_\|_emoji_row_index" -- ':!docs/'` — no hits.
- [ ] `git status` — only intended files changed (do not `git add -A`; a background job may add unrelated "fix" commits — commit each task's files by explicit path).

---

## Self-Review

**Spec coverage:**
- §1 gold dataset + `GOLD_JSONL` → Task 1. ✓
- §2 `dF` metric (OKLab; L-only for dark/bright; min of 5; offset-space graph reused from `export_onnx`) → Task 2 (`_card_distance`) + Task 5 (`_section_cards` palette math). ✓
- §3 threshold accuracy + per-colour table → Task 5 (`per_color`), Task 6 (table + tuning). ✓
- §4 Cards section: model loads → Task 5; `_provenance` → Task 5; `only` key `cards` + default set → Task 5; 5×5 grid + two-line chart + table → Task 6; `report.json` `cards` → Task 5. ✓
- §5 removals (missed texts / keywords / cldr; `_probe` simplification; `_emoji_row_index`; constants) → Task 3. ✓
- §6 `_linechart` multi-series + `.lline2` → Task 4. ✓
- §7 drop `upsample --report` + `--cldr`, delete `report.ts`, trim tests, update `CLAUDE.md` → Task 7 + Task 8. ✓

**Placeholder scan:** The one seed gold row is deliberately malformed with a `[:2]` slice and Step 2 explicitly calls it out for correction; every other code block is complete. No "TBD"/"add error handling"/"similar to Task N".

**Type consistency:** `_card_distance(pred9, gold9, color)` — same 3-arg shape in Task 2 test, Task 2 impl, and Task 5 call. `_hex_to_offsets` / `_offsets_to_hex` names consistent across Tasks 2, 5, 6. `_section_cards(enc, style_head, emoji_head, gen, gold_rows)` — signature in Task 5 interface matches its `build_report` call site. `_linechart(..., series=[(name, values, css_class)], legend=(main, *series))` — Task 4 defines it, Task 6 calls it exactly that way. `CARD_COLORS` defined in Task 5, used in Task 5 and Task 6. `report["cards"]` shape defined in Task 5, consumed in Task 6.

**Deviation from spec (flag to user):** §4 says add `pt/style.pt` / `pt/gen.pt` to `_provenance`. The plan does this (Task 5 Step 2). Consequence: auto-run reports after an `enc`-only training stage will show `pt/gen.pt` missing in the amber banner (Cards genuinely can't be computed then). If that banner noise is unwanted, drop Task 5 Step 2 and let the Cards section's own "not all available" note be the only signal.
