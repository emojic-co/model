# Unified report tool — design

Date: 2026-09-04

## Problem

Six overlapping report/analysis generators, three languages, four output
formats, no single artifact that answers "what do I do next?":

| tool | lang | output |
| --- | --- | --- |
| `test_emoji.py` | py | `report/test-emoji/<ts>-<sha>/` md + json + meta.yml |
| `test_color.py` | py | `report/test-color/<ts>-<sha>/` html + json + meta.yml |
| `tools/data/stat.ts` | bun | stdout md |
| `tools/data/color-analysis.ts` | bun | stdout md |
| `tools/analysis/fails.ts` | bun | stdout md — **already broken** (stale `feeling`/`emoji` schema) |
| `data-quality` skill | — | in-session chat |

Nothing evaluates `StyleHead` or `EmojiHead` on `eval.jsonl` outside the
training loop; no `acc@k`; no colour-conditioning check.

## Decisions (settled with the user)

- **One tool, `tools/report.py`.** Self-contained: loads `enc/style/emoji/gen`
  `.pt` + `data.jsonl` / `train.jsonl` / `eval.jsonl`, computes every section,
  writes **one `report.html` + one `report.json`** per run. `--only` runs a
  subset. `test_emoji.py`, `test_color.py`, `tools/data/stat.ts`,
  `tools/data/color-analysis.ts` (+ its `.test.ts`), `tools/analysis/fails.ts`
  are **deleted**.
- **`.md` reports are gone.** HTML + JSON only.
- **`data-quality` skill is deleted.** The report is objective statistics only —
  no LLM label/palette judgement.
- **Preview tools are untouched** (`tools/data/preview*.ts`), still gitignored
  `preview/`.
- **One folder per run:** `report/<ts>-<model-sha>/{report.html,report.json}`,
  git-tracked. `<ts>` = `%y-%m-%d-%H-%M`, `<model-sha>` = `enc.pt`'s embedded
  `meta["sha"]` (or `nometa`).
- **Auto-runs after training:** `train.py` (after `export_onnx.py`) and
  `train_gan.py` (after `export()`) shell out to `tools/report.py`. Also
  re-runnable by hand after `bun run regen`.
- **Report is linked to a commit + `.pt` set + config** via a provenance header
  and a consistency banner (all `.pt` from one commit? model `train_sha` ==
  current `train.jsonl`?).
- **Colour "match" = threshold hit-rate**, ΔE_OKLab < `COLOR_DELTA_E` (0.15),
  plus the intra-group real-vs-real energy baseline.
- **Missing / shape-mismatched `.pt` is not fatal.** `_load` catches the
  `load_state_dict` `RuntimeError` (config changed since training), records it as
  a provenance issue, and renders the affected section without model metrics.

## Report structure

Bottom-line-first. Big type, tinted metric cards, vertical bar charts, no
horizontal bars. Reference mockup: `report-dummy.html` (this session).

### Header
- one line: `<ts> · model <sha> · train.jsonl <hash8>`
- **consistency banner** (green / amber):
  - green — every present `.pt` shares `meta["sha"]`, and `enc.pt`'s
    `meta["train_sha"]` == `sha256(train.jsonl)[:12]`.
  - amber — otherwise; message names the mismatch ("model trained on a different
    `train.jsonl` snapshot — retrain before trusting §Emojis/§Styles/§Colors";
    "`gen.pt` from a different commit"; "`gen.pt` missing — Colors skipped").

### 1 · Data
- three count tiles: records in `data.jsonl` / `train.jsonl` / `eval.jsonl`
  (raw line counts).

### 2 · Emojis
- **distinct-emoji count** tiles: in `data.jsonl` / `train.jsonl` / `eval.jsonl`
  (raw tokens from the space-separated `emojis` field, open-set — may exceed 352).
- **frequency bar chart**, `train.jsonl` document-frequency, **top 10 + bottom 10
  on a shared linear scale** (bottom bars are near-invisible slivers — that is
  the intended read). Caption: `<n>/352 leaderboard emojis have <20 rows`.
- **retrieval metric cards** on `eval.jsonl` (rows with ≥1 in-vocab emoji):
  `acc@1`, `acc@5`, `acc@10`, `MRR@10`. Card background tinted by verdict.
- **top 10 failed keywords** table from `words.json`: keyword, expected emoji,
  rank of the best expected emoji (full argsort; `—` if expected not in vocab),
  top-3 predicted. Sorted worst-first (unranked, then highest rank).

### 3 · Styles
- **distribution bar chart**, all 21 labels, `train.jsonl` occurrence count,
  sorted desc, rotated x-labels. Caption: `most / least = <r>x`.
- **retrieval metric cards** on `eval.jsonl` (all rows): `acc@1`, `acc@5`,
  `MRR@5`, `mAP@5`. Tinted by verdict.

### 4 · Colors
- **background-colour distribution** bar chart, `train.jsonl`. Each record's two
  `bg` stops → OKLab → mean; assigned to the nearest of **5 named anchors**
  (`red`, `yellow`, `green`, `blue`, `purple`) within ΔE < `COLOR_DELTA_E`, else
  `other`. Bars sorted by count, **`other` always last**, each bar painted its
  anchor hex.
- **palette-realism metric cards** — OKLab energy distance on `eval.jsonl`
  palettes (`[*bg, fg]`, 9-D points): `gen ↔ real`, `real ↔ real` (split-half
  floor), `gap`. `gen` = `ColorGen(enc(text), z)` over the 8 fixed `z` vectors
  (`ENERGY_Z_SAMPLES`, seeded like `test_color.py`).
- **6 colour cards** — "driven by the text, or random?" For each of 6 named
  colours (`red, orange, yellow, green, blue, purple`, each with an anchor hex):
  - feed the bare colour name → `enc` → `ColorGen` over the 8 `z` → decode `bg`
    → mean of the two stops in OKLab → **hit** if ΔE to the anchor <
    `COLOR_DELTA_E`. `hit-rate` over the 8 samples.
  - `baseline` = same anchor vs. `bg` generated for a seeded 200-row `eval.jsonl`
    text sample. `lift = hit-rate − baseline`.
  - `energy gap` = `energy(gen, real) − splithalf(real)` over `train.jsonl` rows
    whose normalised text contains the colour word (≥ 32 rows, else `n/a`).
  - verdict: `✓ text-driven` (`lift ≥ 0.25`), `~ weak` (`0.10 ≤ lift < 0.25`),
    `✗ ≈ random` (`lift < 0.10`).
  - card painted the anchor hex; auto dark/light text by anchor luminance.

## `tools/report.py` — implementation

Comment/docstring-free (repo convention). `typer` CLI, matches the other
entrypoints.

### Path shim
Runs as `uv run python tools/report.py` from repo root, so `sys.path[0]` is
`tools/`. Top of file:

```
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
```

then `from config import ...` etc. Add `tools/report.py` to
`[tool.ruff.lint.per-file-ignores]` = `["E402"]` in `pyproject.toml`.

### CLI

```
uv run python tools/report.py [--only data,emoji,style,color] [--out report]
```

- `--only` — comma list, subset of `{data, emoji, style, color}`; default all.
  Header + provenance always render. A section whose `.pt` is missing renders a
  "model unavailable" stub and is dropped from JSON.
- `--out` — parent dir, default `report`.
- exit 0 on success, 1 if no section could be computed.

### Modules used
`config` (`EMOJIS`, `STYLES`, `ENERGY_Z_SAMPLES`, `SEED`, `TEXT_EMBED_SIZE`,
`MAX_TEXT_LEN`), `data` (`read`, `normalize`, `text_to_tensor`, `hex2rgb`,
`colors2tensor`, `TRAIN_PATH`, `EVAL_PATH`), `model` (`TextEncoder`,
`StyleHead`, `EmojiHead`, `ColorGen`, `rgb_to_oklab`), `runmeta` (`load_pt`,
`run_meta`, `model_slug`). Local re-impl of `ap_at_k` / `mrr_at_k` (3 lines
each, copied from `train.py`) — avoids importing lightning.

### Constants (module level, not `config.py`)

```
COLOR_DELTA_E = 0.20
BG_ANCHORS   = {"red": "#c0392b", "yellow": "#e8d44d", "green": "#4c9a52",
                "blue": "#4a6fd1", "purple": "#7a5aa8"}
CARD_COLORS  = {**BG_ANCHORS-without-changes, "orange": "#e07a3f"}  # 6, ordered
THRESHOLDS = {                       # (good, warn); metric ≥ good → green
  "emoji.acc@1": (.50, .30), "emoji.acc@5": (.70, .50),
  "emoji.acc@10": (.80, .60), "emoji.MRR@10": (.50, .35),
  "style.acc@1": (.60, .45), "style.acc@5": (.85, .70),
  "style.MRR@5": (.60, .45), "style.mAP@5": (.60, .45),
}
ENERGY = {"gen": (.15, .25), "gap": (.03, .10)}   # lower is better → inverted
```

### JSON shape

```json
{
  "generated": "<iso>",
  "provenance": { "ts": "...", "model_sha": "...", "train_sha": "...",
                  "models": { "enc.pt": {<meta>|null}, ... },
                  "consistent": true, "banner": "..." },
  "data":  { "records": { "data": N, "train": N, "eval": N } },
  "emoji": { "distinct": { "data": N, "train": N, "eval": N },
             "top": [["😂", 8420], ...], "bottom": [["🪟", 2], ...], "thin": N,
             "eval": { "n": N, "acc@1": .., "acc@5": .., "acc@10": .., "MRR@10": .. },
             "keywords": { "n": N, "acc@1": .., "acc@3": .., "acc@5": .., "acc@10": ..,
                           "MRR": .., "worst": 10,
                           "words": [ { "keyword": "car", "expected": ["🚗"],
                                        "rank": 346, "top3": [..] }, ... ] } },
  "style": { "distribution": [["Joyful", 12090], ...],
             "eval": { "n": N, "acc@1": .., "acc@5": .., "MRR@5": .., "mAP@5": .. } },
  "color": { "bg_distribution": [["blue", 21300], ..., ["other", 27240]],
             "energy": { "gen": .., "ref": .., "gap": .. },
             "cards": [ { "name": "green", "anchor": "#4c9a52",
                          "hit_rate": .64, "baseline": .15, "lift": .49,
                          "energy_gap": .06, "verdict": "text-driven" }, ... ] }
}
```

Sections absent from `--only` are omitted. `report.html` is rendered from the
same dict — fully self-contained (inline CSS, no CDN), like `test_color.py`
today.

### Determinism
`torch.no_grad()`, `mod.eval()` everywhere. Fixed `torch.Generator().manual_seed(SEED)`
for the `z` bank and every subsample (eval-baseline 200 rows, keyword groups).

## Call-site changes

| file | change |
| --- | --- |
| `train.py` | after `subprocess.run([sys.executable, "export_onnx.py"], check=True)` add `subprocess.run([sys.executable, "tools/report.py"], check=True)` |
| `train_gan.py` | after `export()` add the same `subprocess.run` |
| `loop_emoji.py` | drop `from test_emoji import ...`. Run `tools/report.py --only emoji`, read the newest `report/*/report.json`; gate on `emoji["keywords"]["acc@5"]` (the `words.json` probe, not eval retrieval). Replace `_weak_buckets` / `_failure_detail` with a miss list built from `emoji["keywords"]["words"]`. `_commit` already `git add report`. |
| `tools/data/upsample-emoji-test.ts` | `REPORT_DIR` `./report/test-emoji` → `./report`; `latestReport()` descends `report/*/report.json` by mtime; reads `report.emoji.keywords.words`; `Word.word` → `Word.keyword`. Its `.test.ts` updated to match. |
| `train-modal.py` | `CODE_FILES`: drop `test_emoji.py`, add `tools/report.py`, `energy_keywords.txt`. `add_local_file` for `tools/report.py` → `{REPO}/tools/report.py`. Line 129: `subprocess.run([VENV_PY, "tools/report.py", "--only", "data,emoji,style"], ...)` (no `gen.pt` on the box). `COLLECT_TREES` already has `report`. |
| `package.json` | remove `stat`, `color-analysis`, `fails` script aliases |
| `Taskfile.yml` | `modal-train:reports`: `report/test-emoji` → `report`; `-name '*.md'` → `-name 'report.html'`; drop the `sed`/`grep` metric extraction or point it at `report.json` |
| `.claude/commands/data-quality.md` | **delete** |
| `.claude/settings.json` | already `{}` — no change |

## Files deleted

- `test_emoji.py`, `test_color.py`
- `tools/data/stat.ts`, `tools/data/color-analysis.ts`,
  `tools/data/color-analysis.test.ts`
- `tools/analysis/fails.ts` (then remove the empty `tools/analysis/` dir)
- `.claude/commands/data-quality.md`

## Docs

`CLAUDE.md` — the largest edit:
- Replace the `test_emoji.py` / `test_color.py` bullets with one
  `tools/report.py` bullet (what it evaluates, `--only`, output layout,
  auto-run hook).
- Drop every mention of `stat.ts` / `color-analysis.ts` / `fails.ts` /
  the `data-quality` skill / `report/test-emoji` / `report/test-color` /
  `.md` reports.
- "Generated docs & reports" paragraph: `report/<ts>-<model-sha>/` with
  `report.html` + `report.json`, one folder per run, written by
  `tools/report.py`, auto-run at the end of `train.py` / `train_gan.py`.
- "Environment & commands": the post-training spot-check is now
  `uv run python tools/report.py`.
- The Bun-toolchain paragraph: remove `stat` / `color-analysis` / `fails` from
  the package.json alias list; `regen`-first list loses `fails.ts`.

Out of scope (known-stale, pre-existing): `.claude/commands/update-model-md.md`
(`report/model/*.md`), `model.md`, `data.md`.

## Verification

- `uv run ruff check . && uv run ruff format --check .`
- `uv run python tools/report.py` with the local `.pt` set →
  `report/<ts>-<sha>/report.html` + `report.json`. Open the HTML; every section
  present; `json.load` succeeds; `provenance.consistent` matches
  `git`/`train_sha` reality.
- `uv run python tools/report.py --only data,emoji` → JSON has only
  `provenance`/`data`/`emoji`; exit 0.
- Temporarily move `gen.pt` aside → Colors section shows the "gen.pt missing"
  stub, banner amber, exit 0.
- `uv run python tools/report.py --only style` twice → byte-identical
  `report.json` (determinism), modulo `generated`/`ts`.
- `grep -rn "test_emoji\|test_color\|stat\.ts\|color-analysis\|fails\.ts\|data-quality" --include=*.py --include=*.json --include=*.yml --include=*.md .`
  → only `docs/superpowers/` historical files.
- `bun run regen` still works; `package.json` has no dangling alias.
- `web/` build untouched.

## Tools table (task deliverable)

| tool | category | description | cli flags |
| --- | --- | --- | --- |
| `tools/report.py` | report | the single unified model+data report (html+json) | `--only`, `--out` |
| `tools/data/train.ts` | data | append ~1k LLM-generated annotated rows to `data.jsonl` | — |
| `tools/data/regen.ts` | data | rebuild `train.jsonl`/`eval.jsonl`/`labels.json` from `data.jsonl` | `--seed`, `--n` |
| `tools/data/upsample-emojis.ts` | data | append rows for the rarest leaderboard emojis | `--emojis`, … |
| `tools/data/upsample-emoji-test.ts` | data | append rows for `words.json` misses | `--rank` |
| `tools/data/upsample-colors.ts` | data | append colour-conditioned rows | `--colors` |
| `tools/data/annotate.ts` | data | shared LLM annotation module (no entrypoint) | — |
| `tools/data/preview.ts` | preview | browser preview of `eval.jsonl` rows → `preview/` | — |
| `tools/data/preview-pred.ts` | preview | browser preview of `pred.jsonl` | — |
| `tools/data/preview-model.ts` | preview | browser preview of ONNX model output | — |
| `tools/data/preview-labels.ts` | preview | browser preview of `labels.json` | — |
| `tools/data/preview-card.ts` | preview | browser preview of a single styled card | — |
| `train.py` | training | task + GAN stages, export, then report | — |
| `train_gan.py` | training | GAN stage only vs saved `enc.pt`, export, then report | — |
| `train-modal.py` | training | task stage on Modal + report (`--only data,emoji,style`) | `--cpu`, `--memory` |
| `loop_emoji.py` | training | emoji train→report→upsample loop | `--iterations`, `--target`, `--rank`, `--cpu`, `--memory` |
| `export_onnx.py` | other | write `web/public/{model.onnx,meta.json,config.json}` | — |
| `run.py` | other | inference over first 200 `eval.jsonl` rows → `pred.jsonl` | — |
| `test_runmeta.py` | other | plain-assert check of `runmeta.py` | — |
