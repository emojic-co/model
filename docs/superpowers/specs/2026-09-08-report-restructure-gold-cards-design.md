# Report restructure: drop miss-tables, add gold-standard end-to-end cards

## Goal

Reshape `tools/report.py` so it stops enumerating individual failures and
instead reports:

1. Keyword and CLDR emoji retrieval as **acc@k line charts only** (no
   per-keyword miss tables).
2. A new **Cards** section: an end-to-end test of the shipped inference
   graph against a small hand-authored gold-standard dataset — rendered
   mini-cards, an emoji+style acc@k chart, and per-color threshold
   accuracy plus mean colour distance.

Also drop `upsample --report` and `upsample --cldr` from the data
toolchain (both consume report miss-tables that are going away).

## Non-goals

- No model or training changes.
- No web-app changes.
- The Cards section does not reproduce `web/src/feelings.js` fonts or
  animations — the report is a static HTML file.
- Not a permanent report scope; this is another interim shape.

## 1. `data/gold.jsonl`

New **committed** file (not gitignored, not part of the append-only
`data/data.jsonl` master).

- One JSON object per line, `data/data.jsonl` schema
  (`text`, `emojis` — space-separated string, `styles` — list,
  `bg` — two hex stops, `fg` — one hex) **plus** one field
  `"color"` — one of `red`, `green`, `blue`, `dark`, `bright`.
- 125 rows: exactly 25 per colour.
- Hand-authored by the implementer and self-reviewed twice
  ("double verified"):
  - `text` — a short string (within `MAX_TEXT_LEN` after
    `model/data.py:normalize`) with a strong, unambiguous association
    to its colour.
  - `emojis` — 1–N high-confidence emojis, **every one present in the
    current `data/labels.json` vocab** so emoji acc@k is always
    defined for every gold row.
  - `styles` — 1–N labels from the fixed 21 in `tools/data/styles.ts`.
  - `bg` / `fg` — a palette that plainly reads as the row's colour.
- Add a path constant `GOLD_JSONL = f"{DATA_DIR}/gold.jsonl"` to
  `files.py` and `export const GOLD_JSONL = \`${DATA_DIR}/gold.jsonl\``
  to `files.ts`.

The file is reviewed and approved by the user before it is wired into
the report.

## 2. Distance metric `dF`

A **prediction** for a gold row is produced by the shipped inference
graph, reused verbatim from `model/export_onnx.py`:

- `ExportWrapper(enc, style, emoji, gen)` fed the gold text's char-id
  tensor returns `style_logits`, `emoji_logits`, and `color` of shape
  `[5, 9]` — the same 5 `CONST_Z` constant-noise palettes the ONNX /
  web build bakes in (`normalize(torch.randn(5, TEXT_EMBED_SIZE,
  generator=manual_seed(SEED)))`, mixed as
  `(1 - Z_WEIGHT) * normalize(emb) + Z_WEIGHT * z`, then
  `tanh(gen.net(seed)) * 127.5`).
- `report.py` imports `ExportWrapper` and `CONST_Z` from
  `model.export_onnx` (or a shared helper) rather than re-deriving the
  mix, so the report's colour output is byte-identical to what ships.

Each of the 5 palettes is a flat 9-vector = 3 colour slots
(`bg1`, `bg2`, `text_color`). The gold row likewise has 3 slots
(`bg[0]`, `bg[1]`, `fg`).

Per slot, convert the predicted sRGB-offset vector and the gold hex to
OKLab via `model/color.py:rgb_to_oklab` (it expects sRGB **byte
offsets**, i.e. value − 127.5; gold hex → int → minus 127.5 before the
call), then:

- `red` / `green` / `blue` rows: slot distance = Euclidean norm over
  the full `(L, a, b)` triple.
- `dark` / `bright` rows: slot distance = `abs(L_pred - L_gold)` only.

- **Palette distance** = mean of the 3 slot distances.
- **Row distance `dF`** = **min** of the 5 palette distances
  (best-case palette).
- **Per-colour mean distance** = mean `dF` over that colour's 25 rows.
- **Overall mean distance** = mean `dF` over all 125 rows.

## 3. Threshold accuracy

- `CARD_DIST_THRESHOLD` — a module constant in `tools/report.py`.
  Initial value `0.10`. Tuned **once** by the implementer against a
  real `pt/` run so the numbers are meaningful, then left with the
  chosen value (no per-colour thresholds).
- **Per-colour accuracy** = fraction of that colour's 25 rows whose
  `dF < CARD_DIST_THRESHOLD`.
- **Overall accuracy** = fraction of all 125 rows under threshold.
- Rendered as a table: one row per colour (`accuracy`, `mean
  distance`) plus an `all` summary row.

## 4. `tools/report.py` — new "Cards" section

### Model loading

`build_report` currently loads only `enc.pt` + `emoji.pt`. Add:

- `style.pt` → `StyleHead`, `gen.pt` → `ColorGen`, loaded via the
  existing `_load` helper. Missing or shape-mismatched `style.pt` /
  `gen.pt` are **non-fatal** (same policy as `emoji.pt`): the Cards
  section renders a "enc/style/emoji/gen not all available" note and is
  otherwise skipped.
- `_provenance` adds `pt/style.pt` and `pt/gen.pt` to the paths it
  stats and sha-checks (so a stale/missing style/gen surfaces in the
  banner). `enc.pt` remains the reference for `model_sha`.

### `only` key

- New key `cards`. Default `want` set becomes
  `{data, labels, emoji, cldr, cards}`.
- `report.json` gains a top-level `cards` object.

### `cards` computation (`_section_cards`)

Input: the 4 modules + the parsed `data/gold.jsonl` rows.

For each gold row, run `ExportWrapper` once to get `style_logits`,
`emoji_logits`, `color [5, 9]`. From that:

- Predicted top-1 emoji = `argmax(emoji_logits)` → `EMOJIS[i]`.
- Predicted top-1 style = `argmax(style_logits)` → `STYLES[i]`.
- Palette #1 (`color[0]`) decoded to 3 hex strings (`+127.5`, clamp
  0–255, hex — reuse `model/pred.py:rgb_to_hex` semantics) for the
  mini-card gradient + text colour.
- `dF` per §2, threshold hit per §3.

Aggregates:

- `emoji_acc_at_k` for k in 1..10 — per row, does any in-vocab target
  emoji land in `emoji_logits.topk(k)`? (reuse `_acc_at_k`); mean over
  all 125 rows (all have an in-vocab target by construction).
- `style_acc_at_k` for k in 1..10 — same against the target style
  multi-hot over the 21 styles; mean over all 125 rows.
- Per-colour and overall `{accuracy, mean_distance}` per §2–§3.
- Per-row render payload: `{color, text, emoji, style, bg1, bg2,
  text_color}`.

### `cards` HTML (`_cards_html`)

- `<h2>Cards</h2>` + a one-line note that this is an end-to-end test
  of the shipped graph on `data/gold.jsonl`.
- Per colour, in fixed order `red, green, blue, dark, bright`:
  `<h3>{color}</h3>` then a **5×5 CSS-grid of mini-cards**. Mini-card:
  `linear-gradient(135deg, bg1, bg2)` background, `color: text_color`,
  a large emoji glyph, the gold text, and the predicted style as a
  small caption. Fixed card size (~150px), no web fonts, no animation.
- One acc@k line chart, k on the x-axis 1..10, **two series**: emoji
  acc@k and style acc@k, with a legend.
- The per-colour accuracy / mean-distance table from §3.
- When `_section_cards` returned empty (models missing): just the
  "not available" note.

## 5. `tools/report.py` — removals

- **Missed texts**: delete the `missed_texts` block in
  `_section_emoji` (the `order`/`rank_of` argsort, the `random.Random
  (SEED).sample`, `MISSED_TEXT_N` / `MISSED_TEXT_TOP`) and its table in
  `_emoji_html`. `random` and `SEED` imports drop if otherwise unused.
- **Missed keywords**: delete the misses table in `_emoji_html`. Keep
  the existing `kw["acc_at_k"]` line chart (it already scores only
  keywords with ≥1 in-vocab target, which is exactly "keywords with at
  least one emoji in the vocab"). Retitle to a plain
  "Performance on keywords.json (N words)".
- **Missed CLDR**: delete the misses table in `_cldr_html`; keep the
  `acc_at_k` chart, same filter, retitled.
- `_probe` stops building `misses`: drop the `all_targets` param, the
  `\bword\b` regex, `pattern`/`row_ids`/`pair`, `emoji_freq` /
  `pair_freq`. It returns `{"n", "total", "acc_at_k"}`.
- Delete `_emoji_row_index` (only `_probe`'s miss path used it) and the
  now-unused constants `KEYWORD_MISS_K`, `KEYWORD_TOP`,
  `MISSED_TEXT_N`, `MISSED_TEXT_TOP`. `_cldr_probe` keeps its
  short/no-letter keyword filter (`CLDR_MIN_KEYWORD_LEN`).
- `report.json` shapes: `emoji.keywords` becomes `{n, total,
  acc_at_k}`; `emoji.missed_texts` is gone; `cldr` becomes `{n, total,
  acc_at_k}`.
- Net rendered sections: Data, Labels, Model — Emojis (eval acc@k
  chart + keywords acc@k chart), CLDR (cldr acc@k chart), Cards.

## 6. `_linechart` multi-series support

- Add an optional `series` argument: a list of
  `(label, [values], css_class)` extra lines drawn on the same axes
  with the same x labels, plus a small legend row. `baseline` /
  `legend` behaviour for existing single-line callers is unchanged
  (pass nothing → identical output).
- Add one colour, `.lline2{stroke:#c0392b}` (or similar), to `_STYLE`
  for the second series.
- The gold chart calls `_linechart(emoji_points, series=[("style",
  style_vals, "lline2")], legend=("emoji", "style"))`.

## 7. `tools/data/upsample.ts` — drop `--report` and `--cldr`

- Remove the `--report` and `--cldr` CLI options and their handler
  branches in the mode dispatch.
- Remove `--max-pair-freq`, the `MAX_PAIR_FREQ` constant, its warning
  and validation, and the `FAIL_RANK` constant if it becomes unused.
- Remove the helpers only those two modes used: `failingEmojis`,
  `missedCldrKeywords`, `missedEmojis` (verify each has no other
  caller before deleting), and the
  `import { type Miss, type Report, latestReport } from "./report.ts"`.
- Delete `tools/data/report.ts` (only `upsample.ts` imports it — verify
  with a repo grep first).
- Remove the corresponding cases from `tools/data/upsample.test.ts`
  (`failingEmojis skips keywords whose pair_freq …`,
  `missedCldrKeywords dedupes keywords …`) and any now-unused test
  imports.
- Update the `upsample.ts` paragraph in `CLAUDE.md`: drop the
  `--report` clause ("`--report` targets emoji failing the latest
  report's `data/keywords.json` keyword probe …") and the `--cldr`
  description, and adjust surrounding "mutually exclusive with" lists.

## Testing

Verification only — no training run
(`memory/dont-run-main-to-test.md`).

- `uv run ruff check .` and `uv run ruff format --check .`
- `uv run python model/test_runmeta.py`
- `uv run python model/test_train_cli.py`
- `uv run python tools/report.py --pt pt` against the working-tree
  `pt/` → open `report/<ts>-<sha>/report.html`: confirm the five 5×5
  grids render with gradients + emoji + text, the two-line acc@k chart
  draws, and the per-colour table has plausible numbers. Tune
  `CARD_DIST_THRESHOLD` and re-run.
- `bun test tools/data/upsample.test.ts`
- `cd web && npm test` (sanity; unaffected).

## Risks / notes

- `rgb_to_oklab` input convention is sRGB **byte offsets** (`+127.5`
  happens inside it). Gold hex must be converted to `int - 127.5`
  before the call, matching how predicted `color` values arrive.
- After an `enc`-only training run there is no `gen.pt` / `style.pt`,
  so the auto-run report will show "Cards not available" — expected,
  not a failure.
- `data/gold.jsonl` emoji targets must stay inside the dynamic
  `data/labels.json` vocab; if the vocab shrinks below one of them a
  future report run will score that row's emoji acc@k as a miss. This
  is acceptable (surfaces as lower gold acc@k) but worth a comment in
  the spec, not a guard.
