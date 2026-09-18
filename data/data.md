# data/

Every path here is declared in `../files.py` / `../files.ts` (the source of truth for
which constant maps to which file) — check there for the exact import name. This doc
explains what's actually *inside* each file and which tool reads/writes it.

Row shape used by most `.jsonl` files here: `{text, emojis, styles, bg?, fg?, ...}` —
`emojis` is a space-joined string of emoji glyphs, `styles` an array of style labels,
`bg` a 2-stop gradient (`[start, end]` hex) and `fg` the matching text color.

## Master corpus & splits

- **`data.jsonl`** — the committed master corpus, one annotated text per line. Grows via
  `bun run upsample` (`tools/data/upsample.ts`): run with no subcommand for
  topic-rotation LLM generation (`--count` controls how many texts), or a targeted mode
  (`emojis`, `colors`, `linkedin`, `motivational`, `sarcasm`, `groups`, `balance`);
  all append here via `appendJsonl`. Consumed by `bun run regen` (`tools/data/regen.ts`)
  as the sole input to `train.jsonl`/`eval.jsonl`.
- **`train.jsonl`** / **`eval.jsonl`** — gitignored, rebuilt by `regen.ts` from
  `data.jsonl`: de-duplicated, emoji-frequency-capped (`greedyCap`), filtered to the
  kept emoji vocab, then split (`eval.jsonl` is the held-out slice, size `--n`, default
  1500). Read directly by `model/data.py` / `model/train.py` for training and eval.
- **`labels.json`** — gitignored, the label vocab: `{langs, styles, emojis}`. Written by
  `regen.ts` (emoji list = the frequency-capped vocab; `styles`/`langs` from
  `tools/data/styles.ts` / `langs.ts`). Loaded by `model/config.py` and by the web app's
  ONNX export/runtime and several `tools/` scripts — this is the file that pins which
  emojis/styles the model can predict, so changing it invalidates checkpoints.
- **`cldr-baseline.json`** — gitignored, written by `regen.ts` via
  `tools/analysis/cldr-baseline.ts` (`runBaseline`): accuracy/MRR of simple
  keyword-lookup baselines (`strict`/`forward`/…) against `eval.jsonl`, used by
  `tools/report.py` as a non-model reference point in the report.

## Keyword / term / flag probes

Built by `bun run merge-keywords` (`tools/data/keywords.ts`) from three sources merged
by normalized text (`mergeKeywords`): `cldr.jsonl` + `emojilib.jsonl` + `wa-keywords.json`.
Split by word count into single-word `keywords.jsonl` vs. multi-word `terms.jsonl`;
rows are dropped if none of their emojis are in the current `labels.json` vocab, if the
text is <3 chars, or if every emoji is a flag. Both are gitignored and regenerated as
part of `bun run regen`.

- **`cldr.jsonl`** — CLDR (Unicode) annotation keywords per emoji, LLM-styled
  (style + palette) by `bun run build-cldr` (`tools/data/cldr.ts`), sourced from
  `node_modules/cldr-annotations(-derived)-full` + `emojibase-data`. Also the source for
  `flags.jsonl` (see below) and for the CLDR-coverage comparison in `regen --analysis`.
- **`emojilib.jsonl`** — same idea from the `emojilib` npm package's keyword lists,
  built by `bun run build-emojilib` (`tools/data/emojilib.ts`).
- **`wa-keywords.json`** — `{emoji-name-ish key: [glyphs]}` keyword→emoji map (WhatsApp
  keyword set); loaded read-only by `loadWaKeywords()` in `tools/data/wa-keywords.ts`,
  no generator script in this repo — treat it as a checked-in source dataset.
- **`ii.json`** — a hand/LLM-curated keyword→emoji inverted index, seeded from
  `emojilib.jsonl`'s annotations and extended via
  `bun run build-emojilib -- --merge-ii`. Read by `tools/report.py` for a keyword-probe
  metric in the report; not part of the `regen` pipeline.
- **`keywords.jsonl`** / **`terms.jsonl`** — outputs of the merge described above; used
  as auxiliary training/eval signal (`SamplingSource` in `model/config.py`, `acc@1`
  targets) and read by `tools/report.py`.
- **`flags.jsonl`** — single-flag-emoji rows extracted from `cldr.jsonl` by
  `buildFlags()` in `keywords.ts` (also written on every `regen`/`merge-keywords` run).
  Currently **not wired into training** — `model/config.py` imports `FLAGS_JSONL` but
  its `SamplingSource` entry is commented out.

## Colors

- **`color-names-source.json`** — checked-in source list of `{name, hex}` named colors
  (X11/web color names). Read by `bun run color-terms`
  (`tools/data/color-terms.ts` → `color-names.ts`) to derive gradient palettes.
- **`color_terms.jsonl`** — generated from `color-names-source.json`: one row per color
  name with `emojis: ""`, `styles: []`, and a `bg`/`fg` palette derived from the hex.
  Used as a color-only training signal (`SamplingSource("color", ..., "mse")` in
  `model/config.py`).
- **`colors.jsonl`** — gold text→palette rows (no relation to `color_terms.jsonl`'s
  name list — these are ordinary sentences with an annotated palette), used by
  `tools/report.py` and `tools/analysis/color_label_audit.py` as the ground truth the
  color GAN (`ColorGen`/`ColorCritic`) is scored against.

## Emoji metadata

- **`group.json`** — `{emojibase subgroup key: [glyphs]}`, rebuilt from
  `node_modules/emojibase-data` by `bun run build-groups`
  (`tools/data/build-groups.ts`). Used by `tools/report.py` and
  `tools/analysis/emoji-coverage.ts` to report vocab coverage per emoji category, and by
  `tools/data/upsample.ts` to target under-represented groups.

## Unused / orphaned (candidates for cleanup — see prior audit in this conversation)

- **`pred.jsonl`** — gitignored, declared as `PRED_JSONL` in `files.py`/`files.ts`, but
  no script reads or writes it anymore (`model/pred.py` writes predictions elsewhere).
  Stale since ~2026-09-05.
- **`emoji_popularity.json`** — committed, declared as `EMOJI_POPULARITY_JSON`, but the
  constant has zero importers anywhere in `model/`, `tools/`, or `web/src`.
- **`flags.int.jsonl`** — committed, **not declared** in `files.py`/`files.ts` and not
  referenced by any script. Looks like an intermediate dump from generating
  `flags.jsonl` (richer per-country `meta` field, same flag-emoji shape).
- **`linkedin.jsonl`** — gitignored via `.gitignore`, but the `linkedin` mode in
  `tools/data/upsample.ts` actually appends to `data.jsonl` like every other mode; no
  current code path writes to this filename.
