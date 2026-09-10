# Step 1 — small keyword-search-first model

Prove the pipeline on the smallest possible footprint before scaling vocab, model, or coverage.

## Targets

| Metric | Gate | Scored over |
|---|---|---|
| Exact CLDR keyword Acc@1 | ≥ 0.95 | held-out curated keywords, query text = the keyword |
| Fuzzy CLDR keyword Acc@1 | ≥ 0.90 | auto-generated typo / inflection variants |
| Short-text emoji Acc@1 | ≥ 0.70 | vocab-filtered `data/data.jsonl` slice |

Hit rule everywhere: top-1 emoji ∈ the row's target set (same as `tools/report.py`'s `cldr` probe).

Deliberately **out of scope for step 1**: style, colour palette, the full-size vocab, and the Vocab Coverage / Diversity gates (a ~150-emoji vocab fails those by design — they are lowest priority).

Scope knobs: ~150-emoji vocab · ~250–300 curated keywords · small char-CNN · `emoji` + `fusion` heads only.

---

## 1. Dataset

Everything below is produced deterministically (SEED-seeded mulberry32) by a new `tools/data/regen-step1.ts`, writing into a gitignored `data/step1/`.

### 1a. Vocabulary (~150 emoji) + curated keyword slice (~250–300)

From `data/ii.json` (5,031 CLDR keyword → emoji entries):

1. Keep entries whose key is a single `[a-z]+` token, 3–12 chars, non-stopword (`query_tokens(k) == [k]`).
2. Keep **unambiguous** entries: ≤ 3 target emoji **and** all targets share one emojibase group (drops "bat", "cardinal", "seal", …).
3. Rank survivors: target-set size 1 before 2 before 3; ties broken by the targets' EmojiTracker popularity (`data/emoji_popularity.json`) descending.
4. Walk the ranked list adding each keyword's targets to the vocab; **stop when the vocab reaches ~150 emoji**. Keywords consumed so far = the curated slice.
5. Category floor: before stopping, ensure ≥ 12 keywords from each of animals-nature, food-drink, travel-places, objects, symbols, smileys-emotion; pull the next-ranked keyword from a short group. Add ~20 common country flags (US, GB, FR, DE, JP, IT, ES, BR, IN, CN, CA, …) so `flag_completeness` isn't zero.

Outputs: `data/step1/labels.json` (the ~150-emoji vocab + the fixed 21 styles, unused in step 1), `data/step1/ii.json` (curated slice), `data/step1/kwproj.json`.

### 1b. Exact keyword rows

- Every curated `(keyword, target-set)` → a row `{text: keyword, emojis: target-set}`.
- **Hold out ~20% of keywords** (stratified by emojibase group) → `data/step1/exact_eval.jsonl`. The rest → training. The inverted index still contains all keywords, so exact eval measures index postings ordering + model tie-break on unseen keywords.

### 1c. Fuzzy keyword rows — new `tools/data/fuzz_keywords.ts`

For each curated keyword, deterministically emit variants:
- 1 insertion, 1 deletion, 1 adjacent transposition, 1 keyboard-adjacent substitution
- morphology: plural↔singular (`±s`, `±es`, `y↔ies`), `±ing`, `±ed`
- 0–2 hardcoded common misspellings from a small map

Each variant row: `{text: variant, emojis: <same target set>, base: keyword}`.

Split:
- 2 variants per **training** keyword → `data/step1/fuzzy_train.jsonl` (folded into training — this is what teaches char-level typo tolerance).
- all remaining variants, plus every variant of the held-out keywords → `data/step1/fuzzy_eval.jsonl` (probe only).

### 1d. Short-text slice

- From `data/data.jsonl`: keep rows where `emojis ∩ vocab ≠ ∅` after filtering to the ~150-emoji vocab and `len(normalize(text)) ≤ 32`.
- Seeded split → `data/step1/train.jsonl` + `data/step1/eval.jsonl`.

### 1e. Assembled training set

`data/step1/train.jsonl` = short-text rows **+** exact keyword rows **+** `fuzzy_train.jsonl`, all in the one `(text, emoji-set)` retrieval format. Each row also gets the `kw` field computed from `data/step1/ii.json` (for the fusion heads). Eval files stay separate per metric.

### 1f. Wiring

`files.py` / `model/config.py` gain an `EMOJIC_STEP1=1` branch that points `LABELS_JSON` / `TRAIN_JSONL` / `EVAL_JSONL` at `data/step1/*`. Main pipeline untouched. Add `data/step1/` to `.gitignore`.

---

## 2. Model configuration

Small char-CNN, single retrieval head. `model/config.py` under `EMOJIC_STEP1`:

| param | main | step 1 | note |
|---|---|---|---|
| `MAX_TEXT_LEN` | 42 | 32 | keywords are short; a few short texts clip |
| `CHAR_EMBED_SIZE` | 24 | 16 | |
| `ENCODER_CHANNELS` | [130, 220, 340] | [64, 96] | ~⅕ width |
| `ENCODER_DILATION` | [1, 2, 4] | [1, 2] | receptive field 7 chars — full for a keyword, partial for short text (acceptable in step 1; bump to [1,2,3] → RF 13 if short-text Acc@1 lags) |
| `ENCODER_KERNEL_SIZE` | 3 | 3 | |
| `TEXT_EMBED_SIZE` | 690 | 160 | = Σ channels |
| `EMOJI_EMBED_SIZE` | 64 | 32 | small vocab |
| `DROPOUT_EMOJI` | 0.2 | 0.1 | tiny corpus; watch overfit on `MRR/e/val` |
| heads | style,emoji,critic,fusion | **emoji,fusion** | no style / colour / critic |
| `TASK_BATCH_SIZE` | 512 | 128 | |
| `EPOCHS_TASK` | — | raise (epochs are fast) | early-stop still guards |

Params: encoder + head + `150×32` embed table ≈ well under 200 k — sub-millisecond single-thread wasm.

Loss unchanged: `lse_infonce` multi-positive retrieval for `EmojiHead`; the three detached `FusionHead{Gate,Gain,Mix}` combiners as today. Checkpoint / early-stop on `MRR/fusion/val`.

Train: `train enc --local --heads emoji,fusion -o pt-step1` (non-default `-o` skips the `web/public/` export). Writes `pt-step1/{enc,emoji,emoji_embed,fusion_gate,fusion_gain,fusion_mix}.pt`.

---

## 3. Search fusion

Inference for query text `t`:

1. **Keyword vector `kw(t)`** — non-learned inverted index over `data/step1/ii.json`:
   - `queryTokens(t)` → per token: exact posting hit strength `1.0`; uFuzzy hits (`sim ≥ 0.5`) strength `sim`.
   - `kw[e] = max over (token, key) of strength · IDF(key)`, `IDF = 1/log2(1+df)`.
   - **Postings-ordering fix (primary lever for exact Acc@1):** within a key's posting list, keep CLDR canonical order and give the first-listed (primary) emoji a small additive bonus so it outscores its co-listed siblings. Single-target keywords are already 100%; this is what carries the 2–3-target ones. Mirror the change byte-identically in `regen`'s `kw` builder and `web/src/keywords.js`.

2. **Neural vector `logit_m(t)`** — `EmojiEmbedding.score(EmojiHead(TextEncoder(t)))`, per-row standardized `z(logit_m)`.

3. **Fusion** — the three detached combiners, trained in stage 1, winner chosen on the step-1 eval:
   - `FusionHeadGain` — `logit_m + softplus(β)·kw` — **expected winner**: an exact keyword makes `kw` large and sharp → it owns the argmax (exact Acc@1 tracks the index); free short text makes `kw` ≈ 0 → `logit_m` decides (short-text Acc@1 tracks the model); fuzzy is the blend.
   - `FusionHeadGate` — per-text `a·z(logit_m) + (1−a)·kw`.
   - `FusionHeadMix` — scalar `g`, `(1−g)·z(logit_m) + g·kw`.

4. **Report probes** (`tools/report.py`, step-1 mode = `EMOJIC_STEP1=1`) — for each of `exact_eval` / `fuzzy_eval`, report **fused** (best combiner), **index-only (`kw`)**, and **model-only (`logit_m`)** Acc@{1,5,10} under `report.json.keyword.{exact,fuzzy}`; short-text is the standard `emoji.eval` section over `data/step1/eval.jsonl`. Grading is via the per-iteration `goal/<date>-<sha>.yml` → `report.json.goals.compare` (`keyword.exact.acc@1` ≥ 0.95, `keyword.fuzzy.acc@1` ≥ 0.90, `text.acc@1` ≥ 0.70), **not** hard-coded `status.goals` rows.

### Why the targets are reachable

- **Exact ≥ 0.95** — served by the curated index. Curation removed ambiguity (≤3 same-group targets); postings are primary-first. Single-target keywords = 100%; the ordering bonus + a hand-audit of the ~30 worst multi-target keywords covers the rest. The model only breaks ties.
- **Fuzzy ≥ 0.90** — uFuzzy `sim ≥ 0.5` catches most 1-edit typos and simple inflections; the char-CNN trained on 2 fuzz variants/keyword catches keyboard subs and plural/tense via sub-word char n-grams. `gain` fusion lets whichever component fires win.
- **Short-text ≥ 0.70** — ~150 emoji is ~8× easier than the 1,281 vocab where `fusion_gain` already scores 0.65 Acc@1. A small char-CNN on the vocab-filtered `data/data.jsonl` slice should clear 0.70; if not, `bun run upsample --emojis` the weak ones.

### Build order

1. ✅ `tools/data/fuzz_keywords.ts` — deterministic variant generator (`fuzzVariants(keyword)`).
2. ✅ `tools/data/regen-step1.ts` (`bun run regen-step1`) — curation + `data/step1/{labels.json,ii.json,kwproj.json,train.jsonl,eval.jsonl,exact_eval.jsonl,fuzzy_eval.jsonl}` + `kw` fields with the primary-emoji bonus. All curation knobs are top-of-file constants.
3. ✅ `files.py` + `files.ts` + `model/config.py` — `EMOJIC_STEP1` branch (paths → `data/step1/`; `MAX_TEXT_LEN 32`, `ENCODER_CHANNELS [64,96]`, `ENCODER_DILATION [1,2]`, `CHAR_EMBED_SIZE 16`, `EMOJI_EMBED_SIZE 32`, `DROPOUT_EMOJI 0.1`, `TASK_BATCH_SIZE 128`, `EPOCHS_TASK 1500`). `.gitignore` += `data/step1/`, `pt-step1/`; `package.json` += `regen-step1`.
4. ✅ Primary-emoji bonus — in `regen-step1.ts`'s `kw` builder only (step-1 isolated; `PRIMARY_BONUS = 0.15` on the most-popular in-vocab target). Main `regen.ts` / `web/src/keywords.js` untouched — promoting it to the main pipeline needs a full main retrain and is a separate change.
5. ⬜ `EMOJIC_STEP1=1 train enc --local --heads emoji,fusion -o pt-step1` — **user runs this** (commit the tree first; `train` aborts on a dirty tree).
6. ✅ `tools/report.py` — `EMOJIC_STEP1=1` mode: `report.json.keyword.{exact,fuzzy}` probes (fused / kw-only / model-only Acc@k), `cldr` + `cards` dropped from the default sections, `Keyword search (step 1)` HTML block. Grading via `goal/*.yml`.
7. ⬜ Iterate with the `planning-emojic-improvements` loop against `goal/<date>-<sha>.yml`.

### Deviations from this doc

- Curation ranks each keyword's targets by **EmojiTracker popularity**, not CLDR posting order (`data/ii.json` arrays are not reliably primary-first); `proj[k][0]` = most-used in-vocab target, and that is what gets `PRIMARY_BONUS`.
- The primary bonus is **step-1 only** (see build step 4).
- Probes are graded through the goal-file mechanism, not new `status.goals` rows (see §3.4).
- `curate()` currently yields ~356 keywords / ~170 emoji (the category-floor pass overshoots the ~150 vocab target); tune the constants if a tighter slice is wanted.
- The assembled train set is short-text-dominated (~39k short vs ~285 exact + ~570 fuzzy rows). If exact/fuzzy Acc@1 lags, raise `FUZZY_TRAIN_PER_KW` and/or cap the short-text slice.
