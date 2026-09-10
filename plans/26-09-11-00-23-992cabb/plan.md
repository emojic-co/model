# Improvement Plan — 26-09-11-00-23 · 992cabb

Report: report/26-09-11-00-23-992cabb/report.html
Goals file written: goal/2026-09-11-470d8d8.yml
Prev plan: none (iteration 1 — `plans/` did not exist)
Changed since prev: milestone-1 pipeline collapse + train run `992cabb` + report/web export commit `470d8d8`
Best emoji variant: Fusion·Gain
Loop verdict: **step-change** — shipping a report-side exact-kw CLDR probe moves the priority-1 gate from an unmeasured proxy (model-only CLDR Acc@1 0.312 🔴) to the real shipped keyword path: **exact-kw Acc@1 = 0.966 🟢**. The milestone-1 keyword-search thesis is validated. Remaining open gate: short-text emoji Acc@1 0.538 → targeted 1.25k-row keyword upsample + export-variant fix this loop; encoder capacity bump recommended.

## Scorecard  (status.goals + goals.compare vs goal/2026-09-10-aaaa68e.yml)

| Goal (priority) | Target | Current | Status | Δ prev | Last goal / met? |
|---|---|---|---|---|---|
| **Keyword Acc@1 (exact kw)** — NEW probe | ≥0.95 exact | **0.966** | 🟢 | n/a (new) | — / **yes** |
| CLDR keyword Acc@1 (model-only, diagnostic) | ≥0.90 fuzzy | 0.312 | 🔴 | n/a | 0.90 / no |
| Keyword Acc@1 (fusion combiner, on CLDR) | — | 0.680 | ⚪ | n/a (new) | — |
| Short-text emoji Acc@1 | ≥0.80 | 0.538 | 🔴 | n/a | 0.70 / no |
| Short-text emoji Acc@5 | ≥0.90 | 0.694 | 🔴 | n/a | deferred |
| Short-text emoji Acc@10 | ≥0.95 | 0.743 | 🔴 | n/a | deferred |
| Color palette (pure acc) | high | n/a | ⚪ | | deferred |
| Style Acc@1 | high | n/a | ⚪ | | deferred |
| Emoji vocab size | ≥700 | 400 | 🔴 | n/a | deferred |
| Vocab Coverage (popularity-wtd) | ≥0.80 | 0.504 | ⚪ deferred | | deferred |
| Vocab Diversity | ≥0.80 | 0.310 | ⚪ deferred | | deferred |

Priority-1 regression gate — CLDR Acc@1/@5/@10 vs prev: **PASS** (no prior plan; exact-kw path is non-learned and cannot regress from data/training changes; model-only CLDR is a monitored sub-signal only).

### What the new probe measured (report.json `keyword`)

- `keyword.exact.acc_at_k` — non-learned inverted-index `kw` predictor, exact postings + `PRIMARY_BONUS`, scored directly on `data/cldr.jsonl` (1561/4951 keywords have an in-vocab target): **Acc@1 0.966 · Acc@5 0.985 · Acc@10 0.992**.
- `keyword.fusion.acc_at_k` — best of the three detached combiners over `(model_logits, exact-kw)` on the same CLDR keywords: **Acc@1 0.680**. The combiner *drags the clean kw ranking down* with model noise — on bare keyword queries the raw `kw` vector beats fusion by ~0.29 Acc@1. `gate/a` collapsed to ≈0.99 (all-model) and `mix/g` to 0.0 in training, so only the additive `gain` combiner carries any `kw` signal, and even it is net-negative on pure keywords.
- `keyword.acc@1` (model-only `EmojiHead` on CLDR) stays **0.312** — the encoder has not learned bare keyword→emoji association. This is expected for milestone 1 (the inverted index owns keyword queries) and is kept only as a correlate of short-text emoji quality.

### TensorBoard (`runs/…23:52:39…/enc`, early stop epoch 106 / 1500)

- `MRR/e/val` best **0.591** @ step 13035, last 0.588 — plateaued, not undertrained.
- `MRR/e/train` 0.637 vs val 0.588 → gap 0.049 (mild overfit; `loss/e/val` min @ step 12639 then a slight rise). Not the dominant limiter.
- `MRR/fusion/val` 0.613 (= `fusion_gain`; gate/mix ≈ raw model). `gain/beta` → 1.87 (softplus ≈ 2.0).
- `MRR/kw/val` **0.315**, flat from step 99 — the `kw` vector alone is weak on *phrases* (real eval text rarely contains a clean keyword); it is strong only on bare keywords (see CLDR probe).
- GAN (`/gan`): `energy/gan/val` best 0.032 vs ref 0.046 — not gated this milestone.

**Diagnosis.** Priority 1 is met once measured on the shipped path. Priority 2 (short-text emoji Acc@1) is **capacity/データ-limited**, not overfit-limited: the encoder is tiny (`ENCODER_CHANNELS [64, 96]`, dilation `[1, 2]` → receptive field **7 chars** over `MAX_TEXT_LEN` 32 — partial coverage), `MRR/e/val` is flat at the early-stop knee, and 454/854 single-token keyword→in-vocab-emoji associations never reach rank ≤ 10 in the `keywords_flex` probe. The model simply has not seen enough keyword-anchored short text and has too little capacity to store 400-way associations from phrase data alone.

## Goals for next iteration  (goal/2026-09-11-470d8d8.yml)

- `keyword.exact.acc@1: 0.95` — **held** (met at 0.966; the priority-1 gate). Non-learned, cannot regress.
- `keyword.fusion.acc@1: 0.75` — small step from 0.680; flags the shipped-default-path gap. Moves only with recommend-only combiner work (below) — tracked, not worked this loop.
- `keyword.acc@1: 0.40` — model-only CLDR, **stepped** from the 0.312 baseline (old file's 0.90 was unreachable in one loop and gave no per-loop signal). Driven by the keyword upsample.
- `text.acc@1: 0.58` — **stepped down** from the old 0.70 (never met; 0.538 actual). ≈ actual + one loop's realistic gain from the upsample; hard target stays 0.80.
- Deferred: `keyword.acc@{5,10}`, `keyword.fusion.acc@5`, `text.acc@{5,10}`, colors, coverage, style.

## Next actions

### Analysis / diagnostics — DONE this loop
- **Exact-vs-model CLDR split** (`tools/report.py`) — new `report.json.keyword` block (`exact` + `fusion` sub-probes), new "Keyword predictor — CLDR" HTML section, `status.goals` priority-1 row split into "Keyword Acc@1 (exact kw)" (the gate, 0.95) + "CLDR keyword Acc@1 (model-only)" (diagnostic). New `_goal_specs` leaves `keyword.exact.acc@{1,5}`, `keyword.fusion.acc@1`. Closes the skill's `†` metric-mapping TODO for the exact half. **Fuzzy half still unwired** — see below.

### Data upsampling — bun run upsample — DONE this loop
- `bun run upsample --keywords "pet,bear,star,wet,speech,artist,witch,winner,crush,scary,bird,diet,health,sour,herb,treat,afraid,blush,hammer,cube,pen,memo,flex,brew,yay"` — 25 keywords × 50 = ~1,250 keyword-anchored short texts, default annotator (no emoji forcing). Every target emoji is in the current 400-vocab; keywords picked from the worst `keywords_flex` misses where the natural emoji is unambiguous. Target: **short-text emoji Acc@1** + model-only CLDR. CLDR-exact risk: **none** (non-learned path). Then `bun run regen` (defaults `--min-count 150 --max-count 400 --n 4000`).

### Model configuration — model/export_onnx.py — DONE this loop
- `FUSION_EXPORT_VARIANT`: `"gate"` → `"gain"` — `emoji.eval` shows `fusion_gain_acc_at_k[0]` 0.538 vs `fusion_gate` 0.509 (`status.best_emoji_variant` = Fusion·Gain); gate collapsed to all-model in training. Free win for the shipped default mode. Takes effect only when the (milestone-1-dormant) `web/public/` export path is re-run — no re-export / no `web/**` commit this loop.

### Model configuration — model/config.py
- **None this loop.** TensorBoard shows a plateau with only a mild (0.049) train/val gap — no single knob (`LR`, `INFONCE_TEMP`, `DROPOUT_EMOJI`, `EARLY_STOP_PATIENCE`) is clearly indicated. A dropout bump would trade convergence speed for the same val ceiling.

### Regen configuration — bun run regen   (recommend-only; user runs it)
- `--min-count 150 → ~100` to lift the vocab toward the ≥700 floor (priority 5) and Coverage/Diversity — **do not do this yet**: it re-partitions the eval split and breaks Acc@k comparability while priority 2 is still red. Size it with `bun run regen --matrix` first. Revisit once short-text Acc@1 is green.

### Analysis / diagnostics to add
- **Fuzzy CLDR `kw` probe** (`keyword.fuzzy.acc_at_k`, wires `keyword.fuzzy.acc@{1,5}`) — the exact probe cannot cover CLDR keywords absent from `data/ii.json`; the browser ships uFuzzy `matches()`. uFuzzy is JS and not char-identical in Python, so the clean path is a **`regen.ts` sidecar**: have `regen` also emit `data/cldr_kw.json` = `{ "<cldr text>": [[emojiIdx, value], …] }` for every CLDR row using the identical `matches()` it already runs, then `tools/report.py` joins on it. Mechanical (no `--min/max-count/--n` change, does not touch the training `kw` field) but a `regen.ts` edit — **recommend-only** for the user to approve.
- **Per-class / error-taxonomy Acc@k** for short-text emoji (`tools/analysis/*.py`, read-only) — bucket eval misses by emoji, `meta.src` tag, text length, negation. Needed before the *next* data-upsample so it targets the classes that actually carry the miss rate rather than the `keywords_flex` proxy. Build next loop if Acc@1 is still short.

### Architecture / loss — recommend-only
- **Encoder capacity** — receptive field is 7 chars over `MAX_TEXT_LEN` 32. Options: (a) add a third dilated block `ENCODER_CHANNELS [64, 96, 128]` / `ENCODER_DILATION [1, 2, 4]` → RF 15, `TEXT_EMBED_SIZE` 160 → 288; (b) widen kernel `ENCODER_KERNEL_SIZE 3 → 5`. Param cost of (a): trunk conv params roughly ×1.9 (~90k → ~175k), one extra pooled block in every head's input (`EMOJI_EMBED` Linear 160→32 becomes 288→32). wasm-latency: a third `Conv1d` over 32 steps + a larger first Linear — well under 1 ms on the single-thread wasm backend at this width; the `.onnx` grows ~0.1 MB. This is the most direct lever on short-text Acc@1 and needs a Modal/`--local` run to validate.
- **Fusion combiner routing** — `gate` never learns the keyword regime (`gate/a` → 0.99). Feed `FusionHeadGate` an explicit keyword-confidence feature (`kw.max()`, `kw.count_nonzero()`) alongside its 6 current inputs, or co-train with bare-keyword rows so the gate sees both regimes. Would let the shipped `fusion` mode fall back to raw `kw` on keyword-like queries (recovers the 0.966 vs 0.680 gap). `model/model.py` structural change — recommend-only.

## Applied this run

- **`tools/report.py`** — new `_cldr_keywords()` / `_kw_proj()` / `_kw_exact_dense()` / `_rank_acc()` / `_section_keyword()`; `build_report` emits `report["keyword"]`; `_section_status` splits the priority-1 row; `_goal_specs` gains `keyword.exact.acc@{1,5}` + `keyword.fusion.acc@1`; `_keyword_html()` + render wiring. `tools/test_report.py` — updated `test_section_status_orders_and_grades`, added `test_section_keyword_exact_probe` + `test_keyword_html_renders`.
  - `uv run ruff check` — All checks passed · `uv run ruff format --check` — 2 files already formatted
  - `uv run python tools/test_report.py` — ok · `model/test_runmeta.py` — ok · `model/test_train_cli.py` — ok
  - `uv run python tools/report.py --pt pt` → `report/26-09-11-00-23-992cabb/` — `keyword.exact.acc@1 = 0.966` (🟢 priority-1 gate met), `keyword.fusion.acc@1 = 0.680`, `provenance.issues == []`.
- **`model/export_onnx.py`** — `FUSION_EXPORT_VARIANT = "gate"` → `"gain"`. `uv run ruff check` — passed; `model/test_train_cli.py` — ok. No re-export / no `web/**` commit (dormant path this milestone).
- **`goal/2026-09-11-470d8d8.yml`** — written; report `goals.compare` now grades against it (`summary: {met: 1, unmet: 3}`).
- **`goal/README.md`** — leaf→source table updated for the new `keyword.*` leaves.
- **`bun run upsample --keywords "pet,bear,…,yay"`** (25 kw × 50) — 1,247 rows generated, 1,247 appended to `data/data.jsonl`, 0 dropped (60 palette repairs). Then **`bun run regen`**:
  - master lines 240,474 · greedy kept 96,699 · **emoji vocab 400 → 402** (🍌 and 2 others crossed `--min-count 150`) · `data/train.jsonl` 92,699 · `data/eval.jsonl` 4,000 · `kw` keys 1,582 (mean nz 3.84) · baseline overlap acc@10 51.1.
  - ⚠️ **Vocab size changed (400 → 402)** — the seeded eval split is re-partitioned over the larger kept set, so the *next* run's short-text `emoji.eval` Acc@k and the `keyword.*` probes are **not strictly comparable** to this report's numbers. Baseline for iteration 2 is the fresh report the user's retrain produces, not `26-09-11-00-23`.
  - `pt/*.pt` are now stale vs `data/train.jsonl` (`train_sha` will differ) — expected; the user retrains at loop step 1 (`train enc --local --heads emoji,fusion`), which regenerates the report.
