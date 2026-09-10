---
name: planning-emojic-improvements
description: Use when a training run has finished (train then tools/report.py) and the eval report needs to become a prioritized improvement plan toward the emoji / CLDR / style / color accuracy targets. Also use when asked to review the latest report/ folder or plan the next training iteration.
---

# Planning Emojic Improvements

## Overview

After each `train -> report` iteration, turn the newest `report/<stamp>-<sha>/report.json` into **one** prioritized plan of **concrete, ready-to-run actions** — config diffs, `bun run upsample` commands, `bun run regen` flag changes, data-quality checks, eval-set updates, and (recommend-only) architecture changes — measured against the hard targets below. Write the plan to `plans/<same-stamp>/plan.md` (git-tracked, mirrors `report/`). Optionally apply the single lowest-risk step; never retrain.

`report.json` already carries a priority-ordered scorecard at `status.goals` (rendered as the coloured banner atop `report.html`) and the vocab breakdowns at `status.coverage` / `status.diversity`. The plan's job is to turn that scorecard into the next set of moves — not to re-derive it.

Core principle: **CLDR keyword parity is priority 1 and must never regress.** Every recommendation states its risk to the CLDR probe explicitly. A short-text-emoji gain that costs CLDR accuracy is not an improvement.

**Every train → report → analysis loop must aim for a *visible* jump on a failing goal — not a marginal gain.** Before writing the plan, decide the loop's verdict:
- **step-change** — there is a concrete action (or short stack) with a credible path to a visible move (rule of thumb: ≥ +0.02 Acc@1 on the top failing goal, or clearing a category gate). Write it.
- **blocked — need a diagnostic** — the report cannot explain why the top goal is stuck. Then the plan's #1 action is to **build the analysis that would explain it** (see *Analysis to add*); shipping that diagnostic *is* this loop's deliverable.
- **marginal only** — the best available action is a small tweak. Say so explicitly, and pair it with either a bigger swing (recommend-only, e.g. architecture) or a diagnostic — never hand back a plan whose whole content is a marginal nudge.

## Hard targets

| Target | Metric | Threshold |
|---|---|---|
| Short-text emoji Acc@1 | `emoji.eval` best fusion variant, k=1 | ≥ 0.80 |
| Short-text emoji Acc@5 | same, k=5 | ≥ 0.90 |
| Short-text emoji Acc@10 | same, k=10 | ≥ 0.95 |
| CLDR keyword Acc@1 (exact) | `cldr.acc_at_k[0]` † | ≥ 0.95 |
| CLDR keyword Acc@1 (fuzzy) | `cldr.acc_at_k[0]` † | ≥ 0.90 |
| Color palette | `cards.per_color.*` accuracy / OKLab distance | high (no fixed number) |
| Style | `cards.style_acc_at_k`, TB `MRR/s/val` | high (no fixed number) |
| Vocab size | `len(data/labels.json.emojis)` | ≥ 700 (floor) |
| Vocab **Coverage** (popularity-weighted) | see *Vocabulary coverage & diversity* | ≥ 0.80 |
| Vocab **Diversity** (types + keywords) | see *Vocabulary coverage & diversity* | ≥ 0.80 |

† **TODO — unresolved metric mapping.** `report.json` currently has a single CLDR probe (`cldr` section: `EmojiHead` scored over `data/cldr.jsonl`, no exact-vs-fuzzy split); `status.goals` grades that one number against the fuzzy bar (0.90) and notes the gap. The non-learned keyword predictor is only scored on eval short texts (`emoji.eval.keywords_acc_at_k`), not on CLDR. If CLDR becomes the blocking target, the first recommended action is: extend `tools/report.py` to emit `kw`-predictor Acc@1 on `data/cldr.jsonl` with exact-postings-only vs. uFuzzy-enabled, and split the `status` row in two.

Priority order: **1** CLDR parity (no degradation) · **2** short-text emoji Acc@k · **3** color palette · **4** style · then the **vocab-size floor** · then **Coverage & Diversity (lowest priority of all)**. Coverage/Diversity are a **floor gate, not an optimisation target**: act on a sub-0.80 score **only after every accuracy goal above is green**, and never push either past 0.80. While any higher goal is still failing, the report shows them as `deferred`, not `red`.

## Vocabulary coverage & diversity

Two independent gates on the current vocab `V` = `data/labels.json.emojis` (dynamic from `regen`), **both ≥ 0.80**, **both lowest priority** (see priority note above). Reported as `status.coverage` / `status.diversity` with full sub-breakdowns.

### Coverage — popularity-weighted

"Of the emoji people actually use, weighted by how much they use them, what fraction does `V` hold."

```
Coverage = Σ_{e ∈ V} w(e)  /  Σ_{e ∈ R ∪ V} w(e)
```
- `R` / `row(e)` — Unicode's emoji-frequency ranking (`unicode.org/emoji/frequency.html`): a tier system, row 0 = most used (😂 anchor), each row ≈ half the previous, down to ~row 15 (rows 16–17 are sparse samples). Skin-tone / gender variants are consolidated there. Transcribed **once** into committed `data/emoji_popularity.json` = `{ "<emoji>": <row 0..17> }` (path constant `EMOJI_POPULARITY_JSON` in `files.py`). Match `V` to it after stripping `FE0F` / skin-tone modifiers / ZWJ-gender.
- `w(e) = 2^(-row(e))`; an emoji not in `R` → floor row 17 → ~0 weight.
- Row 0 outweighs row 10 by 2^10, so Coverage is dominated by whether `V` holds the common emoji; dropping only the long tail barely moves it — which is the point.

Report: `Coverage`, ranked-emoji covered per row (`row 0: 8/8 · row 1: 15/16 · …`), and the highest-`w` missing emoji.

### Diversity — breadth across types

```
Diversity = 0.50·keyword_coverage + 0.25·group_balance + 0.25·flag_completeness
```
- **keyword_coverage** = `|{ k ∈ data/ii.json : proj(k) ∩ V ≠ ∅ }| / |{ k : proj(k) ≠ ∅ }|` — fraction of the inverted-index keyword vocabulary that has ≥ 1 target emoji in `V`. Also report the raw covered-keyword **count**. This is the "distinct emotions / animals / objects / concepts covered" measure across every topic at once, no hand-maintained list.
- **group_balance** = normalized inverse-Simpson of `V` over the 8 emojibase groups (`people-body` excluded, flags counted): `(1 / Σ_b share_b²) / 8`, clipped to [0, 1]. 1.0 = even spread; low = piled in one type. **Hard sub-gate:** ≥ 6 of the 8 groups non-empty.
- **flag_completeness** = `|flags ∩ V| / 262` (emoji whose emojibase `label` matches `/^flag: /`).

Report: `Diversity`, the three sub-scores, the per-group vocab share, and covered-keyword count.

### In the report

`tools/report.py:_coverage` / `_diversity` emit `status.coverage` and `status.diversity` (with full sub-breakdowns — covered bands, top missing emoji, per-group shares, uncovered-keyword counts) and two `status.goals` rows at **priority 6–7**. While any higher-priority goal is failing, the score is computed but renders as `deferred` (grey), not `red`; once every accuracy goal is green it becomes an active ≥ 0.80 gate. `data/emoji_popularity.json` is committed (EmojiTracker ranking, `{emoji: score}`); if it's ever missing, Coverage shows `not measurable`.

## Inputs

| Source | Use |
|---|---|
| `report/<newest>-<sha>/report.json` | **Primary** — all Acc@k numbers, plus the `status` block (per-goal target vs current, priority-ordered; also rendered as the coloured banner atop `report.html`). Confirm `provenance.issues == []` first; if not, stop and tell the user the report/`.pt` are inconsistent. |
| `runs/<CONFIG_NAME>/` (gitignored TensorBoard) | Optional — loss curves, `MRR/{e,s,fusion,kw}/val`, `auc/critic/val`, `energy/gan/val`, early-stop epoch. Parse with `uv run tensorboard` or `EventAccumulator`. |
| `model/config.py` + `uv run python model/config.py` | Current hyperparameters, param count, receptive field (15) vs `MAX_TEXT_LEN` (42). |
| `data/labels.json` | Current emoji vocab list + size (dynamic). |
| `node_modules/emojibase-data/en/data.json` (groups + `flag:` labels) · `data/ii.json` (keyword→emoji index) · `data/emoji_popularity.json` (Unicode frequency rows) | Coverage & Diversity inputs (see *Vocabulary coverage & diversity*). |
| newest previous `plans/*/plan.md` | Regression gate — diff the scorecard. |
| `git log --oneline -15` | What changed since the last plan (data grow, config edit, arch). |

### report.json field map

- `status.goals` — priority-ordered scorecard: `{goal, priority, target, current, status (good|amber|red|na), note}` per goal. **Start here.**
- `status.best_emoji_variant` — which fusion variant the short-text rows used.
- `status.coverage` — popularity-weighted vocab coverage: `score`, `covered_bands`, `top_missing`, `passed`.
- `status.diversity` — `score`, `keyword_coverage` (+ `keywords_covered`/`keywords_total`), `group_balance` (+ `effective_groups`, `group_shares`), `flag_completeness` (+ `flags_missing`), `passed`.
- `status.summary` — counts of good/amber/red/na.
- `emoji.eval.acc_at_k` — `EmojiHead` retrieval, eval short texts, k=1..10.
- `emoji.eval.fusion_{gate,gain,mix}_acc_at_k` — same via each detached combiner. Shipped variant = `FUSION_EXPORT_VARIANT` in `model/export_onnx.py` (default `gate`). If another variant clearly wins, that is a free win.
- `emoji.eval.keywords_acc_at_k` — non-learned `kw` vector alone.
- `emoji.eval.baseline.acc_at_k` — `overlap` baseline.
- `cldr.acc_at_k`, `cldr.n`, `cldr.total` — `data/cldr.jsonl` keyword probe.
- `cards.emoji_acc_at_k`, `cards.style_acc_at_k` — shipped-graph end-to-end on the 125-row colours gold set.
- `cards.per_color.<red|green|blue|dark|bright|all>.{pure_accuracy,pure_mean_distance,gt_accuracy,gt_mean_distance}` — palette accuracy per colour.
- `keywords_flex.{candidates,missed,ranked}` — single-token keyword-vocab diagnostic; `ranked` = rank>10 misses, worst first.
- `data.records`, `data.length_distribution` — corpus size + text-length histogram.
- `labels.emojis`, `labels.styles` — integer **counts** in `report.json` (not lists); vocab size for the run. The full lists are in `data/labels.json`.

## Workflow

1. **Locate** the newest `report/` dir. Verify its `-<sha>` matches `git rev-parse --short HEAD` and `provenance.issues == []`. If stale or inconsistent, stop and say so (or run `uv run python tools/report.py` if the user wants a fresh one).
2. **Scorecard** — copy `status.goals` verbatim and add a **Δ vs previous plan** column per goal (`plans/<prev>/plan.md`). Note `status.best_emoji_variant`.
3. **Regression gate** — compare the `CLDR keyword Acc@1` row (and `cldr.acc_at_k[4]`, `[9]`) to the previous plan. Any drop = priority-1 finding, called out at the top of the plan.
4. **Diagnose**, gathering evidence for each action bucket below:
   - *Class balance* — `data.records`; most/least frequent kept emojis; `regen`'s summary (`kw` keys + mean nonzero); which in-vocab emojis have high `keywords_flex` rank / low `emoji.eval` contribution → **upsample** candidates.
   - *Coverage & Diversity* (lowest priority — only if every accuracy goal is green) — `status.coverage` / `status.diversity`: which score is under 0.80, `coverage.top_missing`, `diversity.keyword_coverage` shortfall, weak emojibase groups (`group_shares`), `flags_missing` → **upsample** and/or **regen `--min-count`** candidates. Skip this bucket entirely while any higher goal fails.
   - *Config* — best vs exported fusion variant; which head's val metric lags (`MRR/e` vs `MRR/s` vs `auc/critic` vs `energy/gan`); `INFONCE_TEMP`, `LR`, dropout, `EARLY_STOP_PATIENCE`, batch sizes; early-stop epoch (under/over-training) → **config diff** candidates.
   - *Data quality* — pull a sample from a suspect slice (rows tagged `color:*` / `flag:*` / `neg:*`, or the worst `keywords_flex` misses) with `grep`/`jq` on `data/data.jsonl`; look for mislabeled emojis, wrong-set styles, palette drift → **data-quality check** items (what to inspect, the exact command, what "bad" looks like).
   - *Eval set* — is `data/eval.jsonl` representative? emojis/flags/concepts under-represented vs the vocab; whether to raise `regen --n`; whether a failing target needs its own probe slice in `tools/report.py`. CLDR probe rows come from `data/cldr.jsonl`.
   - *Architecture* (recommend-only) — receptive field 15 vs 42; channel chain / dilation / kernel; fusion combiner shape; **on-device budget**: quantify any param-count / wasm-latency cost.
   - *Diagnostic gap* — for the top failing goal, can the current report + TensorBoard actually explain *why* it fails (which inputs, which classes, confident-vs-unsure errors)? If not, the fix this loop is a new analysis — pick from *Analysis to add*.
5. **Set the loop verdict** (step-change / blocked-need-diagnostic / marginal-only, per the Overview) and **prioritize** — rank actions by (expected gain toward the highest-priority *failing* goal) then (cheapest first: analysis script ≈ config edit > data-gen > Modal GPU). Respect the priority order. Never propose a change that risks CLDR parity without a risk line and a mitigation.
6. **Write** `plans/<report-stamp>/plan.md` from the template below (reuse the report dir's exact `<stamp>`).
7. **Optionally apply** at most one step — see *Applying a low-risk step*. Record what was done (or "none") under *Applied this run*.

## Applying a low-risk step

Apply **at most one** — the top-ranked action that qualifies. Everything else is recommend-only.

**Qualifies:**
- Targeted data upsample for a clearly underperforming in-vocab emoji or keyword cluster: `bun run upsample --emojis "<e1>,<e2>"` / `--keywords "<k1>,<k2>"` / `--rare`, then `bun run regen`. Re-read the regen summary; if vocab size changed, record it (future Acc@k is no longer comparable).
- **Coverage gap fill** — missing flags: `bun run upsample --flags` (add `--count N` for a partial pass). Missing animal/object/emotion/symbol concept: `bun run upsample --keywords "<concept words>"`. Then `bun run regen`. Note: added rows only pull an emoji into the vocab once it clears `regen`'s `--min-count`; a single upsample pass may not be enough.
- **Building `data/emoji_popularity.json`** (Unicode emoji-frequency rows) when Coverage can't be computed for want of it — committed data the report reads, touches no model code.
- **A new read-only analysis script** under `tools/` (e.g. `tools/analysis/*.py`) — loads `pt/*.pt` + `data/*.jsonl`, writes a report/plot or prints a table, **mutates nothing** (no `.pt`, no `data/`, no `web/`). Lint after (`uv run ruff check`). Wiring it into `tools/report.py` as a section is recommend-only (bigger surface, changes the standard report).
- Switching `FUSION_EXPORT_VARIANT` in `model/export_onnx.py` when the report shows another variant clearly wins, then `uv run python model/export_onnx.py`.
- One localized scalar nudge in `model/config.py` — a single knob, small step, only when TensorBoard clearly points to it: one of `LR`, `INFONCE_TEMP`, `DROPOUT_EMOJI`, `DROPOUT_STYLE`, `EARLY_STOP_PATIENCE`.

**Does NOT qualify (recommend-only):**
- Any edit to `ENCODER_CHANNELS` / `ENCODER_DILATION` / `ENCODER_KERNEL_SIZE` / `CHAR_EMBED_SIZE` or `model/model.py` structure.
- Anything touching `normalize` / `CHARS` (invalidates `.pt`, breaks web parity).
- `regen --min-count` / `--max-count` / `--n` changes (re-partition the eval split — break Acc@k comparability; size them with `bun run regen --matrix` first and let the user run it).
- Multi-knob sweeps, or anything needing a Modal GPU run to evaluate.

**Guardrails:**
- A background job auto-commits "fix" commits to this branch (see `[[concurrent-training-pipeline-commits]]`): never `git add -A`; scope any commit to the files you touched.
- After editing a `.py`: `uv run ruff check .` and `uv run ruff format --check .`.
- After a data change: `bun run regen`, then re-read its summary.
- Never run `train` / `train --local` — see `[[dont-run-main-to-test]]`. The retrain is the user's to kick off.
- No comments or docstrings in any code touched — see `[[no-comments-or-docstrings]]`.

## Analysis to add

When the standing report can't explain a failure or a win, recommend (or build, if it's a read-only `tools/analysis/*.py`) one of these. Pick the **one** that most directly unblocks the top failing goal — a loop that ships a diagnostic which explains a plateau counts as progress.

**Emoji / CLDR failures**
- **Per-class Acc@k** — Acc@1/@5 for every emoji in the vocab (and per emojibase group / per `meta.src` tag). Surfaces which classes carry the miss rate vs. a uniform sag.
- **Confusion pairs** — for eval misses, the top-1 predicted emoji vs. the gold set. Ranked confusion list → merge candidates, annotation noise, or a real semantic gap.
- **Never-retrieved set** — target emojis that never reach top-10 for any eval row → coverage/embedding-collapse signal.
- **Error taxonomy** — bucket eval misses by text length, negation (`neg` tag), single- vs multi-emoji rows, topic/`src` tag, style → which slice to upsample or re-annotate.
- **CLDR exact vs fuzzy** — split `cldr` Acc@1 by exact-posting hit vs uFuzzy-only, by keyword length and token count (also closes the metric-mapping TODO).
- **Score-margin / calibration** — histogram of the top-1 logit gap for correct vs wrong rows; are wrong answers confident (needs data/loss) or unsure (needs capacity/temp)?
- **Fusion gate behaviour** — distribution of the learned gate/gain/mix scalar across eval texts; rows where `kw` would have helped but the gate suppressed it (or vice-versa).
- **Embedding neighbours** — nearest emojis in `EmojiEmbedding` space for a set of probe words; k-NN purity by group.

**Colour / style**
- **OKLab scatter** — generated vs real palette clouds per colour tag; per-slot (`bg1/bg2/text_color`) error.
- **Style confusion matrix** — gold vs predicted style on the Cards gold set and on an eval slice.

**Data / corpus**
- **Near-duplicate & annotation-disagreement audit** — rows sharing a `normalize(text)` key with conflicting emoji/style sets; duplicate rate; label entropy per text.
- **Eval representativeness** — vocab coverage of `data/eval.jsonl` (which emojis/flags/concepts are thin or absent) vs `data/train.jsonl`.
- **Cross-run trend** — table of every `plans/*/plan.md` scorecard over time → is the loop actually moving, and where did each jump come from?

## Plan template

```markdown
# Improvement Plan — <stamp> · <sha>

Report: report/<stamp>-<sha>/report.html
Prev plan: plans/<prev-stamp>/plan.md   (or "none")
Changed since prev: <one line from git log>
Best emoji variant: <status.best_emoji_variant>
Loop verdict: <step-change | blocked — need diagnostic | marginal-only> — <one line: the expected visible move, or the diagnostic being shipped>

## Scorecard  (status.goals + Δ vs prev)

| Goal (priority) | Target | Current | Status | Δ prev |
|---|---|---|---|---|
| CLDR keyword Acc@1 | ≥0.95/0.90 | 0.000 | 🔴 | +0.00 |
| Short-text emoji Acc@1 | ≥0.80 | | | |
| Short-text emoji Acc@5 | ≥0.90 | | | |
| Short-text emoji Acc@10 | ≥0.95 | | | |
| Color palette (pure acc) | high | | ⚪ | |
| Style Acc@1 | high | | ⚪ | |
| Emoji vocab size | ≥700 | | | |
| Vocab Coverage (popularity-wtd) | ≥0.80 | | ⚪ deferred | |
| Vocab Diversity | ≥0.80 | | ⚪ deferred | |

Priority-1 regression gate — CLDR Acc@1/@5/@10 vs prev: <PASS / FAIL + numbers>

## Next actions

Each action: **target goal**, expected effect, CLDR risk, cost (config edit | data-gen | Modal GPU), and the exact command/diff. Ordered by priority. Omit a bucket if it has nothing this iteration.

### Model configuration — model/config.py
- `<KNOB>`: `<old>` → `<new>` — because <TB metric / report number>. Target: <goal>. CLDR risk: <…>.

### Data upsampling — bun run upsample
- `bun run upsample --keywords "<…>"` — <N rows, which coverage/Acc@k gap>. Then `bun run regen`.
- `bun run upsample --emojis "<…>"` / `--rare --iter <n>` / `--flags [--count N]` — …

### Regen configuration — bun run regen   (recommend-only; user runs it)
- `--min-count <old→new>` / `--max-count …` / `--n …` — vocab-size / eval-split tradeoff. Run `bun run regen --matrix` first; note that Acc@k stops being comparable across the change.

### Data-quality checks
- Inspect `<slice>` (e.g. `grep '"color": "red"' data/data.jsonl | head -50` / `jq` the worst `keywords_flex` misses) — looking for <mislabeled emojis | wrong-set styles | palette drift>. Action if confirmed: <re-annotate via `bun run upsample --reannotate …` | drop | targeted regen>.

### Evaluation-set updates
- <coverage gap in data/eval.jsonl vs vocab | raise `regen --n` | add a probe slice to tools/report.py for goal <…>>.

### Analysis / diagnostics to add
- <which one from *Analysis to add*> — what failure it will explain, why the current report can't. New `tools/analysis/<name>.py` (read-only) or a `tools/report.py` section (recommend-only).

### Architecture — recommend-only
- <change> — param-count / wasm-latency cost: <quantified>. Needs a Modal GPU run to validate.

## Applied this run

<what was executed, with verification output — or "none, all actions recommend-only">
```

## Common mistakes

- Comparing Acc@k across runs where the emoji vocab size changed — the dynamic vocab shifts the eval split; not comparable. Note vocab size in every plan.
- Optimizing short-text emoji at the cost of the `cldr` probe — violates priority 1.
- Treating `fusion_gate`/`fusion_mix` reading equal to `EmojiHead` as a bug — a combiner collapses to the raw model when its learned scalar is ~0; report the gain from the best variant instead.
- Reading an older `report/` dir than the latest train, or one with non-empty `provenance.issues`.
- Recommending architecture changes without quantifying the param-count / wasm-latency cost.
- Running training to "check" a recommendation — leave that to the user.
- Handing back a plan whose entire content is a marginal nudge because the report didn't obviously point anywhere — that means the missing piece is a diagnostic; recommend building it.
- Proposing "add more data" / "train longer" as the action when no analysis shows *which* data or that the model is under-trained. Name the slice or the metric first.

## When NOT to use

- Mid-training, or before any `report/` exists (run `uv run python tools/report.py` first).
