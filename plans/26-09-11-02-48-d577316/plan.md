# Improvement Plan — 26-09-11-02-48 · d577316

Report: report/26-09-11-02-48-d577316/report.html
Goals file written: goal/2026-09-11-d577316.yml
Prev plan: plans/26-09-11-00-23-992cabb/plan.md
Changed since prev: `feat(data): sample cldr.jsonl into training batches instead of merging` (3977ed6) · `feat: global goals.yml + per-Unicode-group vocab coverage` (6a4b036) · `fix: grade emoji prediction goals against the fusion model, not standalone kw search` (b095914) · encoder widened `[64,96]`→`[80,100]`, `EMOJI_EMBED_SIZE` 32→42 · a retrain (`enc`+`emoji`+`fusion` only) on top of last iteration's 25-keyword upsample.
Best emoji variant: Fusion (single detached per-emoji-diagonal `FusionHead` now — the old gate/gain/mix variant sweep is gone)
Loop verdict: **step-change** — found and fixed the reason last iteration's keyword upsample barely moved anything: `bun run upsample --keywords` never forced the target emoji into its rows (unlike `--emojis`/`--group`/`--flags`), so the model was never actually shown the *correct* keyword→emoji pairing for `pet`/`health`/`hammer`/`memo`/etc. — it only saw whatever emoji the annotator naturally picked. Shipped a `keyword=emoji` forcing syntax and re-ran it against 27 of the worst `keywords_flex` misses. A same-loop `bun run regen` (plain, no flags) also grew the emoji vocab 287→400, restoring it to the iteration-1 value — see the vocab-comparability note below.

## Scorecard  (status.goals + Δ vs prev + goals.compare)

| Goal (priority) | Target | Current (report d577316) | Status | Δ vs prev plan's actual | Last goal / met? |
|---|---|---|---|---|---|
| Exact keyword Acc@1 — fusion (1) | ≥0.95 | 0.7064 | 🔴 | 0.680 → 0.7064 (+0.026) | 0.72 / no |
| Exact keyword Acc@5 — fusion (1) | ≥0.95 | 0.8506 | 🔴 | n/a (new leaf) | — |
| Exact keyword Acc@10 — fusion (1) | ≥0.95 | 0.8901 | 🟡 | n/a (new leaf) | — |
| Full-text emoji Acc@1 (2) | ≥0.80 | 0.5496 | 🔴 | 0.538 → 0.5496 (+0.012) | 0.58 / no |
| Full-text emoji Acc@5 (2) | ≥0.85 | 0.6994 | 🔴 | 0.694 → 0.6994 (+0.005) | deferred |
| Full-text emoji Acc@10 (2) | ≥0.90 | 0.7481 | 🔴 | 0.743 → 0.7481 (+0.005) | deferred |
| Fuzzy keyword Acc@1 (3) | ≥0.90 | n/a | ⚪ | unwired | deferred |
| Colour energy · global (4) | ≤0.01 | n/a | ⚪ | `cards` off this milestone | deferred |
| Style Acc@1 (5) | ≥0.80 | n/a | ⚪ | `cards` off this milestone | deferred |
| Max text len (6) | ≥42 | 32 | 🔴 | unchanged | deferred |
| Emoji vocab size (7) | ≥700 | **400** (was 287 in this report; iteration-1 was also 400) | 🔴 | see note below | deferred |
| Vocab coverage — groups (8) | all ≥ target | 18/100 pass | ⚪ deferred | unchanged | deferred |

Priority-1 regression gate — fusion model on exact keywords vs prev plan's actual: **PASS** (Acc@1 0.680 → 0.7064, Acc@5/@10 are new leaves this report, both currently amber/red against their 0.95 target but improving with `n`).

### Vocab-comparability note (read before trusting the next training run's Acc@k deltas)

`report/26-09-11-02-48-d577316` (the report this plan is derived from) measured everything against a **287-emoji vocab**. Iteration 1's report (`992cabb`) measured against a **400-emoji vocab** — see its own scorecard row "Emoji vocab size … 400". Sometime between those two reports, `data/labels.json` was regenerated with different `--min-count`/`--max-count` flags than the file's own defaults (`MIN_COUNT 150` / `MAX_COUNT 400` in `tools/data/regen.ts`, unchanged in git history), shrinking the vocab to 287 without anyone re-running the plain, documented `bun run regen`. This loop ran plain `bun run regen` (no flags — the standard post-data-change step, not a deliberate min/max-count edit) after the keyword upsample, which restored the vocab to **400**, matching iteration 1. **The next training run's Acc@k numbers are against 400 classes, not 287** — not directly comparable to `26-09-11-02-48-d577316`'s numbers. This is expected churn from the data toolchain, not a regression, and is itself forward progress on the vocab-size goal (287→400, target 700).

## Goals for next iteration  (goal/2026-09-11-d577316.yml)

- `emoji prediction.exact keyword.acc@1: 0.72` — **held** at the prior target (not stepped further): actual moved +0.026 but the vocab jump to 400 (more distractor classes) makes the next run's number not a clean continuation of this trend, so a bigger step would give a false read either way.
- `emoji prediction.full text.acc@1: 0.56` — small step from 0.5496, same reasoning; capped conservatively given the vocab jump could pull this down mechanically (more classes to rank against) even if the underlying model improves.
- Deferred (unchanged from last iteration): `exact keyword.acc@{5,10}`, `fuzzy keyword`, `full text.acc@{5,10}`, style, color, max text len, vocabulary (size + coverage).

## Next actions

### Root-cause diagnosis — DONE this loop

`report.json.keywords_flex.ranked` still showed `pet` (rank 227), `health` (250), `hammer` (208), `memo` (213) as top misses even though last iteration's plan explicitly upsampled all four via `bun run upsample --keywords "pet,health,hammer,memo,…"`. Grepping the appended rows in `data/data.jsonl` (`grep '"keyword":"pet"' data/data.jsonl`) showed why: `{"text":"Morning! How's your pet doing?","emojis":"🌅 🐾",…,"keyword":"pet",…}` — the annotator picked 🌅/🐾, never the `data/ii.json`/CLDR target 🐶 the flex probe grades against. Reading `tools/data/upsample.ts`'s emoji-assignment code (`const target = cands[i].target; if (target) { emojis = [target, …] } else { emojis = label.emojis.join(" ") }`) confirmed `--keywords` mode never sets `cands[i].target` — unlike `--emojis`/`--group`/`--flags`/`--rare`, which all force their target emoji into the row. So every `--keywords` upsample round to date taught the model whatever emoji the free-form annotator associated with the sentence, not the specific keyword→emoji mapping the eval probes score.

### Data tooling fix — tools/data/upsample.ts — DONE this loop

- Added `parseKeywordTargets` (new, tested) alongside the existing `parseKeywords`: `--keywords` now accepts optional `keyword=emoji` pairs (bare `keyword` still works, unchanged behavior). When a target is given it flows into `cands[i].target`, reusing the exact same forced-injection path `--emojis`/`--group` already use (`emojis = [target, ...label.emojis.filter(e => e !== target)]`), and the `target hit / miss` summary line (previously suppressed for all standalone modes) now also prints for `--keywords` runs that pass at least one target.
- Test coverage: `tools/data/upsample.test.ts` — `parseKeywordTargets` trims/dedupes/handles a bare keyword falling back to no target. `bun test tools/data/upsample.test.ts` → 20 pass.
- CLDR-exact risk: **none** — this only changes what gets written to `data/data.jsonl`/what `bun run upsample` does; the non-learned `kw` exact-match path (`data/ii.json` postings) is untouched.

### Data upsampling — bun run upsample — DONE this loop

Re-ran the keyword upsample with forced targets, picking 27 pairs from the worst `keywords_flex.ranked` misses that had exactly one clean in-vocab target emoji (dropping a few where the `ii.json` target felt too indirect to force safely, e.g. `fresh→🪟`, `work→💦`, `bent→🦵`):

```
bun run upsample --keywords "beauty=🌹,proof=🧾,corn=🍿,insect=🐛,date=📅,health=🍎,empty=🫙,pet=🐶,irish=🍀,fancy=👗,sing=🎤,memo=📝,serve=🍦,hammer=🛠️,congee=🥣,flu=🤧,chest=🧰,popper=🎉,entry=🚫,crush=🥰,choo=🚆,major=🔑,rofl=😂,video=📸,grow=🪴,roller=🧳,lamp=🛋️"
```

Result: 1349 texts generated/annotated/appended, `target hit / miss: 209 / 1140` (the annotator's free-form pick agreed with the forced `ii.json` target only ~15% of the time — expected, that's exactly the gap this fix closes). All 27 target emoji were already well above `MAX_COUNT` (400) in raw master supply (441–5637 raw occurrences each), so not every new row survives `regen`'s greedy cap — spot-checked `hammer`/`pet` post-regen: 39/50 and 47/50 kept into `train.jsonl`+`eval.jsonl` respectively, each now literally containing the word plus the correct target emoji.

Then `bun run regen` (plain, defaults) — see the vocab-comparability note above for the 287→400 side effect.

```
master lines read     : 251840
distinct texts        : 250446 (collapsed away 1394)
min-count / max-count : 150 / 400
greedy kept           : 95210 rows (dropped 155236 over max-count)
emoji vocab (>= 150)  : 400 (below min-count: 1267)
  most frequent       : 🎨 400, 🏫 400, 📍 400, 🏆 400, 🐶 400
  least frequent kept : 🔎 164, 🧂 161, 🧀 158, 🥵 155, 🍌 152
-> data/eval.jsonl       : 4000
-> data/train.jsonl      : 91210
-> data/labels.json    : 21 styles, 400 emojis
kw                    : keys=1570, mean nz 3.87
```

### Model configuration — model/config.py

None this loop — no TensorBoard signal pointed at a specific knob beyond what's already been moved (`ENCODER_CHANNELS`/`EMOJI_EMBED_SIZE` were already widened since the last plan). `MRR/e/val` and `MRR/kw/val` were both flat from very early in training (see TensorBoard below), consistent with a data/labeling problem rather than an undertrained or over-regularized model — which is what this loop's fix targets.

### Analysis — TensorBoard (`runs/…/enc`, the run behind report `d577316`)

- Ran the full `EPOCHS_TASK` schedule (22229 steps, no early stop triggered — `MRR/e/val`'s last point (0.5935) equals its running max, i.e. still noisily climbing/plateaued at the cutoff, not clearly over- or under-trained).
- `MRR/e/val` bounces in a narrow 0.574–0.594 band from roughly step ~4000 onward — a plateau, not a monotonic climb.
- `MRR/kw/val` is dead flat at ~0.29 from step 99 to the end — the model-only channel gets essentially no signal from more training on full-sentence data alone; consistent with the keyword-forcing diagnosis above (the *label* was the bottleneck, not the schedule).
- `fusion/w_search_mean/val` fell from 2.89 → 2.29 over training while `fusion/w_dl_mean/val` stayed ~0.79–0.95 — the combiner still weights `kw` ~2.5–3× the model logit per emoji, yet still underperforms the standalone exact-kw search (0.974 Acc@1) by ~27 points on `keyword.fusion.acc@1` (0.706). With per-emoji-diagonal weights and no cross-emoji normalization, a handful of emoji with large raw model logits can still outrank a correctly-matched-but-lower-magnitude `kw` entry for *other* emoji — a scale-mismatch the combiner can't fix per-emoji without seeing many more keyword-shaped rows in training (partially addressed this loop by the forced-emoji keyword upsample).

### Evaluation-set updates

None — `data/eval.jsonl` is rebuilt by the same `bun run regen` above; no separate probe added this loop.

### Data-quality checks

None flagged this loop beyond the upsample-tool bug itself (already fixed above).

### Analysis / diagnostics to add — recommend, not built this loop

- **Fusion-drag breakdown** (`tools/analysis/fusion_exact_kw_drag.py`, read-only) — for every CLDR exact-keyword row where the standalone `kw` vector ranks the gold emoji #1 but the fused score does not, log the gold emoji, the fused top-1 wrong emoji, and each's `w_dl·model_logit` / `w_search·kw` contribution. Would turn the "scale mismatch" hypothesis above from qualitative into a ranked list of the specific emoji whose model logits are dragging fusion down — a much sharper upsample target list than `keywords_flex` misses alone. Deferred this loop only because the keyword-forcing bug was the higher-confidence, cheaper fix to ship first; worth building next if `keyword.fusion.acc@1` is still flat after this loop's retrain.

### Architecture / loss — recommend-only

- `FusionHead`'s per-emoji diagonal combination has no cross-emoji score normalization (e.g. z-scoring `logit_m` per row before combining), which is the most likely fix for the scale-mismatch above. Needs a Modal GPU run to validate and touches `model/model.py` — out of scope to apply here.

## Applied this run

- `tools/data/upsample.ts` — added `parseKeywordTargets` + wired `--keywords keyword=emoji` forcing through the existing target-injection path; `--keywords` help text and the dry-run/summary logging updated to show the pairing.
- `tools/data/upsample.test.ts` — added a test for `parseKeywordTargets`. `bun test tools/data/upsample.test.ts` → 20 pass, 0 fail.
- `bun run upsample --keywords "beauty=🌹,proof=🧾,corn=🍿,insect=🐛,date=📅,health=🍎,empty=🫙,pet=🐶,irish=🍀,fancy=👗,sing=🎤,memo=📝,serve=🍦,hammer=🛠️,congee=🥣,flu=🤧,chest=🧰,popper=🎉,entry=🚫,crush=🥰,choo=🚆,major=🔑,rofl=😂,video=📸,grow=🪴,roller=🧳,lamp=🛋️"` → 1349 rows appended to `data/data.jsonl` (target hit/miss 209/1140).
- `bun run regen` → rebuilt `data/train.jsonl` / `data/eval.jsonl` / `data/labels.json` (400-emoji vocab; see comparability note).
- `bun test tools/data/` → 96 pass, 1 pre-existing failure (`pool.test.ts`, "dedupe drops rows that normalize to empty or over the length cap" — `model/data.py:CHARS` now includes digits/punctuation so `"12345"` no longer normalizes to empty; unrelated to this loop's changes, not fixed here since it touches `normalize`/`CHARS` territory this skill treats as recommend-only).
- Not run: `train` / `train enc --local` (loop step 1 is the user's).

Next step for the user: `train enc --local --heads emoji,fusion` (writes `pt/{enc,emoji,emoji_embed,fusion}.pt`), then `uv run python tools/report.py --pt pt`.
