---
name: planning-emojic-improvements
description: Use when a training run has finished (train then tools/report.py) and the eval report needs to become the next iteration's goals file plus a prioritized, ready-to-run improvement plan toward the emoji / CLDR / style / color accuracy targets. Also use when asked to review the latest report/ folder or plan the next training iteration.
---

# Planning Emojic Improvements

## Overview

`emojic` improves on a fixed **7-step loop**. This skill owns steps 3–6:

| # | Step | Who |
|---|---|---|
| 1 | `train` (full pipeline: `LitEncoder` all four heads + `LitColorGAN`, then a `web/public/` export) — the standard command every iteration, so each report measures the **shipped, functional web app** end to end, not a partial-heads checkpoint | user |
| 2 | `tools/report.py` → `report/<stamp>-<sha>/{report.html,report.json}` | pipeline |
| **3** | **Analysis** — read the newest report + TensorBoard, diagnose the top failing goal | **this skill** |
| **4** | **Write `goal/<YYYY-MM-DD>-<short-sha>.yml`** — the targets the *next* iteration aims to hit | **this skill** |
| **5** | **Derive a concrete plan** → `plans/<stamp>/plan.md` (git-tracked, mirrors `report/`) — current goals/status/gaps, then proposed actions per aspect (Data, Model Configuration, Model Architecture, Training Configuration, Metric Adjustments, Global Goal Adjustments, Current Goal Adjustments), ending in a menu of ranked options — then present them and wait | **this skill** |
| **6** | **Implement the plan** — only the option(s) the user picked; data, config, analysis, regen changes (guardrails below); architecture / loss are recommend-only | **this skill, after the user picks** |
| 7 | back to step 1 | user |

The report already carries one merged, priority-ordered scorecard at the top of `report.html` (mirrored as `report.json` `status.goals`) — goal name, `goals.yml`'s global target, the newest `goal/*.yml`'s current-iteration target, this run's current value, and a good/amber/red/grey status — plus the per-Unicode-group vocab breakdown at `status.vocab_coverage`, and the full leaf-level detail (every gated leaf, not just the priority rows) at `goals.compare` just below it. Steps 3–6 turn that into the next goals file and the next set of moves; they do not re-derive the scorecard.

Core principle: **the fusion model's Acc@1 on exact CLDR keywords (`keyword.fusion.acc_at_k[0]`) is priority 1 and must never regress.** Every recommendation states its risk to that number explicitly. A full-text-emoji gain that costs exact-keyword fusion accuracy is not an improvement. (The standalone inverted-index search `keyword.exact` stays a monitored sub-signal — it is not the gate.)

**Data upsampling is a last resort, not the default move.** The corpus is already large (hundreds of thousands of rows) — the more common bottleneck is *understanding what's already in it*, not a missing concept: a mislabeled slice, an eval set that doesn't probe the failing goal, a config knob, a combiner ignoring a signal it already has. Before a plan proposes `bun run upsample` in any form, it must show why a diagnostic, a data-quality check, or a config nudge could not plausibly close the gap on its own. Upsampling earns its place when the gap is a genuine coverage hole nothing already in the corpus can teach (a whole Unicode group, a keyword→emoji pairing that needs the literal word forced in) — say so explicitly and name the cheaper option that was ruled out and why.

**Always discuss before acting — this skill proposes, it does not decide.** Step 5 ends with 2-4 ranked options, not one locked-in path. Stop there, present them to the user (e.g. via `AskUserQuestion`), and wait for their pick before Step 6 touches anything. Never carry out "every action that qualifies" unattended — a written plan is not standing authorization to execute it.

**Every loop must aim for a *visible* jump on a failing goal — not a marginal gain.** Before writing the plan, decide the loop's verdict:
- **step-change** — there is a concrete action (or short stack) with a credible path to a visible move (rule of thumb: ≥ +0.02 Acc@1 on the top failing goal, or clearing a category gate). Write it and implement it.
- **blocked — need a diagnostic** — the report cannot explain why the top goal is stuck. Then the plan's #1 action is to **build the analysis that would explain it** (see *Tool catalogue → Should have*); shipping that diagnostic *is* this loop's deliverable.
- **marginal only** — the best available action is a small tweak. Say so explicitly, and pair it with either a bigger swing (recommend-only, e.g. architecture) or a diagnostic — never hand back a plan whose whole content is a marginal nudge.

## Hard targets

**`goals.yml` (repo root) is the single source of truth for every target below.** It is
the long-term project-goals statement (`GOALS_YML` in `files.py` / `files.ts`); the
numbers in this table are just a quick reference and can lag the file. Each
`goal/<date>-<sha>.yml` covers **every one of the priority-ordered goals below** — see
*Step 4* for the rule that gives lower-priority goals an easy (hold-current) target
while a higher-priority one is still open.

Every `emoji prediction` leaf measures **the fusion model** (the shipped system: best of
the detached combiners over model logits + kw), by input type. The standalone
inverted-index keyword search (`report.json.keyword.exact` / `keyword.fuzzy`) is a
diagnostic, not a graded goal.

| Target | goals.yml path | report.json source | Threshold (as of writing) |
|---|---|---|---|
| Fusion model — exact keywords Acc@{1,5,10} | `emoji prediction.exact keyword.acc@{1,5,10}` | `keyword.fusion.acc_at_k` (best combiner over model + **exact** kw, on CLDR keywords) | ≥ 0.95 |
| Fusion model — fuzzy keywords Acc@{1,5,10} | `emoji prediction.fuzzy keyword.acc@{1,5,10}` | `keyword.fuzzy_fusion.acc_at_k` † (best combiner over model + **uFuzzy** kw, on CLDR keywords) | ≥ 0.90 / 0.95 / 0.95 |
| Fusion model — full text Acc@{1,5,10} | `emoji prediction.full text.acc@{1,5,10}` | `emoji.eval.<best fusion variant>_acc_at_k` | ≥ 0.80 / 0.85 / 0.90 |
| Style Acc@{1,5,10} | `style prediction.full text.acc@{1,5,10}` | `cards.style_acc_at_k` | ≥ 0.80 / 0.85 / 0.90 |
| Colour energy distance | `color generator.energy distance.{global,red,green,blue,dark,bright}` | `cards.energy` ‡ / `cards.per_color.<c>.gt_mean_distance` | ≤ 0.01 |
| Max text len | `max text len` | `data.max_text_len` (= `config.MAX_TEXT_LEN`) | ≥ 42 |
| Vocab size | `vocabulary.size` | `labels.emojis` | ≥ 700 |
| Per-group vocab coverage | `vocabulary.coverage.<group>` | `status.vocab_coverage.groups.<group>.score` | per-group, see `goals.yml` |

† `keyword.fuzzy_fusion.*` is wired via `tools/analysis/kw-search.ts` (the standalone
uFuzzy keyword search — exact + fuzzy matches, `PRIMARY_BONUS`-ranked, same algorithm as
`regen.ts`/`web/src/keywords.js`), shelled out to by `tools/report.py:_kw_search_rows` and
fused the same way `keyword.fusion` fuses the exact kw vector; it reads back empty (not
fatal) if `bun` or `web/public/kwproj.json` are unavailable when the report runs. ‡ `cards.energy`
(OKLab energy distance) is not computed yet; the per-colour `gt_mean_distance` legs are
wired whenever `pt/style.pt` + `pt/gen.pt` exist — `cards` is now a **default**
`tools/report.py` section (every `train` run trains and ships the GAN, so there's always
a functional web app to measure end to end); it only comes back empty if those two
checkpoints are missing.

Priority order (scorecard priority number in parens): **1** fusion model on exact
keywords Acc@k — no degradation — · **2** fusion model on full text Acc@k · **3** fusion
model on fuzzy keywords Acc@k · **4** colour energy distance · **5** style Acc@k · **6**
max text len · **7** vocab-size floor · **8** per-group vocab coverage (lowest priority of
all). Vocab coverage is a **floor gate, not
an optimisation target**: act on a group below its target **only after every
higher-priority goal is green**. While any higher goal is still failing, the report shows
the coverage row as `deferred` (grey), not `red`.

## Vocabulary coverage — per Unicode group

One gate on the current vocab `V` = `data/labels.json.emojis` (dynamic from `regen`):
for each of the 101 Unicode emoji subgroups, the share of that group's emoji that `V`
holds must clear the group's target in `goals.yml` `vocabulary.coverage`.

```
coverage(group) = |{ e ∈ group : strip_FE0F(e) ∈ V }| / |group|
```

- **Group membership** — committed `data/group.json` = `{ "<subgroup-key>": ["😀", …] }`,
  one entry per unicode.org/emoji/charts/emoji-list.html subgroup (== emojibase
  `messages.json` subgroups). Rebuilt by `bun run build-groups` (reads
  `node_modules/emojibase-data`); `GROUP_JSON` in `files.py` / `files.ts`.
- **Targets** — `goals.yml` `vocabulary.coverage.<group>` (0–1 per group). Tune these
  freely; they encode "how much of each category the vocab should carry" (e.g. flags
  1.0, `person-role` 0.05).
- Empty groups (`food-marine` in the current emojibase) are reported `n/a`, not graded.

### In the report

`tools/report.py:_vocab_coverage` emits `status.vocab_coverage`:
`{ groups: {<g>: {score, covered, total, target, passed}}, groups_passed,
groups_measurable, groups_total, score (= groups_passed/groups_measurable), passed,
top_missing }`, plus one `status.goals` row at **priority 8** (`N/M groups meet target`,
weakest four groups in the note) and the full per-group breakdown as a collapsible table
in `report.html`. While any higher-priority goal is failing it renders `deferred`
(grey). The old popularity-weighted `status.coverage` and composite `status.diversity`
are gone.

## Inputs

| Source | Use |
|---|---|
| `report/<newest>-<sha>/report.json` | **Primary** — all Acc@k numbers, the `status` block (per-goal target vs current, priority-ordered), and `goals.compare` (target-vs-actual vs the last goals file). Confirm `provenance.issues == []` first; if not, stop and tell the user the report/`.pt` are inconsistent. |
| `runs/<CONFIG_NAME>/` (gitignored TensorBoard) | **Required at step 3** — loss curves, `MRR/{e,s,fusion,kw}/val`, per-variant `MRR/fusion_{gate,gain,mix}/val`, `auc/critic/val`, `energy/gan/val`, `gate/a` · `gain/beta` · `mix/g`, early-stop epoch. Read with `uv run tensorboard --logdir runs` or `tensorboard.backend.event_processing.event_accumulator.EventAccumulator`. Note the newest run dir (`ls -t runs`). |
| `goals.yml` (repo root) | **Long-term targets — source of truth.** Every `status` scorecard's global-target column is read from here; the newest `goal/*.yml` covers every priority goal in this tree (easy holds on the ones below focus — see *Step 4*). |
| `goal/<newest>.yml` + `goal/README.md` | The targets this iteration was aiming at, and the schema + leaf→`report.json` source map for the file step 4 writes. |
| `model/config.py` + `uv run python model/config.py` | Current hyperparameters, param count, receptive field (15) vs `MAX_TEXT_LEN` (42). |
| `data/labels.json` | Current emoji vocab list + size (dynamic). |
| `data/group.json` (`bun run build-groups`) · `data/ii.json` (keyword→emoji index) | Per-group vocab-coverage input (see *Vocabulary coverage — per Unicode group*). |
| newest previous `plans/*/plan.md` | Regression gate — diff the scorecard. |
| `git log --oneline -15` | What changed since the last plan (data grow, config edit, arch). |

### report.json field map

- `status.goals` — priority-ordered scorecard, one row per goal, rendered as `report.html`'s opening "Goal status" table: `{goal, priority, dir (max|min), target (global, from goals.yml), iter_target (this iteration, from the newest goal/*.yml — "—" if unset), current, status (good|amber|red|na), note}`. `status` is good when `current` clears the global `target`, amber when it clears only `iter_target` (an intentionally easy iteration ask), red otherwise. **Start here.**
- `status.best_emoji_variant` — which fusion variant the short-text rows used.
- `status.vocab_coverage` — per-Unicode-group vocab coverage: `groups.<g>.{score,covered,total,target,passed}`, `groups_passed` / `groups_measurable` / `groups_total`, `score`, `passed`, `top_missing`.
- `status.summary` — counts of good/amber/red/na.
- `goals.source_file` — path of the goals file this report was graded against (newest `goal/*.yml`).
- `goals.meta` — that file's `meta` block (`rationale`, `based_on`, `deferred`, …).
- `goals.compare` — the full **leaf-level** detail (every gated leaf, including every individual `vocabulary.coverage.<group>` entry, not just the priority rows in `status.goals`); mirrors the `goals:` tree, each leaf `{target, actual, met, dir (max|min), delta}`. `actual`/`met` are `null` when the source is not wired yet (see `goal/README.md` map — `color generator.energy distance.global` is not wired; `emoji prediction.fuzzy keyword.*` is wired but reads back `null` on a run where `bun` or `web/public/kwproj.json` weren't available). Rendered in `report.html` right under the merged "Goal status" table, as "Goals for this iteration — per-leaf detail".
- `goals.summary` — counts of `met` / `unmet` / `unmeasured` leaves.
- `emoji.eval.acc_at_k` — `EmojiHead` retrieval, eval short texts, k=1..10.
- `emoji.eval.fusion_{gate,gain,mix}_acc_at_k` — same via each detached combiner. Shipped variant = `FUSION_EXPORT_VARIANT` in `model/export_onnx.py` (default `gate`). If another variant clearly wins, that is a free win.
- `emoji.eval.keywords_acc_at_k` — non-learned `kw` vector alone.
- `emoji.eval.baseline.acc_at_k` — `overlap` baseline.
- `keyword.{exact,model,fusion,fuzzy,fuzzy_fusion}` — CLDR-keyword probe, each `{n, total, acc_at_k[0..9]}`: `exact`/`fuzzy` are the standalone kw search (Python exact-only vs. `tools/analysis/kw-search.ts` exact+uFuzzy, both diagnostics only), `model` is bare `EmojiHead`, `fusion`/`fuzzy_fusion` are the graded fusion-model numbers (priority-1 / priority-3). `model`/`fusion`/`fuzzy_fusion` need `enc.pt`+`emoji.pt` (+`emoji_embed.pt`+`fusion.pt` for the two fusion columns) to load; `fuzzy`/`fuzzy_fusion` additionally need `bun` + `web/public/kwproj.json` at report time.
- `cldr.acc_at_k`, `cldr.n`, `cldr.total` — `data/cldr.jsonl` keyword probe.
- `cards.emoji_acc_at_k`, `cards.style_acc_at_k` — shipped-graph end-to-end on the 125-row colours gold set.
- `cards.per_color.<red|green|blue|dark|bright|all>.{pure_accuracy,pure_mean_distance,gt_accuracy,gt_mean_distance}` — palette accuracy per colour.
- `keywords_flex.{candidates,missed,ranked}` — single-token keyword-vocab diagnostic; `ranked` = rank>10 misses, worst first.
- `data.records`, `data.length_distribution` — corpus size + text-length histogram.
- `labels.emojis`, `labels.styles` — integer **counts** in `report.json` (not lists); vocab size for the run. The full lists are in `data/labels.json`.

## Workflow

### Step 3 — Analysis

1. **Locate** the newest `report/` dir. Verify its `-<sha>` matches `git rev-parse --short HEAD` and `provenance.issues == []`. If stale or inconsistent, stop and say so (or run `uv run python tools/report.py --pt pt` if the user wants a fresh one).
2. **Read TensorBoard** — open the newest `runs/<CONFIG_NAME>/` (+ `/gan`). Record: which head's val metric lags (`MRR/e` vs `MRR/s` vs `MRR/fusion` vs `auc/critic` vs `energy/gan`), the early-stop epoch (under- vs over-training), the learned fusion scalars (`gate/a` · `gain/beta` · `mix/g`), and any divergence / plateau shape.
3. **Scorecard** — copy `status.goals` verbatim; add a **Δ vs previous plan** column per goal (`plans/<prev>/plan.md`). Note `status.best_emoji_variant`. Read `goals.compare` / `goals.summary` — which of last iteration's targets were `met` / `unmet` / `unmeasured`.
4. **Regression gate** — compare the priority-1 row (fusion model on exact keywords, `keyword.fusion.acc_at_k[0]/[4]/[9]`) and the priority-3 row (fusion on fuzzy keywords, `keyword.fuzzy_fusion.acc_at_k`) to the previous plan; also note `keyword.exact.*` / `keyword.fuzzy.*` (standalone search, exact vs. exact+uFuzzy) and model-only `keyword.model.*` / `cldr.acc_at_k` as sub-signals. Any drop on either fusion row = a priority-1/3 finding, called out at the top of the plan.
5. **Diagnose**, gathering evidence for each action bucket:
   - *Class balance* — `data.records`; most/least frequent kept emojis; `regen`'s summary (`kw` keys + mean nonzero); which in-vocab emojis have high `keywords_flex` rank / low `emoji.eval` contribution. Check first whether the rows that *should* teach this are already in the corpus but mislabeled, never forced into training, or drowned out (a data-quality or config fix) before treating it as an **upsample** candidate.
   - *Vocab coverage* (lowest priority — only if every higher goal is green) — `status.vocab_coverage`: which groups are under their `goals.yml` target (`groups.<g>.passed == false`), `top_missing`, the weakest groups from the `status.goals` note → **`bun run upsample --group <name>`** and/or **regen `--min-count`** candidates. Skip this bucket entirely while any higher goal fails.
   - *Config* — best vs exported fusion variant; the lagging head metric from step 2; `INFONCE_TEMP`, `LR`, dropout, `EARLY_STOP_PATIENCE`, batch sizes; early-stop epoch → **config diff** candidates.
   - *Data quality* — pull a sample from a suspect slice (rows tagged `color:*` / `flag:*` / `neg:*`, or the worst `keywords_flex` misses) with `grep`/`jq` on `data/data.jsonl`; look for mislabeled emojis, wrong-set styles, palette drift → **data-quality check** items (what to inspect, the exact command, what "bad" looks like).
   - *Eval set* — is `data/eval.jsonl` representative? emojis/flags/concepts under-represented vs the vocab; whether to raise `regen --n`; whether a failing target needs its own probe slice in `tools/report.py`. CLDR probe rows come from `data/cldr.jsonl`.
   - *Architecture* (recommend-only) — receptive field 15 vs 42; channel chain / dilation / kernel; fusion combiner shape; **on-device budget**: quantify any param-count / wasm-latency cost.
   - *Diagnostic gap* — for the top failing goal, can the current report + TensorBoard actually explain *why* it fails (which inputs, which classes, confident-vs-unsure errors)? If not, the fix this loop is a new analysis — pick from *Tool catalogue → Should have*.

### Step 4 — Write the goals file

Write `goal/<YYYY-MM-DD>-<short-sha>.yml` (`date +%F`, `git rev-parse --short HEAD`) following `goal/README.md` — same spaced keys as `goals.yml`. `tools/report.py` reads the **newest** `goal/*.yml`, so a new file supersedes the last one; commit it with the plan. If a long-term target itself needs to change, edit `goals.yml` (not the per-iteration file) and say why in the plan.

**Every priority-ordered goal gets a target this iteration — none are omitted.** First find the **focus priority**: the priority number of the highest-priority goal that is still unmet (from `status.goals` / `goals.compare`). Then, per goal leaf:

| Goal's priority vs. the focus priority | Last iteration's result | Next target to write |
|---|---|---|
| at or above focus (this loop is fighting for it, or already won it) | `met` | **hold** — keep it at the value it cleared (or step it toward the hard target if there's obvious headroom). Never lower it. |
| at or above focus | `unmet` but moving | an **achievable next step**: roughly `actual + one loop's expected gain`, capped at the hard target. Don't restate the hard target if it's far — a target the loop can't hit in one step gives no signal. |
| at or above focus | `unmet` and flat / regressed | keep last target; the plan's job is a diagnostic or a bigger swing, not a number change. |
| at or above focus | `unmeasured` (source not wired) | keep the intended target **and** make the plan's #1 action "build the probe that measures it" (see the `†`/`‡` notes and *Tool catalogue*). |
| **below** focus (a lower-priority goal the loop isn't earning its keep on yet) | met or unmet | an **easy target — hold at (or just above) the current measured value**, not a stretch. Never below current (that would read as sanctioning a regression). List it in `meta.deferred` with the one-line reason ("lower priority than the priority-`<n>` focus this iteration"). |

`vocabulary.coverage` (priority 8) follows the same rule but at the aggregate level, not per-group: while it is below focus, either omit the `coverage` branch entirely or hold every currently-gated group at its current score — don't reach for new groups. Only treat it as the focus (start stepping specific group targets) once every higher-priority goal is `good`.

This same per-leaf reasoning (held / stepped / kept flat / unmeasured→probe / easy-hold-below-focus) is restated in the plan's **Current Goal Adjustments** section — the goal file is the machine-read artifact the report grades against, the plan section is the human-read rationale for the same numbers.

### Step 5 — Derive the plan

6. **Set the loop verdict** (step-change / blocked-need-diagnostic / marginal-only, per the Overview) and **prioritize** — rank actions by (expected gain toward the highest-priority *failing* goal) then (cheapest / least-generative first: diagnostic ≈ config edit ≈ data-quality fix > data-gen upsample > Modal GPU). Respect the priority order. Never propose a change that risks the priority-1 exact-keyword fusion Acc@1 without a risk line and a mitigation. Any upsample option must name the non-generative fix that was considered and ruled out.
7. **Write** `plans/<report-stamp>/plan.md` from the template below (reuse the report dir's exact `<stamp>`). Reference the goals file written in step 4. Write the *Abstract* **last** (it summarizes everything else) but it renders **first** in the file, immediately after the metadata block — a short synthesis of the newest report + TensorBoard read from Step 3, not new analysis. *Current goals & status* / *Current gaps* come next; *Proposed actions* follows, organized under the seven fixed aspect headings (Data, Model Configuration, Model Architecture, Training Configuration, Metric Adjustments, Global Goal Adjustments, Current Goal Adjustments) — each a **menu of ranked options**, not a to-do list, every option with its expected effect, cost, and priority-1 risk. Keep every heading even when nothing applies this iteration — write "no change proposed" under it instead of deleting it.
8. **Stop and present the options** — summarize the menu in chat (e.g. via `AskUserQuestion`) and wait for the user to pick one or more before doing anything in Step 6.

### Step 6 — Implement the plan

9. **Execute only what the user picked** in step 8, in priority order if they picked more than one. Record each under *Applied this run* with its verification output. Architecture / `model/model.py` structure / loss-function changes stay **recommend-only** — write them in the plan, don't apply them regardless of what's picked.

**Qualifies to implement** (list in the plan cheapest / least-generative first — this order is also the default option ranking for step 8):
- **A new read-only analysis script** under `tools/analysis/*.py` (or `*.ts`, e.g. `kw-search.ts`) — loads `pt/*.pt` + `data/*.jsonl` (or `web/public/kwproj.json` + `data/labels.json`), writes a report/plot or prints a table, **mutates nothing** (no `.pt`, no `data/`, no `web/`). Wiring it into `tools/report.py` as a permanent section is recommend-only (bigger surface, changes the standard report) — a `.ts` script gets wired via `subprocess` (see `_kw_search_rows`), same as `kw-search.ts` was.
- **Building a probe the goals file needs** — e.g. `cards.energy` (OKLab energy distance) for `color generator.energy distance.global`. Report-side / committed-data only; touches no model weights.
- Switching `FUSION_EXPORT_VARIANT` in `model/export_onnx.py` when the report shows another variant clearly wins, then `uv run python model/export_onnx.py`.
- Localized scalar nudges in `model/config.py` — one small step per knob, only where TensorBoard clearly points: `LR`, `INFONCE_TEMP`, `DROPOUT_EMOJI`, `DROPOUT_STYLE`, `EARLY_STOP_PATIENCE`, `EPOCHS_TASK`.
- **Data-quality fix on rows already in the corpus** — e.g. `bun run upsample --reannotate colors|emojis` on a slice a data-quality check flagged, or fixing a tool that was silently mislabeling/misrouting existing generation (as opposed to generating more rows). Prefer this over fresh generation whenever the gap is in how existing data is used or labeled, not in what concepts exist.
- **Targeted data upsample — last resort.** Only once the options above have been considered and ruled out (or already tried and exhausted) for the failing goal: `bun run upsample --emojis "<e1>,<e2>"` / `--keywords "<k1>,<k2>"` / `--rare`, then `bun run regen`. State in the plan which non-generative option was ruled out and why. Re-read the regen summary; if vocab size changed, record it (future Acc@k is no longer comparable).
- **Coverage gap fill** — the one upsample case that's close to unavoidable, since there's no existing row to recover a missing concept from: a whole under-target Unicode group (`bun run upsample --group <name>`, `--count N` for a random subset), missing flags (`bun run upsample --flags`, `--count N` for a partial pass), or a missing animal/object/emotion/symbol concept (`bun run upsample --keywords "<concept words>"`). Then `bun run regen`. Note: added rows only pull an emoji into the vocab once it clears `regen`'s `--min-count`; a single upsample pass may not be enough.

**Recommend-only (write in the plan, do NOT apply):**
- Any edit to `ENCODER_CHANNELS` / `ENCODER_DILATION` / `ENCODER_KERNEL_SIZE` / `CHAR_EMBED_SIZE` / `TEXT_EMBED_SIZE`, `model/model.py` structure, or the loss (`lse_infonce`, the fusion combiners, GAN losses).
- Anything touching `normalize` / `CHARS` (invalidates `.pt`, breaks web parity).
- `regen --min-count` / `--max-count` / `--n` changes (re-partition the eval split — break Acc@k comparability; size them with `bun run regen --matrix` first and let the user run it).
- Multi-knob config sweeps, or anything needing a Modal GPU run to evaluate.

**Guardrails:**
- **Never execute a Step 6 action before the user has picked from Step 5's options** — write the plan, stop, ask.
- A background job auto-commits "fix" commits to this branch (see `[[concurrent-training-pipeline-commits]]`): never `git add -A`; scope any commit to the files you touched.
- After editing a `.py`: `uv run ruff check .` and `uv run ruff format --check .` (leave `model/train.py`'s pre-existing format alone — it fails `--check` independently of your edits).
- After a data change: `bun run regen`, then re-read its summary.
- Never run `train` / `train --local` — see `[[dont-run-main-to-test]]`. The retrain is the user's to kick off (loop step 1).
- No comments or docstrings in any code touched — see `[[no-comments-or-docstrings]]`.
- Run the python test scripts after touching `tools/report.py` or `model/*`: `uv run python tools/test_report.py` · `model/test_runmeta.py` · `model/test_train_cli.py`. After touching `tools/analysis/*.ts`: `bun test tools/analysis/`.

## Tool catalogue

What the loop can reach for at steps 3 and 6. **Have** = exists today; **Should have** = build it (a read-only `tools/analysis/*.py` qualifies to implement — see step 6) when the standing report can't explain the top failing goal. When blocked, ship the **one** diagnostic that most directly unblocks that goal — a loop that ships it counts as progress.

### Have

| Tool | Command | Gives |
|---|---|---|
| Full report | `uv run python tools/report.py --pt pt` | the merged `status.goals` scorecard (global target + this iteration's target + current value, vs `goals.yml` + newest `goal/*.yml`), `goals.compare` (full per-leaf detail), the 5-way `emoji.eval` Acc@k, `cldr` probe, `status.vocab_coverage`, `keywords_flex` vocab diagnostic, and — as a **default** section now — the Cards end-to-end gold-set (style + colour, over the shipped web-app inference graph). |
| Rebuild group map | `bun run build-groups` | committed `data/group.json` (Unicode subgroup → emoji list) — the per-group vocab-coverage input. Re-run after an `emojibase-data` bump. |
| Upsample a whole group | `bun run upsample --group <name> [--count N] [--dry]` | fresh rows targeting every emoji in Unicode subgroup `<name>` (random `N` with `--count`) — closes a `vocabulary.coverage` gap. Then `bun run regen`. |
| Inference spot-check | `uv run python model/pred.py --pt pt` | `data/pred.jsonl` — top styles/emoji + `fusion_top_labels` for the first 200 eval rows. Eyeball concrete failures. |
| Model stats | `uv run python model/config.py` | channel chain, effective kernels, receptive field (15) vs `MAX_TEXT_LEN` (42), param counts. |
| Regen summary | `bun run regen` | rows read/kept/dropped, vocab size, most/least frequent kept emojis, `kw` key count + mean nonzero. |
| Regen sweep | `bun run regen --matrix` | `kept-rows / emoji-vocab` grid over `--min-count` × `--max-count` — size a vocab change before proposing it. |
| CLDR-vs-train vocab overlap | `bun run regen --analysis` → `docs/regen.md` | which emoji are in `data/cldr.jsonl` only, train only, or both. |
| Upsample dry-run | `bun run upsample --dry <mode>` | what a mode would target/generate, no writes. |
| Corpus preview | `bun run preview data/eval.jsonl` / `bun tools/preview.ts <file>` | rendered cards for a slice. |
| TensorBoard | `uv run tensorboard --logdir runs` | `MRR/{e,s,fusion,kw}/val`, per-variant fusion MRR, `auc/critic/val`, `energy/gan/val`, `gate/a`·`gain/beta`·`mix/g`, early-stop epoch. |
| ONNX export | `uv run python model/export_onnx.py` | refresh `web/public/` after a `FUSION_EXPORT_VARIANT` switch (no retrain). |
| Raw slicing | `grep`/`jq` on `data/data.jsonl` | inspect rows by `color:*` / `flag:*` / `neg:*` / `keyword:*` / `meta.src` tag. |
| Standalone keyword search | `bun run tools/analysis/kw-search.ts <data.jsonl> [--json]` | the production exact+uFuzzy keyword search (same algorithm as `regen.ts`/`web/src/keywords.js`, `PRIMARY_BONUS`-ranked) run over any `data.jsonl`-schema file's full text, reporting Acc@{1,5,10}. `--json` also emits each scored row's sparse `kw` vector. `tools/report.py:_kw_search_rows` shells out to it (on the CLDR keyword set) to populate `report.json.keyword.fuzzy` / `keyword.fuzzy_fusion`; needs `bun` + `web/public/kwproj.json`. |

### Should have — build when a failure is unexplained

**Emoji / CLDR**
- **Per-class Acc@k** — Acc@1/@5 for every emoji in the vocab (and per emojibase group / per `meta.src` tag). Surfaces which classes carry the miss rate vs. a uniform sag.
- **Confusion pairs** — for eval misses, top-1 predicted emoji vs. the gold set. Ranked → merge candidates, annotation noise, or a real semantic gap.
- **Never-retrieved set** — target emojis that never reach top-10 for any eval row → coverage / embedding-collapse signal.
- **Error taxonomy** — bucket eval misses by text length, negation (`neg` tag), single- vs multi-emoji rows, topic / `src` tag, style → which slice to upsample or re-annotate.
- **Score-margin / calibration** — histogram of the top-1 logit gap for correct vs wrong rows; are wrong answers confident (needs data/loss) or unsure (needs capacity/temp)?
- **Fusion gate behaviour** — distribution of the learned gate/gain/mix scalar across eval texts; rows where `kw` would have helped but the gate suppressed it (or vice-versa).
- **Embedding neighbours** — nearest emojis in `EmojiEmbedding` space for probe words; k-NN purity by group.

**Colour / style**
- **OKLab scatter** — generated vs real palette clouds per colour tag; per-slot (`bg1`/`bg2`/`text_color`) error. Also emit `cards.energy` (OKLab energy distance) to wire the `colors.energy` goal leaf.
- **Style confusion matrix** — gold vs predicted style on the Cards gold set and on an eval slice.

**Data / corpus**
- **Near-duplicate & annotation-disagreement audit** — rows sharing a `normalize(text)` key with conflicting emoji/style sets; duplicate rate; label entropy per text.
- **Eval representativeness** — vocab coverage of `data/eval.jsonl` (which emojis/flags/concepts are thin or absent) vs `data/train.jsonl`.
- **Cross-run trend** — table of every `plans/*/plan.md` scorecard (and `goal/*.yml` target) over time → is the loop moving, and where did each jump come from?

## Plan template

```markdown
# Improvement Plan — <stamp> · <sha>

Report: report/<stamp>-<sha>/report.html
Goals file written: goal/<YYYY-MM-DD>-<sha>.yml
Prev plan: plans/<prev-stamp>/plan.md   (or "none")
Changed since prev: <one line from git log>
Best emoji variant: <status.best_emoji_variant>
Loop verdict: <step-change | blocked — need diagnostic | marginal-only> — <one line: the expected visible move, or the diagnostic being shipped>

## Abstract

3-6 sentences, written last but placed first — the TL;DR of Step 3's analysis, for a reader who won't read past it: what the newest report + TensorBoard actually showed (the headline number and its direction vs prev), what's driving it (which head/metric from TensorBoard — under- vs over-training, a plateau, a lagging loss), the loop verdict and why, and the one thing this plan is proposing to do about it. No new claims — every sentence here must be traceable to a `report.json` field or a TensorBoard curve cited later in the plan.

## Current goals & status  (report.html's opening "Goal status" table + Δ vs prev)

Copy `report.html`'s merged, priority-ordered table verbatim (same as `report.json.status.goals`, sorted priority 1→8 — highest priority first) and add the plan-only Δ column:

| Goal (priority) | Global target (goals.yml) | Current target (this iteration's goal/*.yml) | Current value | Status | Δ vs prev plan |
|---|---|---|---|---|---|
| Exact keyword acc@1 (1) | ≥0.95 | ≥0.75 | 0.000 | 🔴 | +0.00 |
| Full-text emoji acc@1 (2) | ≥0.80 | | | | |
| Full-text emoji acc@5 (2) | ≥0.85 | | | | |
| Full-text emoji acc@10 (2) | ≥0.90 | | | | |
| Fuzzy keyword acc@1 (3) | ≥0.90 | | | ⚪ | |
| Color energy · global (4) | ≤0.01 | | | ⚪ | |
| Style acc@1 (5) | ≥0.80 | | | ⚪ | |
| Max text len (6) | ≥42 | | | | |
| Emoji vocab size (7) | ≥700 | | | | |
| Vocab coverage (per Unicode group) (8) | all ≥ target | | | ⚪ deferred | |

Status here is 🟢 good (clears the global target) / 🟡 amber (clears only this iteration's easier target) / 🔴 red (misses even that) / ⚪ na (unmeasured, or `deferred` for vocab coverage while a higher goal is open) — see `_grade_merged` in `tools/report.py`.

Priority-1 regression gate — fusion model on exact keywords Acc@1/@5/@10 (`keyword.fusion`) vs prev: <PASS / FAIL + numbers>

## Current gaps

<per-bucket diagnosis from Step 3 (only the buckets with evidence this iteration): class balance / vocab coverage / config / data quality / eval-set representativeness / architecture headroom / diagnostic gap. Each gap names the goal it's blocking, the evidence (report.json field or TensorBoard curve), and whether the report + TensorBoard can already explain it or a new diagnostic is needed.>

## Proposed actions

Every action below states its **target goal**, expected effect, priority-1 (exact-keyword fusion) risk, and cost (config edit | data-gen | Modal GPU). Ordered cheapest / least-generative first within each heading. Nothing here runs until the user picks from the menu.

### Data
- **Regen params** (recommend-only — user runs it): `--min-count` / `--max-count` / `--n` proposal, or "no change proposed". Size with `bun run regen --matrix` first; note Acc@k stops being comparable across the change.
- **Upsampling** (last resort — avoid if possible; name the non-generative option ruled out first): "no upsample proposed" unless a genuine coverage hole was found. Otherwise: ruled-out option, then `bun run upsample --keywords/--emojis/--rare/--flags/--group …`, then `bun run regen`.
- **`CLDR_WEIGHT`**: current → proposed (or "no change"), because <evidence — `MRR/e/val` vs `MRR/e/train` split, `cldr.acc_at_k` fit-vs-generalization gap>. Priority-1 risk: <…>.
- **Other data adjustments** — data-quality fixes on existing rows (`bun run upsample --reannotate …` or a tool fix) and eval-set representativeness updates, or "none this iteration".

### Model Configuration

Full inventory of every current `model/config.py` capacity/regularization knob that is **not** a structural edit (those are Model Architecture) and **not** a training-loop knob (those are Training Configuration): `EMOJI_EMBED_SIZE`, `STYLE_EMBED_SIZE`, `DROPOUT_EMOJI`, `DROPOUT_STYLE`, `DROPOUT_CRITIC`, `RELU_SLOPE`, `Z_WEIGHT`, `GEN_CHANNELS`, `CRITIC_COLOR_CHANNELS`, `CRITIC_TEXT_CHANNELS`. Every row appears every iteration — unchanged knobs get "—" in the last three columns, not omission. `EMOJI_EMBED_SIZE` / `STYLE_EMBED_SIZE` resize only their own head's projection, not the shared `TEXT_EMBED_SIZE` trunk, so they stay configuration rather than architecture.

| Knob | Current value | Proposed value | Reason for change | Expected impact |
|---|---|---|---|---|
| `EMOJI_EMBED_SIZE` | | | | |
| `STYLE_EMBED_SIZE` | | | | |
| `DROPOUT_EMOJI` | | | | |
| `DROPOUT_STYLE` | | | | |
| `DROPOUT_CRITIC` | | | | |
| `RELU_SLOPE` | | | | |
| `Z_WEIGHT` | | | | |
| `GEN_CHANNELS` | | | | |
| `CRITIC_COLOR_CHANNELS` | | | | |
| `CRITIC_TEXT_CHANNELS` | | | | |

Only rows matching the Step 6 whitelist (`DROPOUT_EMOJI`, `DROPOUT_STYLE`) are auto-implementable if picked; every other row is a recommendation for the user to apply by hand.

### Model Architecture  (avoid unless Model Configuration changes are insufficient)
- <proposed change to `ENCODER_CHANNELS` / `ENCODER_DILATION` / `ENCODER_KERNEL_SIZE` / `CHAR_EMBED_SIZE` / `TEXT_EMBED_SIZE`, `model/model.py` structure, or a loss shape (`lse_infonce`, a fusion combiner, a GAN loss)> — reason: <which Model Configuration knob(s) were tried or would plausibly not be enough, and why>. Expected impact: <…>. Param-count / wasm-latency cost: <quantified, e.g. via `uv run python model/config.py`>. Needs a Modal GPU run to validate. Recommend-only — do not apply. Or "no architecture change proposed".

### Training Configuration

Full inventory of every current `model/config.py` optimization-loop knob: `LR`, `GAN_GEN_LR`, `GAN_CRITIC_LR`, `GAN_GEN_MARGIN`, `GRAD_CLIP_GEN`, `GRAD_CLIP_CRITIC`, `INFONCE_TEMP`, `TASK_BATCH_SIZE`, `GAN_BATCH_SIZE`, `EPOCHS_TASK`, `EPOCHS_GAN`, `VAL_CHECK_INTERVAL`, `EARLY_STOP_PATIENCE`. (`CLDR_WEIGHT` lives in the same file but is proposed under Data, not here — it's a corpus-mix knob, not an optimizer knob.) Every row appears every iteration — unchanged knobs get "—" in the last three columns, not omission.

| Knob | Current value | Proposed value | Reason for change | Expected impact |
|---|---|---|---|---|
| `LR` | | | | |
| `GAN_GEN_LR` | | | | |
| `GAN_CRITIC_LR` | | | | |
| `GAN_GEN_MARGIN` | | | | |
| `GRAD_CLIP_GEN` | | | | |
| `GRAD_CLIP_CRITIC` | | | | |
| `INFONCE_TEMP` | | | | |
| `TASK_BATCH_SIZE` | | | | |
| `GAN_BATCH_SIZE` | | | | |
| `EPOCHS_TASK` | | | | |
| `EPOCHS_GAN` | | | | |
| `VAL_CHECK_INTERVAL` | | | | |
| `EARLY_STOP_PATIENCE` | | | | |

Only rows matching the Step 6 whitelist (`LR`, `INFONCE_TEMP`, `EARLY_STOP_PATIENCE`, `EPOCHS_TASK`) are auto-implementable if picked; every other row (GAN optimizer knobs, batch sizes, grad clip, `VAL_CHECK_INTERVAL`) is a recommendation for the user to apply by hand.

### Metric Adjustments
- **Wrong signal**: <if a current metric grades the wrong thing for the top failing goal, propose the replacement and why it's better — cite the `report.json` field it would change>, or "none this iteration".
- **Missing signal**: <if a metric is missing that would explain the top *Current gaps* entry, propose it (pull from *Tool catalogue → Should have*) and why it matters>, or "none this iteration".

### Global Goal Adjustments  (goals.yml — the long-term targets)
- <if a `goals.yml` leaf looks structurally unreachable given everything observed across iterations, or was cleared with room to spare, propose a revised long-term number with the evidence>, or "no change proposed — targets still appropriate".

### Current Goal Adjustments  (goal/<YYYY-MM-DD>-<sha>.yml — this iteration's targets)
<mirrors the Step 4 table: per gated leaf, held / stepped from X to Y / kept flat (blocked) / unmeasured→build probe, with the one-line reason. Flag explicitly if a leaf looks too ambitious for one loop (soften it here rather than in goal/*.yml directly) or too easy (step it further).>

## Applied this run

<the option(s) the user picked, with verification output — ruff, regen summary, test scripts, export — or "none yet — awaiting the user's pick from the options above">
```

## Common mistakes

- Comparing Acc@k across runs where the emoji vocab size changed — the dynamic vocab shifts the eval split; not comparable. Note vocab size in every plan.
- Optimizing short-text emoji at the cost of the `cldr` probe — violates priority 1.
- Treating `fusion_gate`/`fusion_mix` reading equal to `EmojiHead` as a bug — a combiner collapses to the raw model when its learned scalar is ~0; report the gain from the best variant instead.
- Writing a goals-file target equal to the far-off hard target when the loop can only move a fraction of the way — it gives no per-loop signal. Step it.
- Lowering a goal that was already `met`, or gating `vocabulary.coverage` while a higher-priority goal is still red.
- Editing a target in a `goal/*.yml` when the *long-term* target changed — that belongs in `goals.yml`.
- Reading an older `report/` dir than the latest train, or one with non-empty `provenance.issues`.
- Skipping TensorBoard at step 3 — the report alone can't tell you under- vs over-training or which head's loss is stuck.
- Recommending architecture changes without quantifying the param-count / wasm-latency cost.
- Running training to "check" a recommendation — leave that to the user (loop step 1).
- Handing back a plan whose entire content is a marginal nudge because the report didn't obviously point anywhere — that means the missing piece is a diagnostic; build it this loop.
- Proposing "add more data" / "train longer" as the action when no analysis shows *which* data or that the model is under-trained. Name the slice or the metric first.
- Reaching for `bun run upsample` before checking whether a diagnostic, config nudge, or data-quality fix on existing rows could close the gap — upsampling is the last resort, not the first idea.
- Implementing anything in Step 6 without first stopping at Step 5's option menu and getting the user's pick — a written plan is not standing authorization to run it.
- Omitting a lower-priority goal from `goal/*.yml` entirely — every priority-ordered goal needs a target this iteration; below the focus priority that target is an easy hold-at-current, not silence.
- Writing a stretch target on a goal below the focus priority — that spends the plan's effort on the wrong goal and defeats the point of "easy" targets for goals that aren't this loop's fight.
- Writing the Abstract before Step 3's analysis is done, or letting it assert something no `report.json` field or TensorBoard curve backs up elsewhere in the plan — it's a summary of the analysis, not a place to introduce new claims.
- Leaving a Model Configuration / Training Configuration table incomplete — it must inventory every current `model/config.py` knob in that bucket, not just the ones being changed; an unchanged knob still gets a row ("—" in the last three columns), never omission.
- Proposing a Model Architecture change without first showing which Model Configuration knob was tried (or would plausibly not be enough) — architecture is the last resort within model changes, the same way upsampling is the last resort within data changes.

## When NOT to use

- Mid-training, or before any `report/` exists (run `uv run python tools/report.py --pt pt` first).
