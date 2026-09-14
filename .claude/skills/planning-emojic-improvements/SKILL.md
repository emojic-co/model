---
name: planning-emojic-improvements
description: Use when a training run has finished (train then tools/report.py) and the eval report needs to become a prioritized, ready-to-run improvement plan against goals.yml. Also use when asked to review the latest report/ folder, read TensorBoard logs, or plan the next training iteration.
---

# Planning Emojic Improvements

## Loop

| # | Step | Who |
|---|---|---|
| 1 | `train` (full pipeline, all heads + the color GAN, then a `web/public/` export) | user |
| 2 | `tools/report.py` → `report/<stamp>-<sha>/{report.html,report.json}` | pipeline |
| 3 | **Analyze** — read the newest report, TensorBoard, `goals.yml`, and the current `model/model.py` / `model/train.py` / `model/config.py` | this skill |
| 4 | **Plan** — `plans/<stamp>/plan.md`: current status vs. `goals.yml`, gaps, 1-3 ranked concrete modifications — then stop and present them | this skill |
| 5 | **Implement** — only the option(s) picked; config/data-level changes only, architecture/loss stay recommend-only | this skill, after the pick |
| 6 | back to step 1 | user |

## Principles

- **`goals.yml` (repo root) is the only source of targets.** There is no per-iteration target file — always read `goals.yml` fresh and grade the newest report against it directly. Never restate one of its numbers in this skill or let a plan go stale against it.
- **Priority order** (rank guides focus, but weigh it against how far each is currently from its `goals.yml` target — a badly-missed priority-2 goal can outrank a nearly-met priority-1 one for a single loop's action, but never at the cost of regressing priority 1):
  1. emoji prediction accuracy
  2. color prediction quality
  3. style prediction accuracy
  4. coverage of emoji categories (a floor, not a stretch target — act on it once 1-3 are close to their `goals.yml` targets)
- **Never hardcode a metric field, TB tag, config knob, or number in this skill or in a plan.** Read it live from the report / TensorBoard / `model/config.py` / `model/train.py` every time — these drift, and a stale name silently becomes wrong advice.
- **Data upsampling is a last resort.** Before proposing it, show why a diagnostic, a config nudge, or a fix to data already in the corpus (mislabeled, never trained on, drowned out) couldn't plausibly close the gap.
- **Always discuss before acting.** Step 4 ends in 2-4 ranked options presented to the user (e.g. via `AskUserQuestion`); step 5 never runs unprompted.
- **Aim for a visible move, or ship a diagnostic.** If the report + TensorBoard can't explain why the top gap is stuck, the plan's #1 action is a small read-only diagnostic, not a guess.

## Always read, every loop (live — do not rely on a prior loop's values)

- Newest `report/<stamp>-<sha>/{report.html,report.json}` — confirm its `-<sha>` and provenance match `git rev-parse --short HEAD` before trusting it; stop and say so if stale or inconsistent.
- Newest `runs/` TensorBoard run — both the encoder/heads sub-run and the color-GAN sub-run. List each one's actual logged scalar tags (`EventAccumulator(dir).Tags()['scalars']`) before reading them; don't assume a tag name from a previous loop.
- `goals.yml` — the only target source.
- `model/model.py` — current architecture: the shared encoder's block stack, each head, the GAN generator/critic.
- `model/train.py` — current loss, checkpoint/early-stop monitor logic (which metric each stage actually optimizes for), and any data-mixing/sampling logic.
- `model/config.py`, plus `uv run python model/config.py` — every current hyperparameter and derived stat (param counts, receptive field vs. max text length), live.
- The newest previous `plans/*/plan.md` — regression check against last loop's numbers.
- `git log --oneline` since that previous plan — what actually changed.

## Encoder block capacity — read this report section every loop

The report's **"Model — Encoder block capacity"** section breaks down, per encoder block and per head, how much of that head's output each block actually contributes — separately over short single-token inputs, short phrases, and full sentences. Cross-reference it against `model/config.py`'s current channel/dilation/kernel chain and its receptive-field-vs-max-text-length figure, and reason about:

- **A block whose contribution collapses on full sentences relative to short inputs** — a capacity flag on priority 1, since full sentences are real usage. Check whether it's the block with the longest reach, and whether the encoder's current receptive field actually covers the configured max text length.
- **A block a head barely uses at all vs. one it leans on almost entirely** — that head isn't exploiting the shared trunk's depth. Consider whether that head's own capacity/regularization knob is starving it before proposing an architecture change.
- **Two heads wanting opposite things from the same shared blocks** — a real tension worth naming explicitly, and a poor case for an architecture change that helps one and hurts the other.
- Any fix implied here at the architecture level (channel widths, dilation schedule, kernel size) is recommend-only — quantify its param-count/latency cost via `model/config.py` and flag that it needs a GPU run to validate.

## Workflow

### Analyze
1. Confirm the report is current (see *Always read*).
2. Read both TensorBoard sub-runs: which metric (per head, per GAN sub-part) is lagging; train-vs-val shape (under- vs. over-fit); the early-stop point.
3. Grade the report's current values against `goals.yml` per the priorities above; check the regression gate — priority 1 must never have dropped vs. the previous plan.
4. Read the block-capacity section per the section above.
5. Diagnose the top gap (by priority, weighted by current distance from target): does the evidence already explain it? If not, the plan's job is a diagnostic.

### Plan — `plans/<report-stamp>/plan.md`
- **Abstract** (write last, place first) — a synthesis only; every sentence must trace to a report field or TB curve cited below.
- **Current status** — the report's own scorecard verbatim (field names as this run actually has them), plus a Δ-vs-previous-plan column, ordered by the priorities above.
- **Gaps** — per priority, with evidence, and whether it's diagnostic-blocked or actionable now.
- **1-3 ranked concrete modifications** — the plan's real output. Each one states:
  - the exact file + knob/section to change, as read live this turn (never from memory or a prior plan),
  - current value → proposed value (or the shape of the change, if not a scalar),
  - the reasoning tying it to the evidence above,
  - an estimated effect (direction + rough magnitude) on its target goal,
  - the risk to priority 1 and how it's mitigated,
  - whether it's directly implementable this loop or recommend-only (see *Implement*).
- **`goals.yml` adjustments** — only if a target looks already cleared with room to spare, or structurally unreachable across repeated loops; otherwise "no change proposed."

Stop here and present the menu (e.g. via `AskUserQuestion`); wait for the user's pick before touching anything.

### Implement
Only the picked option(s), in priority order. Record verification output under *Applied this run*.

**Directly implementable:**
- A read-only diagnostic under `tools/analysis/*.py`/`*.ts` — reads `.pt`/data files, mutates nothing.
- A single scalar nudge to one `model/config.py` knob, only where this loop's report/TB evidence points to it.
- A data-quality fix on rows already in the corpus (not fresh generation).
- `uv run python model/export_onnx.py` to refresh `web/public/` after a report/export-only change.
- A targeted `bun run upsample` + `bun run regen`, only after ruling out the above (state what was ruled out and why).

**Recommend-only — write it, never apply it:**
- Any `model/model.py` structural change or loss-shape change.
- Anything touching text normalization or the char vocab (invalidates checkpoints, breaks web parity).
- Corpus re-partitioning (`regen`'s count/size knobs) — breaks Acc@k comparability; size with a dry-run sweep first and let the user run it.
- Multi-knob sweeps, or anything needing a GPU run to evaluate.

## Guardrails
- Never run `train` / `train --local` — that's the user's step; verify with `ruff` and the existing report only.
- A background job auto-commits to this branch — never `git add -A`; scope commits to files you touched.
- After a `.py` edit: `uv run ruff check .` and `--format --check` (skip `model/train.py`'s format check — pre-existing).
- After a data edit: `bun run regen`, re-read its summary.
- No comments or docstrings in touched code.
- After touching `tools/report.py` or `model/*`: run the existing Python test scripts. After `tools/analysis/*.ts`: `bun test tools/analysis/`.

## Common mistakes
- Restating a report field, TB tag, config value, or variable name here or in a plan instead of reading it live — it drifts, and stale advice looks confident. This is the reason this skill has no numbers or knob names in it.
- Comparing any vocab-size-dependent metric across a vocab-size change without saying so.
- Improving a lower priority at a stated or unstated cost to a higher one.
- Skipping TensorBoard — the report alone can't show under/over-fitting or which head/sub-part is stuck.
- Proposing "more data" or "train longer" without a specific slice or metric pointing there.
- Reaching for upsampling before ruling out a diagnostic, config nudge, or existing-data fix.
- Proposing an architecture change without first showing a config-level knob was tried or is clearly insufficient, plus a quantified param/latency cost.
- Skipping the "Model — Encoder block capacity" section, or reasoning about it without checking it against the current `model/config.py` receptive-field figure.
- Implementing anything before the user has picked from the plan's menu.

## When NOT to use
- Before any report exists for the current checkout (run `uv run python tools/report.py --pt pt` first), or mid-training.
