---
description: Diagnose the feeling classifier and recommend the ONE smallest change most likely to raise eval.jsonl feeling accuracy toward >90%
allowed-tools: [Bash, Read, Write, Edit]
---

# feeling_90_per

**Goal:** `acc/f/val` — top-1 feeling accuracy on the fixed gold holdout `eval.jsonl`
(`data.py:split`) — **> 0.90**.

One invocation = **one diagnostic pass + one written recommendation**. This command
**does not edit model/training source and does not train.** It reads the current
state, reconciles the last recommendation against what actually happened, and
names the single smallest, lowest-risk change most likely to move `acc/f/val`
next. Reaching 0.90 in one step is not expected; each recommended step should be
independently accuracy-positive.

The only file this command writes is the ledger `report/feeling_90_per.md`.
`config.py` guarantees `EPOCHS % EVAL_EPOCHS == 0` — any epoch recommendation must
keep that true.

## 1. Read the ledger and reconcile the last step

Read `report/feeling_90_per.md` (the running log of every change recommended, its
before/after `acc/f/val`, and whether it helped). Create it from the skeleton at
the bottom of this file if it is missing.

If the newest `runs/*/` directory is **newer than the last ledger row**, a
recommendation was acted on: measure its `acc/f/val` (step 2) and fill in the
row's `after` + `outcome` (`kept` if it beat the prior best by > 0.005,
`no-gain` otherwise, `reverted` if the source no longer shows the change).
Never re-recommend a change already in the ledger unless its note says
"retry bigger".

## 2. Measure the current accuracy

```bash
uv run python - <<'EOF'
import glob, os
from tensorboard.backend.event_processing import event_accumulator
runs = sorted(glob.glob("runs/*/"), key=os.path.getmtime)
for run in runs[-4:]:
    ea = event_accumulator.EventAccumulator(run, size_guidance={event_accumulator.SCALARS: 0})
    ea.Reload()
    print("\nrun:", os.path.basename(run.rstrip("/")))
    for tag in ea.Tags()["scalars"]:
        ev = ea.Scalars(tag); v = [e.value for e in ev]
        agg = max(v) if "acc" in tag else min(v)
        print(f"  {tag}: n={len(ev)} first={v[0]:.4f} last={v[-1]:.4f} best={agg:.4f}")
EOF
```

The **current accuracy** is the best `acc/f/val` across all runs that match the
current `config.py` / `model.py` (ignore stale-architecture runs). Note also:
`acc/f/train` at the last epoch, `loss/f/train` vs `loss/f/val`, and whether
`loss/f/train` is still descending at the final step.

Majority-class baseline on `eval.jsonl` is **0.143** (7 feelings, balanced ~112
rows each). Anything near that means no signal; the current model is well above it.

### Per-class eval accuracy + confusion (the reports don't show this)

```bash
uv run python - <<'EOF'
import collections, torch
from data import FEELING, _load, EVAL_PATH, collate_fn
from model import Model
m = Model(); m.load_state_dict(torch.load("model.pt", map_location="cpu")); m.eval()
rows = [s for _, s in _load(EVAL_PATH)]
x, _, y = collate_fn(rows)
with torch.no_grad():
    pred = m(x)[0].argmax(-1)
per = collections.Counter(); tot = collections.Counter(); conf = collections.Counter()
for t, p in zip(y.tolist(), pred.tolist()):
    tot[t] += 1; per[t] += (t == p); conf[(FEELING[t], FEELING[p])] += 1
print("overall", sum(per.values()) / sum(tot.values()))
for i, f in enumerate(FEELING):
    print(f"  {f:9} {per[i]:3}/{tot[i]:3} = {per[i]/tot[i]:.3f}")
print("top confusions:", conf.most_common(8))
EOF
```

## 3. Gather context (read, don't run)

- `config.py`, `model.py`, `train.py` — the current architecture and hyper-params
  (these drift; trust the code). Note especially: `EPOCHS`, `CHANNELS`,
  `CHAR_EMBED_SIZE`, `LR`, `WEIGHT_DECAY`, conv `padding`, the pooling op
  (`torch.max` over time), the loss (`nn.CrossEntropyLoss`), the optimizer
  (`Adam`, no schedule).
- newest `report/*.md` — the behavioral batteries (name prompts, negations).
  Sanity signal only; the metric is `acc/f/val`.
- `model.md` — standing improvement notes. May lag the code; cross-check.
- `git log --oneline -15` and `git diff` — what changed since the measured run.
- Train corpus shape (never read `data.jsonl` whole):

  ```bash
  python3 -c "
  import json, collections
  fe=collections.Counter(); n=0
  for l in open('data.jsonl'):
      l=l.strip()
      if not l: continue
      r=json.loads(l); n+=1; fe[{'Excited':'Happy'}.get(r['feeling'],r['feeling'])]+=1
  print('rows', n)
  [print(f'  {k:10} {v:6} {v/n:.3f}') for k,v in fe.most_common()]
  "
  ```

## 4. Diagnose the bottleneck (one paragraph)

Pick the dominant one from the step-2/3 evidence:

| Signal | Diagnosis | Look in menu at |
| --- | --- | --- |
| `acc/f/train` ≤ `acc/f/val`, or `loss/f/train` still falling at the last epoch, or `loss/f/train` ≫ `loss/f/val` | **Underfitting** — not enough capacity / epochs / step count | Tier 1, then Tier 2 (schedule) |
| `loss/f/val` rises off a clear minimum while `loss/f/train` keeps falling | **Overfitting** | Tier 3 (data), then regularization note |
| `acc/f/train` plateaus well *below* `acc/f/val` and `loss/f/train` stalls high | **Label-noise ceiling** in `data.jsonl` | Tier 3 item 11 |
| One or two feelings far below the rest in the per-class table; confusions cluster on the majority classes (Happy/Neutral) | **Train-prior / class imbalance** or a **structural** blind spot | Tier 2 item 10, then Tier 4 |
| `not <feeling>` and word-order cases fail in the report | **Order-blind trunk** (global max-pool = bag of n-grams) | Tier 4 |

## 5. Recommend ONE change

Take the **highest item not already in the ledger** that matches the diagnosis.
One knob per invocation. Prefer `config.py` over `train.py`/`model.py`; prefer a
small `model.py` edit over a new module. Do **not** recommend a Tier 4
architecture change unless the ledger shows **either** every Tier 1–3 item tried
**or** two consecutive `no-gain` steps.

### Tier 1 — `config.py` one-liners (start here for underfitting)

1. **`EPOCHS` → 150** (and set `EVAL_EPOCHS` to a divisor, e.g. 3). The best run
   on record trained 100 epochs with `loss/f/train` still falling; a shorter
   `EPOCHS` undertrains. Zero risk.
2. **Widen `CHANNELS`** `(32, 16, 16)` → `(64, 64, 64)` → `(128, 96, 64)`. The
   narrow tail feeds a 7-way head after a lossy global max-pool — the tightest
   bottleneck under an underfit diagnosis. One width per invocation; log the rest.
3. **`CHAR_EMBED_SIZE` 16 → 32.**
4. **`WEIGHT_DECAY` 1e-4 → 0** while underfitting (decay only subtracts capacity
   here). Tiny effect; bundle-worthy, low priority alone.
5. **`LR`** single-value probe: try `0.003` (steadier) or `0.02` (faster). Adam
   is already reasonable, so low expected value on its own — pair with item 6.

### Tier 2 — small `train.py` / `model.py` edits

6. **LR schedule** in `configure_optimizers`: `CosineAnnealingLR(T_max=EPOCHS)` or
   `ReduceLROnPlateau` on `acc/f/val`. Pairs with item 1; standard underfit lever.
7. **Conv `padding=0` → `padding=1`** in `model.py`. Three k3 convs + 2 pools
   shrink a 42-char field to ~9 steps; short `eval.jsonl` texts lose most of it.
8. **Pool `mean ⧺ max`** over the time axis instead of `max` alone
   (`model.py`, the `torch.max(out, dim=-1)` line); double the head's
   `in_features`. `max` keeps only the single peak window per channel.
9. **`label_smoothing=0.05`** on `nn.CrossEntropyLoss`. Mostly calibration;
   skip while the model is underfitting.
10. **Correct the train prior**: `data.jsonl` is ~2.2× imbalanced (Happy 0.24,
    Neutral 0.21, rest ~0.10) while `eval.jsonl` is balanced. Use
    `nn.CrossEntropyLoss(weight=inv_freq)` or a `WeightedRandomSampler`.
    Recommend when the per-class table shows the rare feelings lagging.

### Tier 3 — data (raises the ceiling; slower, uses other skills)

11. **`/data-quality`** — fixes mislabeled/broken rows in place, then retrain.
    Recommend when the diagnosis is label-noise ceiling.
12. **Grow / rebalance the corpus** — `bun feeling2emoji.ts` then
    `bun gen_labels.ts` (adds ~500 texts each for the least-covered feelings),
    or `/add-samples`.

### Tier 4 — architecture (only when Tiers 1–3 are exhausted or 2× no-gain)

13. **Add one order-sensitive layer** over the conv features: a single
    `nn.GRU(bidirectional=True)` or a small self-attention block, then pool.
    Restores the word-order / negation sensitivity a bag-of-n-grams trunk lacks.
14. **Swap the trunk** for a 2-layer tiny transformer encoder (d_model 64,
    2 heads) → mean-pool → linear head.

**Regularization note:** only after capacity is added (Tier 1/2) *and*
`loss/f/val` starts rising off a minimum: `nn.Dropout(0.1–0.2)` after the conv
stack, and restore `WEIGHT_DECAY` to 1e-4.

## 6. Write the recommendation

Append one row to `report/feeling_90_per.md` and print the same content. The row:

- **date**, **current `acc/f/val`** (+ which run), **diagnosis** (one line).
- **change**: the exact edit — file, symbol, old value → new value.
- **why**: the step-2/3 evidence that motivates it (numbers, not adjectives).
- **expected**: rough direction/size of the `acc/f/val` move and the risk.
- **verify next**: what the next invocation should read to judge it
  (`acc/f/val` best vs the current best; per-class table; `loss` gap).
- **after** / **outcome**: left blank — filled in by the next run (step 1).

## 7. Report back

2–3 sentences: current `acc/f/val` vs the 0.90 target and the 0.143 baseline, the
diagnosis, the single recommended change, and what it is expected to do.

## Ledger skeleton (`report/feeling_90_per.md`)

```markdown
# feeling_90_per progress

Target: `acc/f/val` > 0.90 on eval.jsonl (majority-class baseline 0.143).
Baseline: <acc> — <run name> — <date>.

| date | acc/f/val (run) | diagnosis | change (file: old → new) | why | expected | after | outcome |
| --- | --- | --- | --- | --- | --- | --- | --- |
```
