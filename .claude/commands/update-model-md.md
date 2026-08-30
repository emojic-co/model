---
description: Regenerate model.md from the latest behavioral report, tensorboard logs, and current source
allowed-tools: [Read, Write, Edit, Bash]
---

# Update model.md

Rewrite `model.md` — the model-improvement notes — so every claim in it is
backed by the **latest** behavioral report, the **latest** tensorboard run, and
the **current** source. Do not train, do not run `main.py`/`train.py`.

## 1. Gather evidence

**Latest behavioral report** — pick the newest `report/model/*.md` (names sort
lexically: `MM-DD-HH:MM.md`) and read it in full:

```bash
ls -1 report/model/*.md | sort | tail -1
```

Note its summary line (feelings N/8, negations avoided N/8 + opposite N/6,
emoji top-1/top-3/top-5), the per-emoji table (rank / count / top-k), and which
emojis / keyword groups fail.

**Latest tensorboard run** — newest dir in `runs/` (its name encodes the config).
Extract the scalar trajectories:

```bash
uv run python - <<'EOF'
import glob, os
from tensorboard.backend.event_processing import event_accumulator
run = sorted(glob.glob("runs/*/"), key=os.path.getmtime)[-1]
print("run:", os.path.basename(run.rstrip("/")))
ea = event_accumulator.EventAccumulator(run, size_guidance={event_accumulator.SCALARS: 0})
ea.Reload()
for tag in ea.Tags()["scalars"]:
    ev = ea.Scalars(tag)
    print(f"{tag}: n={len(ev)} first={ev[0].value:.4f}@{ev[0].step} "
          f"last={ev[-1].value:.4f}@{ev[-1].step} min={min(e.value for e in ev):.4f}")
EOF
```

Read the train-vs-eval gap: which losses keep falling, which flatten or rise
(over/underfitting), where the best checkpoint lands.

**Current source** — read `model.py`, `train.py`, `data.py`, `config.py`,
`labels.json`, `test_model.py`. The architecture and hyperparameters drift;
trust the code, not the previous `model.md`.

**Dataset shape** — never read `data.jsonl` whole:

```bash
wc -l data.jsonl
python3 -c "
import json,collections,statistics
rows=[json.loads(l) for l in open('data.jsonl')]
fe=collections.Counter(r['feeling'] for r in rows); em=collections.Counter(r['emoji'] for r in rows)
mc=em.most_common()
print('rows', len(rows))
print('feelings', fe.most_common())
print('n emoji classes', len(mc), 'median', statistics.median(em.values()), 'max', mc[0], 'min', mc[-1])
print('tail', mc[-12:])
"
```

## 2. Rewrite model.md

Regenerate the file. Keep the existing skeleton — **1. Where the model stands**
(a battery-results table), **2. Root causes**, **3. Recommendations
(prioritised)** with impact tiers, **4. Suggested order of work** — but:

- Open with a line naming the exact report file and tensorboard run name this
  pass is based on, plus the source files read.
- Replace every number with the current one. If a metric named in the old
  file no longer exists in the code (e.g. an accuracy scalar that's no longer
  logged), drop it.
- Reconcile claims against the code you just read. Correct anything stale
  (architecture, head design, loss, `config.py` values, label counts,
  `CLAUDE.md` mismatches).
- Fold in the tensorboard read: state the train/eval loss trajectory and what
  the gap implies.
- Carry forward recommendations that still apply; mark as done / delete ones
  the code now implements; add new ones the latest evidence motivates.
- Every quantitative claim must trace to the report, the tb scalars, the code,
  or the dataset snippet above. No invented figures.

## 3. Report

Lint-check nothing (it's Markdown). In 2–3 sentences say which report and run
you used and what changed in the analysis since the previous version.
