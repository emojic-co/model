# `goal/` — per-iteration goal files

One file per train → report → analysis loop, committed:

```
goal/<YYYY-MM-DD>-<short-sha>.yml
```

Written by the `planning-emojic-improvements` skill at **loop step 4** (after reading the
report + TensorBoard, before deriving the plan). It states the targets the *next*
iteration aims to hit. `tools/report.py` reads the **newest** `goal/*.yml` (lexical sort —
ISO date prefix keeps it chronological) and writes a target-vs-actual comparison into
`report.json` under `goals`.

## Schema

```yaml
meta:
  based_on: <doc or plan this derives from, optional>
  written_after_report: <report/<dir> name, or null>
  rationale: <one line - the bottleneck this iteration targets>
  deferred: [<domains intentionally not gated this iteration>]

goals:            # every leaf optional - omit a domain you are not gating
  keyword:
    "acc@1": 0.90    # exact-keyword probe = CLDR keywords
    "acc@5": 0.95
  text:
    "acc@1": 0.70
    "acc@5": 0.85
    "acc@10": 0.90
  colors:
    energy: 0.03      # max OKLab energy distance vs real palettes
    red: 0.20         # max mean OKLab dF to the GT palette, per colour tag
    green: 0.10
    blue: 0.10
    dark: 0.10
    bright: 0.10
  coverage:
    vocab: 700        # min emoji vocab size
    diversity: 0.60   # min status.diversity.score
```

## Leaf → report.json source, and comparison direction

| goal path | dir | report.json source | wired? |
|---|---|---|---|
| `keyword.acc@{1,5,10}` | ≥ | `cldr.acc_at_k[{0,4,9}]` (CLDR keyword probe) | yes |
| `text.acc@{1,5,10}` | ≥ | `emoji.eval.<best fusion variant>_acc_at_k[{0,4,9}]` | yes |
| `colors.energy` | ≤ | `cards.energy` | no — `cards` does not compute OKLab energy yet |
| `colors.{red,green,blue,dark,bright}` | ≤ | `cards.per_color.<c>.gt_mean_distance` | yes |
| `coverage.vocab` | ≥ | `labels.emojis` (count) | yes |
| `coverage.diversity` | ≥ | `status.diversity.score` | yes |

`report.json.goals.compare` mirrors the `goals:` tree; each leaf becomes
`{target, actual, met, dir, delta}` (`actual`/`met` are `null` when the source is not
wired). `report.json.goals.summary` counts `met` / `unmet` / `unmeasured`.
