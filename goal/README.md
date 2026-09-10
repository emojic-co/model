# `goal/` — per-iteration goal files

One file per train → report → analysis loop, committed:

```
goal/<YYYY-MM-DD>-<short-sha>.yml
```

Written by the `planning-emojic-improvements` skill at **loop step 4** (after reading the
report + TensorBoard, before deriving the plan). It states the targets the *next*
iteration aims to hit and is a **strict subset** of the repo-root `goals.yml` — the
single long-term goals statement. `tools/report.py` reads the **newest** `goal/*.yml`
(lexical sort — ISO date prefix keeps it chronological) and writes a target-vs-actual
comparison into `report.json` under `goals`. (`goals.yml` itself is what
`report.json.status` grades against.)

## Schema

Mirror the `goals.yml` tree, keeping only the branches this iteration gates. Keys are
`goals.yml`'s literal spaced strings.

```yaml
meta:
  based_on: <doc or plan this derives from, optional>
  written_after_report: <report/<dir> name, or null>
  rationale: <one line - the bottleneck this iteration targets>
  deferred: [<goals.yml paths / branches intentionally not gated this iteration>]

goals:                       # every branch optional - omit what you are not gating
  emoji prediction:
    exact keyword:
      "acc@1": 0.95
      "acc@5": 0.95
      "acc@10": 0.95
    fuzzy keyword:
      "acc@1": 0.90
    full text:
      "acc@1": 0.70
      "acc@5": 0.85
      "acc@10": 0.90
  style prediction:
    full text:
      "acc@1": 0.80
  color generator:
    energy distance:
      global: 0.03
      red: 0.20
  max text len: 42
  vocabulary:
    size: 700
    coverage:
      animal-mammal: 0.45      # per-Unicode-group vocab share
```

## Leaf → report.json source, and comparison direction

| goal path (flattened) | dir | report.json source | wired? |
|---|---|---|---|
| `emoji prediction.exact keyword.acc@{1,5,10}` | ≥ | `keyword.exact.acc_at_k[{0,4,9}]` (non-learned kw predictor on CLDR, exact postings — **priority-1 gate**) | yes |
| `emoji prediction.fuzzy keyword.acc@{1,5,10}` | ≥ | `keyword.fuzzy.acc_at_k[...]` (uFuzzy-enabled kw predictor on CLDR) | no — needs a uFuzzy `kw` sidecar from `regen.ts` |
| `emoji prediction.full text.acc@{1,5,10}` | ≥ | `emoji.eval.<best fusion variant>_acc_at_k[{0,4,9}]` | yes |
| `style prediction.full text.acc@{1,5,10}` | ≥ | `cards.style_acc_at_k[{0,4,9}]` | yes (only when the `cards` section runs) |
| `color generator.energy distance.global` | ≤ | `cards.energy` | no — `cards` does not compute OKLab energy yet |
| `color generator.energy distance.{red,green,blue,dark,bright}` | ≤ | `cards.per_color.<c>.gt_mean_distance` | yes (only when the `cards` section runs) |
| `max text len` | ≥ | `data.max_text_len` (= `config.MAX_TEXT_LEN`) | yes |
| `vocabulary.size` | ≥ | `labels.emojis` (count) | yes |
| `vocabulary.coverage.<group>` | ≥ | `status.vocab_coverage.groups.<group>.score` (`|vocab ∩ group| / |group|`) | yes |

`report.json.goals.compare` mirrors the `goals:` tree; each leaf becomes
`{target, actual, met, dir, delta}` (`actual`/`met` are `null` when the source is not
wired). `report.json.goals.summary` counts `met` / `unmet` / `unmeasured`.
