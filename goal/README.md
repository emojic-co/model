# `goal/` — per-iteration goal files

One file per train → report → analysis loop, committed:

```
goal/<YYYY-MM-DD>-<short-sha>.yml
```

Written by the `planning-emojic-improvements` skill at **loop step 4** (after reading the
report + TensorBoard, before deriving the plan). It states the targets the *next*
iteration aims to hit against the repo-root `goals.yml` — the single long-term goals
statement. `tools/report.py` reads the **newest** `goal/*.yml` (lexical sort — ISO date
prefix keeps it chronological) and merges each leaf's target into `report.json`'s
priority-ordered `status.goals` (`current target` alongside `goals.yml`'s `global
target`), and separately writes the full leaf-level comparison into `report.json` under
`goals`. (`goals.yml` itself is what both tables' `global target` column grades against.)

## Schema

Every priority-ordered goal (the same rows `status.goals` / `report.html`'s "Goal
status" table carries — `emoji prediction.*`, `style prediction.*`, `color generator.*`,
`max text len`, `vocabulary.size`) gets a target **every iteration**, even one this loop
isn't actively working: find the highest-priority goal that is still unmet, then —

- every goal **at or above** that priority gets a real, stepped target (see the table in
  `## Workflow → Step 4` of the skill: hold / step / keep flat / build-probe);
- every goal **below** that priority (lower-priority, i.e. the loop isn't earning its
  keep there yet) gets an **easy target — hold it at (or just above) its current
  measured value**, not a stretch target. This is not "omit it" — it's a deliberately
  unambitious number, so the report's merged table shows it 🟡 amber (on track with the
  easy ask) rather than 🔴 red, while the real fight stays on the goal(s) that are
  actually blocking.

`vocabulary.coverage` is the one exception to per-goal-row granularity: it is a single
row in the priority table (priority 8, "floor gate") backed by ~100 per-Unicode-group
leaves. Only gate the specific groups this loop is actually working; the report's
merged table computes an aggregate "N/M groups (this iteration)" figure from whichever
group leaves are present, and while any higher-priority goal is open the row still
renders `deferred` (grey) regardless of what's in `coverage`.

Keys are `goals.yml`'s literal spaced strings.

```yaml
meta:
  based_on: <doc or plan this derives from, optional>
  written_after_report: <report/<dir> name, or null>
  rationale: <one line - the bottleneck this iteration targets>
  deferred: [<goals.yml paths / branches given an easy hold-current target this iteration>]

goals:                       # every priority-row goal present; vocabulary.coverage is the exception (gate only the groups this loop targets)
  emoji prediction:
    exact keyword:           # priority 1 — the loop's focus this iteration
      "acc@1": 0.75
      "acc@5": 0.80
      "acc@10": 0.85
    fuzzy keyword:            # unmet, lower priority than the focus above — held easy
      "acc@1": 0.0
    full text:                 # unmet, lower priority — held easy at current
      "acc@1": 0.55
      "acc@5": 0.65
      "acc@10": 0.75
  style prediction:
    full text:                 # unmeasured (cards off) — held at 0 until wired
      "acc@1": 0.0
  color generator:
    energy distance:
      global: 0.05              # unmeasured — held until cards.energy is wired
      red: 0.20
      green: 0.20
      blue: 0.20
      dark: 0.20
      bright: 0.20
  max text len: 32               # already met at MAX_TEXT_LEN — held, not stepped past the hard target's headroom
  vocabulary:
    size: 400                    # unmet, lower priority — held at current vocab size
    coverage:
      animal-mammal: 0.45      # per-Unicode-group vocab share — only if this loop gates coverage
```

## Leaf → report.json source, and comparison direction

| goal path (flattened) | dir | report.json source | wired? |
|---|---|---|---|
| `emoji prediction.exact keyword.acc@{1,5,10}` | ≥ | `keyword.exact.acc_at_k[{0,4,9}]` — exact-match inverted-index keyword search scored against CLDR keywords; **priority-1 gate** | yes |
| `emoji prediction.fuzzy keyword.acc@{1,5,10}` | ≥ | `keyword.fuzzy.acc_at_k[...]` — uFuzzy-matched keyword search scored against CLDR keywords | yes — via `tools/analysis/kw-search.ts`, shelled out to by `tools/report.py:_kw_search_rows` (only when `bun` + `web/public/kwproj.json` are available) |
| `emoji prediction.full text.acc@{1,5,10}` | ≥ | `emoji.eval.acc_at_k[{0,4,9}]` — the `EmojiHead` classifier on eval short texts | yes |
| `style prediction.full text.acc@{1,5,10}` | ≥ | `cards.style_acc_at_k[{0,4,9}]` | yes (only when the `cards` section runs) |
| `color generator.energy distance.global` | ≤ | `cards.energy` | no — `cards` does not compute OKLab energy yet |
| `color generator.energy distance.{red,green,blue,dark,bright}` | ≤ | `cards.per_color.<c>.gt_mean_distance` | yes (only when the `cards` section runs) |
| `max text len` | ≥ | `data.max_text_len` (= `config.MAX_TEXT_LEN`) | yes |
| `vocabulary.size` | ≥ | `labels.emojis` (count) | yes |
| `vocabulary.coverage.<group>` | ≥ | `status.vocab_coverage.groups.<group>.score` (`|vocab ∩ group| / |group|`) | yes |

Each leaf under `emoji prediction` measures a different retrieval method for
the same task (predict emojis from text): the exact- and fuzzy-keyword leaves
go through the inverted-index keyword search, the full-text leaf goes through
the `EmojiHead` classifier. `report.json.keyword.model` (the `EmojiHead`
scored on CLDR keyword text) is a report **diagnostic**, not a graded goal.

`report.json.goals.compare` mirrors the `goals:` tree; each leaf becomes
`{target, actual, met, dir, delta}` (`actual`/`met` are `null` when the source is not
wired). `report.json.goals.summary` counts `met` / `unmet` / `unmeasured`. Rendered in
`report.html` as "Goals for this iteration — per-leaf detail", right under the merged
"Goal status" table (which carries the same targets one row per priority goal, next to
`goals.yml`'s global target and this run's current value — see
`tools/report.py:_section_status`).
