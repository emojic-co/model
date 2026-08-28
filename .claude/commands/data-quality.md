---
description: Sample 500 rows from data.jsonl, write a data-quality report (label correctness, text quality, coverage), then rewrite the rows the sample flagged as broken
allowed-tools: [Bash, Read, Write, Edit]
---

# Data quality

Judge the health of `data.jsonl` from a **500-row random sample**. Produce one
report covering label correctness, text quality, label coverage, and text-style
coverage, then repair the flagged rows in place (step 5). Do not train, do not
run `train.py`/`test_model.py`. Steps 1-4 are read-only; only step 5 writes
`data.jsonl`, and only to rewrite rows the sample judged broken — it never adds,
deletes, or reorders rows.

The label sets live in `labels.json` (`feelings`, `emojis`). The training
pipeline runs every `text` through `normalize` in `data.py` first
(`re` collapse of whitespace and 3+ char runs, lowercase, then **drop every
char not in `CHARS`** = `a-z` + `!?:()@$%&* ` + space). So digits, commas,
periods, apostrophes, quotes, in-text emoji and accented letters never reach the
model — text whose meaning depends on them is effectively corrupted.

## 1. Draw the sample

Never read `data.jsonl` whole. Sample 500 rows to a file next to the report:

```bash
mkdir -p report/data
STAMP=$(date +%m-%d-%H:%M)
shuf -n 500 data.jsonl > "report/data/$STAMP.sample.jsonl"   # fallback: sort -R data.jsonl | head -n 500
wc -l "report/data/$STAMP.sample.jsonl"
```

Read that sample file in full. It is your evidence for every judgment below.

## 2. Mechanical stats

Run once; these numbers are not up for interpretation, only the sample feeds the
judging in step 3.

```bash
uv run python - "report/data/$STAMP.sample.jsonl" <<'EOF'
import collections, json, statistics, sys
from data import normalize, CHARS

sample = [json.loads(l) for l in open(sys.argv[1], encoding="utf-8")]
labels = json.load(open("labels.json", encoding="utf-8"))
FEEL, EMO = labels["feelings"], labels["emojis"]

# ---- full-corpus coverage (counting only, never enters context) ----
full = collections.Counter(), collections.Counter()
with open("data.jsonl", encoding="utf-8") as f:
    for line in f:
        r = json.loads(line)
        full[0][r["feeling"]] += 1
        full[1][r["emoji"]] += 1
fe_all, em_all = full
print("== FULL CORPUS ==")
print("rows", sum(fe_all.values()))
print("feelings", dict(sorted(fe_all.items(), key=lambda x: -x[1])))
print("emoji classes present", len(em_all), "/", len(EMO))
missing = [e for e in EMO if e not in em_all]
print("emoji absent:", " ".join(missing) if missing else "none")
mc = em_all.most_common()
print("emoji top10", mc[:10])
print("emoji bottom10", mc[-10:])
lo = min(em_all.values()); hi = max(em_all.values())
print(f"emoji imbalance max/min = {hi}/{lo} = {hi/lo:.1f}x")
off = [f for f in fe_all if f not in FEEL] + [f for f in FEEL if f not in fe_all]
print("feeling labels off-vocab or unused:", off or "none")

# ---- sample-level ----
print("\n== SAMPLE (500) ==")
sfe = collections.Counter(r["feeling"] for r in sample)
sem = collections.Counter(r["emoji"] for r in sample)
print("feelings", dict(sorted(sfe.items(), key=lambda x: -x[1])))
print("distinct emojis in sample", len(sem))

# ---- normalize damage ----
casualties = []
for r in sample:
    t = r["text"]; n = normalize(t)
    raw = "".join(c for c in t.lower() if not c.isspace())
    kept = "".join(c for c in n if not c.isspace())
    lost = [c for c in raw if c not in kept and c not in CHARS]
    if any(ch.isdigit() for ch in t) or (raw and len(kept) / max(len(raw), 1) < 0.85):
        casualties.append((t, n, "".join(sorted(set(lost)))))
print(f"normalize casualties: {len(casualties)}/500")
for t, n, lost in casualties[:15]:
    print(f"  drop[{lost}]  {t!r} -> {n!r}")

# ---- exact + near duplicates in sample ----
norm_texts = collections.Counter(normalize(r["text"]) for r in sample)
dups = {k: v for k, v in norm_texts.items() if v > 1}
print(f"exact-after-normalize duplicate texts in sample: {sum(dups.values()) - len(dups)} extra rows across {len(dups)} texts")
for k, v in list(sorted(dups.items(), key=lambda x: -x[1]))[:10]:
    print(f"  x{v}  {k!r}")

# ---- length distribution (words) ----
wl = [len(r["text"].split()) for r in sample]
print("word count: min %d  p25 %d  median %d  p75 %d  max %d" % (
    min(wl), statistics.quantiles(wl, n=4)[0], statistics.median(wl),
    statistics.quantiles(wl, n=4)[2], max(wl)))
buckets = collections.Counter(
    "1-3" if w <= 3 else "4-7" if w <= 7 else "8-15" if w <= 15 else "16+"
    for w in wl)
print("length buckets", dict(buckets))
EOF
```

If `from data import normalize, CHARS` fails, find the current module
(`grep -rn "def normalize" *.py`) and adjust — the source drifts.

## 3. Judge the sample

Work through the 500 rows in **10 batches of 50**. Keep a running tally; do not
try to hold all 500 in mind at once. For each batch record counts and keep the
sharpest 2-3 examples per failure type.

### 3a. Label correctness

For every row ask two independent questions:

- **Feeling fit** — does `feeling` match the emotion a person would read from
  `text`? Neutral is the correct label for genuinely affect-free text; don't
  force one of the 6 strong feelings onto it. Mark *wrong* only when a
  different label is clearly better, *weak* when defensible but not the best.
- **Emoji fit** — does `emoji` plausibly illustrate `text`? The palette is
  large and many emojis are near-synonyms, so accept any reasonable member of
  the right cluster; mark *wrong* only when the emoji points at a different
  topic or valence than the text.

Tally: `emoji_ok`, `emoji_weak`, `emoji_wrong`; `feeling_ok`, `feeling_weak`,
`feeling_wrong` (out of 500). Note any **systematic** pattern — a specific
emoji or feeling that is mislabeled again and again, or a text template that
always gets the same wrong label.

### 3b. Text quality

Flag each row for at most the worst issue it has:

- **broken** — grammar/spelling errors, truncation, template artifacts
  (`{feeling}`, trailing `-`, JSON crumbs), non-English, empty/near-empty.
- **normalize-fragile** — reads fine now but step 2 flagged it as a casualty
  (leans on digits/punctuation/in-text emoji that `normalize` deletes).
- **low-content** — grammatical but says nothing a label can hang on
  (`ok`, `hm`, `there it is`).
- **clean** — everything else.

Tally the four counts; keep examples for the first three.

## 4. Text-style coverage

Classify the sample (a 100-row skim is enough if 500 is too slow) across these
axes and give an approximate share per bucket:

- **register**: formal · neutral · casual · slang/net-speak
- **form**: 1st-person feeling statement · narrative/recount · dialogue/quote ·
  observation/aphorism · question
- **device**: plain · exclamation · all-caps · in-text emoji · profanity
- **apparent age register**: child · teen · adult · indeterminate

Then name the **gaps**: styles that are barely present or absent (e.g. no
formal writing, no long multi-sentence messages, everything is 1st-person
present tense, one age register dominates). Style monoculture is a finding even
when every individual row is fine.

## 5. Fix the flagged records

Rewrite the rows your step 3 judgment flagged, **in place**. This step changes
`data.jsonl`; everything above it must be finished first.

### 5a. What is in scope

Only sampled rows you flagged as one of:

- **broken** (step 3b) — grammar/spelling errors, truncation, template
  artifacts (`{feeling}`, trailing `-`, JSON crumbs), non-English, empty.
- **emoji or feeling clearly wrong** (step 3a `*_wrong` — **not** `*_weak`).
- **low-content** (step 3b).
- **exact/near-duplicate texts** (step 2's duplicate list).

**Out of scope — never touch these:**

- **normalize-fragile** rows. `normalize` / `CHARS` may change later, so never
  edit a row because of what `normalize` currently deletes.
- `*_weak` labels, `clean` rows, style-coverage gaps, anything you are not
  confident is wrong.

### 5b. Rules

- **Rewrite only. Never delete a row, add a row, or reorder rows.** The line
  count of `data.jsonl` is identical before and after.
- Per flagged row, produce a corrected `{text, emoji, feeling}`:
  - **broken** → fix the text; keep its meaning and both labels.
  - **wrong label** → change only the offending `emoji` / `feeling` to the
    better fit you recorded in step 3a; leave `text` alone.
  - **low-content** → rewrite `text` into a concrete short message that fits
    the existing labels.
  - **duplicate** → rewrite all but one copy into distinct texts; keep labels.
- If you cannot fix a row with confidence, **leave it untouched** and list it
  under "unfixed" in the report.

### 5c. Apply the fixes

Write one correction per line to `report/data/$STAMP.fixes.jsonl`, each
`{"old": <row verbatim from the sample>, "new": <corrected row>}`. The `old`
object must be the exact `{text, emoji, feeling}` from the sample file. For a
duplicate you are splitting, emit one line per copy you are changing (same
`old`, different `new`); they are consumed in file order.

Snapshot first so the invariant checks compare against the real pre-step state:

```bash
cp data.jsonl "report/data/$STAMP.data-before.jsonl"
```

```bash
uv run python - data.jsonl "report/data/$STAMP.fixes.jsonl" <<'EOF'
import collections, json, sys

data_path, fixes_path = sys.argv[1], sys.argv[2]
DUMP = dict(ensure_ascii=False, sort_keys=True, separators=(",", ":"))
key = lambda r: (r["text"], r["emoji"], r["feeling"])

queues = collections.defaultdict(collections.deque)
n_fix = 0
for line in open(fixes_path, encoding="utf-8"):
    line = line.strip()
    if not line:
        continue
    f = json.loads(line)
    queues[key(f["old"])].append(f["new"])
    n_fix += 1

out, applied = [], 0
for raw in open(data_path, encoding="utf-8"):
    r = json.loads(raw)
    q = queues.get(key(r))
    if q:
        out.append(json.dumps(q.popleft(), **DUMP) + "\n")
        applied += 1
    else:
        out.append(raw)  # untouched rows pass through byte-for-byte

leftover = sum(len(q) for q in queues.values())
if leftover:
    sys.exit(f"ERROR: {leftover} corrections matched no row; nothing written")

with open(data_path, "w", encoding="utf-8") as f:
    f.writelines(out)
print(f"corrections: {n_fix}  applied: {applied}")
EOF
```

Then verify the edit is exactly what you intended:

```bash
BEFORE="report/data/$STAMP.data-before.jsonl"
test "$(wc -l < "$BEFORE")" = "$(wc -l < data.jsonl)" && echo "row count unchanged"
diff "$BEFORE" data.jsonl | grep -c '^> '   # changed lines; must equal corrections applied
diff "$BEFORE" data.jsonl | grep '^[<>]' | head -40   # eyeball every change
```

Because step 5 never reorders rows, `diff` shows only in-place `Nc N` hunks.

Changed-line count must equal the script's `applied:` count and the row count
must be identical. If either is off, `cp "$BEFORE" data.jsonl` and redo the
fixes file. Do not commit or retrain here — leave the corrected corpus for the
next `uv run main.py`.

## 6. Write the report

Write the report to **both** `report/data/$STAMP.md` (timestamped archive) and
`data.md` at the repo root (stable copy of the latest run, identical content —
overwrite it every time). Use this skeleton:

```markdown
# Data quality report — <YYYY-MM-DD HH:MM>

- Sample: 500 of <N> rows (`report/data/<STAMP>.sample.jsonl`)
- Label correctness: emoji <emoji_ok>/500 ok · <emoji_weak> weak · <emoji_wrong> wrong; feeling <feeling_ok>/500 ok · <feeling_weak> weak · <feeling_wrong> wrong
- Text quality: <clean>/500 clean · <broken> broken · <normalize_fragile> normalize-fragile · <low_content> low-content
- Label coverage: feelings <k>/8 present · emojis <m>/<total> present · imbalance <r>x
- Style coverage: <one-line gist + biggest gap>
- Fixes applied: <f> rows rewritten (<b> broken · <l> labels · <c> low-content · <d> dedup) · <u> left unfixed

## 1. Label correctness
<rates; systematic patterns first, then a table of the worst individual rows>

| text | labeled emoji / feeling | better fit | note |
| --- | --- | --- | --- |

## 2. Text quality
- broken: <n> — <examples>
- normalize-fragile: <n> — <examples with `before -> after`>
- low-content: <n> — <examples>
- exact/near duplicates: <n>

## 3. Label coverage
### Feelings
| feeling | corpus count | corpus share | sample count |
### Emojis
- present <m>/<total>; absent: <list or "none">
- top 10: <...>
- bottom 10: <...>
- imbalance max/min = <hi>/<lo> = <r>x

## 4. Text-style coverage
| axis | buckets (approx share) |
| --- | --- |
| register | ... |
| form | ... |
| device | ... |
| age register | ... |

Gaps: <bullets>

## 5. Fixes applied
- rewritten: <f> rows (<b> broken · <l> labels · <c> low-content · <d> dedup); fixes file `report/data/<STAMP>.fixes.jsonl`
- unfixed (flagged but not confidently fixable): <u> — <examples>

| before (text — emoji / feeling) | after | why |
| --- | --- | --- |

## 6. Verdict & recommendations
<3-6 prioritised bullets: what to fix in the generator / labels / normalize, in impact order>
```

Every number must trace to step 2's output, your step 3/4 tally, or step 5's
`applied:` count — no invented figures. Nothing to lint (Markdown). Confirm both
files were written (`report/data/$STAMP.md` and `data.md`).

## 7. Report back

In 2-3 sentences: the sample size and corpus size, the headline correctness and
coverage numbers, how many rows step 5 rewrote, and the single biggest
data-quality problem you found.
