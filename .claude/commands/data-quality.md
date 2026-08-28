---
description: Sample 500 rows from data.jsonl and write a data-quality report (label correctness, text quality, coverage)
allowed-tools: [Bash, Read, Write]
---

# Data quality

Judge the health of `data.jsonl` from a **500-row random sample**. Produce one
report covering label correctness, text quality, label coverage, and text-style
coverage. Do not train, do not run `train.py`/`test_model.py`, do not touch
`data.jsonl`.

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

## 5. Write the report

Write `report/data/$STAMP.md` with this skeleton:

```markdown
# Data quality report — <YYYY-MM-DD HH:MM>

- Sample: 500 of <N> rows (`report/data/<STAMP>.sample.jsonl`)
- Label correctness: emoji <emoji_ok>/500 ok · <emoji_weak> weak · <emoji_wrong> wrong; feeling <feeling_ok>/500 ok · <feeling_weak> weak · <feeling_wrong> wrong
- Text quality: <clean>/500 clean · <broken> broken · <normalize_fragile> normalize-fragile · <low_content> low-content
- Label coverage: feelings <k>/8 present · emojis <m>/<total> present · imbalance <r>x
- Style coverage: <one-line gist + biggest gap>

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

## 5. Verdict & recommendations
<3-6 prioritised bullets: what to fix in the generator / labels / normalize, in impact order>
```

Every number must trace to step 2's output or your step 3/4 tally — no invented
figures. Nothing to lint (Markdown).

## 6. Report back

In 2-3 sentences: the sample size and corpus size, the headline correctness and
coverage numbers, and the single biggest data-quality problem you found.
