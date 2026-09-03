---
description: Read-only data-quality report for data.jsonl (the append-only master) — refreshes tools/data/stat.ts distributions, runs structural checks (schema, read() survival, raw length, style set, emoji vocab, duplicate/union-merge keys, palette), then judges label + palette quality on a 200-row sample in-session. Writes one report to report/data-quality/, never modifies any data file.
allowed-tools: [Bash, Read, Write]
---

# Data quality

Judge the health of `data.jsonl` (the append-only master that `bun run regen`
derives `train.jsonl` / `eval.jsonl` / `labels.json` from) and write **one**
report to `report/data-quality/<MM-DD-HH:MM>.md`. Distribution stats come from
`tools/data/stat.ts`; this skill adds structural checks and an in-session label
judgment on top.

**This skill is strictly read-only.** It never writes, deletes, reorders, or
relabels a row in `train.jsonl`, `eval.jsonl`, `data.jsonl`, or `labels.json`,
and it never writes a root `data.md`. The only file it creates is the report
(plus one throwaway sample file next to it). Label judgment is
**keep-or-drop reasoning only** — you decide whether a stored label is right,
you never rewrite one.

`data.jsonl` is the only committed data file. If `labels.json` (or `train.jsonl`
/ `eval.jsonl`) is missing or stale, run `bun run regen` first.

Reference points, all read from source so the report stays honest:

- `data.py:read` accepts a row only with `text` + `emojis` + `styles` + `bg` +
  `fg`. `emojis` is a **single space-separated string**; `styles` is a list.
  A row is silently dropped when its `normalize`d text is empty or longer than
  `MAX_TEXT_LEN` (`config.py`, currently 42), or when no style survives the
  `STYLES` membership filter. Emojis are filtered to the `labels.json` vocab
  (a row may legitimately keep zero).
- `labels.json` → `styles` is the **fixed 21-entry closed set**, `emojis` is the
  current top-`TOP_EMOJIS` frequency leaderboard, rebuilt from `data.jsonl` by
  `bun run regen` (open-set in the corpus).
- Corpus text-length spec is **4–48 code points** on the raw (un-normalized)
  `text` — wider than the model's `MAX_TEXT_LEN`, so both are checked.
- `bg` = two `#rrggbb` stops, `fg` = one `#rrggbb`.

## 1. Refresh distributions

```bash
mkdir -p report/data-quality
STAMP=$(date +%m-%d-%H:%M)
bun run tools/data/stat.ts
STAT=$(ls -t report/data-stat/*.md | head -1)
echo "distribution report: $STAT"
```

Read `$STAT`. Its `data.jsonl` numbers (row counts, style distribution,
text-length histogram + out-of-range count, emoji distribution, top-label
coverage) are lifted into section 1 of the report **by reference** — do not
recompute them.

## 2. Structural checks

One read-only pass over the full file. Counts only; a bounded number of
example texts per issue is fine, whole-file dumps are not.

```bash
uv run python - <<'EOF'
import collections, json, re

from config import MAX_TEXT_LEN
from data import normalize

LABELS = json.load(open("labels.json", encoding="utf-8"))
STYLE_SET, EMOJI_SET = set(LABELS["styles"]), set(LABELS["emojis"])
PATH = "data.jsonl"
MIN_RAW, MAX_RAW = 4, 48
REQUIRED = ("text", "emojis", "styles", "bg", "fg")
HEX = re.compile(r"^#[0-9a-fA-F]{6}$")


def load(path):
    rows, bad = [], 0
    for line in open(path, encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            bad += 1
    return rows, bad


def palette_ok(r):
    bg, fg = r.get("bg"), r.get("fg")
    return (
        isinstance(bg, list) and len(bg) == 2
        and all(isinstance(x, str) and HEX.match(x) for x in bg)
        and isinstance(fg, str) and bool(HEX.match(fg))
    )


rows, bad = load(PATH)
n = len(rows)
print(f"==== {PATH} ====")
print(f"rows {n}   json parse failures {bad}")

missing = collections.Counter()
for r in rows:
    for k in REQUIRED:
        if k not in r:
            missing[k] += 1
print("missing required key", dict(missing) or "none")

schema_ok = [r for r in rows if all(k in r for k in REQUIRED)]
nt = {id(r): normalize(r["text"]) for r in schema_ok}
in_set = {id(r): [s for s in r["styles"] if s in STYLE_SET] for r in schema_ok}
norm_empty = sum(1 for r in schema_ok if not nt[id(r)])
too_long = sum(1 for r in schema_ok if nt[id(r)] and len(nt[id(r)]) > MAX_TEXT_LEN)
no_style = sum(1 for r in schema_ok if not in_set[id(r)])
survive = sum(
    1 for r in schema_ok
    if nt[id(r)] and len(nt[id(r)]) <= MAX_TEXT_LEN and in_set[id(r)]
)
print(f"normalize->empty {norm_empty}   norm len>{MAX_TEXT_LEN} {too_long}   "
      f"no in-set style {no_style}")
print(f"ROWS THAT SURVIVE data.py:read {survive}  ({n - survive} lost)")

raw_lens = sorted(len([*r["text"]]) for r in schema_ok)
oor = [r["text"] for r in schema_ok
       if not (MIN_RAW <= len([*r["text"]]) <= MAX_RAW)]
if raw_lens:
    print(f"raw len  min {raw_lens[0]} median {raw_lens[len(raw_lens)//2]} "
          f"max {raw_lens[-1]}")
print(f"raw len outside {MIN_RAW}-{MAX_RAW}: {len(oor)}"
      + (f"  e.g. {oor[:5]}" if oor else ""))

off_style = collections.Counter(
    s for r in schema_ok for s in r["styles"] if s not in STYLE_SET
)
print("styles off the closed set", dict(off_style.most_common(20)) or "none")
spr = collections.Counter(min(len(in_set[id(r)]), 3) for r in schema_ok)
print(f"in-set styles/row 0/1/2/3+ {spr[0]}/{spr[1]}/{spr[2]}/{spr[3]}")

toks = [(r, r["emojis"].split()) for r in schema_ok if isinstance(r["emojis"], str)]
distinct = {e for _, es in toks for e in es}
mentions = [e for _, es in toks for e in es]
oov = [e for e in mentions if e not in EMOJI_SET]
all_in = sum(1 for _, es in toks if es and all(e in EMOJI_SET for e in es))
zero_after = sum(1 for _, es in toks if es and not [e for e in es if e in EMOJI_SET])
print(f"distinct emoji tokens {len(distinct)} ({len(distinct - EMOJI_SET)} outside "
      f"labels.json)   mentions {len(mentions)} / oov {len(oov)}")
print("  top oov", collections.Counter(oov).most_common(8))
print(f"rows all emojis in vocab {all_in}/{len(toks)}   "
      f"rows -> 0 emoji after filter {zero_after}")

raw_dupes = collections.Counter(r["text"] for r in schema_ok)
raw_extra = sum(v - 1 for v in raw_dupes.values() if v > 1)
norm_map = collections.Counter(nt[id(r)] for r in schema_ok)
norm_extra = sum(v - 1 for v in norm_map.values() if v > 1)
keys_multi = sum(1 for v in norm_map.values() if v > 1)
print(f"exact raw-text dupes {raw_extra} extra rows   "
      f"normalize-collapsed dupes {norm_extra} extra rows")
print(f"normalized keys with 2+ rows (union-merged by regen) {keys_multi} "
      f"/ {len(norm_map)} keys")
print("  e.g.", [t for t, v in raw_dupes.most_common(5) if v > 1])

bad_pal = [r["text"] for r in schema_ok if not palette_ok(r)]
print(f"malformed bg/fg {len(bad_pal)}"
      + (f"  e.g. {bad_pal[:5]}" if bad_pal else ""))
EOF
```

If `from data import ...` or `from config import ...` fails, the modules have
drifted — `grep -n "def normalize\|MAX_TEXT_LEN" data.py config.py` and adjust.

## 3. Draw the judging sample

Never read `data.jsonl` whole. Sample to a file next to the report:

```bash
shuf -n 200 data.jsonl > "report/data-quality/$STAMP.sample.jsonl"
wc -l "report/data-quality/$STAMP.sample.jsonl"
```

(If the file has fewer rows than asked, `shuf` returns all of them — fine.)

## 4. Judge label + palette quality — in this session

Read the sample file. Work through it in **chunks of ~50**, keeping a
running tally; do not try to hold a whole sample in mind at once. For
each row, with `styles` / `emojis` / `bg` / `fg` visible, decide — **judgment
only, never edit**:

- **styles** — is every stored style a label a reader would clearly assign to
  `text`? The set is closed and small, so a wrong pick is usually
  unambiguous. Per row: `ok` (all right), `weak` (defensible but not the label
  you'd choose), `wrong` (at least one clearly off). Also note **missing** —
  an obvious style the row should carry and doesn't.
- **emojis** — is each emoji a defensible illustration of `text` (right topic
  and valence; near-synonyms in the right cluster are fine)? Per row:
  `ok` / `has-weak` / `has-wrong`. Also flag **over-labeled** (filler emoji
  that adds nothing) and **under-labeled** (an obvious emoji absent).
- **palette** — `bg` reads as one gradient (not a clash), `fg` stays readable
  over both `bg` stops, and the palette plausibly matches the mood/imagery of
  `text`. Per row: `ok` / `weak` (muddy or only loosely on-mood) /
  `wrong` (clash, unreadable `fg`, or off-mood).

For each failure type keep the **sharpest 2–3 examples** (verbatim `text` +
the stored labels + what you'd expect). Watch for **systematic** patterns: a
specific style or emoji misapplied again and again, a phrasing template that
always draws the same wrong label, one style dominating far past its share,
palettes collapsing to a single look. A monoculture is a finding even when
each individual row is fine.

## 5. Write the report

Write to `report/data-quality/$STAMP.md` **only**. No root `data.md`, no other
file. Skeleton:

```markdown
# Data quality report — <YYYY-MM-DD HH:MM>

- File: `data.jsonl` <N> rows
- Distributions: `<path to the report/data-stat/*.md refreshed this run>`
- Trainable after `data.py:read`: <a> (<N-a> lost)
- Structural: <single worst finding in one line>
- Label quality — sample <n>: styles <ok>/<weak>/<wrong> · emoji rows <ok>/<has-weak>/<has-wrong> · palette <ok>/<weak>/<wrong>
- Biggest problem: <one sentence>

## 1. Distributions
<key numbers pulled from the stat.ts report, with its path; no recomputation>

## 2. Structural checks
### Schema & `read()` survival
| check | count |
| --- | ---: |
| rows | |
| json parse failures | |
| missing required key | |
| normalize → empty | |
| normalized len > MAX_TEXT_LEN (<MAX_TEXT_LEN>) | |
| no in-set style after filter | |
| normalized keys with 2+ rows | |
| **rows that survive data.py:read** | |

### Raw text length (code points)
- outside 4–48: <n> (<pct>) — examples: …

### Style label set
- off the 21-style closed set: <list with counts, or none>
- in-set styles per row (0/1/2/3+): …

### Emoji vocab
- distinct tokens: <n> (<k> outside labels.json)
- mentions / out-of-vocab: … — top OOV: …
- rows with all emojis in vocab: <pct>
- rows left with 0 emojis after the vocab filter: <n>

### Duplicates
- exact raw-text: <extra> rows — examples: …
- normalize-collapsed: <extra>

### Palette
- malformed `bg`/`fg`: <n> — examples: …

## 3. Label quality (sample <n> of <N>)
### Styles
<rates; systematic patterns first, then worst individual rows>

| text | stored styles | better fit | note |
| --- | --- | --- | --- |

### Emojis
<rates; over/under-labeled; systematic patterns; then worst rows>

| text | stored emojis | issue |
| --- | --- | --- |

### Palette
<rates; then worst rows>

| text | bg / fg | issue |
| --- | --- | --- |

## 4. Systematic patterns
<bullets across the sample: labels/emojis repeatedly misapplied, templates
that always draw the same wrong label, style or palette monoculture>

## 5. Verdict & recommendations
<3–6 prioritised bullets, impact order — generator prompt (`tools/data/train.ts`),
annotator prompt (`tools/data/annotate.ts`), `labels.json` / `tools/data/styles.ts`,
`normalize`, dedup>
```

Every number must trace to the section 1 stat report, the section 2 script
output, or your section 3/4 tally — no invented figures. It is Markdown, so
there is nothing to lint. Confirm the report file exists and that
`data.jsonl` and `labels.json` are byte-identical to before this run
(`git status --short` shows only new files under `report/`; `train.jsonl` /
`eval.jsonl` are gitignored).

## 6. Report back

In 2–3 sentences: the master row count and how many survive `data.py:read`, the
headline style / emoji / palette quality rates for the sample, the single
biggest data-quality problem, and the path to the written report.
