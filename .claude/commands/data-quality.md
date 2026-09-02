---
description: Read-only data-quality report for train.jsonl + eval.jsonl — refreshes tools/data/stat.ts distributions, runs structural checks (schema, read() survival, vocab drift, dupes, train/eval leakage, palette, provenance), then judges label + palette quality on a 200/100-row sample in-session. Writes one report to report/data-quality/, never modifies any data file.
allowed-tools: [Bash, Read, Write]
---

# Data quality

Judge the health of `train.jsonl` and `eval.jsonl` side by side and write **one**
report to `report/data-quality/<MM-DD-HH:MM>.md`. Distribution stats come from
`tools/data/stat.ts`; this skill adds structural checks and an in-session label
judgment on top.

**This skill is strictly read-only.** It never writes, deletes, reorders, or
relabels a row in `train.jsonl`, `eval.jsonl`, `data.jsonl`, or `labels.json`,
and it never writes a root `data.md`. The only file it creates is the report
(plus two throwaway sample files next to it). Label judgment is
**keep-or-drop reasoning only** — you decide whether a stored label is right,
you never rewrite one.

Reference points, all read from source so the report stays honest:

- `data.py:read` accepts a row only with `text` + `emojis` + `styles` + `bg` +
  `fg`. `emojis` is a **single space-separated string**; `styles` is a list.
  A row is silently dropped when its `normalize`d text is empty or longer than
  `MAX_TEXT_LEN` (`config.py`, currently 42), or when no style survives the
  `STYLES` membership filter. Emojis are filtered to the `labels.json` vocab
  (a row may legitimately keep zero).
- `labels.json` → `styles` is the **fixed 21-entry closed set**, `emojis` is the
  current top-320 frequency leaderboard (open-set in the corpus).
- Corpus text-length spec is **4–48 code points** on the raw (un-normalized)
  `text` — wider than the model's `MAX_TEXT_LEN`, so both are checked.
- `bg` = two `#rrggbb` stops, `fg` = one `#rrggbb`.
- Rows written since the coverage update carry `meta` (`src`, `v`, `at`,
  `model`, `params`, plus `topic` on train rows); older rows have none.

## 1. Refresh distributions

```bash
mkdir -p report/data-quality
STAMP=$(date +%m-%d-%H:%M)
bun run tools/data/stat.ts
STAT=$(ls -t report/data-stat/*.md | head -1)
echo "distribution report: $STAT"
```

Read `$STAT`. Its numbers (row counts, style distribution, text-length
histogram + out-of-range count, emoji distribution, top-label coverage) are
lifted into section 1 of the report **by reference** — do not recompute them.

## 2. Structural checks

One read-only pass over both full files. Counts only; a bounded number of
example texts per issue is fine, whole-file dumps are not.

```bash
uv run python - <<'EOF'
import collections, json, re

from config import MAX_TEXT_LEN
from data import normalize

LABELS = json.load(open("labels.json", encoding="utf-8"))
STYLE_SET, EMOJI_SET = set(LABELS["styles"]), set(LABELS["emojis"])
FILES = ["train.jsonl", "eval.jsonl"]
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


norm_by_file = {}
for path in FILES:
    rows, bad = load(path)
    n = len(rows)
    print(f"\n==== {path} ====")
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
    trainable = sum(
        1 for r in schema_ok
        if nt[id(r)] and len(nt[id(r)]) <= MAX_TEXT_LEN and in_set[id(r)]
    )
    print(f"normalize->empty {norm_empty}   norm len>{MAX_TEXT_LEN} {too_long}   "
          f"no in-set style {no_style}")
    print(f"ROWS USED FOR TRAINING {trainable}  ({n - trainable} lost)")

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
    print(f"exact raw-text dupes {raw_extra} extra rows   "
          f"normalize-collapsed dupes {norm_extra} extra rows")
    print("  e.g.", [t for t, v in raw_dupes.most_common(5) if v > 1])

    bad_pal = [r["text"] for r in schema_ok if not palette_ok(r)]
    print(f"malformed bg/fg {len(bad_pal)}"
          + (f"  e.g. {bad_pal[:5]}" if bad_pal else ""))

    with_meta = sum(1 for r in rows if isinstance(r.get("meta"), dict))
    src = collections.Counter(
        r["meta"].get("src") for r in rows if isinstance(r.get("meta"), dict)
    )
    ver = collections.Counter(
        r["meta"].get("v") for r in rows if isinstance(r.get("meta"), dict)
    )
    print(f"rows with meta {with_meta}/{n}   src {dict(src) or 'none'}   "
          f"v {dict(ver) or 'none'}")
    if path == "train.jsonl":
        topic = collections.Counter(
            r["meta"].get("topic") for r in rows
            if isinstance(r.get("meta"), dict) and r["meta"].get("topic")
        )
        print(f"distinct meta.topic {len(topic)}  "
              f"min {min(topic.values(), default=0)} max {max(topic.values(), default=0)}")

    norm_by_file[path] = set(norm_map)

leak = norm_by_file["train.jsonl"] & norm_by_file["eval.jsonl"]
print("\n==== leakage ====")
print(f"normalized texts in BOTH files {len(leak)}"
      + (f"  e.g. {list(leak)[:5]}" if leak else ""))
EOF
```

If `from data import ...` or `from config import ...` fails, the modules have
drifted — `grep -n "def normalize\|MAX_TEXT_LEN" data.py config.py` and adjust.

## 3. Draw the judging samples

Never read either corpus whole. Sample to files next to the report:

```bash
shuf -n 200 train.jsonl > "report/data-quality/$STAMP.train-sample.jsonl"
shuf -n 100 eval.jsonl  > "report/data-quality/$STAMP.eval-sample.jsonl"
wc -l "report/data-quality/$STAMP".*-sample.jsonl
```

(If a file has fewer rows than asked, `shuf` returns all of them — fine.)

## 4. Judge label + palette quality — in this session

Read both sample files. Work through them in **chunks of ~50**, keeping a
running tally per file; do not try to hold a whole sample in mind at once. For
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

- Files: `train.jsonl` <N> rows · `eval.jsonl` <M> rows
- Distributions: `<path to the report/data-stat/*.md refreshed this run>`
- Trainable after `data.py:read`: train <a> (<N-a> lost) · eval <b> (<M-b> lost)
- Structural: <single worst finding in one line>
- Label quality — train sample <n>: styles <ok>/<weak>/<wrong> · emoji rows <ok>/<has-weak>/<has-wrong> · palette <ok>/<weak>/<wrong>
- Label quality — eval sample <n>: styles … · emoji … · palette …
- Biggest problem: <one sentence>

## 1. Distributions
<key numbers pulled from the stat.ts report, with its path; no recomputation>

## 2. Structural checks
### Schema & `read()` survival
| check | train | eval |
| --- | ---: | ---: |
| rows | | |
| json parse failures | | |
| missing required key | | |
| normalize → empty | | |
| normalized len > MAX_TEXT_LEN (<MAX_TEXT_LEN>) | | |
| no in-set style after filter | | |
| **rows used for training** | | |

### Raw text length (code points)
- outside 4–48: train <n> (<pct>) · eval <n> (<pct>) — examples: …

### Style label set
- off the 21-style closed set: <list with counts, or none>
- in-set styles per row (0/1/2/3+): train … · eval …

### Emoji vocab
- distinct tokens: train <n> (<k> outside labels.json) · eval <n> (<k>)
- mentions / out-of-vocab: train … · eval … — top OOV: …
- rows with all emojis in vocab: train <pct> · eval <pct>
- rows left with 0 emojis after the vocab filter: train <n> · eval <n>

### Duplicates
- exact raw-text: train <extra> rows · eval <extra> — examples: …
- normalize-collapsed: train <extra> · eval <extra>

### train / eval leakage
- normalized texts in both files: <n> — examples: …

### Palette
- malformed `bg`/`fg`: train <n> · eval <n> — examples: …

### Provenance (`meta`)
- rows with meta: train <pct> · eval <pct>
- by `src`: … · by `v`: …
- train `topic`: <distinct> topics, min/max rows per topic

## 3. Label quality — train (sample <n> of <N>)
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

## 4. Label quality — eval (sample <n> of <M>)
<same three subsections>

## 5. Systematic patterns
<bullets across both samples: labels/emojis repeatedly misapplied, templates
that always draw the same wrong label, style or palette monoculture>

## 6. Verdict & recommendations
<3–6 prioritised bullets, impact order — generator prompt (`tools/data/train.ts`),
annotator prompt (`tools/data/annotate.ts`), `labels.json` / `tools/data/styles.ts`,
`normalize`, dedup>
```

Every number must trace to the section 1 stat report, the section 2 script
output, or your section 3/4 tally — no invented figures. It is Markdown, so
there is nothing to lint. Confirm the report file exists and that
`train.jsonl`, `eval.jsonl`, `data.jsonl`, and `labels.json` are byte-identical
to before this run (`git status --short` should show only the new files under
`report/`).

## 6. Report back

In 2–3 sentences: the two row counts and how many survive `data.py:read`, the
headline style / emoji / palette quality rates for each sample, the single
biggest data-quality problem, and the path to the written report.
