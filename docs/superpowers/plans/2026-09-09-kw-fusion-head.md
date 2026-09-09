# KW-Fusion Rework Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace `FusionHead`'s 18+4 hand-crafted feature residual with a
three-head design (`EmojiHead` + `KWHead` blended by a learned scalar gate,
scored against a shared emoji-embedding table), fed by a length-200 soft-TF
keyword vector instead of the per-emoji flex plumbing.

**Architecture:** A new `EmojiEmbedding` module owns the `[V, 64]` label table +
bias, shared by `EmojiHead` (text -> 64-d query) and `KWHead` (TF vector -> 64-d
query). `FusionHead` is now just `Linear(TEXT_EMBED_SIZE + N, 1) -> sigmoid`
producing a per-row gate `a`; the fused logits are
`(a * q_txt.detach() + (1-a) * q_kw.detach()) @ E.detach().t() + bias.detach()`,
so `loss/fusion` trains only the gate. The lexical signal is a length-N
(`N = 200`) raw soft-TF vector over a fixed global keyword vocabulary, produced
by the three parity-locked ranker surfaces (`tools/data/flexrank.ts`,
`model/flexrank.py`, `web/src/flexrank.js`).

**Tech Stack:** Python 3.13 + PyTorch/Lightning (`uv`), Bun/TypeScript data
toolchain, Vite/React web app with `onnxruntime-web`.

**Spec:** `docs/superpowers/specs/2026-09-09-kw-fusion-head-design.md`

## Global Constraints

- `V = len(EMOJIS)` from `data/labels.json` (957 today, dynamic). `N = 200`
  (`KW_VOCAB_SIZE`). `EMOJI_EMBED_SIZE = 64`. `TEXT_EMBED_SIZE = 620`
  (`sum(ENCODER_CHANNELS) = 120+200+300`).
- No comments or docstrings in source (keep `type: ignore` / `noqa` / shebangs).
  See `no-comments-or-docstrings` memory.
- Never verify by running training. Verify with `ruff`, the plain-assert test
  scripts, `bun test`, and `npm test`. See `dont-run-main-to-test` memory. The
  full `train --local` in the final task is behavioral sign-off, run once.
- `uv` only — no `pip install`. Bun for the TS toolchain.
- Commit only files you changed for the task; never `git add -A` (a background
  job auto-commits to this branch). See `concurrent-training-pipeline-commits`.
- Keyword-vocab selection and `tf_vec` must be byte-identical across
  `tools/data/flexrank.ts` / `model/flexrank.py` / `web/src/flexrank.js`,
  locked by `web/src/flexrank.fixture.json` (dense length-N vectors, values
  rounded to 3 decimals).
- `MIN_FUZZY_SCORE = 0.66`, `FUZZY_MIN_LEN = 4`, `KW_MIN_LEN = 4`,
  `KW_MAX_LEN = 12`.
- `.pt` files are gitignored — no checkpoint migration; retrain from scratch.
- Between Task 2 and Task 11 the repo is in a knowingly-broken intermediate
  state (`model/flexrank.py` consumers, `pred.py`, `report.py`, the web app
  reference the old `flex.json` schema). Run only the current task's tests, not
  the whole suite, until Task 12 completes.

---

## File Structure

**New files:**
- `tools/data/flexrank.test.ts` — `bun test` unit coverage for `selectKwVocab`
  + `tfVec`.
- `model/test_data_flex.py` — plain-assert coverage for `_row_tf` / `FLEX_N` /
  dataset tensor shape.
- `model/test_model_heads.py` — plain-assert coverage for `EmojiEmbedding`,
  `EmojiHead`, `KWHead`, `FusionHead` (shapes, init equivalence, gate bias).

**Modified files:**
- `tools/data/config.ts` — `KW_VOCAB_SIZE`, `KW_MIN_LEN`, `KW_MAX_LEN`,
  `MIN_FUZZY_SCORE`.
- `tools/data/flexrank.ts` — rewrite: `selectKwVocab`, `tfVec`, `buildJson`
  emits `{ kw_vocab }`. Drop `rank`, `flexq`, `FLEX_COLS`, `FlexHit`, `FlexQ`.
- `tools/data/regen.ts` — `flex_tf` row field, `--kw-n` / `--no-kw` flags, new
  `flex.json` + fixture writes, `Row` type / `BASE_FIELDS` / `toLine`.
- `web/public/flex.json` — regenerated: `{ "kw_vocab": string[200] }`.
- `web/src/flexrank.fixture.json` — regenerated: `{ kw_vocab, cases:[{text, tf}] }`.
- `model/flexrank.py` — `class FlexRanker` keeps only `tf_vec(text) -> list[float]`.
- `model/test_flexrank.py` — replay `tf` fixture vectors.
- `web/src/flexrank.js` — `makeFlexRanker(flexJson)` returns `{ tfVec }`.
- `web/src/flexrank.test.js` — replay `tf` fixture vectors.
- `model/data.py` — `KW_VOCAB`, `FLEX_N`, `record.flex_tf`, `_row_tf`,
  `EmojiDataset.flex_tf`, `__getitem__`. Delete `_row_flex`, `scatter_flex`,
  `FLEX_MAX_K`, `FLEX_RAW_DIM`, `FLEXQ_DIM`, `_FLEXQ_KEYS`.
- `model/config.py` — add `DROPOUT_KW`; remove `FUSION_HIDDEN`,
  `DROPOUT_FUSION`, `FUSION_INT_CLAMP`, `FUSION_INT_EMBED_SIZE`, `fus_str`, the
  `FUSION:` `CONFIG_PARTS` entry.
- `files.py` — `EMOJI_EMBED_PT`, `KW_PT`.
- `model/model.py` — `EmojiEmbedding`, rewrite `EmojiHead` (returns q_txt),
  `KWHead`, rewrite `FusionHead` (gate). Delete `fusion_features`, `_FUSION_EPS`,
  the int-embedding tables.
- `model/train.py` — `LitEncoder` (`emoji_embed` attr, batch tuple, 3 losses,
  MRR logs, checkpoint key), `_train_encoder` save block, `_run_local` gan
  `_require_pt`, `train_remote` / `_run_remote` byte params.
- `model/test_train_cli.py` — required-pt assertion updates if any.
- `model/export_onnx.py` — `ExportWrapper` (2 inputs, 4 outputs), `_load`
  `EmojiEmbedding` + `KWHead`, `meta.json` keys, abort check target.
- `model/pred.py` — load `emoji_embed.pt` / `kw.pt`, build `flex_tf`, gate.
- `tools/report.py` — thread `emb` through sections, EmojiHead/KWHead/Fusion
  3-way, delete FlexRank curve + `_cldr_flex_smoke` + `FLEX_SMOKE_MIN`.
- `tools/test_report.py` — rename FlexRank->KWHead assertions, drop smoke test.
- `web/src/model.js` — build `flex_tf`, 2-input ONNX call, `keywords` mode ->
  `kw_logits`.
- `web/src/model.test.js` — input/output shape updates.
- `CLAUDE.md`, `docs/model.md` — prose.

---

## Task 1: TS ranker — keyword vocab + `tfVec`

**Files:**
- Modify: `tools/data/config.ts`
- Modify: `tools/data/flexrank.ts` (full rewrite of the exported surface)
- Create: `tools/data/flexrank.test.ts`

**Interfaces:**
- Consumes: `loadCldrAnnotations()` from `tools/data/cldr.ts`, `makeIdf`,
  `queryTokens`, `stripVS`, `FUZZY_MIN_LEN` from
  `tools/analysis/cldr-baseline.ts`.
- Produces:
  - `tools/data/config.ts`: `export const KW_VOCAB_SIZE = 200`,
    `KW_MIN_LEN = 4`, `KW_MAX_LEN = 12`, `MIN_FUZZY_SCORE = 0.66`.
  - `tools/data/flexrank.ts`:
    - `export type FlexJson = { kw_vocab: string[] }`
    - `export async function buildFlexRanker(): Promise<{ kwVocab: string[];
      tfVec: (text: string) => number[]; buildJson: () => FlexJson }>`
    - `tfVec(text)` returns a dense `number[]` of length `kwVocab.length`,
      each cell `r3`-rounded.

- [ ] **Step 1: Add constants to `tools/data/config.ts`**

Append near the other exported constants:

```ts
export const KW_VOCAB_SIZE = 200
export const KW_MIN_LEN = 4
export const KW_MAX_LEN = 12
export const MIN_FUZZY_SCORE = 0.66
```

- [ ] **Step 2: Write the failing test `tools/data/flexrank.test.ts`**

```ts
import { describe, expect, it } from "bun:test"
import { KW_VOCAB_SIZE } from "./config"
import { buildFlexRanker } from "./flexrank.ts"

describe("kw ranker", () => {
  it("selects a fixed-size single-word 4..12-char vocab", async () => {
    const r = await buildFlexRanker()
    expect(r.kwVocab.length).toBe(KW_VOCAB_SIZE)
    for (const k of r.kwVocab) {
      expect(k).toMatch(/^[a-z0-9]+$/)
      expect(k.length).toBeGreaterThanOrEqual(4)
      expect(k.length).toBeLessThanOrEqual(12)
    }
    const sorted = [...r.kwVocab].sort()
    expect(new Set(r.kwVocab).size).toBe(r.kwVocab.length)
    expect(sorted).not.toEqual(r.kwVocab) // idf-ordered, not alphabetical
  })

  it("tfVec: exact hit = 1.0, fuzzy hit graded, miss = 0", async () => {
    const r = await buildFlexRanker()
    const kw = r.kwVocab[0]
    const v = r.tfVec(kw)
    expect(v.length).toBe(r.kwVocab.length)
    expect(v[0]).toBe(1.0)
    expect(r.tfVec("zzzznotawordzzzz").every((x) => x === 0)).toBe(true)
  })

  it("buildJson emits only kw_vocab", async () => {
    const r = await buildFlexRanker()
    expect(Object.keys(r.buildJson())).toEqual(["kw_vocab"])
    expect(r.buildJson().kw_vocab).toEqual(r.kwVocab)
  })
})
```

- [ ] **Step 3: Run the test to verify it fails**

Run: `bun test tools/data/flexrank.test.ts`
Expected: FAIL — `buildFlexRanker` still has the old signature / returns `rank`.

- [ ] **Step 4: Rewrite `tools/data/flexrank.ts`**

Replace the entire file body (keep the imports it still needs) with:

```ts
import {
  FUZZY_MIN_LEN,
  makeIdf,
  queryTokens,
  stripVS,
} from "../analysis/cldr-baseline.ts"
import { loadCldrAnnotations } from "./cldr.ts"
import { KW_MAX_LEN, KW_MIN_LEN, KW_VOCAB_SIZE, MIN_FUZZY_SCORE } from "./config"

export type FlexJson = { kw_vocab: string[] }

const r3 = (x: number) => Number(x.toFixed(3))

function overlap(w: string, k: string): number {
  if (w === k) return 1.0
  if (w.length < FUZZY_MIN_LEN || k.length < FUZZY_MIN_LEN) return 0.0
  if (!(w.startsWith(k) || k.startsWith(w))) return 0.0
  const r = Math.min(w.length, k.length) / Math.max(w.length, k.length)
  return r >= MIN_FUZZY_SCORE ? r : 0.0
}

export async function buildFlexRanker(): Promise<{
  kwVocab: string[]
  tfVec: (text: string) => number[]
  buildJson: () => FlexJson
}> {
  const annotations = await loadCldrAnnotations()
  const kwTokens: string[][] = []
  const candidates = new Set<string>()
  for (const [, keywords] of annotations) {
    const toks = keywords.map((w) => w.toLowerCase())
    kwTokens.push(toks)
    for (const kw of toks) {
      if (/\s/.test(kw)) continue
      if (kw.length < KW_MIN_LEN || kw.length > KW_MAX_LEN) continue
      candidates.add(kw)
    }
  }
  const idf = makeIdf(kwTokens)
  const kwVocab = [...candidates]
    .sort((a, b) => idf(b) - idf(a) || (a < b ? -1 : a > b ? 1 : 0))
    .slice(0, KW_VOCAB_SIZE)

  const tfVec = (text: string): number[] => {
    const q = queryTokens(text)
    return kwVocab.map((k) => {
      let s = 0
      for (const w of q) s += overlap(w, k)
      return r3(s)
    })
  }

  return { kwVocab, tfVec, buildJson: () => ({ kw_vocab: kwVocab }) }
}
```

Note: `stripVS` import stays only if still referenced elsewhere in the file;
if not, drop it and let Task 2 re-add it where `regen.ts` needs it. Confirm
`makeIdf`'s signature (`(docs: string[][]) => (word: string) => number`) — it is
called with the full per-glyph keyword-string lists, matching today's usage.

- [ ] **Step 5: Run the test to verify it passes**

Run: `bun test tools/data/flexrank.test.ts`
Expected: PASS (3 tests).

- [ ] **Step 6: Commit**

```bash
git add tools/data/config.ts tools/data/flexrank.ts tools/data/flexrank.test.ts
git commit -m "feat(flexrank): keyword-vocab selection + tfVec, drop per-emoji rank"
```

---

## Task 2: `regen.ts` — emit `flex_tf`, regenerate `flex.json` + fixture

**Files:**
- Modify: `tools/data/regen.ts:1-60` (imports, `Row` type, `BASE_FIELDS`),
  `:175-190` (`toLine`), `:295-392` (flag defs + flex block)
- Regenerate + commit: `web/public/flex.json`, `web/src/flexrank.fixture.json`

**Interfaces:**
- Consumes: `buildFlexRanker` from Task 1.
- Produces: train/eval rows carry `"flex_tf": [[kwIdx, value], ...]` (sparse,
  `value > 0` only). `web/public/flex.json` = `{ "kw_vocab": string[200] }`.
  `web/src/flexrank.fixture.json` = `{ "kw_vocab": string[200], "cases":
  [{ "text": string, "tf": number[200] }] }`.

- [ ] **Step 1: Update `Row` type + `BASE_FIELDS` + imports**

In `tools/data/regen.ts`:
- Change the import `import type { FlexHit, FlexQ } from "./flexrank.ts"` to
  `import { buildFlexRanker } from "./flexrank.ts"` (remove the duplicate
  `buildFlexRanker` import lower in the file at ~line 175).
- `Row` type: replace `flexsearch?: FlexHit[]` / `flexq?: FlexQ` with
  `flex_tf?: [number, number][]`.
- `BASE_FIELDS`: replace `"flexsearch", "flexq"` with `"flex_tf"`.
- Replace `const FLEX_K = 32` with `const KW_N = 200` (only used for the log
  line / flag default now).

- [ ] **Step 2: Update `toLine`**

```ts
export function toLine(r: Row): string {
  const base =
    r.bg && r.fg
      ? { text: r.text, emojis: r.emojis, styles: r.styles, bg: r.bg, fg: r.fg }
      : { text: r.text, emojis: r.emojis, styles: r.styles }
  const withExtra = r.extra ? { ...base, ...r.extra } : base
  return JSON.stringify(
    r.flex_tf ? { ...withExtra, flex_tf: r.flex_tf } : withExtra,
  )
}
```

- [ ] **Step 3: Update the CLI flags**

- Rename `--flex-k <n>` -> `--kw-n <n>` with description
  `` `keyword-vocab size for the soft-TF fusion signal (default ${KW_N})` ``
  (informational only — `buildFlexRanker` reads `KW_VOCAB_SIZE` from config).
- Rename `--no-flexsearch` -> `--no-kw` with description
  `"skip the soft-TF fusion signal on train/eval rows"`.

- [ ] **Step 4: Rewrite the flex block (`~line 355-386`)**

```ts
  const useKw = options.kw !== false
  let kwLine = "flex_tf               : skipped (--no-kw)"
  if (useKw) {
    console.log("computing soft-TF fusion vectors...")
    const ranker = await buildFlexRanker()
    let nzSum = 0
    for (const r of split) {
      const dense = ranker.tfVec(r.text)
      const pairs = dense
        .map((v, i) => [i, v] as [number, number])
        .filter(([, v]) => v > 0)
      r.flex_tf = pairs
      nzSum += pairs.length
    }
    await writeFileAtomic(FLEX_JSON, JSON.stringify(ranker.buildJson()) + "\n")
    const cases = FLEX_FIXTURE_TEXTS.map((text) => ({
      text,
      tf: ranker.tfVec(text),
    }))
    await writeFileAtomic(
      FLEX_FIXTURE_JSON,
      JSON.stringify({ kw_vocab: ranker.kwVocab, cases }, null, 2) + "\n",
    )
    const denom = split.length || 1
    kwLine =
      `flex_tf               : N=${ranker.kwVocab.length}, ` +
      `mean nz ${(nzSum / denom).toFixed(2)}`
  }
```

Replace the later `console.log(flexLine)` with `console.log(kwLine)`.

- [ ] **Step 5: Run regen and inspect output**

Run: `bun run regen`
Expected: completes; prints `flex_tf : N=200, mean nz ...`.

Then:
```bash
head -c 400 web/public/flex.json
python3 -c "import json;d=json.load(open('web/public/flex.json'));print(list(d),len(d['kw_vocab']))"
python3 -c "import json;r=json.loads(open('data/eval.jsonl').readline());print('flex_tf' in r, r.get('flex_tf')[:3])"
python3 -c "import json;f=json.load(open('web/src/flexrank.fixture.json'));print(len(f['kw_vocab']),len(f['cases']),len(f['cases'][0]['tf']))"
```
Expected: `flex.json` keys `['kw_vocab']`, length 200; eval row has `flex_tf`;
fixture has 200 kw_vocab, 8 cases, each `tf` length 200.

- [ ] **Step 6: Commit**

```bash
git add tools/data/regen.ts web/public/flex.json web/src/flexrank.fixture.json
git commit -m "feat(regen): emit sparse flex_tf rows + kw_vocab flex.json"
```

(`data/train.jsonl` / `data/eval.jsonl` / `data/labels.json` are gitignored —
not committed.)

---

## Task 3: `model/flexrank.py` — Python `tf_vec` + conformance test

**Files:**
- Modify: `model/flexrank.py` (full rewrite)
- Modify: `model/test_flexrank.py`

**Interfaces:**
- Consumes: `web/public/flex.json` (`{ kw_vocab }`),
  `web/src/flexrank.fixture.json` (`{ kw_vocab, cases:[{text, tf}] }`).
- Produces: `class FlexRanker` with `__init__(self, flex_json_path)` and
  `tf_vec(self, text: str) -> list[float]` (dense, length `len(kw_vocab)`,
  each cell rounded to 3 decimals). Module constants `FUZZY_MIN_LEN = 4`,
  `MIN_FUZZY_SCORE = 0.66`. `query_tokens(text)` retained (unchanged).

- [ ] **Step 1: Rewrite `model/flexrank.py`**

```python
import json
import re
from pathlib import Path

FUZZY_MIN_LEN = 4
MIN_FUZZY_SCORE = 0.66
STOPWORDS = set(
    (
        "a an the to of in on at is it its i you we they he she this that for and or but"
        " not with my your me am are was were be been being do does did have has had"
        " will would can could just so if"
    ).split()
)
_NON = re.compile(r"[^a-z0-9\s]")
_WS = re.compile(r"\s+")


def query_tokens(text: str) -> list[str]:
    t = _NON.sub(" ", text.lower())
    return [w for w in _WS.split(t) if len(w) >= 2 and w not in STOPWORDS]


def _overlap(w: str, k: str) -> float:
    if w == k:
        return 1.0
    if len(w) < FUZZY_MIN_LEN or len(k) < FUZZY_MIN_LEN:
        return 0.0
    if not (w.startswith(k) or k.startswith(w)):
        return 0.0
    r = min(len(w), len(k)) / max(len(w), len(k))
    return r if r >= MIN_FUZZY_SCORE else 0.0


def _r3(x: float) -> float:
    return float(f"{x:.3f}")


class FlexRanker:
    def __init__(self, flex_json_path):
        j = json.loads(Path(flex_json_path).read_text(encoding="utf-8"))
        self.kw_vocab = j["kw_vocab"]

    def tf_vec(self, text: str) -> list[float]:
        q = query_tokens(text)
        out = []
        for k in self.kw_vocab:
            s = 0.0
            for w in q:
                s += _overlap(w, k)
            out.append(_r3(s))
        return out
```

- [ ] **Step 2: Rewrite `model/test_flexrank.py`**

```python
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import json

from files import FLEX_JSON
from model.flexrank import FlexRanker

FIX = Path("web/src/flexrank.fixture.json")


def test_fixture_parity():
    fx = json.loads(FIX.read_text(encoding="utf-8"))
    r = FlexRanker(FLEX_JSON)
    assert r.kw_vocab == fx["kw_vocab"], "kw_vocab drift vs fixture"
    for case in fx["cases"]:
        got = r.tf_vec(case["text"])
        exp = case["tf"]
        assert len(got) == len(exp), (case["text"], len(got), len(exp))
        for i, (g, e) in enumerate(zip(got, exp, strict=True)):
            assert abs(g - e) < 1e-6, (case["text"], i, g, e)
    print(f"ok: {len(fx['cases'])} cases")


if __name__ == "__main__":
    test_fixture_parity()
```

- [ ] **Step 3: Run the test**

Run: `uv run python model/test_flexrank.py`
Expected: PASS — `ok: 8 cases`.

- [ ] **Step 4: Lint**

Run: `uv run ruff check model/flexrank.py model/test_flexrank.py && uv run ruff format --check model/flexrank.py model/test_flexrank.py`
Expected: clean.

- [ ] **Step 5: Commit**

```bash
git add model/flexrank.py model/test_flexrank.py
git commit -m "feat(flexrank.py): tf_vec surface, fixture parity test"
```

---

## Task 4: `web/src/flexrank.js` — JS `tfVec` + parity test

**Files:**
- Modify: `web/src/flexrank.js` (full rewrite)
- Modify: `web/src/flexrank.test.js`

**Interfaces:**
- Consumes: `../public/flex.json`, `./flexrank.fixture.json`.
- Produces: `export function makeFlexRanker(flexJson)` returning
  `{ kwVocab: string[], tfVec: (text: string) => number[] }` — `tfVec` dense,
  length `kwVocab.length`, `r3`-rounded.

- [ ] **Step 1: Rewrite `web/src/flexrank.js`**

```js
const FUZZY_MIN_LEN = 4
const MIN_FUZZY_SCORE = 0.66
const STOPWORDS = new Set(
  (
    'a an the to of in on at is it its i you we they he she this that for and or but' +
    ' not with my your me am are was were be been being do does did have has had' +
    ' will would can could just so if'
  ).split(' '),
)
const r3 = (x) => Number(x.toFixed(3))

function queryTokens(text) {
  return text
    .toLowerCase()
    .replace(/[^a-z0-9\s]/g, ' ')
    .split(/\s+/)
    .filter((w) => w.length >= 2 && !STOPWORDS.has(w))
}

function overlap(w, k) {
  if (w === k) return 1
  if (w.length < FUZZY_MIN_LEN || k.length < FUZZY_MIN_LEN) return 0
  if (!(w.startsWith(k) || k.startsWith(w))) return 0
  const r = Math.min(w.length, k.length) / Math.max(w.length, k.length)
  return r >= MIN_FUZZY_SCORE ? r : 0
}

export function makeFlexRanker(flexJson) {
  const kwVocab = flexJson.kw_vocab
  return {
    kwVocab,
    tfVec(text) {
      const q = queryTokens(text)
      return kwVocab.map((k) => {
        let s = 0
        for (const w of q) s += overlap(w, k)
        return r3(s)
      })
    },
  }
}
```

- [ ] **Step 2: Rewrite `web/src/flexrank.test.js`**

```js
import { describe, expect, it } from 'vitest'
import flexJson from '../public/flex.json'
import fixture from './flexrank.fixture.json'
import { makeFlexRanker } from './flexrank'

describe('flexrank parity', () => {
  const r = makeFlexRanker(flexJson)

  it('kw_vocab matches fixture', () => {
    expect(r.kwVocab).toEqual(fixture.kw_vocab)
  })

  for (const c of fixture.cases) {
    it(`tfVec matches fixture: ${JSON.stringify(c.text)}`, () => {
      const got = r.tfVec(c.text)
      expect(got.length).toBe(c.tf.length)
      got.forEach((v, i) => expect(v).toBeCloseTo(c.tf[i], 6))
    })
  }
})
```

- [ ] **Step 3: Run the test**

Run: `cd web && npm test -- flexrank`
Expected: PASS — 1 + 8 tests.

- [ ] **Step 4: Commit**

```bash
git add web/src/flexrank.js web/src/flexrank.test.js
git commit -m "feat(web/flexrank): tfVec surface, fixture parity test"
```

---

## Task 5: `model/data.py` — `flex_tf` path

**Files:**
- Modify: `model/data.py:1-20` (constants), `:84-123` (`record` + `read`),
  `:125-178` (`_row_flex` / `scatter_flex` -> `_row_tf`), `:238-258`
  (`EmojiDataset`)
- Create: `model/test_data_flex.py`

**Interfaces:**
- Consumes: `web/public/flex.json` (`kw_vocab`), rows with
  `"flex_tf": [[idx, val], ...]`.
- Produces:
  - `model/data.py` module constants `KW_VOCAB: list[str]`,
    `FLEX_N: int = len(KW_VOCAB)`.
  - `record.flex_tf: list = field(default_factory=list)` (replaces
    `flexsearch` / `flexq`).
  - `_row_tf(row) -> torch.Tensor` shape `[FLEX_N]` float32.
  - `EmojiDataset.__getitem__` returns
    `(text, emoji, style, colors, flex_tf)` — 5-tuple.
  - Deletes `_row_flex`, `scatter_flex`, `FLEX_MAX_K`, `FLEX_RAW_DIM`,
    `FLEXQ_DIM`, `_FLEXQ_KEYS`.

- [ ] **Step 1: Write the failing test `model/test_data_flex.py`**

```python
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from model.data import FLEX_N, KW_VOCAB, _row_tf


def test_flex_n_matches_vocab():
    assert FLEX_N == len(KW_VOCAB)
    assert FLEX_N == 200


def test_row_tf_scatters_sparse_pairs():
    t = _row_tf({"flex_tf": [[3, 1.0], [7, 0.667]]})
    assert t.shape == (FLEX_N,)
    assert t.dtype == torch.float32
    assert t[3].item() == 1.0
    assert abs(t[7].item() - 0.667) < 1e-6
    assert t.sum().item() == 1.667


def test_row_tf_empty():
    assert _row_tf({"flex_tf": []}).sum().item() == 0.0
    assert _row_tf({}).sum().item() == 0.0


if __name__ == "__main__":
    test_flex_n_matches_vocab()
    test_row_tf_scatters_sparse_pairs()
    test_row_tf_empty()
    print("ok")
```

- [ ] **Step 2: Run it to verify it fails**

Run: `uv run python model/test_data_flex.py`
Expected: FAIL — `ImportError: cannot import name 'FLEX_N'`.

- [ ] **Step 3: Edit `model/data.py`**

Replace lines 12-15:

```python
FLEX_MAX_K = 32
FLEX_RAW_DIM = 10
FLEXQ_DIM = 5
_FLEXQ_KEYS = ("tokens", "matched", "sum", "max", "cand")
```

with:

```python
with open(FLEX_JSON, encoding="utf-8") as _f:
    KW_VOCAB: list[str] = json.load(_f)["kw_vocab"]
FLEX_N = len(KW_VOCAB)
```

and add `FLEX_JSON` to the `from files import ...` line.

In the `record` dataclass, replace:

```python
    flexsearch: list = field(default_factory=list)
    flexq: dict = field(default_factory=dict)
```

with:

```python
    flex_tf: list = field(default_factory=list)
```

In `read`, replace the `yield record(...)` tail:

```python
                yield record(
                    text,
                    emojis,
                    styles,
                    [*bg, fg],
                    d.get("flex_tf") or [],
                )
```

Delete `_row_flex` and `scatter_flex` entirely; add:

```python
def _row_tf(row) -> torch.Tensor:
    pairs = row.get("flex_tf") if isinstance(row, dict) else row.flex_tf
    out = torch.zeros(FLEX_N, dtype=torch.float32)
    for i, v in pairs or []:
        out[int(i)] = float(v)
    return out
```

In `EmojiDataset.__init__`, replace the `flex = [_row_flex(r) ...]` block:

```python
        self.flex_tf = torch.stack([_row_tf(r) for r in records])
```

In `__getitem__`, return the 5-tuple:

```python
    def __getitem__(self, idx):
        return (
            self.text[idx],
            self.emoji[idx],
            self.style[idx],
            self.colors[idx],
            self.flex_tf[idx],
        )
```

Check whether `field` is still used elsewhere in the file; if `record` was its
only user, drop `field` from the `dataclasses` import and keep `dataclass`.

- [ ] **Step 4: Run the test to verify it passes**

Run: `uv run python model/test_data_flex.py`
Expected: PASS — `ok`.

- [ ] **Step 5: Lint**

Run: `uv run ruff check model/data.py model/test_data_flex.py && uv run ruff format --check model/data.py model/test_data_flex.py`
Expected: clean.

- [ ] **Step 6: Commit**

```bash
git add model/data.py model/test_data_flex.py
git commit -m "feat(data): flex_tf tensor path, drop per-emoji flex scatter"
```

---

## Task 6: `model/config.py` + `files.py`

**Files:**
- Modify: `model/config.py:47-64` (EMOJI / FUSION blocks), `:140-150`
  (`CONFIG_PARTS`)
- Modify: `files.py:16-21`

**Interfaces:**
- Produces: `model/config.py` exports `DROPOUT_KW = 0.1`; no longer exports
  `FUSION_HIDDEN`, `DROPOUT_FUSION`, `FUSION_INT_CLAMP`,
  `FUSION_INT_EMBED_SIZE`. `files.py` exports `EMOJI_EMBED_PT = f"{PT_DIR}/emoji_embed.pt"`,
  `KW_PT = f"{PT_DIR}/kw.pt"`.

- [ ] **Step 1: Edit `model/config.py`**

In the `# EMOJI` block, after `DROPOUT_EMOJI = 0.2` add:

```python
DROPOUT_KW = 0.1
```

and change `emj_str`:

```python
emj_str = " ".join([str(p) for p in (EMOJI_EMBED_SIZE, DROPOUT_EMOJI, DROPOUT_KW)])
```

Delete the whole `# FUSION` block (`FUSION_HIDDEN`, `DROPOUT_FUSION`,
`FUSION_INT_CLAMP`, `FUSION_INT_EMBED_SIZE`, `fus_str`).

In `CONFIG_PARTS`, delete the `f"FUSION: {fus_str}",` line.

- [ ] **Step 2: Verify the stats module still runs**

Run: `uv run python model/config.py`
Expected: prints the model-stats table, no `NameError`.

- [ ] **Step 3: Edit `files.py`**

After `EMOJI_PT = f"{PT_DIR}/emoji.pt"` add:

```python
EMOJI_EMBED_PT = f"{PT_DIR}/emoji_embed.pt"
KW_PT = f"{PT_DIR}/kw.pt"
```

- [ ] **Step 4: Lint**

Run: `uv run ruff check model/config.py files.py && uv run ruff format --check model/config.py files.py`
Expected: clean.

- [ ] **Step 5: Commit**

```bash
git add model/config.py files.py
git commit -m "feat(config): DROPOUT_KW; drop FUSION_* knobs; emoji_embed/kw pt paths"
```

---

## Task 7: `model/model.py` — shared table + three heads

**Files:**
- Modify: `model/model.py:8-30` (imports), `:90-200` (`EmojiHead` ..
  `FusionHead`)
- Create: `model/test_model_heads.py`

**Interfaces:**
- Consumes: `FLEX_N` from `model.data`; `DROPOUT_KW` from `model.config`.
- Produces:
  - `class EmojiEmbedding(nn.Module)` — `self.embed = nn.Embedding(len(EMOJIS),
    EMOJI_EMBED_SIZE)`, `self.bias = nn.Parameter(torch.zeros(len(EMOJIS)))`,
    `score(self, q) -> q @ self.embed.weight.t() + self.bias`.
  - `class EmojiHead(nn.Module)` — `forward(self, text_embedding) -> q_txt`
    `[B, 64]` (`Dropout(DROPOUT_EMOJI) -> Linear(TEXT_EMBED_SIZE, 64,
    bias=False)`). No `embed` / `bias` attrs anymore.
  - `class KWHead(nn.Module)` — `forward(self, tf_vec) -> q_kw` `[B, 64]`
    (`Dropout(DROPOUT_KW) -> Linear(FLEX_N, 64, bias=False)`, weight zero-init).
  - `class FusionHead(nn.Module)` — `self.net = nn.Linear(TEXT_EMBED_SIZE +
    FLEX_N, 1)` (weight zero-init, bias `+4.0`); `forward(self, text_embedding,
    tf_vec) -> a` `[B]` in `(0, 1)`.
  - Deletes `fusion_features`, `_FUSION_EPS`.

- [ ] **Step 1: Write the failing test `model/test_model_heads.py`**

```python
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from model.config import EMOJIS, TEXT_EMBED_SIZE
from model.data import FLEX_N
from model.model import EmojiEmbedding, EmojiHead, FusionHead, KWHead

B = 4


def test_shapes():
    emb = EmojiEmbedding()
    q_txt = EmojiHead()(torch.randn(B, TEXT_EMBED_SIZE))
    q_kw = KWHead()(torch.randn(B, FLEX_N))
    assert q_txt.shape == (B, 64)
    assert q_kw.shape == (B, 64)
    assert emb.score(q_txt).shape == (B, len(EMOJIS))


def test_kwhead_zero_init():
    q_kw = KWHead()(torch.randn(B, FLEX_N))
    assert torch.allclose(q_kw, torch.zeros_like(q_kw))


def test_gate_starts_near_one():
    a = FusionHead()(torch.randn(B, TEXT_EMBED_SIZE), torch.randn(B, FLEX_N))
    assert a.shape == (B,)
    assert (a > 0.97).all() and (a < 1.0).all()


def test_untrained_fusion_matches_emojihead():
    torch.manual_seed(0)
    emb = EmojiEmbedding()
    eh = EmojiHead().eval()
    kw = KWHead().eval()
    gate = FusionHead().eval()
    x = torch.randn(B, TEXT_EMBED_SIZE)
    tf = torch.randn(B, FLEX_N)
    q_txt = eh(x)
    q_kw = kw(tf)
    a = gate(x, tf).unsqueeze(-1)
    fused = emb.score(a * q_txt + (1 - a) * q_kw)
    base = emb.score(q_txt)
    assert (fused - base).abs().max() < 0.05 * base.abs().max()


if __name__ == "__main__":
    test_shapes()
    test_kwhead_zero_init()
    test_gate_starts_near_one()
    test_untrained_fusion_matches_emojihead()
    print("ok")
```

- [ ] **Step 2: Run it to verify it fails**

Run: `uv run python model/test_model_heads.py`
Expected: FAIL — `ImportError: cannot import name 'EmojiEmbedding'`.

- [ ] **Step 3: Edit `model/model.py`**

Imports: add `DROPOUT_KW` to the `model.config` import; add `FLEX_N` to the
`from model.data import ...` line. Remove `FUSION_HIDDEN`, `DROPOUT_FUSION`,
`FUSION_INT_CLAMP`, `FUSION_INT_EMBED_SIZE` from the `model.config` import.

Replace `class EmojiHead` and everything down through `class FusionHead`
(the `_FUSION_EPS`, `fusion_features`, old `FusionHead`) with:

```python
class EmojiEmbedding(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(len(EMOJIS), EMOJI_EMBED_SIZE)
        self.bias = nn.Parameter(torch.zeros(len(EMOJIS)))

    def score(self, q: torch.Tensor) -> torch.Tensor:
        return q @ self.embed.weight.t() + self.bias


class EmojiHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Dropout(p=DROPOUT_EMOJI),
            nn.Linear(TEXT_EMBED_SIZE, EMOJI_EMBED_SIZE, bias=False))

    def forward(self, text_embedding: torch.Tensor) -> torch.Tensor:
        return self.net(text_embedding)


class KWHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Dropout(p=DROPOUT_KW),
            nn.Linear(FLEX_N, EMOJI_EMBED_SIZE, bias=False))
        nn.init.zeros_(self.net[1].weight)  # type: ignore

    def forward(self, tf_vec: torch.Tensor) -> torch.Tensor:
        return self.net(tf_vec)


class FusionHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Linear(TEXT_EMBED_SIZE + FLEX_N, 1)
        nn.init.zeros_(self.net.weight)
        nn.init.constant_(self.net.bias, 4.0)

    def forward(
        self, text_embedding: torch.Tensor, tf_vec: torch.Tensor
    ) -> torch.Tensor:
        x = torch.cat([text_embedding, tf_vec], dim=-1)
        return torch.sigmoid(self.net(x)).squeeze(-1)
```

Keep `StyleHead` unchanged (it stays a self-contained head with its own embed
table — only the emoji side splits).

- [ ] **Step 4: Run the test to verify it passes**

Run: `uv run python model/test_model_heads.py`
Expected: PASS — `ok`.

- [ ] **Step 5: Lint**

Run: `uv run ruff check model/model.py model/test_model_heads.py && uv run ruff format --check model/model.py model/test_model_heads.py`
Expected: clean.

- [ ] **Step 6: Commit**

```bash
git add model/model.py model/test_model_heads.py
git commit -m "feat(model): EmojiEmbedding + EmojiHead/KWHead/FusionHead gate"
```

---

## Task 8: `model/train.py` — LitEncoder, save block, Modal wiring

**Files:**
- Modify: `model/train.py:63-76` (imports), `:193-270` (`LitEncoder.__init__` /
  `_step`), `:325-333` (`configure_optimizers`), `:497-528` (`_train_encoder`
  save), `:592-598` (`_run_local` gan `_require_pt`), `:715-760`
  (`train_remote` params + uploads), `:843-875` (`_run_remote`)

**Interfaces:**
- Consumes: `EmojiEmbedding`, `KWHead` from `model.model`; `EMOJI_EMBED_PT`,
  `KW_PT` from `files`.
- Produces: batch tuple is `(text, emoji, style, colors, flex_tf)`. `LitEncoder`
  has `self.emoji_embed` (when `emoji` selected). Stage-1 writes
  `emoji_embed.pt` and (fusion on) `kw.pt`. Checkpoint monitor unchanged key
  string `MRR/fusion/val`; `MRR/flex/val` log removed, `MRR/kw/val` added.

- [ ] **Step 1: Imports**

In the `from model.model import (...)` group add `EmojiEmbedding`, `KWHead`.
In the `from model.data import (...)` group remove `scatter_flex`. In the
`from files import (...)` group add `EMOJI_EMBED_PT`, `KW_PT`.

- [ ] **Step 2: `LitEncoder.__init__`**

Replace the head-construction block:

```python
        self.enc = TextEncoder()
        if "style" in self.heads:
            self.style = StyleHead()
        if "emoji" in self.heads:
            self.emoji_embed = EmojiEmbedding()
            self.emoji = EmojiHead()
        if "critic" in self.heads:
            self.critic = ColorCritic()
        if "fusion" in self.heads:
            self.kw = KWHead()
            self.fusion = FusionHead()
```

- [ ] **Step 3: `_step` — batch tuple + emoji/kw/fusion**

Change the unpack:

```python
        text, emoji, style, colors, flex_tf = batch
```

In the `"emoji"` branch, produce `q_txt` then score:

```python
        if "emoji" in self.heads:
            q_txt = self.emoji(enc)
            emoji_logits = self.emoji_embed.score(q_txt)
            loss_emoji = lse_infonce(emoji_logits, emoji, INFONCE_TEMP)
            loss = loss + loss_emoji
            self._log(f"loss/e/{split}", loss_emoji, bs)
            has_e = emoji.sum(dim=-1) > 0
            n_e = int(has_e.sum())
            if n_e:
                rr = mrr(emoji_logits[has_e], emoji[has_e])
                emoji_mrr = rr.mean()
                buf = self._val_rr if split == "val" else self._trn_rr
                buf.append(rr.detach())
            else:
                emoji_mrr = torch.zeros((), device=emoji.device)
            self._log(f"MRR/e/{split}", emoji_mrr, max(n_e, 1))
```

Replace the entire `"fusion"` branch with:

```python
        if "fusion" in self.heads:
            q_kw = self.kw(flex_tf)
            kw_logits = self.emoji_embed.score(q_kw)
            loss_kw = lse_infonce(kw_logits, emoji, INFONCE_TEMP)
            loss = loss + loss_kw
            self._log(f"loss/kw/{split}", loss_kw, bs)

            a = self.fusion(enc, flex_tf).unsqueeze(-1)
            q_fused = a * q_txt.detach() + (1 - a) * q_kw.detach()
            w = self.emoji_embed.embed.weight.detach()
            b = self.emoji_embed.bias.detach()
            fusion_logits = q_fused @ w.t() + b
            loss_fusion = lse_infonce(fusion_logits, emoji, INFONCE_TEMP)
            loss = loss + loss_fusion
            self._log(f"loss/fusion/{split}", loss_fusion, bs)

            if n_e:
                krr = mrr(kw_logits[has_e], emoji[has_e]).mean()
                frr = mrr(fusion_logits[has_e], emoji[has_e]).mean()
            else:
                krr = torch.zeros((), device=emoji.device)
                frr = torch.zeros((), device=emoji.device)
            self._log(f"MRR/kw/{split}", krr, max(n_e, 1))
            self._log(f"MRR/fusion/{split}", frr, max(n_e, 1))
```

(The `critic` branch and everything else in `_step` is unchanged.)

- [ ] **Step 4: `configure_optimizers`**

```python
    def configure_optimizers(self):
        params = list(self.enc.parameters())
        if "emoji" in self.heads:
            params += list(self.emoji_embed.parameters())
        for h in self.heads:
            params += list(getattr(self, h).parameters())
        return optim.Adam(params, lr=LR)
```

- [ ] **Step 5: `_train_encoder` save block**

After `save_pt(mod.enc.state_dict(), ...)`:

```python
    save_pt(mod.enc.state_dict(), str(out_dir / "enc.pt"), stage="enc")
    if "emoji" in heads:
        save_pt(
            mod.emoji_embed.state_dict(),
            str(out_dir / "emoji_embed.pt"),
            stage="enc",
        )
    for h in ALL_HEADS:
        if h in heads:
            save_pt(getattr(mod, h).state_dict(), str(
                out_dir / f"{h}.pt"), stage="enc")
```

(`ALL_HEADS` still `("style", "emoji", "critic", "fusion")`; the loop now also
writes `kw.pt` because `kw` is a `LitEncoder` attribute only when fusion is on —
wait: `kw` is NOT in `ALL_HEADS`. Add an explicit line:)

```python
    if "fusion" in heads:
        save_pt(mod.kw.state_dict(), str(out_dir / "kw.pt"), stage="enc")
```

- [ ] **Step 6: `_run_local` gan required-pt list**

```python
        _require_pt(
            pt_dir,
            [
                "enc.pt",
                "critic.pt",
                "style.pt",
                "emoji.pt",
                "emoji_embed.pt",
                "kw.pt",
                "fusion.pt",
            ],
        )
```

- [ ] **Step 7: Modal `train_remote` + `_run_remote`**

In `train_remote(...)` signature add `emoji_embed_bytes: bytes | None = None`
and `kw_bytes: bytes | None = None`. In its `uploads` dict add
`EMOJI_EMBED_PT: emoji_embed_bytes, KW_PT: kw_bytes`.

In `_run_remote`, the `pt_bytes` dict and the `stage == "gan"` branch: add
`"emoji_embed_bytes": None` / `"kw_bytes": None` to the dict, add
`EMOJI_EMBED_PT, KW_PT` to the `for name in (...)` existence check and the tuple
of `read_bytes()` reads.

- [ ] **Step 8: Run the CLI test + lint**

Run: `uv run python model/test_train_cli.py`
Expected: PASS (heads semantics unchanged).

Run: `uv run ruff check model/train.py && uv run ruff format --check model/train.py`
Expected: clean.

- [ ] **Step 9: Commit**

```bash
git add model/train.py
git commit -m "feat(train): shared emoji_embed + KWHead co-train, gated fusion loss"
```

---

## Task 9: `model/export_onnx.py`

**Files:**
- Modify: `model/export_onnx.py` (imports, `FLEX_COLS`, `ExportWrapper`,
  `export_onnx`, `export_web`, `export`)

**Interfaces:**
- Consumes: `EMOJI_EMBED_PT`, `KW_PT` from `files`; `EmojiEmbedding`, `KWHead`
  from `model.model`; `FLEX_N`, `KW_VOCAB` from `model.data`.
- Produces: `web/public/model.onnx` with inputs `input` `[b, MAX_TEXT_LEN]` +
  `flex_tf` `[b, FLEX_N]`; outputs `style_logits`, `emoji_logits`, `kw_logits`,
  `fusion_logits`, `color`. `meta.json` gains `flex_kw` (list) + `flex_n`;
  loses `flex_cols` / `flex_k`.

- [ ] **Step 1: Imports + constants**

- `from files import (...)`: add `EMOJI_EMBED_PT`, `KW_PT`.
- `from model.data import ...`: replace `FLEX_MAX_K, FLEX_RAW_DIM, FLEXQ_DIM`
  with `FLEX_N, KW_VOCAB`.
- `from model.model import ...`: add `EmojiEmbedding, KWHead`.
- Delete the `FLEX_COLS = [...]` list.

- [ ] **Step 2: `ExportWrapper`**

```python
class ExportWrapper(nn.Module):
    def __init__(self, enc, style, emoji_embed, emoji, kw, gen, fusion):
        super().__init__()
        self.enc = enc
        self.style = style
        self.emoji_embed = emoji_embed
        self.emoji = emoji
        self.kw = kw
        self.gen = gen
        self.fusion = fusion
        self.register_buffer("z", CONST_Z)

    def forward(self, x, flex_tf):
        emb = self.enc(x)
        style_logits = self.style(emb)
        q_txt = self.emoji(emb)
        emoji_logits = self.emoji_embed.score(q_txt)
        q_kw = self.kw(flex_tf)
        kw_logits = self.emoji_embed.score(q_kw)
        a = self.fusion(emb, flex_tf).unsqueeze(-1)
        fusion_logits = self.emoji_embed.score(a * q_txt + (1 - a) * q_kw)
        seed = (1 - Z_WEIGHT) * normalize(emb) + Z_WEIGHT * self.z
        color = torch.tanh(self.gen.net(seed)) * 127.5 + 127.5
        return style_logits, emoji_logits, kw_logits, fusion_logits, color
```

- [ ] **Step 3: `export_onnx` dummy + names**

```python
    dummy = (
        torch.zeros(1, MAX_TEXT_LEN, dtype=torch.long),
        torch.zeros(1, FLEX_N, dtype=torch.float32),
    )
    ...
        input_names=["input", "flex_tf"],
        output_names=["style_logits", "emoji_logits", "kw_logits",
                      "fusion_logits", "color"],
        ...
        dynamic_axes={
            "input": {0: "batch"},
            "flex_tf": {0: "batch"},
            "style_logits": {0: "batch"},
            "emoji_logits": {0: "batch"},
            "kw_logits": {0: "batch"},
            "fusion_logits": {0: "batch"},
        },
```

- [ ] **Step 4: `export_web` meta**

Replace `"flex_cols": FLEX_COLS, "flex_k": FLEX_MAX_K,` with:

```python
        "flex_kw": KW_VOCAB,
        "flex_n": FLEX_N,
```

- [ ] **Step 5: `export()` — loads + abort check**

```python
    enc = _load(TextEncoder(), ENC_PT)
    style = _load(StyleHead(), STYLE_PT)
    emoji_embed = _load(EmojiEmbedding(), EMOJI_EMBED_PT)
    emoji = _load(EmojiHead(), EMOJI_PT)
    kw = _load(KWHead(), KW_PT)
    gen = _load(ColorGen(), GEN_PT)
    fusion = _load(FusionHead(), FUSION_PT)

    if style.embed.weight.shape[0] != len(STYLES):
        raise SystemExit(
            f"style.pt has {style.embed.weight.shape[0]} styles, "
            f"{LABELS_JSON} has {len(STYLES)} -- retrain or restore {LABELS_JSON}"
        )
    if emoji_embed.embed.weight.shape[0] != len(EMOJIS):
        raise SystemExit(
            f"emoji_embed.pt has {emoji_embed.embed.weight.shape[0]} emojis, "
            f"{LABELS_JSON} has {len(EMOJIS)} -- retrain or restore {LABELS_JSON}"
        )

    _strip_spectral_norm(enc)
    wrapper = ExportWrapper(
        enc, style, emoji_embed, emoji, kw, gen, fusion).eval()
    export_web(wrapper)
```

- [ ] **Step 6: Lint (cannot run export — no `.pt` yet)**

Run: `uv run ruff check model/export_onnx.py && uv run ruff format --check model/export_onnx.py`
Expected: clean. (Functional check happens in Task 14 after retrain.)

- [ ] **Step 7: Commit**

```bash
git add model/export_onnx.py
git commit -m "feat(export): 2-input/5-output graph with kw_logits + flex_tf"
```

---

## Task 10: `model/pred.py`

**Files:**
- Modify: `model/pred.py:11-15` (imports), `:63-115` (`predict`)

**Interfaces:**
- Consumes: `EmojiEmbedding`, `KWHead` from `model.model`; `FlexRanker.tf_vec`;
  `_row_tf`-equivalent inline build.
- Produces: `data/pred.jsonl` rows unchanged in shape (`fusion_top_labels`
  present when `kw.pt` + `fusion.pt` + `flex.json` all exist).

- [ ] **Step 1: Imports**

- `from model.data import ...`: replace `_row_flex, scatter_flex` with
  `FLEX_N`.
- `from model.model import ...`: add `EmojiEmbedding, KWHead`.

- [ ] **Step 2: `predict` — load block**

```python
    enc = _load(TextEncoder(), pt_dir / "enc.pt")
    gen = _load(ColorGen(), pt_dir / "gen.pt")
    style = _load(StyleHead(), pt_dir / "style.pt")
    emoji_embed = _load(EmojiEmbedding(), pt_dir / "emoji_embed.pt")
    emoji = _load(EmojiHead(), pt_dir / "emoji.pt")

    fusion = kw = ranker = None
    if (
        (pt_dir / "fusion.pt").exists()
        and (pt_dir / "kw.pt").exists()
        and Path(FLEX_JSON).exists()
    ):
        fusion = _load(FusionHead(), pt_dir / "fusion.pt")
        kw = _load(KWHead(), pt_dir / "kw.pt")
        ranker = FlexRanker(FLEX_JSON)
    else:
        print(
            "kw.pt / fusion.pt / flex.json missing -- skipping fusion_top_labels",
            file=sys.stderr,
        )
```

- [ ] **Step 3: `predict` — per-text inference**

```python
            emb = enc(text_tensor)
            styles = top_labels(style(emb), STYLES, min_k=1, max_k=3)
            q_txt = emoji(emb)
            emoji_logits = emoji_embed.score(q_txt)
            emojis = top_labels(emoji_logits, EMOJIS, min_k=1, max_k=1)
            ...
            if fusion is not None:
                tf = torch.zeros(1, FLEX_N)
                for i, v in enumerate(ranker.tf_vec(text)):
                    tf[0, i] = v
                q_kw = kw(tf)
                a = fusion(emb, tf).unsqueeze(-1)
                fusion_logits = emoji_embed.score(
                    a * q_txt + (1 - a) * q_kw)
                record["fusion_top_labels"] = top_labels(
                    fusion_logits, EMOJIS, min_k=1, max_k=1
                )
```

- [ ] **Step 4: Lint**

Run: `uv run ruff check model/pred.py && uv run ruff format --check model/pred.py`
Expected: clean.

- [ ] **Step 5: Commit**

```bash
git add model/pred.py
git commit -m "feat(pred): kw.pt + gated fusion for fusion_top_labels"
```

---

## Task 11: `tools/report.py`

**Files:**
- Modify: `tools/report.py:20-30` (imports), `:75-135` (`_flex_ranker` ..
  `_cldr_flex_smoke`), `:255-300` (`_section_emoji` / `_section_cldr`),
  `:440-460` (provenance banner), `:740-800` (`_emoji_html` / `_cldr_html`),
  and every `build()` call site of `_section_emoji` / `_section_cldr`
- Modify: `tools/test_report.py:75-165`

**Interfaces:**
- Consumes: `EmojiEmbedding`, `KWHead` from `model.model`; `FLEX_N` from
  `model.data`; `FlexRanker.tf_vec`.
- Produces: `report.json` `emoji.eval` has `kw_acc_at_k` (replaces
  `flex_acc_at_k`) + `fusion_acc_at_k`. `cldr` section loses `flex_acc_at_k`.
  HTML legend `("EmojiHead", "KWHead", "Fusion")`.

- [ ] **Step 1: Update `tools/test_report.py` expectations first (failing)**

- In the `_linechart` 3-series test (`:78-86`): rename the series/legend
  strings `"FlexRank"` -> `"KWHead"`; assertion `">FlexRank<"` -> `">KWHead<"`.
- In the emoji-html test (`:95-110`): rename `"flex_acc_at_k"` ->
  `"kw_acc_at_k"`; assertions `">FlexRank<"` -> `">KWHead<"`; keep
  `">Fusion<"`.
- Delete `test_cldr_html_flex_series`, `test_cldr_flex_smoke`,
  `test_flex_section_keys` and their calls in `__main__`.
- Add `test_kw_section_keys`:

```python
def test_kw_section_keys():
    from files import FLEX_JSON
    from pathlib import Path

    if not Path(FLEX_JSON).exists():
        print("skip test_kw_section_keys (no flex.json)")
        return
    from model.flexrank import FlexRanker

    r = FlexRanker(FLEX_JSON)
    v = r.tf_vec("pizza time with friends tonight")
    assert len(v) == len(r.kw_vocab) == 200
```

Run: `uv run python tools/test_report.py`
Expected: FAIL (import / attribute errors from renamed report internals).

- [ ] **Step 2: `tools/report.py` imports + delete dead helpers**

- Imports: `from model.data import EVAL_PATH, TRAIN_PATH, read, text_to_tensor`
  (drop `_row_flex`, `scatter_flex`; keep `read`, `text_to_tensor`; add
  `FLEX_N`). `from model.model import ColorGen, EmojiEmbedding, EmojiHead,
  FusionHead, KWHead, StyleHead, TextEncoder`.
- Delete `FLEX_SMOKE_MIN`, `_flex_tensors`, `_flex_logits`, `_cldr_flex_smoke`.
- Keep `@cache _flex_ranker()` (now returns a `FlexRanker` with `tf_vec`).
- Add:

```python
@cache
def _kw_head():
    if not Path(KW_PT).exists():
        return None
    head, err = _load(KWHead(), KW_PT)
    return None if err else head


@cache
def _emoji_embed():
    if not Path(EMOJI_EMBED_PT).exists():
        return None
    m, err = _load(EmojiEmbedding(), EMOJI_EMBED_PT)
    return None if err else m
```

(add `EMOJI_EMBED_PT`, `KW_PT` to the `from files import` group.)

- [ ] **Step 3: `_kw_fusion_acc` (replaces `_flex_fusion_acc`)**

```python
def _tf_batch(records):
    fr = _flex_ranker()
    tf = torch.zeros(len(records), FLEX_N)
    for i, r in enumerate(records):
        for j, v in enumerate(fr.tf_vec(r.text)):
            tf[i, j] = v
    return tf


def _kw_fusion_acc(records, tgt, enc_emb, q_txt):
    fr = _flex_ranker()
    emb = _emoji_embed()
    kw = _kw_head()
    if fr is None or emb is None or kw is None or q_txt is None:
        return None, None
    tf = _tf_batch(records)
    with torch.no_grad():
        q_kw = kw(tf)
        kw_acc = [
            _acc_at_k(emb.score(q_kw), tgt, k).mean().item() for k in EMOJI_KS
        ]
        fusion_acc = None
        head = _fusion_head()
        if head is not None:
            a = head(enc_emb, tf).unsqueeze(-1)
            fl = emb.score(a * q_txt + (1 - a) * q_kw)
            fusion_acc = [
                _acc_at_k(fl, tgt, k).mean().item() for k in EMOJI_KS
            ]
    return kw_acc, fusion_acc
```

- [ ] **Step 4: `_section_emoji` / `_section_cldr` + callers**

`_section_emoji` and `_section_cldr` and `_keyword_probe` / `_cldr_probe` need
the `EmojiEmbedding` to turn `EmojiHead` output into logits. Simplest: give
each a local helper. In `_section_emoji`:

```python
def _section_emoji(enc, head, eval_records):
    emb = _emoji_embed()
    if enc is None or head is None or emb is None:
        return {}
    ...
        with torch.no_grad():
            q_txt = head(enc(texts))
            logits = emb.score(q_txt)
        kw_acc, fusion_acc = _kw_fusion_acc(rows, tgt, enc(texts), q_txt)
        d["eval"] = {
            "n": len(rows),
            "acc_at_k": [_acc_at_k(logits, tgt, k).mean().item()
                         for k in EMOJI_KS],
            "kw_acc_at_k": kw_acc,
            "fusion_acc_at_k": fusion_acc,
            "baseline": _cldr_baseline(),
        }
```

Audit `_keyword_probe(enc, head)` and `_cldr_probe(enc, head)`: wherever they
call `head(enc(...))` and treat it as logits, wrap with `_emoji_embed().score(...)`.
`_section_cldr` drops the `res["flex_acc_at_k"] = _cldr_flex_smoke()` line
entirely (return `dict(_cldr_probe(enc, head))`).

- [ ] **Step 5: `_emoji_html` / `_cldr_html` + provenance banner**

- `_emoji_html`: `e.get("flex_acc_at_k")` -> `e.get("kw_acc_at_k")`; series
  label `"FlexRank"` -> `"KWHead"`; legend tuple string likewise.
- `_cldr_html`: delete the `fx = d.get("flex_acc_at_k")` block and the
  `FlexRank acc@1 ...` note — render just the EmojiHead `acc@k` line chart.
- Provenance banner (`:454`): delete the `fs = report["cldr"].get(...)` /
  `FLEX_SMOKE_MIN` check. In whatever list enumerates required `.pt` for the
  banner, add `emoji_embed.pt` and `kw.pt`.

- [ ] **Step 6: Run the report test + lint**

Run: `uv run python tools/test_report.py`
Expected: PASS.

Run: `uv run ruff check tools/report.py tools/test_report.py && uv run ruff format --check tools/report.py tools/test_report.py`
Expected: clean.

- [ ] **Step 7: Commit**

```bash
git add tools/report.py tools/test_report.py
git commit -m "feat(report): EmojiHead/KWHead/Fusion 3-way, drop raw-FlexRank curve"
```

---

## Task 12: `web/src/model.js` + masthead toggle

**Files:**
- Modify: `web/src/model.js` (flex input build, ONNX `run` inputs, decode)
- Modify: whichever component reads `emojiMode` and picks the logit output
  (grep `emojiMode` / `emoji_logits` / `fusion_logits` under `web/src/`)
- Modify: `web/src/model.test.js`

**Interfaces:**
- Consumes: `makeFlexRanker(flexJson).tfVec` from Task 4; `meta.json`
  `flex_n` / `flex_kw`.
- Produces: ONNX session run with `{ input, flex_tf }`; consumers read one of
  `emoji_logits` / `kw_logits` / `fusion_logits` by `emojiMode`
  (`model` / `keywords` / `fusion`).

- [ ] **Step 1: grep the current wiring**

Run: `grep -rn "flex\|emojiMode\|emoji_logits\|fusion_logits\|flexRaw\|flexQ" web/src`
Record every call site the edits below must cover.

- [ ] **Step 2: `web/src/model.js`**

- Where the model inputs are assembled, replace the `flex` `[1,V,10]` +
  `flex_q` `[1,5]` tensors with a single `flex_tf` `Float32Array` of length
  `meta.flex_n` from `ranker.tfVec(normalizedText)`, wrapped as an ORT tensor
  `new ort.Tensor('float32', arr, [1, meta.flex_n])`.
- `session.run({ input, flex_tf })`.
- Return `{ styleLogits, emojiLogits, kwLogits, fusionLogits, color }` from the
  four/five named outputs.

- [ ] **Step 3: masthead toggle**

In the component that maps `emojiMode` to a ranking: `fusion` ->
`fusionLogits`, `model` -> `emojiLogits`, `keywords` -> `kwLogits` (drop the
old `flexrank.js` `.rank()` path — `keywords` is now a model output). Keep the
`localStorage` key `emojiMode` + the legacy `emojiSource` migration.

- [ ] **Step 4: `web/src/model.test.js`**

Update any assertion about input names / output names / tensor shapes to the
new `{ input, flex_tf }` -> 5 outputs shape. If the test loads the real
`model.onnx`, guard it to skip when the file's IO signature is the old one
(it will be refreshed in Task 14).

- [ ] **Step 5: Run web tests + build**

Run: `cd web && npm test`
Expected: PASS.

Run: `cd web && npm run build`
Expected: succeeds.

- [ ] **Step 6: Commit**

```bash
git add web/src/model.js web/src/model.test.js web/src/<toggle-component>
git commit -m "feat(web): flex_tf model input, keywords toggle -> kw_logits"
```

---

## Task 13: Docs

**Files:**
- Modify: `CLAUDE.md` (the `FusionHead` sentences in the Project paragraph; the
  `model/train.py` bullet's fusion/`--heads`/checkpoint-key text; the
  `model/config.py` bullet; the `model/export_onnx.py` bullet; the
  `data/labels.json` / regen / `web/` bullets mentioning `flexsearch` /
  `flex_cols` / `flex.json` schema / the 3-way toggle)
- Modify: `docs/model.md` via the `update-model-md` skill

**Interfaces:** none (prose only).

- [ ] **Step 1: Edit `CLAUDE.md`**

Rewrite the `FusionHead` description to: shared `EmojiEmbedding` table; three
heads (`EmojiHead` text->64d, `KWHead` tf->64d, `FusionHead` gate); fused =
`score(a*q_txt + (1-a)*q_kw)` with `q_txt`/`q_kw`/table detached in the fused
path so `loss/fusion` trains only the gate; `E`/`bias` co-trained by
`loss/emoji` + `loss/kw`; the lexical input is a length-200 raw soft-TF vector
over the highest-IDF single-word 4..12-char CLDR keywords (`flex_tf`, sparse on
rows); `flex.json` is now `{ kw_vocab }`; `meta.json` has `flex_kw` / `flex_n`;
ONNX inputs `input` + `flex_tf`, outputs add `kw_logits`; checkpoint order uses
`MRR/kw/val` where it used `MRR/flex/val`; stage-1 writes `emoji_embed.pt` +
`kw.pt`; `gan` requires them too. Remove mentions of `fusion_features`, the
18+4 features, `FUSION_INT_*`, `FUSION_HIDDEN`, `flexsearch`/`flexq` row
fields, `flex_cols`/`flex_k`, `scatter_flex`, the raw-FlexRank report curve and
the CLDR FlexRank smoke test / `FLEX_SMOKE_MIN`.

- [ ] **Step 2: Regenerate `docs/model.md`**

Invoke the `update-model-md` skill.

- [ ] **Step 3: Commit**

```bash
git add CLAUDE.md docs/model.md
git commit -m "docs: KW-fusion rework"
```

---

## Task 14: Integration — regen, retrain, export, commit web assets

**Files:**
- Regenerate (gitignored): `data/train.jsonl`, `data/eval.jsonl`,
  `data/labels.json`
- Regenerate + commit: `web/public/model.onnx`, `web/public/meta.json`,
  `web/public/config.json`

**Interfaces:** none — this is behavioral sign-off.

- [ ] **Step 1: Full static verification**

```bash
uv run ruff check . && uv run ruff format --check .
uv run python model/test_runmeta.py
uv run python model/test_train_cli.py
uv run python model/test_flexrank.py
uv run python model/test_data_flex.py
uv run python model/test_model_heads.py
uv run python tools/test_report.py
bun test tools/data/flexrank.test.ts
cd web && npm test && npm run build && cd ..
```
Expected: all green.

- [ ] **Step 2: Regen (fresh artifacts on the new schema)**

Run: `bun run regen`
Expected: prints `flex_tf : N=200, mean nz ...`; `data/labels.json` has 21
styles + the emoji vocab.

- [ ] **Step 3: Retrain stage 1 + 2 (long-running — this is the sign-off)**

Run: `uv run python model/train.py --local`
Watch in TensorBoard (`uv run tensorboard --logdir runs`):
- `MRR/fusion/val` — the checkpoint/early-stop key — should track
  `>= max(MRR/e/val, MRR/kw/val)`.
- `MRR/e/val`, `MRR/kw/val` logged alongside.
- Stage 2: `energy/gan/val` vs `energy/gan/ref`.
Note (per `exportbest-clobbers-model-pt` memory): `pt/*.pt` and `web/public/`
are overwritten even if the run is worse — judge by the logs, not the files.
This writes `pt/enc.pt` / `style.pt` / `emoji.pt` / `emoji_embed.pt` / `kw.pt` /
`critic.pt` / `fusion.pt` / `gen.pt` and refreshes `web/public/` via
`export_onnx.py`.

- [ ] **Step 4: Spot-check pred + report**

```bash
uv run python model/pred.py --pt pt
uv run python tools/report.py
```
Expected: `data/pred.jsonl` rows carry `fusion_top_labels`; the report's
`report/<ts>-<sha>/report.html` Model-Emojis chart shows the EmojiHead /
KWHead / Fusion 3-way overlay and the Cards section renders.

- [ ] **Step 5: Commit the refreshed web model**

```bash
git add web/public/model.onnx web/public/meta.json web/public/config.json
git commit -m "chore(web): refresh model.onnx for KW-fusion graph"
```

- [ ] **Step 6: Finish the branch**

Invoke `superpowers:finishing-a-development-branch`.

---

## Self-Review

**Spec coverage:**
- Shared `EmojiEmbedding` + `emoji_embed.pt` — Tasks 7, 8, 9, 10, 11.
- `EmojiHead` returns q_txt / `KWHead` / linear `FusionHead` gate — Task 7.
- 64-d blend, `q_txt`/`q_kw`/table detached in fused path — Tasks 7 (test), 8
  (`_step`), 9 (`ExportWrapper`), 10 (`pred`), 11 (`report`).
- Gate init `a ~= 0.98` — Task 7 (`test_gate_starts_near_one`,
  `test_untrained_fusion_matches_emojihead`).
- `kw_vocab` selection (single-word, 4..12, top-200 by IDF, alpha tiebreak) —
  Task 1.
- `tf_vec` overlap rule + `MIN_FUZZY_SCORE = 0.66` — Tasks 1, 3, 4.
- `flex.json = { kw_vocab }`, `meta.json` `flex_kw`/`flex_n` — Tasks 2, 9.
- sparse `flex_tf` row field — Tasks 2, 5.
- conformance fixture locks `tf` vectors — Tasks 2, 3, 4.
- training losses `emoji` + `kw` + `fusion`, `MRR/kw/val` replaces
  `MRR/flex/val`, checkpoint order — Task 8.
- `--heads` unchanged, `fusion` requires `emoji` — unchanged (Task 8 keeps
  `_parse_heads`), covered by `test_train_cli.py`.
- `gan` required-pt adds `emoji_embed.pt` + `kw.pt` — Task 8.
- ONNX 2 inputs / 5 outputs incl. `kw_logits` — Task 9.
- web `flex_tf` input, `keywords` toggle -> `kw_logits` — Task 12.
- report 3-way overlay, drop FlexRank curve + smoke — Task 11.
- config `DROPOUT_KW`, drop `FUSION_*` — Task 6.
- `files.py` `EMOJI_EMBED_PT` / `KW_PT` — Task 6.
- CLDR-baseline (`cldr-baseline.ts`) untouched — respected (no task touches it).
- docs — Task 13.
- migration (regen -> retrain -> export -> commit web) — Task 14.

**Placeholder scan:** Task 11 Step 4 says "audit `_keyword_probe` /
`_cldr_probe`" — this is a real instruction (wrap `head(enc(...))` in
`_emoji_embed().score(...)`), with the exact transform given; the executor must
grep those two functions. Task 12 Step 3 references `<toggle-component>` —
resolved by the Step 1 grep. Both are bounded lookups, not open design.

**Type consistency:**
- `FlexRanker.tf_vec(text) -> list[float]` — Tasks 3, 10, 11 all call it the
  same way; `.kw_vocab` attr used in Tasks 3, 11.
- `buildFlexRanker(): { kwVocab, tfVec, buildJson }` — Task 1 defines, Task 2
  consumes exactly those names.
- `makeFlexRanker(flexJson) -> { kwVocab, tfVec }` — Task 4 defines, Task 12
  consumes.
- `EmojiEmbedding.score(q)`, `EmojiHead(x) -> q_txt`, `KWHead(tf) -> q_kw`,
  `FusionHead(text_embedding, tf_vec) -> a` — Task 7 defines; Tasks 8, 9, 10,
  11 use these exact signatures.
- `_row_tf(row) -> Tensor[FLEX_N]`, `FLEX_N`, `KW_VOCAB` — Task 5 defines;
  Tasks 8 (via batch), 9, 11 use `FLEX_N`; Task 9 uses `KW_VOCAB`.
- batch tuple `(text, emoji, style, colors, flex_tf)` — Task 5 (`__getitem__`)
  and Task 8 (`_step` unpack) agree.
