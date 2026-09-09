# FusionHead Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a learned FusionHead that re-ranks `EmojiHead` logits using the CLDR keyword-retrieval signal, trained end to end, shipped in the ONNX graph, and selectable in the web app alongside raw-model and keyword-only ranking.

**Architecture:** FusionHead is a residual per-label re-ranker — `fused = emoji_logit + g(features)` with `g`'s last layer zero-initialised, so an untrained head reproduces `EmojiHead`. It consumes a **detached** copy of `emoji_logits`, so the encoder and raw `EmojiHead` train byte-identically to today. Per-emoji flex features (`flexsearch`/`flexq` from `regen`) are stored sparse and scattered to `[B, V, 10]` per batch. The feature transform is one pure-torch function traced straight into ONNX; the browser only runs a deterministic ranker (`web/src/flexrank.js`) reading a shared `web/public/flex.json`.

**Tech Stack:** PyTorch + Lightning, Typer, ONNX (opset 18), Modal; Bun/TypeScript for the data toolchain; React + Vite + onnxruntime-web for the app; `uv` for Python deps; `ruff` for lint.

**Spec:** `docs/superpowers/specs/2026-09-09-fusion-head-design.md`

## Global Constraints

- Python 3.13; package management is `uv` only (`uv add` / `uv sync`), never `pip install`.
- No comments or docstrings in source (keep `type: ignore` / `noqa` / shebangs). Applies to Python and TypeScript.
- `torch` is a `cpu`/`gpu` conflicting dependency-group split; CPU / `--local` runs must stay byte-identical (all CUDA-only knobs already gated on `torch.cuda.is_available()`).
- `model/train.py` aborts on a dirty git tree in every mode — commit each task before running any training smoke.
- `.pt` files and `data/train.jsonl` / `data/eval.jsonl` / `data/labels.json` are gitignored artifacts from `bun run regen`; `web/public/flex.json` **is** committed.
- `normalize` / `encode` / the flex ranker must stay byte-identical across TS, Python, and JS — locked by `web/src/flexrank.fixture.json`.
- Emoji vocab size is dynamic (`len(EMOJIS)` from `data/labels.json`); never hardcode it. Call it `V` below.
- `FLEX_MAX_K = 32` (matches `regen --flex-k` default); the per-emoji raw vector is 10 wide; `flexq` is 5 wide.
- New config values (exact): `FUSION_HIDDEN = 48`, `DROPOUT_FUSION = 0.1`, `FUSION_INT_CLAMP = 16`, `FUSION_INT_EMBED_SIZE = 8`.
- New head name is `fusion`; `ALL_HEADS = ("style", "emoji", "critic", "fusion")`; `fusion` requires `emoji` in the selected set.
- Checkpoint/early-stop monitor is `MRR/fusion/val` (mode `max`) whenever `fusion` is a selected head.

---

## File Structure

| File | Responsibility |
|---|---|
| `tools/data/flexrank.ts` (modify) | ranker returns `{ rank, buildJson }`; `buildJson(vocab)` emits the `flex.json` object |
| `tools/data/regen.ts` (modify) | call `ranker.rank(...)`; write `web/public/flex.json`; summary line |
| `files.ts` / `files.py` (modify) | add `FLEX_JSON`, `FUSION_PT` |
| `web/src/flexrank.js` (create) | JS ranker over `flex.json`; `buildFlexInputs(text) -> {flexRaw, flexQ}` |
| `web/src/flexrank.fixture.json` (create) | shared parity fixture (`text -> {flexsearch, flexq}`) |
| `web/src/flexrank.test.js` (create) | fixture parity + hand cases |
| `model/flexrank.py` (create) | Python ranker over `flex.json`; mirrors `flexrank.ts` |
| `model/test_flexrank.py` (create) | asserts the shared fixture |
| `model/config.py` (modify) | 4 `FUSION_*` consts; `CONFIG_NAME` |
| `model/model.py` (modify) | `fusion_features()` + `FusionHead` |
| `model/data.py` (modify) | `read()` captures flex; sparse tensors in `EmojiDataset`; `scatter_flex()` |
| `model/train.py` (modify) | `fusion` head plumbing; `_step` loss + 3 MRRs; monitor; save `fusion.pt`; gan `_require_pt`/upload |
| `model/export_onnx.py` (modify) | load `fusion.pt`; `flex`/`flex_q` inputs; `fusion_logits` output; `meta.json` keys |
| `model/pred.py` (modify) | fused predictions column |
| `web/src/hooks/useOnnx.js` (modify) | fetch `flex.json`; build `flex`/`flex_q` tensors; return `fusion` + `flexRaw` |
| `web/src/App.jsx` (modify) | 3-way emoji-mode toggle (`fusion`/`model`/`keywords`), default `fusion` |
| `web/src/cldrEmojis.ts` + `web/src/data/cldr-emoji-index.json` (delete) | replaced by `flexrank.js` + `flex.json` |
| `tools/report.py` (modify) | `model/flexrank.py` use; 3-way charts; cards chart split; `.lline3`; smoke-test banner |
| `tools/test_report.py` (modify) | new keys |
| `model/test_train_cli.py` (modify) | `--heads` fusion cases |
| `CLAUDE.md` (modify) | FusionHead, `fusion` head, `flex.json`, parity surfaces |

---

## Phase 1 — flex.json + parity rankers

### Task 1: `flex.json` emitted by regen

**Files:**
- Modify: `tools/data/flexrank.ts`
- Modify: `tools/data/regen.ts`
- Modify: `files.ts` (add `FLEX_JSON`)
- Test: `tools/data/flexrank.test.ts` (create)

**Interfaces:**
- Consumes: existing `buildFlexRanker(k)` internals (`glyphs`, `kwTokens`, `idf`, `prefix4`, `kwCount`).
- Produces:
  - `buildFlexRanker(k): Promise<{ rank: FlexRanker; buildJson: (vocab: Map<string,string>) => FlexJson }>`
    where `FlexRanker = (text: string, vocab: Map<string,string>) => FlexResult` (unchanged signature).
  - `type FlexJson = { emojis: string[]; keywords: string[][]; idf: Record<string, number>; idf_default: number }`
  - `export const FLEX_JSON = "web/public/flex.json"` in `files.ts`.

- [ ] **Step 1: Write the failing test** — `tools/data/flexrank.test.ts`

```ts
import { expect, test } from "bun:test"
import { buildFlexRanker } from "./flexrank.ts"
import { stripVS } from "../analysis/cldr-baseline.ts"

test("buildJson emits vocab keywords + idf map consistent with the ranker", async () => {
  const { rank, buildJson } = await buildFlexRanker(32)
  const vocabOrig = ["🍕", "🐞", "🧁"]
  const vocab = new Map(vocabOrig.map((e) => [stripVS(e), e]))
  const j = buildJson(vocab)

  expect(j.emojis).toEqual(vocabOrig)
  expect(j.keywords.length).toBe(3)
  expect(j.keywords.every((ks) => Array.isArray(ks) && ks.length > 0)).toBe(true)
  expect(typeof j.idf_default).toBe("number")

  // every keyword token of a vocab emoji has an idf entry
  for (const ks of j.keywords) for (const k of ks) expect(k in j.idf).toBe(true)

  // buildJson does not perturb ranking
  const before = rank("cheese pizza time", vocab)
  expect(before.flexsearch[0][0]).toBe("🍕")
})
```

- [ ] **Step 2: Run test to verify it fails**

Run: `bun test tools/data/flexrank.test.ts`
Expected: FAIL — `buildFlexRanker(...)` currently resolves to a function, `.rank` / `.buildJson` undefined.

- [ ] **Step 3: Refactor `buildFlexRanker` to return `{ rank, buildJson }`**

In `tools/data/flexrank.ts`, keep all index construction as-is. Change the return:

```ts
export type FlexJson = {
  emojis: string[]
  keywords: string[][]
  idf: Record<string, number>
  idf_default: number
}

// ...inside buildFlexRanker, after `const idf = makeIdf(kwTokens)` and the
// existing `const rank: FlexRanker = (text, vocab) => { ... }` (rename the
// returned closure to `rank`):

  const allKeywords = new Set<string>()
  for (const toks of kwTokens) for (const kw of toks) allKeywords.add(kw)
  const idfMap: Record<string, number> = {}
  for (const kw of allKeywords) idfMap[kw] = idf(kw)
  const idfDefault = Math.log((glyphs.length + 1) / 1) + 1

  const glyphToIdx = new Map(glyphs.map((g, i) => [g, i]))
  const buildJson = (vocab: Map<string, string>): FlexJson => {
    const emojis: string[] = []
    const keywords: string[][] = []
    for (const [stripped, orig] of vocab) {
      const gi = glyphToIdx.get(stripped)
      emojis.push(orig)
      keywords.push(gi === undefined ? [] : kwTokens[gi])
    }
    return { emojis, keywords, idf: idfMap, idf_default: idfDefault }
  }

  return { rank, buildJson }
```

(`makeIdf`'s formula is `Math.log((n + 1) / (df.get(word) ?? 0 + 1)) + 1` with `n = docs.length`; a `df` of 0 gives `Math.log((n+1)/1)+1` — that is `idf_default`. Verify against `cldr-baseline.ts:makeIdf` and match it exactly.)

- [ ] **Step 4: Update `regen.ts` call sites + write `flex.json`**

In `tools/data/regen.ts`:
- import `FLEX_JSON` from `files.ts`.
- change `const ranker = await buildFlexRanker(flexK)` usage: `ranker(r.text, vocabMap)` → `ranker.rank(r.text, vocabMap)`.
- after the flex loop (still inside `if (useFlex)`), write the json:

```ts
    const flexJson = ranker.buildJson(vocabMap)
    await writeFileAtomic(FLEX_JSON, JSON.stringify(flexJson) + "\n")
```

- add to the summary block: `flexLine` already reports mean list; append `, flex.json ${flexJson.emojis.length} emoji`.

- [ ] **Step 5: Run tests**

Run: `bun test tools/data/flexrank.test.ts tools/analysis/cldr-baseline.test.ts`
Expected: PASS (12 + new).

Run: `bun run regen`
Expected: writes `web/public/flex.json`; summary line mentions it.

Run: `python3 -c "import json;d=json.load(open('web/public/flex.json'));print(len(d['emojis']), len(d['idf']), d['idf_default'])"`
Expected: `957 <big> <float>` (emoji count matches `data/labels.json`).

- [ ] **Step 6: Commit**

```bash
git add tools/data/flexrank.ts tools/data/flexrank.test.ts tools/data/regen.ts files.ts web/public/flex.json
git commit -m "feat: regen writes web/public/flex.json for the fusion ranker

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01NZUMgpJ5J1UKZiV76KUbR6"
```

---

### Task 2: shared parity fixture

**Files:**
- Create: `web/src/flexrank.fixture.json`
- Create: `tools/data/gen-flex-fixture.ts`

**Interfaces:**
- Consumes: `buildFlexRanker` from Task 1.
- Produces: `web/src/flexrank.fixture.json` — `{ vocab: string[], cases: [{ text: string, flexsearch: [string, ...number[]][], flexq: {tokens,matched,sum,max,cand} }] }`. Tasks 3 and 8 assert against this.

- [ ] **Step 1: Write the fixture generator**

`tools/data/gen-flex-fixture.ts`:

```ts
import { buildFlexRanker } from "./flexrank.ts"
import { stripVS } from "../analysis/cldr-baseline.ts"
import { LABELS_JSON } from "../../files.ts"
import { writeFileAtomic } from "./io.ts"

const TEXTS = [
  "Cupcakes are in the break room",
  "I found a ladybug in my book",
  "pizza time with friends tonight",
  "feeling anxious about the meeting",
  "the dog is running fast in the park",
  "not good at all",
  "Woof!",
  "xyzzy qwerty",
]

const vocabOrig = JSON.parse(await Bun.file(LABELS_JSON).text()).emojis as string[]
const vocab = new Map(vocabOrig.map((e) => [stripVS(e), e]))
const { rank } = await buildFlexRanker(32)
const cases = TEXTS.map((text) => {
  const { flexsearch, flexq } = rank(text, vocab)
  return { text, flexsearch, flexq }
})
await writeFileAtomic(
  "web/src/flexrank.fixture.json",
  JSON.stringify({ vocab: vocabOrig, cases }, null, 2) + "\n",
)
console.log(`wrote web/src/flexrank.fixture.json (${cases.length} cases)`)
```

- [ ] **Step 2: Generate it**

Run: `bun run regen` (so `flex.json` + labels are current), then
Run: `bun tools/data/gen-flex-fixture.ts`
Expected: `web/src/flexrank.fixture.json` written; open it and confirm case `"Cupcakes are in the break room"` has `flexsearch[0][0] === "💔"` and `flexq.tokens === 3`.

- [ ] **Step 3: Commit**

```bash
git add tools/data/gen-flex-fixture.ts web/src/flexrank.fixture.json
git commit -m "test: shared flex-ranker parity fixture

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01NZUMgpJ5J1UKZiV76KUbR6"
```

---

### Task 3: `model/flexrank.py`

**Files:**
- Create: `model/flexrank.py`
- Create: `model/test_flexrank.py`

**Interfaces:**
- Consumes: `web/public/flex.json` (Task 1), `web/src/flexrank.fixture.json` (Task 2).
- Produces:
  - `FlexRanker(flex_json_path: str | Path)` with:
    - `.rank(text: str) -> list[tuple[str, float, float, int, int, float, int, int, int, int]]`
      — `(emoji, score, score_norm, exact, fuzzy, best_idf, n_kw, kw_len, word_len, overlap)`, sorted, `score>0` only, cap 32.
    - `.flexq(text: str) -> dict` — `{"tokens","matched","sum","max","cand"}`.
  - Module consts mirrored from `cldr-baseline.ts`: `FUZZY_MIN_LEN = 4`, `FUZZY_MAX_LEN_DELTA = 3`, `FUZZY_WEIGHT = 0.6`, and the `STOPWORDS` set (copy verbatim from `tools/analysis/cldr-baseline.ts`).

- [ ] **Step 1: Write the failing test** — `model/test_flexrank.py`

```python
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from files import FLEX_JSON
from model.flexrank import FlexRanker

FIX = Path("web/src/flexrank.fixture.json")


def test_matches_shared_fixture():
    fix = json.loads(FIX.read_text())
    r = FlexRanker(FLEX_JSON)
    for case in fix["cases"]:
        got = r.rank(case["text"])
        exp = case["flexsearch"]
        assert len(got) == len(exp), case["text"]
        for g, e in zip(got, exp, strict=True):
            assert g[0] == e[0], (case["text"], g, e)
            for gi, ei in zip(g[1:], e[1:], strict=True):
                assert abs(float(gi) - float(ei)) < 1e-6, (case["text"], g, e)
        q = r.flexq(case["text"])
        for k, v in case["flexq"].items():
            assert abs(q[k] - v) < 1e-6, (case["text"], k, q[k], v)


if __name__ == "__main__":
    test_matches_shared_fixture()
    print("ok")
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run python model/test_flexrank.py`
Expected: FAIL — `ModuleNotFoundError: model.flexrank`.

- [ ] **Step 3: Implement `model/flexrank.py`**

Port the algorithm from `tools/data/flexrank.ts` exactly. Key correspondences:
- `query_tokens(text)`: `text.lower()`, `re.sub(r"[^a-z0-9\s]", " ", ...)`, split on whitespace, drop `len < 2` and `STOPWORDS`.
- `fuzzy_match(a, b)`: `len(a) >= 4 and len(b) >= 4 and abs(len(a)-len(b)) <= 3 and (a.startswith(b) or b.startswith(a))`.
- Build from `flex.json`: `glyphs = emojis` (use as-is — vocab already stripped on the JS/TS side via `stripVS`; here `flex.json.emojis` are the original label strings, so also build a `strip_vs` and key the inverted index by stripped form to match `flexrank.ts` which keys by `stripVS(glyph)`). **Copy `stripVS` regex** `[︎️]` → `strip_vs`.
- `kw_tokens[i] = flex.json.keywords[i]` (already lowercased).
- `kw_to_glyphs: dict[str, set[int]]`, `prefix4: dict[str, set[str]]` (key = `kw[:4]` for `len(kw) >= 4`), `kw_count[i] = len(kw_tokens[i])`.
- `idf(w) = flex_json["idf"].get(w, flex_json["idf_default"])`.
- Per text: replicate the `bump` accumulator with fields `score, exact, fuzzy, best_idf, best_exact, kw_len, word_len, overlap`; exact pass then fuzzy pass (fuzzy picks the max-overlap keyword per glyph, `overlap = min(len(w), len(kw))`); skip a glyph's fuzzy contribution if it was an exact hit for the same token.
- `sum` / `max` over all accumulator entries (pre-filter); sort by `score desc, (exact+fuzzy) desc, glyph-index asc`; vocab filter is implicit (only vocab emoji are in `flex.json`); cap 32.
- Round `score`, `score_norm`, `best_idf` to 3 decimals with `round(x, 3)` to match `Number(x.toFixed(3))`.
- `matched` for `flexq`: count query tokens present in the set of all keyword *words* (`w for kw in all keywords for w in kw.split()`).

Implementation (no docstrings/comments):

```python
import json
import math
import re
from pathlib import Path

FUZZY_MIN_LEN = 4
FUZZY_MAX_LEN_DELTA = 3
FUZZY_WEIGHT = 0.6
STOPWORDS = set(
    (
        "a an the to of in on at is it its i you we they he she this that for and or but"
        " not with my your me am are was were be been being do does did have has had"
        " will would can could just so if"
    ).split()
)
_VS = re.compile(r"[︎️]")
_NON = re.compile(r"[^a-z0-9\s]")
_WS = re.compile(r"\s+")


def strip_vs(s: str) -> str:
    return _VS.sub("", s)


def query_tokens(text: str) -> list[str]:
    t = _NON.sub(" ", text.lower())
    return [w for w in _WS.split(t) if len(w) >= 2 and w not in STOPWORDS]


def fuzzy_match(a: str, b: str) -> bool:
    if len(a) < FUZZY_MIN_LEN or len(b) < FUZZY_MIN_LEN:
        return False
    if abs(len(a) - len(b)) > FUZZY_MAX_LEN_DELTA:
        return False
    return a.startswith(b) or b.startswith(a)


def _r3(x: float) -> float:
    return round(x, 3)


class FlexRanker:
    def __init__(self, flex_json_path):
        j = json.loads(Path(flex_json_path).read_text(encoding="utf-8"))
        self.emojis = j["emojis"]
        self.keywords = j["keywords"]
        self._idf = j["idf"]
        self._idf_default = j["idf_default"]
        self.glyphs = [strip_vs(e) for e in self.emojis]
        self.kw_count = [len(k) for k in self.keywords]
        self.kw_to_glyphs: dict[str, set[int]] = {}
        self.prefix4: dict[str, set[str]] = {}
        self._global_kw: set[str] = set()
        for gi, toks in enumerate(self.keywords):
            for kw in toks:
                for w in kw.split():
                    if w:
                        self._global_kw.add(w)
                self.kw_to_glyphs.setdefault(kw, set()).add(gi)
                if len(kw) >= FUZZY_MIN_LEN:
                    self.prefix4.setdefault(kw[:FUZZY_MIN_LEN], set()).add(kw)

    def idf(self, w: str) -> float:
        return self._idf.get(w, self._idf_default)

    def _score(self, text: str):
        q = query_tokens(text)
        acc: dict[int, dict] = {}

        def bump(gi, s, tok_idf, is_exact, word_len, kw_len, overlap):
            e = acc.get(gi)
            if e is None:
                e = {
                    "score": 0.0, "exact": 0, "fuzzy": 0, "best_idf": -1.0,
                    "best_exact": False, "kw_len": 0, "word_len": 0, "overlap": 0,
                }
                acc[gi] = e
            e["score"] += s
            if is_exact:
                e["exact"] += 1
            else:
                e["fuzzy"] += 1
            win = (
                tok_idf > e["best_idf"]
                or (tok_idf == e["best_idf"] and is_exact and not e["best_exact"])
                or (
                    tok_idf == e["best_idf"]
                    and is_exact == e["best_exact"]
                    and overlap > e["overlap"]
                )
            )
            if win:
                e["best_idf"] = tok_idf
                e["best_exact"] = is_exact
                e["kw_len"] = kw_len
                e["word_len"] = word_len
                e["overlap"] = overlap

        for w in q:
            w_idf = self.idf(w)
            exact = self.kw_to_glyphs.get(w)
            if exact:
                for gi in exact:
                    bump(gi, w_idf, w_idf, True, len(w), len(w), len(w))
            if len(w) >= FUZZY_MIN_LEN:
                bucket = self.prefix4.get(w[:FUZZY_MIN_LEN])
                if bucket:
                    fuzz: dict[int, tuple[str, int]] = {}
                    for kw in bucket:
                        if kw == w or not fuzzy_match(w, kw):
                            continue
                        ov = min(len(w), len(kw))
                        for gi in self.kw_to_glyphs[kw]:
                            cur = fuzz.get(gi)
                            if cur is None or ov > cur[1]:
                                fuzz[gi] = (kw, ov)
                    for gi, (kw, ov) in fuzz.items():
                        if exact and gi in exact:
                            continue
                        bump(gi, w_idf * FUZZY_WEIGHT, w_idf, False, len(w), len(kw), ov)

        total = sum(v["score"] for v in acc.values())
        mx = max((v["score"] for v in acc.values()), default=0.0)
        matched = sum(1 for w in q if w in self._global_kw)
        return q, acc, total, mx, matched

    def rank(self, text: str):
        _q, acc, _t, mx, _m = self._score(text)
        ranked = sorted(
            acc.items(),
            key=lambda kv: (-kv[1]["score"], -(kv[1]["exact"] + kv[1]["fuzzy"]), kv[0]),
        )
        out = []
        for gi, v in ranked:
            out.append((
                self.emojis[gi],
                _r3(v["score"]),
                _r3(v["score"] / mx) if mx else 0.0,
                v["exact"],
                v["fuzzy"],
                _r3(v["best_idf"]),
                self.kw_count[gi],
                v["kw_len"],
                v["word_len"],
                v["overlap"],
            ))
            if len(out) >= 32:
                break
        return out

    def flexq(self, text: str) -> dict:
        q, _acc, total, mx, matched = self._score(text)
        return {
            "tokens": len(q),
            "matched": matched,
            "sum": _r3(total),
            "max": _r3(mx),
            "cand": len(_acc),
        }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run python model/test_flexrank.py`
Expected: `ok`.

Run: `uv run ruff check model/flexrank.py model/test_flexrank.py && uv run ruff format --check model/flexrank.py model/test_flexrank.py`
Expected: clean.

- [ ] **Step 5: Add `files.py` constant**

In `files.py`: `FLEX_JSON = f"{WEB_PUBLIC_DIR}/flex.json"` and `FUSION_PT = f"{PT_DIR}/fusion.pt"`.

- [ ] **Step 6: Commit**

```bash
git add model/flexrank.py model/test_flexrank.py files.py
git commit -m "feat: model/flexrank.py — Python flex ranker over flex.json

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01NZUMgpJ5J1UKZiV76KUbR6"
```

---

### Task 4: `web/src/flexrank.js`

**Files:**
- Create: `web/src/flexrank.js`
- Create: `web/src/flexrank.test.js`
- Delete: `web/src/cldrEmojis.ts`, `web/src/data/cldr-emoji-index.json`

**Interfaces:**
- Consumes: `flex.json` object (parsed), `web/src/flexrank.fixture.json`.
- Produces:
  - `makeFlexRanker(flexJson) -> { rank(text), flexRaw(text), flexQ(text) }`
    - `rank(text)` → array of `[emoji, score, score_norm, exact, fuzzy, best_idf, n_kw, kw_len, word_len, overlap]`, sorted, cap 32 (parity with fixture).
    - `flexRaw(text)` → `Float32Array(V * 10)` — per vocab emoji, the 10-vector incl. `rank_recip = 1/(pos+1)` at index 5 (see layout note), zeros for absent. **Layout for the model input** is `[score, score_norm, exact, fuzzy, best_idf, rank_recip, n_kw, kw_len, word_len, overlap]` — note this **reorders** `best_idf`/`rank_recip`: `rank_recip` replaces the ranker's raw `rank`-less slot. Keep `rank()` output in the *ranker's* 10-field order for the fixture; `flexRaw()` emits the *model* order.
    - `flexQ(text)` → `Float32Array([tokens, matched, sum, max, cand])`.

  > Layout reconciliation: the ranker's per-entry tuple is
  > `[emoji, score, score_norm, exact, fuzzy, best_idf, n_kw, kw_len, word_len, overlap]` (10 incl. emoji).
  > The **model** per-emoji vector is 10 numeric fields:
  > `[score, score_norm, exact, fuzzy, best_idf, rank_recip, n_kw, kw_len, word_len, overlap]`
  > where `rank_recip = 1/(list_position + 1)`. `data.py`, `flexrank.py` callers, and `flexrank.js`
  > all build the model vector by inserting `rank_recip` after `best_idf`.

- [ ] **Step 1: Write the failing test** — `web/src/flexrank.test.js`

```js
import { describe, it, expect } from 'vitest'
import fixture from './flexrank.fixture.json'
import flexJson from '../public/flex.json'
import { makeFlexRanker } from './flexrank'

describe('flexrank parity', () => {
  const r = makeFlexRanker(flexJson)
  for (const c of fixture.cases) {
    it(`matches fixture: ${JSON.stringify(c.text)}`, () => {
      const got = r.rank(c.text)
      expect(got.length).toBe(c.flexsearch.length)
      got.forEach((row, i) => {
        const exp = c.flexsearch[i]
        expect(row[0]).toBe(exp[0])
        for (let k = 1; k < row.length; k++) expect(row[k]).toBeCloseTo(exp[k], 6)
      })
      const q = r.flexQ(c.text)
      expect([...q]).toEqual([
        c.flexq.tokens, c.flexq.matched,
        expect.closeTo(c.flexq.sum, 6), expect.closeTo(c.flexq.max, 6), c.flexq.cand,
      ])
    })
  }

  it('flexRaw is V*10 and puts rank_recip after best_idf', () => {
    const raw = r.flexRaw('cheese pizza time')
    expect(raw.length).toBe(flexJson.emojis.length * 10)
    const pizza = flexJson.emojis.findIndex((e) => e === '🍕')
    expect(raw[pizza * 10 + 5]).toBeCloseTo(1, 6) // rank_recip of the #1 hit
  })
})
```

- [ ] **Step 2: Run to verify it fails**

Run: `cd web && npx vitest run src/flexrank.test.js`
Expected: FAIL — `makeFlexRanker` not found.

- [ ] **Step 3: Implement `web/src/flexrank.js`**

Port `model/flexrank.py` step 3 to JS (same algorithm, same rounding — `Number(x.toFixed(3))`). Skeleton:

```js
const FUZZY_MIN_LEN = 4
const FUZZY_MAX_LEN_DELTA = 3
const FUZZY_WEIGHT = 0.6
const STOPWORDS = new Set(
  ('a an the to of in on at is it its i you we they he she this that for and or but' +
    ' not with my your me am are was were be been being do does did have has had' +
    ' will would can could just so if').split(' '),
)
const VS = /[︎️]/g
const r3 = (x) => Number(x.toFixed(3))
const stripVS = (s) => s.replace(VS, '')

function queryTokens(text) {
  return text
    .toLowerCase()
    .replace(/[^a-z0-9\s]/g, ' ')
    .split(/\s+/)
    .filter((w) => w.length >= 2 && !STOPWORDS.has(w))
}
function fuzzyMatch(a, b) {
  if (a.length < FUZZY_MIN_LEN || b.length < FUZZY_MIN_LEN) return false
  if (Math.abs(a.length - b.length) > FUZZY_MAX_LEN_DELTA) return false
  return a.startsWith(b) || b.startsWith(a)
}

export function makeFlexRanker(flexJson) {
  const emojis = flexJson.emojis
  const keywords = flexJson.keywords
  const idfMap = flexJson.idf
  const idfDefault = flexJson.idf_default
  const kwCount = keywords.map((k) => k.length)
  const kwToGlyphs = new Map()
  const prefix4 = new Map()
  const globalKw = new Set()
  keywords.forEach((toks, gi) => {
    for (const kw of toks) {
      for (const w of kw.split(/\s+/)) if (w) globalKw.add(w)
      let s = kwToGlyphs.get(kw)
      if (!s) kwToGlyphs.set(kw, (s = new Set()))
      s.add(gi)
      if (kw.length >= FUZZY_MIN_LEN) {
        const p = kw.slice(0, FUZZY_MIN_LEN)
        let b = prefix4.get(p)
        if (!b) prefix4.set(p, (b = new Set()))
        b.add(kw)
      }
    }
  })
  const idf = (w) => (w in idfMap ? idfMap[w] : idfDefault)

  function scoreAll(text) {
    const q = queryTokens(text)
    const acc = new Map()
    const bump = (gi, s, tokIdf, isExact, wLen, kLen, ov) => {
      let e = acc.get(gi)
      if (!e) {
        e = { score: 0, exact: 0, fuzzy: 0, bestIdf: -1, bestExact: false, kwLen: 0, wordLen: 0, overlap: 0 }
        acc.set(gi, e)
      }
      e.score += s
      if (isExact) e.exact += 1
      else e.fuzzy += 1
      const win =
        tokIdf > e.bestIdf ||
        (tokIdf === e.bestIdf && isExact && !e.bestExact) ||
        (tokIdf === e.bestIdf && isExact === e.bestExact && ov > e.overlap)
      if (win) {
        e.bestIdf = tokIdf
        e.bestExact = isExact
        e.kwLen = kLen
        e.wordLen = wLen
        e.overlap = ov
      }
    }
    for (const w of q) {
      const wIdf = idf(w)
      const exact = kwToGlyphs.get(w)
      if (exact) for (const gi of exact) bump(gi, wIdf, wIdf, true, w.length, w.length, w.length)
      if (w.length >= FUZZY_MIN_LEN) {
        const bucket = prefix4.get(w.slice(0, FUZZY_MIN_LEN))
        if (bucket) {
          const fuzz = new Map()
          for (const kw of bucket) {
            if (kw === w || !fuzzyMatch(w, kw)) continue
            const ov = Math.min(w.length, kw.length)
            for (const gi of kwToGlyphs.get(kw)) {
              const cur = fuzz.get(gi)
              if (!cur || ov > cur.ov) fuzz.set(gi, { kw, ov })
            }
          }
          for (const [gi, m] of fuzz) {
            if (exact && exact.has(gi)) continue
            bump(gi, wIdf * FUZZY_WEIGHT, wIdf, false, w.length, m.kw.length, m.ov)
          }
        }
      }
    }
    let sum = 0
    let max = 0
    for (const v of acc.values()) {
      sum += v.score
      if (v.score > max) max = v.score
    }
    const matched = q.filter((w) => globalKw.has(w)).length
    return { q, acc, sum, max, matched }
  }

  function rankedEntries(text) {
    const { acc, max } = scoreAll(text)
    return [...acc.entries()]
      .sort(
        (a, b) =>
          b[1].score - a[1].score ||
          (b[1].exact + b[1].fuzzy) - (a[1].exact + a[1].fuzzy) ||
          a[0] - b[0],
      )
      .slice(0, 32)
      .map(([gi, v], pos) => ({ gi, v, pos, max }))
  }

  return {
    rank(text) {
      return rankedEntries(text).map(({ gi, v, max }) => [
        emojis[gi],
        r3(v.score),
        max ? r3(v.score / max) : 0,
        v.exact,
        v.fuzzy,
        r3(v.bestIdf),
        kwCount[gi],
        v.kwLen,
        v.wordLen,
        v.overlap,
      ])
    },
    flexRaw(text) {
      const out = new Float32Array(emojis.length * 10)
      for (const { gi, v, pos, max } of rankedEntries(text)) {
        const o = gi * 10
        out[o + 0] = r3(v.score)
        out[o + 1] = max ? r3(v.score / max) : 0
        out[o + 2] = v.exact
        out[o + 3] = v.fuzzy
        out[o + 4] = r3(v.bestIdf)
        out[o + 5] = 1 / (pos + 1)
        out[o + 6] = kwCount[gi]
        out[o + 7] = v.kwLen
        out[o + 8] = v.wordLen
        out[o + 9] = v.overlap
      }
      return out
    },
    flexQ(text) {
      const { q, acc, sum, max, matched } = scoreAll(text)
      return Float32Array.from([q.length, matched, r3(sum), r3(max), acc.size])
    },
  }
}
```

- [ ] **Step 4: Delete the old CLDR keyword path**

```bash
git rm web/src/cldrEmojis.ts web/src/data/cldr-emoji-index.json
```

Remove the `import { cldrEmojis } from './cldrEmojis'` line from `App.jsx` (App.jsx is rewired in Task 12; leaving the import now breaks the build, so do the App.jsx edits from Task 12's Step "rewire" here if executing in order, or keep this delete paired with Task 12). **Ordering note:** run Task 12 immediately after this task, or temporarily comment the import.

- [ ] **Step 5: Run tests**

Run: `cd web && npx vitest run src/flexrank.test.js`
Expected: PASS for every fixture case + the `flexRaw` layout test.

- [ ] **Step 6: Commit**

```bash
git add web/src/flexrank.js web/src/flexrank.test.js
git commit -m "feat: web/src/flexrank.js — browser flex ranker (replaces cldrEmojis)

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01NZUMgpJ5J1UKZiV76KUbR6"
```

---

## Phase 2 — model, config, data

### Task 5: config

**Files:**
- Modify: `model/config.py`

**Interfaces:**
- Produces: `FUSION_HIDDEN = 48`, `DROPOUT_FUSION = 0.1`, `FUSION_INT_CLAMP = 16`, `FUSION_INT_EMBED_SIZE = 8`; folded into `CONFIG_NAME`.

- [ ] **Step 1: Add the constants**

In `model/config.py`, next to `EMOJI_EMBED_SIZE` / `DROPOUT_EMOJI`:

```python
FUSION_HIDDEN = 48
DROPOUT_FUSION = 0.1
FUSION_INT_CLAMP = 16
FUSION_INT_EMBED_SIZE = 8
```

Find the `CONFIG_NAME` assembly (the `emj_str`-style hyperparam string) and append a fusion fragment:

```python
fus_str = " ".join(
    str(p) for p in (FUSION_HIDDEN, DROPOUT_FUSION, FUSION_INT_CLAMP, FUSION_INT_EMBED_SIZE)
)
```

and include `fus_str` in the `CONFIG_NAME` f-string next to the other `*_str` fragments.

- [ ] **Step 2: Verify it imports + prints**

Run: `uv run python model/config.py`
Expected: the model-stats table prints without error; `CONFIG_NAME` (top of output or via `python -c`) contains the four fusion values.

Run: `uv run ruff check model/config.py && uv run ruff format --check model/config.py`
Expected: clean.

- [ ] **Step 3: Commit**

```bash
git add model/config.py
git commit -m "feat: FusionHead hyperparameters in config

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01NZUMgpJ5J1UKZiV76KUbR6"
```

---

### Task 6: `FusionHead` + `fusion_features`

**Files:**
- Modify: `model/model.py`
- Create: `model/test_model_fusion.py`

**Interfaces:**
- Consumes: `FUSION_*` from `model.config`; `RELU_SLOPE`; `EMOJIS` (for `V`).
- Produces:
  - `fusion_features(emoji_logit, flex_raw, flexq) -> tuple[Tensor, Tensor]`
    - `emoji_logit`: `[B, V]`; `flex_raw`: `[B, V, 10]`
      (`[score, score_norm, exact, fuzzy, best_idf, rank_recip, n_kw, kw_len, word_len, overlap]`);
      `flexq`: `[B, 5]` (`[tokens, matched, sum, max, cand]`).
    - returns `(scalars [B, V, 18], int_idx [B, V, 4] long)`.
  - `class FusionHead(nn.Module)` with `forward(emoji_logit, flex_raw, flexq) -> [B, V]` (fused logits).

- [ ] **Step 1: Write the failing test** — `model/test_model_fusion.py`

```python
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from model.model import FusionHead, fusion_features
from model.data import EMOJIS

V = len(EMOJIS)


def _inputs(b=4):
    logit = torch.randn(b, V)
    raw = torch.zeros(b, V, 10)
    raw[:, :5, 0] = torch.rand(b, 5) + 0.1          # score > 0 for first 5 -> present
    raw[:, :5, 1] = torch.rand(b, 5)
    raw[:, :5, 2] = 1.0
    raw[:, :5, 5] = torch.tensor([1.0, 0.5, 1 / 3, 0.25, 0.2])
    raw[:, :5, 6:] = torch.randint(0, 40, (b, 5, 4)).float()
    q = torch.tensor([[3.0, 2.0, 12.0, 6.0, 8.0]]).repeat(b, 1)
    return logit, raw, q


def test_feature_shapes():
    logit, raw, q = _inputs()
    scal, idx = fusion_features(logit, raw, q)
    assert scal.shape == (4, V, 18)
    assert idx.shape == (4, V, 4)
    assert idx.dtype == torch.long
    assert idx.min() >= 0 and idx.max() <= 16


def test_zero_init_is_identity():
    torch.manual_seed(0)
    head = FusionHead()
    with torch.no_grad():
        head.net[-1].weight.zero_()
        head.net[-1].bias.zero_()
    logit, raw, q = _inputs()
    out = head(logit, raw, q)
    assert torch.allclose(out, logit, atol=1e-5)


def test_forward_shape_and_grad():
    head = FusionHead()
    logit, raw, q = _inputs()
    out = head(logit, raw, q)
    assert out.shape == (4, V)
    out.sum().backward()
    assert head.net[0].weight.grad is not None


if __name__ == "__main__":
    test_feature_shapes()
    test_zero_init_is_identity()
    test_forward_shape_and_grad()
    print("ok")
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run python model/test_model_fusion.py`
Expected: FAIL — `ImportError: cannot import name 'FusionHead'`.

- [ ] **Step 3: Implement in `model/model.py`**

Add imports: `FUSION_HIDDEN, DROPOUT_FUSION, FUSION_INT_CLAMP, FUSION_INT_EMBED_SIZE` from `model.config`.

```python
_EPS = 1e-6


def fusion_features(emoji_logit, flex_raw, flexq):
    score = flex_raw[..., 0]
    score_norm = flex_raw[..., 1]
    exact = flex_raw[..., 2]
    fuzzy = flex_raw[..., 3]
    best_idf = flex_raw[..., 4]
    rank_recip = flex_raw[..., 5]
    n_kw = flex_raw[..., 6]
    kw_len = flex_raw[..., 7]
    word_len = flex_raw[..., 8]
    overlap = flex_raw[..., 9]
    present = (score > 0).float()

    q = flexq.unsqueeze(1)
    tokens = q[..., 0]
    matched = q[..., 1]
    qsum = q[..., 2]
    qmax = q[..., 3]
    qcand = q[..., 4]

    scal = torch.stack(
        [
            emoji_logit,
            present,
            torch.log1p(score.clamp(min=0.0)),
            score_norm,
            score / (qsum + _EPS),
            exact,
            fuzzy,
            exact / (tokens + _EPS),
            fuzzy / (tokens + _EPS),
            best_idf / 10.0,
            rank_recip,
            overlap / (word_len + _EPS),
            tokens.expand_as(present),
            matched.expand_as(present),
            (matched / (tokens + _EPS)).expand_as(present),
            torch.log1p(qsum.clamp(min=0.0)).expand_as(present),
            qmax.expand_as(present),
            torch.log1p(qcand.clamp(min=0.0)).expand_as(present),
        ],
        dim=-1,
    )
    ints = torch.stack([n_kw, kw_len, word_len, overlap], dim=-1)
    idx = ints.round().clamp(0, FUSION_INT_CLAMP).long()
    return scal, idx


class FusionHead(nn.Module):
    def __init__(self):
        super().__init__()
        c = FUSION_INT_CLAMP + 1
        e = FUSION_INT_EMBED_SIZE
        self.emb_nkw = nn.Embedding(c, e)
        self.emb_kwlen = nn.Embedding(c, e)
        self.emb_wlen = nn.Embedding(c, e)
        self.emb_ovl = nn.Embedding(c, e)
        self.net = nn.Sequential(
            nn.Linear(18 + 4 * e, FUSION_HIDDEN),
            nn.LeakyReLU(negative_slope=RELU_SLOPE),
            nn.Dropout(p=DROPOUT_FUSION),
            nn.Linear(FUSION_HIDDEN, 1),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, emoji_logit, flex_raw, flexq):
        scal, idx = fusion_features(emoji_logit, flex_raw, flexq)
        emb = torch.cat(
            [
                self.emb_nkw(idx[..., 0]),
                self.emb_kwlen(idx[..., 1]),
                self.emb_wlen(idx[..., 2]),
                self.emb_ovl(idx[..., 3]),
            ],
            dim=-1,
        )
        g = self.net(torch.cat([scal, emb], dim=-1)).squeeze(-1)
        return emoji_logit + g
```

- [ ] **Step 4: Run to verify it passes**

Run: `uv run python model/test_model_fusion.py`
Expected: `ok`.

Run: `uv run ruff check model/model.py model/test_model_fusion.py && uv run ruff format --check model/model.py model/test_model_fusion.py`
Expected: clean.

- [ ] **Step 5: Commit**

```bash
git add model/model.py model/test_model_fusion.py
git commit -m "feat: FusionHead residual re-ranker + fusion_features transform

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01NZUMgpJ5J1UKZiV76KUbR6"
```

---

### Task 7: data loading — sparse flex tensors + `scatter_flex`

**Files:**
- Modify: `model/data.py`
- Create: `model/test_data_flex.py`

**Interfaces:**
- Consumes: train/eval jsonl rows carrying `flexsearch` (list of `[emoji, score, score_norm, exact, fuzzy, best_idf, n_kw, kw_len, word_len, overlap]`) and `flexq` (`{tokens,matched,sum,max,cand}`).
- Produces:
  - `FLEX_MAX_K = 32`, `FLEX_RAW_DIM = 10`, `FLEXQ_DIM = 5` module consts.
  - `record` gains `flexsearch: list` and `flexq: dict` fields.
  - `EmojiDataset.__getitem__` returns `(text, emoji, style, colors, flex_idx, flex_raw, flexq)`:
    - `flex_idx`: `long[FLEX_MAX_K]`, vocab index per retrieved (in-vocab) emoji in list order, `-1` pad.
    - `flex_raw`: `float32[FLEX_MAX_K, FLEX_RAW_DIM]`, the **model-order** vector (`rank_recip` inserted after `best_idf`), zero pad.
    - `flexq`: `float32[FLEXQ_DIM]`.
  - `scatter_flex(flex_idx, flex_raw) -> Tensor [B, V, FLEX_RAW_DIM]` — dense, zeros where absent.

- [ ] **Step 1: Write the failing test** — `model/test_data_flex.py`

```python
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from model.data import (
    EMOJIS,
    FLEX_MAX_K,
    FLEX_RAW_DIM,
    FLEXQ_DIM,
    _row_flex,
    scatter_flex,
)

V = len(EMOJIS)


def test_row_flex_builds_model_order_vector():
    e0, e1 = EMOJIS[0], EMOJIS[1]
    row = {
        "flexsearch": [
            [e0, 5.0, 1.0, 1, 0, 7.0, 9, 5, 5, 5],
            [e1, 2.5, 0.5, 0, 1, 8.5, 17, 6, 5, 5],
        ],
        "flexq": {"tokens": 3, "matched": 2, "sum": 7.5, "max": 5.0, "cand": 2},
    }
    idx, raw, q = _row_flex(row)
    assert idx.shape == (FLEX_MAX_K,) and raw.shape == (FLEX_MAX_K, FLEX_RAW_DIM)
    assert q.tolist() == [3.0, 2.0, 7.5, 5.0, 2.0]
    assert idx[0].item() == 0 and idx[1].item() == 1 and idx[2].item() == -1
    # model order: [score, score_norm, exact, fuzzy, best_idf, rank_recip, n_kw, kw_len, word_len, overlap]
    assert raw[0].tolist() == [5.0, 1.0, 1.0, 0.0, 7.0, 1.0, 9.0, 5.0, 5.0, 5.0]
    assert abs(raw[1][5].item() - 0.5) < 1e-6  # rank_recip = 1/(pos+1) = 1/2


def test_row_flex_missing_fields():
    idx, raw, q = _row_flex({})
    assert idx.tolist() == [-1] * FLEX_MAX_K
    assert raw.abs().sum().item() == 0.0
    assert q.tolist() == [0.0] * FLEXQ_DIM


def test_scatter_flex_places_rows_by_index():
    idx = torch.full((2, FLEX_MAX_K), -1, dtype=torch.long)
    idx[0, 0] = 3
    idx[1, 0] = 5
    idx[1, 1] = 3
    raw = torch.zeros(2, FLEX_MAX_K, FLEX_RAW_DIM)
    raw[0, 0, 0] = 9.0
    raw[1, 0, 0] = 4.0
    raw[1, 1, 0] = 2.0
    dense = scatter_flex(idx, raw)
    assert dense.shape == (2, V, FLEX_RAW_DIM)
    assert dense[0, 3, 0].item() == 9.0
    assert dense[1, 5, 0].item() == 4.0
    assert dense[1, 3, 0].item() == 2.0
    assert dense[0, 0, 0].item() == 0.0


if __name__ == "__main__":
    test_row_flex_builds_model_order_vector()
    test_row_flex_missing_fields()
    test_scatter_flex_places_rows_by_index()
    print("ok")
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run python model/test_data_flex.py`
Expected: FAIL — `ImportError` for `FLEX_MAX_K` / `_row_flex` / `scatter_flex`.

- [ ] **Step 3: Implement in `model/data.py`**

Add near the top consts:

```python
FLEX_MAX_K = 32
FLEX_RAW_DIM = 10
FLEXQ_DIM = 5
```

`record` gains fields (dataclass): `flexsearch: list` and `flexq: dict` with `field(default_factory=...)` — or just append two positional fields and update the one `record(...)` construction in `read()`.

In `read()`'s `match d` case, after building `emojis`/`styles`, capture the extras:

```python
                flexsearch = d.get("flexsearch") or []
                flexq = d.get("flexq") or {}
                yield record(text, emojis, styles, [*bg, fg], flexsearch, flexq)
```

`_row_flex`:

```python
_FLEXQ_KEYS = ("tokens", "matched", "sum", "max", "cand")


def _row_flex(row: dict):
    idx = torch.full((FLEX_MAX_K,), -1, dtype=torch.long)
    raw = torch.zeros(FLEX_MAX_K, FLEX_RAW_DIM, dtype=torch.float32)
    pos = 0
    for entry in row.get("flexsearch") or []:
        if pos >= FLEX_MAX_K:
            break
        emoji = entry[0]
        vi = emoji2idx.get(emoji)
        if vi is None:
            continue
        score, score_norm, exact, fuzzy, best_idf, n_kw, kw_len, word_len, overlap = entry[1:]
        idx[pos] = vi
        raw[pos] = torch.tensor(
            [
                score, score_norm, exact, fuzzy, best_idf,
                1.0 / (pos + 1),
                n_kw, kw_len, word_len, overlap,
            ],
            dtype=torch.float32,
        )
        pos += 1
    q = row.get("flexq") or {}
    flexq = torch.tensor(
        [float(q.get(k, 0.0)) for k in _FLEXQ_KEYS], dtype=torch.float32
    )
    return idx, raw, flexq
```

> Note: `rank_recip = 1/(pos+1)` uses `pos` — the **kept** (in-vocab) position, which
> matches the fixture / JS only when no out-of-vocab emoji precede it in the list. Since
> `flex.json` and `regen`'s vocab filter are the same vocab, the training jsonl's
> `flexsearch` lists are already vocab-filtered — every entry is in `emoji2idx` and `pos`
> equals the list index. The `vi is None` skip is a guard for a stale jsonl; if it ever
> fires, `rank_recip` shifts by one for later entries (acceptable, logged nowhere).

`scatter_flex`:

```python
def scatter_flex(flex_idx: torch.Tensor, flex_raw: torch.Tensor) -> torch.Tensor:
    b = flex_idx.size(0)
    v = len(EMOJIS)
    dense = flex_raw.new_zeros(b, v, FLEX_RAW_DIM)
    safe = flex_idx.clamp(min=0)
    mask = (flex_idx >= 0).unsqueeze(-1)
    src = flex_raw * mask
    dense.scatter_add_(1, safe.unsqueeze(-1).expand(-1, -1, FLEX_RAW_DIM), src)
    return dense
```

(`scatter_add_` with the `-1`→`0` clamp + masked src means pad slots add zero to row 0; a real entry at vocab-index 0 still lands correctly. Distinct retrieved emoji never collide.)

`EmojiDataset.__init__`: stack the three new tensors:

```python
        flex = [_row_flex(_raw_row(r)) for r in records]
```

— but `records` are `record` dataclasses, not dicts. Add the two fields to `record` and build from them:

```python
        self.flex_idx = torch.stack([_row_flex_from_record(r)[0] for r in records])
```

Simpler: make `_row_flex` accept the `record`:

```python
def _row_flex(r):
    fs = getattr(r, "flexsearch", None) or []
    fq = getattr(r, "flexq", None) or {}
    ...
```

and adapt the test's dict inputs by having `_row_flex` accept either a dict or an object (`fs = r["flexsearch"] if isinstance(r, dict) else r.flexsearch`). Keep the test as written.

`__getitem__` returns the 7-tuple; `eval_data_loader` / `train_data_loader` unchanged (default collate stacks fine).

- [ ] **Step 4: Run to verify it passes**

Run: `uv run python model/test_data_flex.py`
Expected: `ok`.

Run: `bun run regen && uv run python -c "from model.data import train_ds; ds=train_ds(); x=ds[0]; print(len(x), x[4].shape, x[5].shape, x[6].shape)"`
Expected: `7 torch.Size([32]) torch.Size([32, 10]) torch.Size([5])`.

Run: `uv run ruff check model/data.py model/test_data_flex.py && uv run ruff format --check model/data.py model/test_data_flex.py`
Expected: clean.

- [ ] **Step 5: Commit**

```bash
git add model/data.py model/test_data_flex.py
git commit -m "feat: load flexsearch/flexq into sparse per-row tensors + scatter_flex

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01NZUMgpJ5J1UKZiV76KUbR6"
```

---

## Phase 3 — training

### Task 8: `fusion` head in `LitEncoder`

**Files:**
- Modify: `model/train.py`
- Modify: `model/test_train_cli.py`

**Interfaces:**
- Consumes: `FusionHead` (Task 6), `scatter_flex` (Task 7), `EMOJIS`.
- Produces:
  - `ALL_HEADS = ("style", "emoji", "critic", "fusion")`.
  - `_parse_heads` rejects a set containing `fusion` but not `emoji`.
  - `LitEncoder._step` logs `loss/fusion/{split}`, `MRR/fusion/{split}`, `MRR/flex/{split}`; existing `MRR/e/{split}` unchanged.
  - `_train_encoder` monitor: `"MRR/fusion/val"` when `"fusion" in heads`.
  - `_train_encoder` writes `fusion.pt` (stage `"enc"`).

- [ ] **Step 1: Write failing CLI tests** — add to `model/test_train_cli.py`

```python
def test_heads_fusion_requires_emoji():
    from model.train import _parse_heads
    import typer

    assert _parse_heads("emoji,fusion") == ("emoji", "fusion")
    try:
        _parse_heads("style,fusion")
    except typer.BadParameter:
        pass
    else:
        raise AssertionError("fusion without emoji should raise")


def test_all_heads_includes_fusion():
    from model.train import ALL_HEADS

    assert ALL_HEADS == ("style", "emoji", "critic", "fusion")
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run python model/test_train_cli.py`
Expected: FAIL — `ALL_HEADS` lacks `fusion`; `_parse_heads` has no emoji-dependency check.

- [ ] **Step 3: Implement in `model/train.py`**

- `ALL_HEADS: tuple[str, ...] = ("style", "emoji", "critic", "fusion")`.
- In `_parse_heads`, after the `bad` check:

```python
    if "fusion" in got and "emoji" not in got:
        raise typer.BadParameter("--heads: fusion requires emoji")
```

- Imports: `from model.model import ... FusionHead`; `from model.data import ... scatter_flex`.
- `LitEncoder.__init__`: `if "fusion" in self.heads: self.fusion = FusionHead()`. Add `self._val_frr: list = []`, `self._trn_frr: list = []` (fusion RR buffers) and `self._val_xrr`, `self._trn_xrr` (flex RR buffers) alongside the existing `_val_rr`.
- `_step` signature already unpacks `text, emoji, style, colors = batch`; change to
  `text, emoji, style, colors, flex_idx, flex_raw, flexq = batch`.
- After the `"emoji"` block computes `emoji_logits` and its MRR, add:

```python
        if "fusion" in self.heads:
            flex_dense = scatter_flex(flex_idx, flex_raw)
            fusion_logits = self.fusion(emoji_logits.detach(), flex_dense, flexq)
            loss_fusion = lse_infonce(fusion_logits, emoji, INFONCE_TEMP)
            loss = loss + loss_fusion
            self._log(f"loss/fusion/{split}", loss_fusion, bs)

            flex_logits = torch.where(
                flex_dense[..., 0] > 0, flex_dense[..., 0], flex_dense.new_full((), -1e9)
            )
            has_e = emoji.sum(dim=-1) > 0
            n_e = int(has_e.sum())
            if n_e:
                frr = mrr(fusion_logits[has_e], emoji[has_e])
                xrr = mrr(flex_logits[has_e], emoji[has_e])
                (self._val_frr if split == "val" else self._trn_frr).append(frr.detach())
                (self._val_xrr if split == "val" else self._trn_xrr).append(xrr.detach())
                self._log(f"MRR/fusion/{split}", frr.mean(), n_e)
                self._log(f"MRR/flex/{split}", xrr.mean(), n_e)
```

- `on_*_epoch_start` / `on_*_epoch_end`: clear + (optionally) aggregate the new buffers. Minimal: in `on_validation_epoch_start` add `self._val_frr.clear(); self._val_xrr.clear()` and the train counterparts; in `_epoch_metrics`, if `"fusion" in self.heads and rr_buf` you may additionally `self.log(f"MRR/fusion/{split}", torch.cat(self._val_frr).mean())` for a clean epoch value — but the per-step `_log(on_epoch=True)` already produces an epoch mean, so this is optional. Keep it minimal: just clear.
- `configure_optimizers`: the existing `for h in self.heads: params += getattr(self, h).parameters()` already picks up `fusion`. No change.
- `_train_encoder` monitor:

```python
    monitor = (
        "MRR/fusion/val"
        if "fusion" in heads
        else "F1/val"
        if {"emoji", "critic"} <= set(heads)
        else "MRR/e/val"
        if "emoji" in heads
        else "MRR/s/val"
        if "style" in heads
        else "auc/critic/val"
    )
```

- `_train_encoder` save loop: `ALL_HEADS` now includes `fusion`, so the existing
  `for h in ALL_HEADS: if h in heads: save_pt(getattr(mod, h).state_dict(), out_dir/f"{h}.pt", stage="enc")`
  writes `fusion.pt` automatically. Verify the loop uses `ALL_HEADS` (it does).

- [ ] **Step 4: Run to verify CLI tests pass**

Run: `uv run python model/test_train_cli.py`
Expected: PASS.

Run: `uv run python model/test_runmeta.py && uv run ruff check model/train.py && uv run ruff format --check model/train.py`
Expected: clean.

- [ ] **Step 5: Smoke — one short local epoch**

Requires a clean tree, so commit first (Step 6), then:

Run: `bun run regen && EMOJIC_TASK_BATCH_SIZE=256 uv run python model/train.py enc --local --heads emoji,fusion` — let it reach the first validation, then Ctrl-C.
Expected: TensorBoard `runs/<CONFIG_NAME>/enc` shows `MRR/e/val`, `MRR/flex/val`, `MRR/fusion/val`, `loss/fusion/*`. No shape errors. `MRR/e/val` numerically unchanged vs a `--heads emoji` run (detach works).

- [ ] **Step 6: Commit** (before the smoke, then amend if needed)

```bash
git add model/train.py model/test_train_cli.py
git commit -m "feat: fusion head in LitEncoder — loss, MRR/e|flex|fusion, monitor

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01NZUMgpJ5J1UKZiV76KUbR6"
```

---

### Task 9: GAN-stage plumbing for `fusion.pt`

**Files:**
- Modify: `model/train.py`

**Interfaces:**
- Consumes: `FUSION_PT` from `files.py`.
- Produces: `gan` stage requires `fusion.pt`; Modal `gan` dispatch uploads it.

- [ ] **Step 1: Add to the local gan requirement**

In `_run_local`, the `Stage.gan` branch:

```python
        _require_pt(pt_dir, ["enc.pt", "critic.pt", "style.pt", "emoji.pt", "fusion.pt"])
```

- [ ] **Step 2: Add to the Modal gan upload**

- `from files import ... FUSION_PT`.
- `_run_remote`: in the `stage == "gan"` branch add `FUSION_PT` to the existence check loop and `"fusion_bytes": Path(FUSION_PT).read_bytes()` to `pt_bytes`; add `"fusion_bytes": None` to the default dict.
- `train_remote` signature: add `fusion_bytes: bytes | None = None`; add `FUSION_PT: fusion_bytes` to the `uploads` dict.

- [ ] **Step 3: Verify CLI still parses**

Run: `uv run python model/test_train_cli.py && uv run python -m modal --help >/dev/null && uv run ruff check model/train.py`
Expected: clean; `import model.train` succeeds (`uv run python -c "import model.train"`).

- [ ] **Step 4: Commit**

```bash
git add model/train.py files.py
git commit -m "feat: gan stage requires + uploads fusion.pt

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01NZUMgpJ5J1UKZiV76KUbR6"
```

---

## Phase 4 — export + web

### Task 10: ONNX export with `fusion_logits`

**Files:**
- Modify: `model/export_onnx.py`
- Create: `model/test_export_onnx.py`

**Interfaces:**
- Consumes: `FusionHead`, `FUSION_PT`, `fusion_features` (via `FusionHead`), `len(EMOJIS)`.
- Produces:
  - `ExportWrapper.forward(x, flex, flex_q)` → `(style_logits, emoji_logits, fusion_logits, color)`.
  - `web/public/model.onnx` with inputs `["input", "flex", "flex_q"]`, outputs `["style_logits", "emoji_logits", "fusion_logits", "color"]`.
  - `web/public/meta.json` gains `"flex_cols"` (10-name list) and `"flex_k"` (32).

- [ ] **Step 1: Write the failing test** — `model/test_export_onnx.py`

```python
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import onnxruntime as ort

from files import EMOJI_PT, ENC_PT, FUSION_PT, GEN_PT, STYLE_PT
from model.config import MAX_TEXT_LEN
from model.data import EMOJIS


def test_onnx_has_fusion_io(tmp_path=Path("web/public")):
    for p in (ENC_PT, STYLE_PT, EMOJI_PT, GEN_PT, FUSION_PT):
        if not Path(p).exists():
            print(f"skip: {p} missing")
            return
    from model.export_onnx import export

    export()
    sess = ort.InferenceSession("web/public/model.onnx")
    names_in = {i.name for i in sess.get_inputs()}
    names_out = {o.name for o in sess.get_outputs()}
    assert names_in == {"input", "flex", "flex_q"}, names_in
    assert "fusion_logits" in names_out and "emoji_logits" in names_out

    v = len(EMOJIS)
    out = sess.run(
        None,
        {
            "input": np.zeros((1, MAX_TEXT_LEN), dtype=np.int64),
            "flex": np.zeros((1, v, 10), dtype=np.float32),
            "flex_q": np.zeros((1, 5), dtype=np.float32),
        },
    )
    by = {o.name: r for o, r in zip(sess.get_outputs(), out, strict=True)}
    # zero flex -> present=0, and zero-init last layer -> fusion == emoji
    assert np.allclose(by["fusion_logits"], by["emoji_logits"], atol=1e-4)


if __name__ == "__main__":
    test_onnx_has_fusion_io()
    print("ok")
```

- [ ] **Step 2: Run to verify it fails / skips**

Run: `uv run python model/test_export_onnx.py`
Expected: either `skip: pt/fusion.pt missing` (before a full train) or FAIL on the IO assertion. If it prints `skip`, that is an acceptable red state for this step — proceed to implement; the smoke in Task 14 exercises it for real.

- [ ] **Step 3: Implement in `model/export_onnx.py`**

- `from files import ... FUSION_PT`; `from model.model import ... FusionHead`.
- `from model.data import ... FLEX_RAW_DIM, FLEXQ_DIM` and `from model.config import ... FUSION*` not needed (FusionHead pulls its own).
- `ExportWrapper.__init__(self, enc, style, emoji, gen, fusion)`: store `self.fusion = fusion`.
- `ExportWrapper.forward(self, x, flex, flex_q)`:

```python
        emb = self.enc(x)
        style_logits = self.style(emb)
        emoji_logits = self.emoji(emb)
        fusion_logits = self.fusion(emoji_logits, flex, flex_q)
        seed = (1 - Z_WEIGHT) * normalize(emb) + Z_WEIGHT * self.z
        color = torch.tanh(self.gen.net(seed)) * 127.5 + 127.5
        return style_logits, emoji_logits, fusion_logits, color
```

- `export_onnx()`: dummy inputs + names:

```python
    v = emoji_logits_width  # = wrapper.emoji.embed.weight.shape[0]
    dummy = (
        torch.zeros(1, MAX_TEXT_LEN, dtype=torch.long),
        torch.zeros(1, v, FLEX_RAW_DIM, dtype=torch.float32),
        torch.zeros(1, FLEXQ_DIM, dtype=torch.float32),
    )
    torch.onnx.export(
        wrapper, dummy, str(dst),
        input_names=["input", "flex", "flex_q"],
        output_names=["style_logits", "emoji_logits", "fusion_logits", "color"],
        opset_version=ONNX_OPSET,
        dynamo=False,
        dynamic_axes={
            "input": {0: "batch"},
            "flex": {0: "batch"},
            "flex_q": {0: "batch"},
            "style_logits": {0: "batch"},
            "emoji_logits": {0: "batch"},
            "fusion_logits": {0: "batch"},
        },
    )
```

  Pass `v` into `export_onnx` (e.g. read `wrapper.emoji.embed.weight.shape[0]`).

- `export()`: `fusion = _load(FusionHead(), FUSION_PT)`; add a vocab guard mirroring the emoji one:

```python
    if fusion.net[-1].out_features != 1 or emoji.embed.weight.shape[0] != len(EMOJIS):
        ...
```

  (the meaningful check is that `emoji.pt` and `fusion.pt` came from the same run — reuse the existing `emoji.embed.weight.shape[0] != len(EMOJIS)` SystemExit; add `fusion` to `ExportWrapper(...)`).

- `export_web()` `meta` dict: add
  `"flex_cols": ["score", "score_norm", "exact", "fuzzy", "best_idf", "rank_recip", "n_kw", "kw_len", "word_len", "overlap"]`,
  `"flex_k": 32`.

- [ ] **Step 4: `ruff` + import check**

Run: `uv run ruff check model/export_onnx.py model/test_export_onnx.py && uv run ruff format --check model/export_onnx.py model/test_export_onnx.py`
Run: `uv run python -c "import model.export_onnx"`
Expected: clean.

- [ ] **Step 5: Commit**

```bash
git add model/export_onnx.py model/test_export_onnx.py
git commit -m "feat: export fusion_logits + flex/flex_q inputs to ONNX

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01NZUMgpJ5J1UKZiV76KUbR6"
```

---

### Task 11: `useOnnx` — flex inputs + fusion output

**Files:**
- Modify: `web/src/hooks/useOnnx.js`
- Modify: `web/src/model.test.js` (add a small unit for the ranking selector — or put it in Task 12)

**Interfaces:**
- Consumes: `makeFlexRanker` (Task 4), `flex.json`, ONNX `flex`/`flex_q` inputs + `fusion_logits` output.
- Produces: `predict(text)` returns `{ feeling, emoji, fusion, keywordRank, palettes, ms }` where
  - `emoji`, `fusion` are `sigmoid(...)` arrays over the vocab,
  - `keywordRank` is `string[]` (emoji, best-first) from `flexRanker.rank`.

- [ ] **Step 1: Load `flex.json` + build the ranker**

In `useOnnx.js` `useEffect`, extend the `Promise.all`:

```js
        const [m, c, fj] = await Promise.all([
          fetch(BASE + 'meta.json').then((r) => r.json()),
          fetch(BASE + 'config.json').then((r) => r.json()),
          fetch(BASE + 'flex.json').then((r) => r.json()),
        ])
```

Store `flexRankerRef.current = makeFlexRanker(fj)` (import `makeFlexRanker` from `../flexrank`).

- [ ] **Step 2: Feed flex inputs, read fusion output**

```js
  const predict = useCallback(async (text) => {
    const m = metaRef.current
    const fr = flexRankerRef.current
    const ids = encode(text, m, char2idxRef.current)
    const norm = normalize(text, char2idxRef.current)
    const V = m.emojis.length
    const flexRaw = fr.flexRaw(norm)
    const flexQ = fr.flexQ(norm)
    const t0 = performance.now()
    const out = await sessionRef.current.run({
      input: new ort.Tensor('int64', ids, [1, m.max_text_len]),
      flex: new ort.Tensor('float32', flexRaw, [1, V, 10]),
      flex_q: new ort.Tensor('float32', flexQ, [1, 5]),
    })
    const ms = performance.now() - t0
    return {
      feeling: sigmoid(out.style_logits.data),
      emoji: sigmoid(out.emoji_logits.data),
      fusion: sigmoid(out.fusion_logits.data),
      keywordRank: fr.rank(norm).map((row) => row[0]),
      palettes: decodeColorList(out.color.data),
      ms,
    }
  }, [])
```

Import `normalize` from `../model` in `useOnnx.js`.

- [ ] **Step 3: Manual check**

Run: `cd web && npm run build`
Expected: builds (needs `web/public/flex.json` + a fusion-enabled `model.onnx`; if `model.onnx` is still the old 1-input graph, the build succeeds but runtime `predict` throws — fine until Task 14 regenerates assets).

- [ ] **Step 4: Commit**

```bash
git add web/src/hooks/useOnnx.js
git commit -m "feat: useOnnx feeds flex inputs and returns fusion + keyword ranking

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01NZUMgpJ5J1UKZiV76KUbR6"
```

---

### Task 12: `App.jsx` — 3-way emoji mode toggle

**Files:**
- Modify: `web/src/App.jsx`
- Modify: `web/src/nav.test.js` or add `web/src/App.mode.test.js` (create) for the mode → ranking selector

**Interfaces:**
- Consumes: `predict` result `{ emoji, fusion, keywordRank, ... }` (Task 11).
- Produces:
  - `EMOJI_MODE_KEY = 'emojiMode'`; values `'fusion' | 'model' | 'keywords'`; default `'fusion'`.
  - `initialEmojiMode()` — reads `localStorage`, migrating legacy `emojiSource` `'cldr'` → `'keywords'`, `'model'` → `'model'`, absent → `'fusion'`.
  - `pickEmojiList(mode, scores, meta, slots)` — pure helper returning `[{emoji, p}]`.

- [ ] **Step 1: Write the failing test** — `web/src/App.mode.test.js`

```js
import { describe, it, expect } from 'vitest'
import { pickEmojiList } from './App.jsx'

const meta = { emojis: ['🍕', '🐞', '🧁'] }

describe('pickEmojiList', () => {
  const scores = {
    emoji: [0.1, 0.9, 0.4],
    fusion: [0.8, 0.2, 0.5],
    keywordRank: ['🧁', '🍕'],
  }
  it('model mode ranks by emoji score', () => {
    expect(pickEmojiList('model', scores, meta, 2).map((x) => x.emoji)).toEqual(['🐞', '🧁'])
  })
  it('fusion mode ranks by fusion score', () => {
    expect(pickEmojiList('fusion', scores, meta, 2).map((x) => x.emoji)).toEqual(['🍕', '🧁'])
  })
  it('keywords mode uses keywordRank order', () => {
    expect(pickEmojiList('keywords', scores, meta, 3).map((x) => x.emoji)).toEqual(['🧁', '🍕'])
  })
})
```

- [ ] **Step 2: Run to verify it fails**

Run: `cd web && npx vitest run src/App.mode.test.js`
Expected: FAIL — `pickEmojiList` not exported.

- [ ] **Step 3: Rewire `App.jsx`**

- Remove `import { cldrEmojis } from './cldrEmojis'`.
- Replace the `EMOJI_SOURCE_KEY` / `initialEmojiSource` block:

```js
const EMOJI_MODE_KEY = 'emojiMode'
const EMOJI_MODES = ['fusion', 'model', 'keywords']

export function initialEmojiMode() {
  try {
    const m = localStorage.getItem(EMOJI_MODE_KEY)
    if (EMOJI_MODES.includes(m)) return m
    const legacy = localStorage.getItem('emojiSource')
    if (legacy === 'cldr') return 'keywords'
    if (legacy === 'model') return 'model'
  } catch {}
  return 'fusion'
}

export function pickEmojiList(mode, scores, meta, slots) {
  if (!scores || !meta) return []
  if (mode === 'keywords') {
    return (scores.keywordRank || [])
      .slice(0, slots)
      .map((emoji, i) => ({ emoji, p: 1 - i / slots }))
  }
  const arr = mode === 'fusion' ? scores.fusion : scores.emoji
  return [...arr.keys()]
    .sort((a, b) => arr[b] - arr[a])
    .slice(0, slots)
    .map((idx) => ({ emoji: meta.emojis[idx], p: arr[idx] }))
}
```

- `const [emojiMode, setEmojiMode] = useState(initialEmojiMode)`.
- `useEffect` persisting: `localStorage.setItem(EMOJI_MODE_KEY, emojiMode)`.
- Replace the `predictedEmoji` / emoji-list `useMemo` (the `useCldrEmojis ? ...` branches) with `pickEmojiList(emojiMode, { ...scores, keywordRank: scores?.keywordRank }, meta, emojiSlots)` and `predictedEmoji = pickEmojiList(emojiMode, scores, meta, 1)[0]?.emoji ?? null`.
- The toggle group render (`['model', 'cldr'].map(...)`) → `EMOJI_MODES.map((mode) => (<button ... aria-pressed={emojiMode === mode} onClick={() => setEmojiMode(mode)}>{mode}</button>))`. Labels: `fusion` / `model` / `keywords`.
- `scores` shape from `predict` now carries `emoji`, `fusion`, `keywordRank` — `setScores(logits)` already stores the whole object; ensure downstream `scores.emoji` / `scores.feeling` references still resolve (they do; `feeling` unchanged).

- [ ] **Step 4: Run tests**

Run: `cd web && npm test`
Expected: PASS (`App.mode.test.js`, `flexrank.test.js`, `model.test.js`, `feelings.test.js`, `fit.test.js`, `nav.test.js`).

- [ ] **Step 5: Manual smoke** (assets regenerated in Task 14; for now check it renders)

Run: `cd web && npm run dev` → open, type a phrase, confirm three toggle buttons render and switching them reorders the emoji list without console errors (fusion/model will need the new `model.onnx` — expect a runtime error until Task 14; `keywords` works immediately off `flex.json`).

- [ ] **Step 6: Commit**

```bash
git add web/src/App.jsx web/src/App.mode.test.js
git commit -m "feat: 3-way emoji mode toggle (fusion default / model / keywords)

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01NZUMgpJ5J1UKZiV76KUbR6"
```

---

## Phase 5 — report, pred, docs, end-to-end

### Task 13: report — 3-way charts + FlexRank smoke test

**Files:**
- Modify: `tools/report.py`
- Modify: `tools/test_report.py`

**Interfaces:**
- Consumes: `model/flexrank.py:FlexRanker` (Task 3), `FusionHead` (Task 6), `scatter_flex` (Task 7), `FLEX_JSON`, `FUSION_PT`.
- Produces:
  - `_flex_and_fusion(texts, emoji_logits) -> (flex_acc_at_k, fusion_acc_at_k)` given target multi-hot.
  - Every emoji `acc@k` line chart renders 3 series (`EmojiHead` / `FlexRank` / `Fusion`).
  - `report.json` sections gain `flex_acc_at_k` / `fusion_acc_at_k`.
  - Header banner flags `cldr` `flex_acc_at_k[0] < 0.7`.
  - `.lline3` CSS class.

- [ ] **Step 1: Write / extend a failing test** — `tools/test_report.py`

```python
def test_flexrank_smoke_and_threeway_keys():
    from files import FLEX_JSON
    from pathlib import Path

    if not Path(FLEX_JSON).exists():
        return
    from model.flexrank import FlexRanker

    r = FlexRanker(FLEX_JSON)
    # CLDR keyword text -> its own emoji should be #1
    ranked = r.rank("pizza")
    assert ranked and ranked[0][0].strip() in ("🍕",), ranked[:3]


def test_report_json_has_threeway(tmp_path):
    from pathlib import Path
    from files import FUSION_PT, FLEX_JSON

    if not (Path(FUSION_PT).exists() and Path(FLEX_JSON).exists()):
        return
    import json
    from tools.report import build_report  # or whatever the top-level builder is

    rep = build_report()
    assert "flex_acc_at_k" in rep["emoji"]["eval"]
    assert "fusion_acc_at_k" in rep["emoji"]["eval"]
    assert "flex_acc_at_k" in rep["cldr"]
```

(If `tools/report.py` has no single `build_report()` entry, adapt the test to call the section functions directly — inspect the file first and match names.)

- [ ] **Step 2: Run to verify it fails / skips**

Run: `uv run python tools/test_report.py`
Expected: `test_flexrank_smoke_and_threeway_keys` passes or skips; the threeway test skips (no `fusion.pt`) — that is acceptable red for now.

- [ ] **Step 3: Implement in `tools/report.py`**

- Imports: `from model.flexrank import FlexRanker`; `from model.model import FusionHead`; `from model.data import scatter_flex, _row_flex, EMOJIS`; `from files import FLEX_JSON, FUSION_PT`.
- Module-level: `_FLEX = FlexRanker(FLEX_JSON) if Path(FLEX_JSON).exists() else None`; load `FusionHead` lazily like the other heads (`_load(FusionHead(), FUSION_PT)`).
- Helper:

```python
def _flex_fusion_curves(texts, tgt, emoji_logits, fusion_head):
    if _FLEX is None:
        return None, None
    idxs, raws = [], []
    for t in texts:
        idx, raw, _q = _row_flex({"flexsearch": [[e, *v] for (e, *v) in _FLEX.rank(t)],
                                  "flexq": _FLEX.flexq(t)})
        idxs.append(idx)
        raws.append(raw)
    flex_idx = torch.stack(idxs)
    flex_raw = torch.stack(raws)
    dense = scatter_flex(flex_idx, flex_raw)
    flex_logits = torch.where(dense[..., 0] > 0, dense[..., 0], torch.full_like(dense[..., 0], -1e9))
    flex_acc = [_acc_at_k(flex_logits, tgt, k).mean().item() for k in EMOJI_KS]
    fusion_acc = None
    if fusion_head is not None:
        qs = torch.stack([torch.tensor(
            [_FLEX.flexq(t)[k] for k in ("tokens", "matched", "sum", "max", "cand")],
            dtype=torch.float32) for t in texts])
        with torch.no_grad():
            fl = fusion_head(emoji_logits, dense, qs)
        fusion_acc = [_acc_at_k(fl, tgt, k).mean().item() for k in EMOJI_KS]
    return flex_acc, fusion_acc
```

  > Note `_row_flex` here re-derives `rank_recip` from position — the `[[e, *v] ...]`
  > rebuild passes the ranker's 10-field entry (no `rank_recip`); `_row_flex` inserts it.
  > This matches training exactly.

- `_section_emoji`: after computing `logits` on `texts`, call `_flex_fusion_curves(texts, tgt, logits, fusion_head)`; add `"flex_acc_at_k"` / `"fusion_acc_at_k"` to `d["eval"]`. Same for `_keyword_probe`.
- `_section_cldr` / `_cldr_probe`: same; also stash `flex_acc_at_k` at the section root so the banner can read it.
- `_section_cards`: compute the two extra curves on the gold `texts`; the cards section already has `emoji_acc` / `style_acc` — add `flex_acc` / `fusion_acc`. Split the chart (see chart step).
- `_provenance` / `_header_html`: if `d["cldr"].get("flex_acc_at_k", [1])[0] < 0.7`, append an issue string `"FlexRank acc@1 on cldr.jsonl is {x:.2f} (<0.70) — flex.json / ranker parity looks broken"`.
- Charts (`_emoji_html`, `_cldr_html`, `_cards_html`): where `_linechart(points, ...)` is called for an emoji curve, pass
  `series=[("FlexRank", d[...]["flex_acc_at_k"], "lline2"), ("Fusion", d[...]["fusion_acc_at_k"], "lline3")]`
  (drop a series if `None`) and `legend=("EmojiHead", "FlexRank", "Fusion")`.
  For `_emoji_html` eval chart that also passes `baseline=`, extend `_linechart`'s legend branch to append a 4th dashed `"CLDR"` entry (small edit: in the `if legend and series` branch, after the loop, if `baseline is not None` add the dashed swatch).
- `_cards_html`: change the current single emoji+style chart into two — the emoji chart with the 3 series above, and a second `_linechart(style_points)` labelled "Style acc@k — gold set".
- CSS: in the `<style>` block near `.linechart .lline2`, add `.linechart .lline3{fill:none;stroke:#0a9c8b;stroke-width:2.5}`.

- [ ] **Step 4: Run**

Run: `uv run python tools/test_report.py && uv run ruff check tools/report.py && uv run ruff format --check tools/report.py`
Expected: clean.

Run (if a full set of `.pt` exists): `uv run python tools/report.py` → open `report/<ts>/report.html`, confirm eval / keywords / cldr / cards emoji charts each show three lines + legend; cards has a separate style chart; cldr FlexRank `acc@1` ≈ 1.

- [ ] **Step 5: Commit**

```bash
git add tools/report.py tools/test_report.py
git commit -m "feat: report — EmojiHead/FlexRank/Fusion 3-way acc@k charts + cldr smoke test

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01NZUMgpJ5J1UKZiV76KUbR6"
```

---

### Task 14: `pred.py` fused column + full end-to-end

**Files:**
- Modify: `model/pred.py`
- Modify: `CLAUDE.md`

**Interfaces:**
- Consumes: `FusionHead`, `FlexRanker`, `scatter_flex`, `_row_flex`.
- Produces: `data/pred.jsonl` rows gain `fusion_top_labels` (same threshold rule as `emoji`).

- [ ] **Step 1: Add fused predictions to `pred.py`**

Load `FusionHead` from `<--pt>/fusion.pt`; build `FlexRanker(FLEX_JSON)`. For the batch of 200 texts, build `flex_idx`/`flex_raw`/`flexq` via `_row_flex({"flexsearch": [[e,*v] for (e,*v) in fr.rank(t)], "flexq": fr.flexq(t)})`, `dense = scatter_flex(...)`, `fusion_logits = fusion(emoji_logits, dense, flexq_stack)`, then apply the existing `top_labels` threshold helper to `sigmoid(fusion_logits)` → `fusion_top_labels`. Write it alongside `emoji`/`style` in each output row.

- [ ] **Step 2: Run**

Run: `uv run ruff check model/pred.py && uv run ruff format --check model/pred.py`
Expected: clean.

- [ ] **Step 3: Update `CLAUDE.md`**

Add to the model paragraph: FusionHead (`model/model.py:FusionHead`) — residual re-ranker on detached `EmojiHead` logits + the `flexsearch`/`flexq` features, `fused = emoji_logit + g(features)` with `g` zero-init; trained in stage 1 as a 4th head (`--heads ... ,fusion`; `ALL_HEADS` now 4), `lse_infonce`, checkpoint/early-stop on `MRR/fusion/val`; `MRR/e/val` + `MRR/flex/val` logged alongside. Add `fusion.pt` to the stage-1 outputs and the `gan`/export required `.pt` list. Document `web/public/flex.json` (written by `regen`, committed) and the three ranker parity surfaces (`tools/data/flexrank.ts`, `model/flexrank.py`, `web/src/flexrank.js`) locked by `web/src/flexrank.fixture.json`. Update the export section: ONNX now has inputs `input` / `flex` / `flex_q` and a 4th output `fusion_logits`; `meta.json` gains `flex_cols` / `flex_k`. Web: the emoji toggle is 3-way (`fusion` default / `model` / `keywords`), `keywords` now uses `web/src/flexrank.js` (the old `cldrEmojis.ts` + `cldr-emoji-index.json` are removed).

- [ ] **Step 4: End-to-end**

```bash
git add -A && git commit -m "feat: pred.py fused column; CLAUDE.md fusion docs

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01NZUMgpJ5J1UKZiV76KUbR6"
bun run regen
bun tools/data/gen-flex-fixture.ts   # refresh fixture against current vocab
cd web && npm test && cd ..
uv run python model/test_flexrank.py
uv run python model/test_model_fusion.py
uv run python model/test_data_flex.py
uv run python model/test_train_cli.py
uv run python model/test_runmeta.py
uv run ruff check . && uv run ruff format --check .
git add web/src/flexrank.fixture.json web/public/flex.json && git commit -m "chore: refresh flex fixture + flex.json" || true
train --local        # full: enc(4 heads) + gan + export + report
uv run python model/pred.py --pt pt
```

Expected:
- `MRR/fusion/val` ≥ `max(MRR/e/val, MRR/flex/val)` in TensorBoard.
- `web/public/model.onnx` has 3 inputs / 4 outputs; `model/test_export_onnx.py` passes.
- `report/<ts>/report.html` shows the 3-way charts; cldr FlexRank acc@1 ≈ 1; no smoke-test banner.
- `cd web && npm run build` succeeds; `npm run dev` → toggling `fusion`/`model`/`keywords` reorders emoji live, inference stays interactive.
- `data/pred.jsonl` rows have `fusion_top_labels`.

- [ ] **Step 5: Commit the regenerated web assets**

```bash
git add web/public/ && git commit -m "chore: regenerate web assets with fusion graph + flex.json

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01NZUMgpJ5J1UKZiV76KUbR6"
```

---

## Self-Review

**Spec coverage**

| Spec section | Task(s) |
|---|---|
| `flex.json` written by regen (shape, idf, ownership) | 1 |
| Three parity rankers + shared fixture | 1, 2, 3, 4 |
| Per-emoji 10-vector incl. `rank_recip` reorder | 4 (layout note), 6, 7 |
| Config `FUSION_*` | 5 |
| `FusionHead` residual + zero-init + int embeddings (clamp 16 / dim 8) | 6 |
| `fusion_features` pure-torch transform (18 scalars + 4×8 embed) | 6 |
| Sparse flex storage + `scatter_flex` | 7 |
| `fusion` in `ALL_HEADS`; requires `emoji`; default run | 8 |
| detached `emoji_logits`; `loss/fusion`; `MRR/e`+`MRR/flex`+`MRR/fusion` | 8 |
| monitor → `MRR/fusion/val` | 8 |
| `fusion.pt` saved; gan `_require_pt` + Modal upload | 8, 9 |
| ONNX: `flex`/`flex_q` inputs, `fusion_logits` output, `meta.flex_cols`/`flex_k` | 10 |
| Web: load `flex.json`, feed inputs, read fusion | 11 |
| Web: 3-way toggle, default fusion, legacy migration, remove `cldrEmojis` | 4 (delete), 12 |
| Report: 3-way charts on eval/keywords/cldr/cards; cards chart split; `.lline3` | 13 |
| Report: cldr FlexRank smoke-test banner | 13 |
| `model/flexrank.py` used by report + pred | 3, 13, 14 |
| `pred.py` fused column | 14 |
| CLAUDE.md | 14 |
| Verification / rollout (`regen` for `flex.json`, retrain) | 14 |

**Placeholder scan:** no TBD/TODO; every code step carries real code. The report test (Task 13 Step 1) says "adapt to match names after inspecting the file" — this is a genuine unknown (the report's top-level builder name is not visible in the excerpts read); the executor must open `tools/report.py` and wire the assertions to the real section functions. Flagged, not hidden.

**Type consistency:**
- Ranker per-entry tuple order `[emoji, score, score_norm, exact, fuzzy, best_idf, n_kw, kw_len, word_len, overlap]` — used identically in Tasks 1, 3, 4, 7, 13, 14.
- Model per-emoji vector order (`rank_recip` after `best_idf`) — Tasks 4 (`flexRaw`), 6 (`fusion_features` indices 0–9), 7 (`_row_flex`) agree.
- `scatter_flex(flex_idx, flex_raw) -> [B, V, 10]` — Tasks 7, 8, 10, 13, 14.
- `FusionHead.forward(emoji_logit, flex_raw, flexq)` — Tasks 6, 8, 10, 13, 14 (note: `flex_raw` here is the **dense** `[B,V,10]` from `scatter_flex`, not the sparse `[32,10]`; consistent everywhere).
- `flexq` field order `("tokens","matched","sum","max","cand")` — Tasks 3, 6, 7, 13.
- `_row_flex` accepts a dict (Tasks 7 test, 13, 14) and a `record` (Task 7 dataset build) — Task 7 Step 3 explicitly handles both.
- Monitor string `"MRR/fusion/val"` — Task 8 (`_step` logs it, `_train_encoder` monitors it).
