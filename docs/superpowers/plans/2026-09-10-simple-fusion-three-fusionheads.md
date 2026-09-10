# Simple Fusion: Three FusionHeads Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the learned `KWHead` + gate `FusionHead` + soft-TF `flexrank` stack with a standalone `EmojiHead`, an offline-computed uFuzzy keyword payload, and three tiny detached learned combiners (`FusionHeadGate` / `FusionHeadGain` / `FusionHeadMix`) trained in parallel for comparison.

**Architecture:** `EmojiHead` trains alone (`lse_infonce`). `tools/data/regen.ts` runs uFuzzy (`@leeoniya/uFuzzy`) over `data/ii.json` for every train/eval row and writes a sparse per-emoji `kw` score field + `web/public/kwproj.json`; the browser runs the same uFuzzy live. Word-splitting is the already-shared `normalize` + `.split(" ")` — no bespoke tokenizer. Each of the three `FusionHead*` modules takes detached `(logit_m, kw)` and returns fused per-emoji scores; `tools/report.py` compares them on `data/eval.jsonl`; `model/export_onnx.py` bakes one winner variant into `meta.json`.

**Tech Stack:** PyTorch + Lightning + Typer (`model/`), Bun + TypeScript (`tools/data/`), Vite + React + onnxruntime-web + `@leeoniya/uFuzzy` (`web/`). `uv` for Python, `bun` for tooling, `npm` for `web/`.

**Spec:** `docs/superpowers/specs/2026-09-10-simple-fusion-three-fusionheads-design.md` — read it alongside this plan.

## Global Constraints

- **No comments or docstrings** in source. Keep existing `# type: ignore` / `# noqa` / shebangs; do not add new explanatory comments. (Test files may keep the existing `"""..."""` on the Typer `main()` command only where one already exists.)
- **Package management:** `uv` only for Python (`uv add` / `uv sync`, never `pip`); `bun add` at repo root; `npm` inside `web/`. `torch` stays a `cpu`/`gpu` conflicting group split — do not touch `pyproject.toml`'s `[tool.uv]` blocks.
- **Determinism:** `regen` must stay deterministic for a fixed `data/data.jsonl` + flags (both shuffles use a `SEED`-seeded mulberry32; `SEED = 42` in `tools/data/config.ts`).
- **Clean git tree:** `model/train.py` aborts on a dirty tree in every mode. Commit after every task.
- **Do not run a training run to verify.** Behavioural verification is a deferred `train enc --local`; per-task gates are `ruff`, the plain-assert test scripts, `bun test`, and `npm test`.
- **Char/normalize parity:** do not touch `model/data.py:normalize`, `CHARS`, `PAD_IDX`, or `INFONCE_TEMP`. `normalize` (`tools/data/normalize.ts` / `web/src/model.js` / `model/data.py`) is already byte-identical across languages — the keyword word-split reuses it verbatim.
- **uFuzzy pin:** `@leeoniya/uFuzzy@^1.0.19`, option `{ intraIns: 1 }`, similarity floor `sim >= 0.5` — the exact same literals in `tools/data/regen.ts` and `web/src/keywords.js`.
- **`N = len(EMOJIS)`** (currently 957) is the emoji vocab size; all fusion tensors are `[B, N]`.
- Concurrent auto-commit job: scope every `git add` to the exact files in the task. Never `git add -A` / `git add .`.
- Emoji vocab order is `data/labels.json` `emojis` == `meta.json` `emojis` == `model.config.EMOJIS`.

---

## File Structure

**New:**
- `model/tokenize.py` — `query_tokens` + `STOPWORDS`, Python port used **only** by the `report.py` keyword-vocab diagnostic (off the training/inference path; no cross-language parity burden).
- `model/test_tokenize.py` — plain-assert test for the above.
- `web/src/keywords.js` — `makeKeywordPredictor(kwprojJson)` → `predict(text): Float32Array(N)`.
- `web/src/keywords.test.js` — vitest, a few `text → expected emoji` cases.
- `web/src/fusion.js` — `makeFusion(metaFusion)` → `fuse(emojiLogits, kwArr): Float32Array(N)`.
- `web/src/fusion.test.js` — vitest for the three variant formulas.

**Modified:**
- `model/model.py` — add `FusionHeadGate` / `FusionHeadGain` / `FusionHeadMix` + `_z` helper; delete `KWHead`, `FusionHead`.
- `model/data.py` — `record.flex_tf` → `record.kw`; `_row_tf` → `_row_kw` over `N`; drop `FLEX_JSON` import + `KW_VOCAB` / `FLEX_N`.
- `model/config.py` — drop `DROPOUT_KW`.
- `model/train.py` — `self.fusion` becomes an `nn.ModuleDict` of the three heads; detached fusion loss; per-variant logging; `MRR/fusion/{split}` = best of three; write `fusion_{gate,gain,mix}.pt`; gan required-file list + Modal byte plumbing.
- `model/export_onnx.py` — one input (`input`), three outputs (`style_logits`, `emoji_logits`, `color`); `FUSION_EXPORT_VARIANT` constant; `meta.json` `fusion` block; drop `flex_kw` / `flex_n`.
- `model/pred.py` — `fusion_top_labels` from `FusionHeadGate` + the row's `kw`.
- `tools/report.py` — import `query_tokens` from `model.tokenize`; 5-way `acc@k` overlay on `data/eval.jsonl`; `report.json` keys.
- `tools/data/regen.ts` — uFuzzy `kw` payload (per `normalize(text).split(" ")` word); write `web/public/kwproj.json`; drop `flexrank` / `kwvocab` / `flex.json` / fixture.
- `files.py` — drop `FLEX_JSON`, `KW_PT`, `FUSION_PT`; add `FUSION_GATE_PT`, `FUSION_GAIN_PT`, `FUSION_MIX_PT`, `KWPROJ_JSON`.
- `files.ts` — drop `FLEX_JSON`, `FLEX_FIXTURE_JSON`; add `KWPROJ_JSON`.
- `model/test_model_heads.py` — rewrite for the three heads.
- `model/test_train_cli.py` — rewrite the fusion step test.
- `web/src/hooks/useOnnx.js` — fetch `kwproj.json`; no `flex_tf` tensor; compute `kw` + `fusion` client-side.
- `web/src/App.jsx` — `scores.fusion` / `scores.kw` are now raw score arrays.
- `web/package.json` — add `@leeoniya/uFuzzy`, remove `flexsearch`.
- `package.json` (root) — add `@leeoniya/uFuzzy`, remove `flexsearch` devDependency.
- `CLAUDE.md` — update the fusion / keyword / flexrank prose.

**Deleted:**
- `model/flexrank.py`, `model/test_flexrank.py`
- `tools/data/flexrank.ts`, `tools/data/kwvocab.ts`
- `web/src/flexrank.js`, `web/src/flexrank.test.js`, `web/src/flexrank.fixture.json`
- `web/public/flex.json`

**Execution order:** Task 3 (`regen`) must run before Task 4's `ruff`/test gate, because `model/data.py` after Task 4 no longer reads `flex.json` and `data/train.jsonl` needs a `kw` field. If running out of order, re-run `bun run regen` before Task 4's gate.

---

## Task 1: `model/tokenize.py` — Python tokenizer for the report diagnostic

`model/flexrank.py` is deleted in Task 4, but `tools/report.py`'s keyword-vocab
diagnostic still needs `query_tokens` (to filter `data/ii.json` keys to single
tokens). This task lifts it into its own Python-only module. It is **not** on the
keyword-predictor path — that path uses uFuzzy + `normalize` + `.split(" ")` and
needs no tokenizer.

**Files:**
- Create: `model/tokenize.py`
- Test: `model/test_tokenize.py`

**Interfaces:**
- Produces: `query_tokens(text: str) -> list[str]`; `STOPWORDS: set[str]`. Identical behaviour to the current `model/flexrank.py:query_tokens` (lowercase, `[^a-z0-9\s]` → space, split on whitespace, keep tokens with `len >= 2` not in `STOPWORDS`).

- [ ] **Step 1: Write the failing test**

Create `model/test_tokenize.py`:

```python
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from model.tokenize import STOPWORDS, query_tokens


def test_splits_and_lowercases():
    assert query_tokens("The Dog is Running FAST") == ["dog", "running", "fast"]


def test_strips_punctuation_and_short_tokens():
    assert query_tokens("pizza-time, y'all! ok?") == ["pizza", "time", "all", "ok"]


def test_drops_stopwords():
    assert "the" in STOPWORDS and "with" in STOPWORDS
    assert query_tokens("a cat and the hat") == ["cat", "hat"]


def test_digits_kept():
    assert query_tokens("bus 42 now") == ["bus", "42", "now"]


if __name__ == "__main__":
    test_splits_and_lowercases()
    test_strips_punctuation_and_short_tokens()
    test_drops_stopwords()
    test_digits_kept()
    print("ok")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run python model/test_tokenize.py`
Expected: `ModuleNotFoundError: No module named 'model.tokenize'`

- [ ] **Step 3: Write minimal implementation**

Create `model/tokenize.py` (port verbatim from `model/flexrank.py`):

```python
import re

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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run python model/test_tokenize.py`
Expected: `ok`

- [ ] **Step 5: Lint**

Run: `uv run ruff check model/tokenize.py model/test_tokenize.py && uv run ruff format --check model/tokenize.py model/test_tokenize.py`
Expected: no errors. (If `ruff format --check` complains, run `uv run ruff format model/tokenize.py model/test_tokenize.py` and re-check.)

- [ ] **Step 6: Commit**

```bash
git add model/tokenize.py model/test_tokenize.py
git commit -m "feat(tokenize): model/tokenize.py — query_tokens port for report diagnostics"
```

---

## Task 2: Three `FusionHead*` modules in `model/model.py`

**Files:**
- Modify: `model/model.py` (add classes near the current `FusionHead`, around line 126; do NOT delete `KWHead` / `FusionHead` yet)
- Test: `model/test_model_heads.py` (full rewrite)

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces:
  - `FusionHeadGate()` — `forward(logit_m: Tensor[B, N], kw: Tensor[B, N]) -> Tensor[B, N]`. Modules `self.bn = nn.BatchNorm1d(6, affine=False)`, `self.lin = nn.Linear(6, 1)` (weight + bias zero-init). Sets `self.last_a: Tensor` (scalar, detached mean gate) on every forward.
  - `FusionHeadGain()` — `forward(logit_m, kw) -> Tensor[B, N]`. Param `self.raw_beta = nn.Parameter(torch.zeros(()))`.
  - `FusionHeadMix()` — `forward(logit_m, kw) -> Tensor[B, N]`. Param `self.raw_g = nn.Parameter(torch.full((), -2.0))`.
  - Module-level helper `_z(x: Tensor) -> Tensor` — per-row standardize over the last axis, `(x - mean) / (std + 1e-6)`.
  - Neutral-init behaviour with `kw == 0`: `FusionHeadGain` → `fused == logit_m`; `FusionHeadMix` → `fused == (1 - g) * _z(logit_m)` with `g = sigmoid(-2)`; `FusionHeadGate` → `fused == 0.5 * _z(logit_m)`.

- [ ] **Step 1: Write the failing test**

Replace the entire contents of `model/test_model_heads.py`:

```python
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from torch.nn.functional import softplus

from model.config import EMOJIS
from model.model import FusionHeadGain, FusionHeadGate, FusionHeadMix, _z

B = 4
N = len(EMOJIS)


def _inputs(requires_grad=False):
    torch.manual_seed(0)
    logit_m = torch.randn(B, N, requires_grad=requires_grad)
    kw = torch.zeros(B, N)
    kw[:, 3] = 0.7
    kw[:, 9] = 0.4
    return logit_m, kw


def test_shapes():
    logit_m, kw = _inputs()
    for head in (FusionHeadGate(), FusionHeadGain(), FusionHeadMix()):
        assert head(logit_m, kw).shape == (B, N)


def test_gain_neutral_at_init():
    logit_m, kw = _inputs()
    head = FusionHeadGain().eval()
    zero_kw = torch.zeros(B, N)
    assert torch.allclose(head(logit_m, zero_kw), logit_m)
    assert torch.isclose(softplus(head.raw_beta), torch.tensor(softplus(torch.zeros(()))))


def test_mix_neutral_at_init():
    logit_m, kw = _inputs()
    head = FusionHeadMix().eval()
    g = torch.sigmoid(torch.tensor(-2.0))
    zero_kw = torch.zeros(B, N)
    assert torch.allclose(head(logit_m, zero_kw), (1 - g) * _z(logit_m), atol=1e-5)


def test_gate_neutral_at_init():
    logit_m, kw = _inputs()
    head = FusionHeadGate().eval()
    zero_kw = torch.zeros(B, N)
    assert torch.allclose(head(logit_m, zero_kw), 0.5 * _z(logit_m), atol=1e-4)
    assert torch.isclose(head.last_a, torch.tensor(0.5), atol=1e-4)


def test_kw_zero_is_noop_per_emoji():
    logit_m, kw = _inputs()
    for head in (FusionHeadGain().eval(), FusionHeadMix().eval(), FusionHeadGate().eval()):
        full = head(logit_m, kw)
        none = head(logit_m, torch.zeros(B, N))
        untouched = (kw == 0).all(dim=0)
        assert torch.allclose(full[:, untouched], none[:, untouched], atol=1e-4)


def test_heads_are_transparent_to_logit_grad_and_train_their_params():
    logit_m, kw = _inputs(requires_grad=True)
    for head in (FusionHeadGate(), FusionHeadGain(), FusionHeadMix()):
        lm = logit_m.detach().clone().requires_grad_(True)
        out = head(lm, kw)
        out.sum().backward()
        assert lm.grad is not None and lm.grad.abs().sum() > 0
        pg = [p.grad for p in head.parameters()]
        assert pg and all(g is not None for g in pg)


if __name__ == "__main__":
    test_shapes()
    test_gain_neutral_at_init()
    test_mix_neutral_at_init()
    test_gate_neutral_at_init()
    test_kw_zero_is_noop_per_emoji()
    test_heads_are_transparent_to_logit_grad_and_train_their_params()
    print("ok")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run python model/test_model_heads.py`
Expected: `ImportError: cannot import name 'FusionHeadGain' from 'model.model'`

- [ ] **Step 3: Write minimal implementation**

In `model/model.py`: add `softplus` to the `torch.nn.functional` import group (currently imports `normalize`, `tanh`). Then insert, immediately **after** the existing `FusionHead` class (leave `FusionHead` and `KWHead` in place for now):

```python
def _z(x: torch.Tensor) -> torch.Tensor:
    return (x - x.mean(-1, keepdim=True)) / (x.std(-1, keepdim=True) + 1e-6)


class FusionHeadGain(nn.Module):
    def __init__(self):
        super().__init__()
        self.raw_beta = nn.Parameter(torch.zeros(()))

    def forward(self, logit_m: torch.Tensor, kw: torch.Tensor) -> torch.Tensor:
        return logit_m + softplus(self.raw_beta) * kw


class FusionHeadMix(nn.Module):
    def __init__(self):
        super().__init__()
        self.raw_g = nn.Parameter(torch.full((), -2.0))

    def forward(self, logit_m: torch.Tensor, kw: torch.Tensor) -> torch.Tensor:
        g = torch.sigmoid(self.raw_g)
        return (1 - g) * _z(logit_m) + g * kw


class FusionHeadGate(nn.Module):
    def __init__(self):
        super().__init__()
        self.bn = nn.BatchNorm1d(6, affine=False)
        self.lin = nn.Linear(6, 1)
        nn.init.zeros_(self.lin.weight)
        nn.init.zeros_(self.lin.bias)
        self.register_buffer("last_a", torch.zeros(()), persistent=False)

    def _features(self, logit_m: torch.Tensor, kw: torch.Tensor) -> torch.Tensor:
        top2 = logit_m.topk(2, dim=-1).values
        p = torch.softmax(logit_m, dim=-1)
        ent = -(p * torch.log(p + 1e-9)).sum(-1)
        return torch.stack(
            [
                logit_m.max(-1).values,
                top2[:, 0] - top2[:, 1],
                ent,
                kw.max(-1).values,
                (kw > 0).sum(-1).float(),
                kw.sum(-1),
            ],
            dim=-1,
        )

    def forward(self, logit_m: torch.Tensor, kw: torch.Tensor) -> torch.Tensor:
        a = torch.sigmoid(self.lin(self.bn(self._features(logit_m, kw))))
        self.last_a = a.detach().mean()
        return a * _z(logit_m) + (1 - a) * kw
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run python model/test_model_heads.py`
Expected: `ok`

- [ ] **Step 5: Lint + regression**

Run: `uv run ruff check model/ && uv run ruff format --check model/model.py model/test_model_heads.py`
Run: `uv run python model/test_train_cli.py` — expected `ok` (the old `FusionHead`/`KWHead` path is untouched).
Run: `uv run python model/test_runmeta.py` — expected `ok`.

- [ ] **Step 6: Commit**

```bash
git add model/model.py model/test_model_heads.py
git commit -m "feat(model): FusionHeadGate/Gain/Mix — three detached learned combiners"
```

---

## Task 3: uFuzzy keyword payload in `regen.ts`

**Files:**
- Modify: `tools/data/regen.ts`, `files.ts`, `package.json` (root)
- Delete: `tools/data/flexrank.ts`, `tools/data/kwvocab.ts`

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces:
  - Each train/eval row in `data/train.jsonl` / `data/eval.jsonl` gains `"kw": [[emojiIdx, value], ...]` (value > 0, rounded to 3 dp; `emojiIdx` in `data/labels.json` `emojis` order). Skipped under `--no-kw`.
  - `web/public/kwproj.json`: `{ "proj": { "<keyword>": [<emojiIdx>, ...], ... } }` — every `data/ii.json` key whose emoji list has ≥1 entry in the current vocab, mapped to those vocab indices. Skipped under `--no-kw`.
  - `files.ts`: `export const KWPROJ_JSON = \`${WEB_PUBLIC_DIR}/kwproj.json\``.
  - Root `package.json`: `@leeoniya/uFuzzy` in `dependencies`.

- [ ] **Step 1: Swap the dependency**

Run: `bun add @leeoniya/uFuzzy && bun remove flexsearch`
Expected: `@leeoniya/uFuzzy` under `dependencies`, `flexsearch` gone from `devDependencies`, `bun.lock` updated.
(If `bun remove flexsearch` errors because something still imports it, that is Task 3 Step 4's `git rm` — run `bun add @leeoniya/uFuzzy` now and `bun remove flexsearch` after Step 4.)

- [ ] **Step 2: Update `files.ts`**

In `files.ts`: delete the `FLEX_JSON` and `FLEX_FIXTURE_JSON` exports. Add after `WEB_SRC_DIR`:

```ts
export const KWPROJ_JSON = `${WEB_PUBLIC_DIR}/kwproj.json`
```

- [ ] **Step 3: Rewrite the `regen.ts` keyword section**

Read `tools/data/regen.ts` fully first. Then:

1. Imports (top of file): delete `import { buildFlexRanker } from "./flexrank.ts"` and `import { keywordVocabFromIiJson, keywordVocabFromReport } from "./kwvocab.ts"`. Add:

```ts
import uFuzzy from "@leeoniya/uFuzzy"
import { readFileSync } from "node:fs"
import { II_JSON } from "../../files.ts"
```

(`normalize` is already imported from `./normalize.ts`. If `readFileSync` / `II_JSON` are already imported, reuse them. `existsSync` is already imported.)

2. In the `files.ts` import list within `regen.ts`, replace `FLEX_FIXTURE_JSON, FLEX_JSON` with `KWPROJ_JSON`.

3. Delete the `FLEX_FIXTURE_TEXTS` constant (lines ~13-22).

4. `Row` type: rename `flex_tf?: [number, number][]` → `kw?: [number, number][]`. In `BASE_FIELDS`, replace `"flex_tf"` with `"kw"`.

5. `toLine` / the row-serialize path (line ~182): replace `r.flex_tf ? { ...withExtra, flex_tf: r.flex_tf } : withExtra` with `r.kw ? { ...withExtra, kw: r.kw } : withExtra`.

6. `--no-kw` option help text (line ~296-299): change to `"skip the uFuzzy keyword score field (kw) on train/eval rows"`.

7. Replace the whole `if (useKw) { ... }` block (lines ~347-378) with:

```ts
  const useKw = options.kw !== false
  let kwLine = "kw                    : skipped (--no-kw)"
  if (useKw) {
    console.log("computing uFuzzy keyword scores...")
    const emojiIdx = new Map(emojis.map((e, i) => [e, i]))
    const ii = JSON.parse(readFileSync(II_JSON, "utf8")) as Record<string, string[]>
    const proj: Record<string, number[]> = {}
    for (const [k, es] of Object.entries(ii)) {
      const idxs = es.map((e) => emojiIdx.get(e)).filter((i): i is number => i !== undefined)
      if (idxs.length) proj[k] = idxs
    }
    const keys = Object.keys(proj)
    const weight = new Map(keys.map((k) => [k, 1 / Math.log2(1 + proj[k].length)]))
    const uf = new uFuzzy({ intraIns: 1 })
    const kwVec = (text: string): [number, number][] => {
      const best = new Map<string, number>()
      for (const word of normalize(text).split(" ")) {
        if (word.length < 3) continue
        const idxs = uf.filter(keys, word)
        if (!idxs || !idxs.length) continue
        const info = uf.info(idxs, keys, word)
        for (let i = 0; i < info.idx.length; i++) {
          const k = keys[info.idx[i]]
          const sim = info.chars[i] / k.length
          if (sim >= 0.5 && sim > (best.get(k) ?? 0)) best.set(k, sim)
        }
      }
      const acc = new Map<number, number>()
      for (const [k, sim] of best) {
        const v = sim * weight.get(k)!
        for (const e of proj[k]) {
          if (v > (acc.get(e) ?? 0)) acc.set(e, v)
        }
      }
      return [...acc.entries()]
        .map(([i, v]) => [i, Number(v.toFixed(3))] as [number, number])
        .filter(([, v]) => v > 0)
        .sort((a, b) => a[0] - b[0])
    }
    let nzSum = 0
    for (const r of split) {
      r.kw = kwVec(r.text)
      nzSum += r.kw.length
    }
    await writeFileAtomic(KWPROJ_JSON, JSON.stringify({ proj }) + "\n")
    const denom = split.length || 1
    kwLine = `kw                    : keys=${keys.length}, mean nz ${(nzSum / denom).toFixed(2)}`
  }
```

Note `normalize` here is `tools/data/normalize.ts` (already imported) — the same
lowercase/collapse/trim/drop-non-vocab transform as `model/data.py:normalize`. If
`uf.info` returns arrays under different names in the installed version, check its
shape with a one-off `console.log(Object.keys(info))` — the plan assumes
`{ idx, start, chars, terms, ... }` (uFuzzy ≥ 1.0).

8. Confirm `kwLine` is still printed in the summary block (it was; the variable name is unchanged).

- [ ] **Step 4: Delete the dead modules**

Run: `git rm tools/data/flexrank.ts tools/data/kwvocab.ts`
Run: `grep -rn "flexrank\|kwvocab\|buildFlexRanker\|keywordVocabFrom\|flexsearch" tools/ --include="*.ts"`
Expected: no matches. (If `tools/preview.ts` or another tool imports `flexrank.ts`, stop and report — the spec did not anticipate that consumer.)
If Step 1's `bun remove flexsearch` was deferred, run it now.

- [ ] **Step 5: Run regen, verify the payload**

Run: `bun run regen`
Expected: completes; summary line shows `kw : keys=<n>, mean nz <x>` with `keys` in the low thousands and `mean nz` > 0. If `mean nz` is 0 or implausibly high, inspect `uf.info` field names (Step 3 note).

Run: `python3 -c "import json; r=[json.loads(l) for l in open('data/eval.jsonl')]; assert all('kw' in x for x in r), 'missing kw'; assert any(x['kw'] for x in r), 'all kw empty'; print('rows', len(r), 'with-kw', sum(1 for x in r if x['kw']))"`
Expected: prints counts; no assertion error. `with-kw` should be a large fraction of `rows`.

Run: `python3 -c "import json; d=json.load(open('web/public/kwproj.json')); p=d['proj']; print('keys', len(p)); assert all(isinstance(v,list) and v for v in p.values())"`
Expected: prints key count; no error.

- [ ] **Step 6: Verify determinism**

```bash
cp data/train.jsonl /tmp/t1.jsonl && cp data/eval.jsonl /tmp/e1.jsonl && cp data/labels.json /tmp/l1.json
bun run regen
diff -q data/train.jsonl /tmp/t1.jsonl && diff -q data/eval.jsonl /tmp/e1.jsonl && diff -q data/labels.json /tmp/l1.json && echo DETERMINISTIC
```

Expected: `DETERMINISTIC` (files byte-identical across two runs).

- [ ] **Step 7: Commit**

```bash
git add tools/data/regen.ts files.ts package.json bun.lock web/public/kwproj.json
git rm tools/data/flexrank.ts tools/data/kwvocab.ts web/public/flex.json
git commit -m "feat(regen): uFuzzy keyword payload (kw field + kwproj.json), drop flexrank/kwvocab"
```

(`data/train.jsonl` / `data/eval.jsonl` / `data/labels.json` are gitignored — do not add them. If the lockfile is `bun.lockb`, add that instead. `web/public/flex.json` deletion lands here since `regen` no longer writes it; `web/src/flexrank.fixture.json` is deleted in Task 6 alongside its test.)

---

## Task 4: Python core — FusionHead stack swap, `kw` plumbing, ONNX, pred, deletions

Large but atomic: the import graph (`FLEX_N` / `KWHead` / `FusionHead` / `KW_PT` / `FUSION_PT` / `FLEX_JSON`) couples `data.py`, `model.py`, `train.py`, `export_onnx.py`, `pred.py`, `report.py`, and `files.py`. The tree is red between steps; the gate is at the end.

**Files:**
- Modify: `model/data.py`, `model/config.py`, `model/model.py`, `model/train.py`, `model/export_onnx.py`, `model/pred.py`, `tools/report.py`, `files.py`, `model/test_train_cli.py`
- Delete: `model/flexrank.py`, `model/test_flexrank.py`

**Interfaces:**
- Consumes: `FusionHeadGate` / `FusionHeadGain` / `FusionHeadMix` / `_z` from Task 2; the `kw` row field from Task 3.
- Produces:
  - `model/data.py`: `record.kw: list` (sparse `[[idx, val], ...]`); `_row_kw(row) -> torch.Tensor` shape `[len(EMOJIS)]`; `EmojiDataset.__getitem__` returns `(text, emoji, style, colors, kw)` with `kw` a `[N]` float tensor.
  - `files.py`: `FUSION_GATE_PT`, `FUSION_GAIN_PT`, `FUSION_MIX_PT` (= `pt/fusion_{gate,gain,mix}.pt`), `KWPROJ_JSON` (= `web/public/kwproj.json`). No `FLEX_JSON` / `KW_PT` / `FUSION_PT`.
  - `model/train.py`: `LitEncoder.fusion` is `nn.ModuleDict({"gate":..., "gain":..., "mix":...})` when `"fusion"` in heads; `_step` unpacks 5-tuple with `kw` last; writes `fusion_gate.pt` / `fusion_gain.pt` / `fusion_mix.pt`.
  - `model/export_onnx.py`: `FUSION_EXPORT_VARIANT = "gate"`; `ExportWrapper(enc, style, emoji_embed, emoji, gen).forward(x) -> (style_logits, emoji_logits, color)`.
  - `model/pred.py`: `fusion_top_labels` column when `pt/fusion_gate.pt` exists and the row carries `kw`.

- [ ] **Step 1: `files.py`**

Delete the `KW_PT` and `FUSION_PT` lines and the `FLEX_JSON` line. Add near the other `PT_DIR` constants:

```python
FUSION_GATE_PT = f"{PT_DIR}/fusion_gate.pt"
FUSION_GAIN_PT = f"{PT_DIR}/fusion_gain.pt"
FUSION_MIX_PT = f"{PT_DIR}/fusion_mix.pt"
```

Add near `II_JSON`:

```python
KWPROJ_JSON = f"{WEB_PUBLIC_DIR}/kwproj.json"
```

- [ ] **Step 2: `model/config.py`**

Remove the `DROPOUT_KW` assignment. Run `grep -rn "DROPOUT_KW" model/ tools/` and remove any remaining references (the `KWHead` that used it is deleted in Step 4).

- [ ] **Step 3: `model/data.py`**

- Delete `FLEX_JSON` from the `from files import ...` line (keep `EVAL_JSONL`, `TRAIN_JSONL`).
- Delete the `with open(FLEX_JSON...) as _f: KW_VOCAB ...` block and `FLEX_N = len(KW_VOCAB)` (lines ~12-14).
- `record` dataclass: replace `flex_tf: list = field(default_factory=list)` with `kw: list = field(default_factory=list)`.
- `read`: change the final `yield` — `d.get("flex_tf") or []` → `d.get("kw") or []`.
- Replace `_row_tf` with:

```python
def _row_kw(row) -> torch.Tensor:
    pairs = row.get("kw") if isinstance(row, dict) else row.kw
    out = torch.zeros(len(EMOJIS), dtype=torch.float32)
    for i, v in pairs or []:
        out[int(i)] = float(v)
    return out
```

- `EmojiDataset.__init__`: `self.flex_tf = torch.stack([_row_tf(r) for r in records])` → `self.kw = torch.stack([_row_kw(r) for r in records])`.
- `EmojiDataset.__getitem__`: last element `self.flex_tf[idx]` → `self.kw[idx]`.
- Run `grep -rn "FLEX_N\|KW_VOCAB\|_row_tf\|flex_tf" model/data.py` → no matches.

- [ ] **Step 4: `model/model.py` — delete the dead heads**

- Delete the `KWHead` class and the `FusionHead` class (the `_z` helper + the three new classes stay).
- Remove `DROPOUT_KW` from the `from model.config import (...)` list.
- Remove `FLEX_N` from the `from model.data import ...` line (keep `COLOR_DIM`, `EMOJIS`, `PAD_IDX`, `STYLES`, `VOCAB_SIZE`).
- Run `grep -n "KWHead\|FusionHead\b\|FLEX_N\|DROPOUT_KW" model/model.py` → matches only `FusionHeadGate/Gain/Mix`.

- [ ] **Step 5: `model/train.py` — fusion wiring**

- Imports: from `files` drop `FUSION_PT`, `KW_PT`; add `FUSION_GATE_PT`, `FUSION_GAIN_PT`, `FUSION_MIX_PT`. From `model.model` drop `FusionHead`, `KWHead`; add `FusionHeadGain`, `FusionHeadGate`, `FusionHeadMix`. Drop `FLEX_JSON` if imported.
- `LitEncoder.__init__`: replace

```python
        if "fusion" in self.heads:
            self.kw = KWHead()
            self.fusion = FusionHead()
```

with

```python
        if "fusion" in self.heads:
            self.fusion = nn.ModuleDict(
                {
                    "gate": FusionHeadGate(),
                    "gain": FusionHeadGain(),
                    "mix": FusionHeadMix(),
                }
            )
```

- `_step`: unpack `text, emoji, style, colors, kw = batch`. Replace the whole `if "fusion" in self.heads:` block with:

```python
        if "fusion" in self.heads:
            lm = emoji_logits.detach()
            frrs = []
            for name, head in self.fusion.items():
                fused = head(lm, kw)
                loss_v = lse_infonce(fused, emoji, INFONCE_TEMP)
                loss = loss + loss_v
                self._log(f"loss/fusion_{name}/{split}", loss_v, bs)
                if n_e:
                    frr = mrr(fused[has_e], emoji[has_e]).mean()
                else:
                    frr = torch.zeros((), device=emoji.device)
                frrs.append(frr)
                self._log(f"MRR/fusion_{name}/{split}", frr, max(n_e, 1))
            self._log(f"MRR/fusion/{split}", torch.stack(frrs).max(), max(n_e, 1))
            if n_e:
                krr = mrr(kw[has_e], emoji[has_e]).mean()
            else:
                krr = torch.zeros((), device=emoji.device)
            self._log(f"MRR/kw/{split}", krr, max(n_e, 1))
            self._log(f"gate/a/{split}", self.fusion["gate"].last_a, bs)
            self._log(
                f"gain/beta/{split}",
                torch.nn.functional.softplus(self.fusion["gain"].raw_beta),
                bs,
            )
            self._log(f"mix/g/{split}", torch.sigmoid(self.fusion["mix"].raw_g), bs)
```

- `_step`, `emoji` block: delete the `if "fusion" not in self.heads:` guard so `loss = loss + loss_emoji` always runs when `"emoji"` is selected. (`emoji_logits` and `n_e` / `has_e` are already computed above the fusion block — keep that order.)
- `configure_optimizers`: delete the `if "fusion" in self.heads: params += list(self.kw.parameters())` block. The `for h in self.heads: params += list(getattr(self, h).parameters())` loop already covers `self.fusion` (the `ModuleDict`).
- `_train_encoder` save block: replace

```python
    if "fusion" in heads:
        save_pt(mod.kw.state_dict(), str(out_dir / "kw.pt"), stage="enc")
    for h in ALL_HEADS:
        if h in heads:
            save_pt(getattr(mod, h).state_dict(), str(
                out_dir / f"{h}.pt"), stage="enc")
```

with

```python
    for h in ALL_HEADS:
        if h in heads and h != "fusion":
            save_pt(getattr(mod, h).state_dict(), str(
                out_dir / f"{h}.pt"), stage="enc")
    if "fusion" in heads:
        for name in ("gate", "gain", "mix"):
            save_pt(
                mod.fusion[name].state_dict(),
                str(out_dir / f"fusion_{name}.pt"),
                stage="enc",
            )
```

- `_run_local`, `stage == Stage.gan` branch: in the `_require_pt(pt_dir, [...])` list replace `"kw.pt", "fusion.pt"` with `"fusion_gate.pt", "fusion_gain.pt", "fusion_mix.pt"`.
- `train_remote` signature: replace `kw_bytes` / `fusion_bytes` params with `fusion_gate_bytes`, `fusion_gain_bytes`, `fusion_mix_bytes`. In its `uploads` dict replace `KW_PT: kw_bytes, ... FUSION_PT: fusion_bytes` with the three `FUSION_*_PT: fusion_*_bytes` entries.
- `_run_remote`: in `pt_bytes` replace `"kw_bytes"` / `"fusion_bytes"` keys with `"fusion_gate_bytes"` / `"fusion_gain_bytes"` / `"fusion_mix_bytes"` (all `None`); in the `stage == "gan"` block replace `KW_PT, ... FUSION_PT` in the existence-check tuple with the three `FUSION_*_PT`, and the `pt_bytes = {...}` reads likewise.
- Run `grep -n "kw\.pt\|fusion\.pt\|KW_PT\|FUSION_PT\|self\.kw\|KWHead\|FusionHead\b" model/train.py` → no matches (only `FusionHead{Gate,Gain,Mix}` and `fusion_{gate,gain,mix}.pt` / `FUSION_*_PT`).

- [ ] **Step 6: `model/export_onnx.py`**

- Imports: from `files` drop `FUSION_PT`, `KW_PT`; add `FUSION_GAIN_PT`, `FUSION_GATE_PT`, `FUSION_MIX_PT`. From `model.data` drop `FLEX_N`, `KW_VOCAB` (keep `CHARS`, `PAD_IDX`). From `model.model` drop `FusionHead`, `KWHead`; add `FusionHeadGain`, `FusionHeadGate`, `FusionHeadMix`. Add `from torch.nn.functional import softplus` (or reuse `normalize` import group).
- Add module constants after `COLOR_SAMPLES`:

```python
FUSION_EXPORT_VARIANT = "gate"
_FUSION_CLS = {
    "gate": FusionHeadGate,
    "gain": FusionHeadGain,
    "mix": FusionHeadMix,
}
_FUSION_PT = {
    "gate": FUSION_GATE_PT,
    "gain": FUSION_GAIN_PT,
    "mix": FUSION_MIX_PT,
}
```

- `ExportWrapper`: drop the `kw` and `fusion` ctor params + attributes. New `forward`:

```python
    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        emb = self.enc(x)
        style_logits = self.style(emb)
        emoji_logits = self.emoji_embed.score(self.emoji(emb))
        seed = (1 - Z_WEIGHT) * normalize(emb) + Z_WEIGHT * self.z
        color = torch.tanh(self.gen.net(seed)) * 127.5 + 127.5
        return style_logits, emoji_logits, color
```

- `export_onnx`: `dummy = (torch.zeros(1, MAX_TEXT_LEN, dtype=torch.long),)`; `input_names=["input"]`; `output_names=["style_logits", "emoji_logits", "color"]`; `dynamic_axes` = `{"input": {0: "batch"}, "style_logits": {0: "batch"}, "emoji_logits": {0: "batch"}}`.
- Add:

```python
def _fusion_meta(variant: str, mod: nn.Module) -> dict:
    if variant == "gate":
        return {
            "variant": "gate",
            "bn_mean": mod.bn.running_mean.tolist(),
            "bn_var": mod.bn.running_var.tolist(),
            "w": mod.lin.weight.detach().flatten().tolist(),
            "b": float(mod.lin.bias.detach()),
        }
    if variant == "gain":
        return {"variant": "gain", "beta": float(softplus(mod.raw_beta.detach()))}
    return {"variant": "mix", "g": float(torch.sigmoid(mod.raw_g.detach()))}
```

- `export_web`: in the `meta` dict, drop `"flex_kw"` and `"flex_n"`; add `"fusion": _fusion_meta(FUSION_EXPORT_VARIANT, wrapper.fusion_head)`.
- `export()`: drop the `kw = _load(KWHead(), KW_PT)` and `fusion = _load(FusionHead(), FUSION_PT)` lines. After loading `gen`, add:

```python
    fusion_head = _load(
        _FUSION_CLS[FUSION_EXPORT_VARIANT](), _FUSION_PT[FUSION_EXPORT_VARIANT]
    )
```

Build the wrapper as `ExportWrapper(enc, style, emoji_embed, emoji, gen).eval()` and set `wrapper.fusion_head = fusion_head` before `export_web(wrapper)` (kept off the module tree so it is not traced into the ONNX graph).
- Run `grep -n "flex\|FLEX\|KWHead\|FusionHead\b\|kw_logits\|fusion_logits" model/export_onnx.py` → no matches.

- [ ] **Step 7: `model/pred.py`**

- Imports: drop `from files import FLEX_JSON`; from `model.data` drop `FLEX_N`, add `read`, `_row_kw`, keep `EMOJIS`, `STYLES`, `normalize`, `text_to_tensor`; drop `from model.flexrank import FlexRanker`; from `model.model` drop `FusionHead`, `KWHead`, add `FusionHeadGate`. Add `from files import EVAL_JSONL, FUSION_GATE_PT`.
- Replace `read_texts` with a row reader that keeps `kw`:

```python
def read_rows(lines: list[str]) -> list[dict]:
    rows = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        d = json.loads(line)
        text = normalize(d["text"])[:MAX_TEXT_LEN]
        if text:
            rows.append({"text": text, "kw": d.get("kw") or []})
    return rows
```

- `predict(rows: list[dict], pt_dir: Path)`: replace the `fusion = kw = ranker = None` block with:

```python
    gate = None
    if (pt_dir / "fusion_gate.pt").exists():
        gate = _load(FusionHeadGate(), pt_dir / "fusion_gate.pt")
    else:
        print("fusion_gate.pt missing -- skipping fusion_top_labels", file=sys.stderr)
```

- In the loop, `for row in tqdm(rows, ...)`, `text = row["text"]`. Replace the `if fusion is not None:` block with:

```python
            if gate is not None and row["kw"]:
                kw_vec = _row_kw(row).unsqueeze(0)
                fused = gate(emoji_logits.detach(), kw_vec)
                record["fusion_top_labels"] = top_labels(
                    fused, EMOJIS, min_k=1, max_k=1
                )
```

- `main`: `records = predict(read_rows(lines), pt)`.
- Run `grep -n "flex\|FLEX\|FlexRanker\|KWHead\|FusionHead\b" model/pred.py` → no matches.

- [ ] **Step 8: `tools/report.py` — import swap only**

- In `_flex_keyword_candidates`, change `from model.flexrank import query_tokens` → `from model.tokenize import query_tokens`.
- Run `grep -n "flexrank\|FLEX_JSON\|flex\.json" tools/report.py` → no matches. (The 5-way overlay is Task 5; nothing else changes here.)

- [ ] **Step 9: Delete the dead Python modules**

```bash
git rm model/flexrank.py model/test_flexrank.py
grep -rn "model.flexrank\|model/flexrank\|import flexrank" model/ tools/
```

Expected: no matches.

- [ ] **Step 10: Rewrite the `test_train_cli.py` fusion test**

In `model/test_train_cli.py`, replace `test_fusion_step_end_to_end_grad` (whole function) with:

```python
def test_fusion_step_detaches_trunk_and_trains_heads():
    from model.config import MAX_TEXT_LEN
    from model.data import EMOJIS, STYLES

    torch.manual_seed(0)
    m = T.LitEncoder(heads=("style", "emoji", "critic", "fusion"))
    logged = {}
    m.log = lambda name, val, *a, **k: logged.__setitem__(name, val)

    b = 4
    text = torch.randint(1, 5, (b, MAX_TEXT_LEN))
    emoji = torch.zeros(b, len(EMOJIS))
    emoji[:, 0] = 1.0
    style = torch.zeros(b, len(STYLES))
    style[:, 0] = 1.0
    colors = torch.zeros(b, 9)
    kw = torch.zeros(b, len(EMOJIS))
    kw[:, 1] = 0.5

    loss = m._step((text, emoji, style, colors, kw), "train")
    loss.backward()

    for name in ("gate", "gain", "mix"):
        grads = [p.grad for p in m.fusion[name].parameters()]
        assert grads and all(g is not None for g in grads), name
        assert any(g.abs().sum() > 0 for g in grads), name
        assert f"loss/fusion_{name}/train" in logged
    assert m.emoji.net[1].weight.grad is not None
    assert m.emoji_embed.embed.weight.grad is not None
    assert next(m.enc.parameters()).grad is not None
    assert "MRR/fusion/train" in logged
    assert "gate/a/train" in logged
    assert "mix/g/train" in logged
```

Update the `main()` command body: replace the `test_fusion_step_end_to_end_grad()` call with `test_fusion_step_detaches_trunk_and_trains_heads()`.

- [ ] **Step 11: Gate — lint + every Python check**

```bash
uv run ruff check . && uv run ruff format --check .
uv run python model/test_runmeta.py
uv run python model/test_tokenize.py
uv run python model/test_model_heads.py
uv run python model/test_train_cli.py
uv run python tools/test_report.py
```

Expected: ruff clean; each script prints `ok` (or `test_report.py` prints its `ok` / `skip ...` lines and exits 0).

If `data/train.jsonl` errors as missing a `kw` field, re-run `bun run regen` (Task 3) and retry.

- [ ] **Step 12: Commit**

```bash
git add model/data.py model/config.py model/model.py model/train.py model/export_onnx.py model/pred.py tools/report.py files.py model/test_train_cli.py
git rm model/flexrank.py model/test_flexrank.py
git commit -m "feat(model): swap KWHead/FusionHead for three detached FusionHeads; drop flex_tf/flexrank"
```

---

## Task 5: `tools/report.py` — 5-way `acc@k` overlay

**Files:**
- Modify: `tools/report.py`

**Interfaces:**
- Consumes: `pt/fusion_{gate,gain,mix}.pt` (optional, from a training run); `model.data._row_kw`; `model.model.FusionHead{Gate,Gain,Mix}`.
- Produces: the "Model → Emojis" `data/eval.jsonl` chart gains series `EmojiHead`, `Keywords`, `Fusion·Gate`, `Fusion·Gain`, `Fusion·Mix`; `report.json.emoji` gains keys `keywords`, `fusion_gate`, `fusion_gain`, `fusion_mix` (each a length-10 `acc@k` list, or absent when the `.pt` is missing / shape-mismatched).

- [ ] **Step 1: Read the current emoji section**

Read `tools/report.py` around `_section_emoji` / the `data/eval.jsonl` `acc@k` computation and how it loads optional `.pt` (grep `emoji_embed`, `acc_at_k`, `_load`, `EVAL`). Note the existing helper that turns ranked logits into an `acc@k` curve — reuse it.

- [ ] **Step 2: Add a helper to load the fusion heads**

Near the other `@cache` `.pt` loaders in `tools/report.py`:

```python
@cache
def _fusion_heads() -> dict:
    from model.model import FusionHeadGain, FusionHeadGate, FusionHeadMix

    specs = {
        "gate": (FusionHeadGate, FUSION_GATE_PT),
        "gain": (FusionHeadGain, FUSION_GAIN_PT),
        "mix": (FusionHeadMix, FUSION_MIX_PT),
    }
    out = {}
    for name, (cls, path) in specs.items():
        if not Path(path).exists():
            continue
        try:
            sd, _ = load_pt(path)
            mod = cls()
            mod.load_state_dict(sd)
            mod.eval()
            out[name] = mod
        except (RuntimeError, KeyError):
            continue
    return out
```

Add `FUSION_GATE_PT`, `FUSION_GAIN_PT`, `FUSION_MIX_PT` to the `from files import ...` block. Confirm `load_pt` and `from functools import cache` are already imported (they are used elsewhere in the file).

- [ ] **Step 3: Compute the extra curves in the eval section**

Where the eval `acc@k` for `EmojiHead` is computed (it already has `enc`, `emoji_head`, `emb` = `_emoji_embed()`, the eval `text` tensor batch, and the multi-hot `emoji` targets), add:

```python
    heads = _fusion_heads()
    with torch.no_grad():
        logit_m = emb.score(emoji_head(enc(eval_text)))
        kw_dense = torch.stack([_row_kw(r) for r in _eval_rows_raw()])
        series = {"emojihead": _acc_at_k(logit_m, eval_targets)}
        series["keywords"] = _acc_at_k(kw_dense, eval_targets)
        for name, head in heads.items():
            fused = head(logit_m.detach(), kw_dense)
            series[f"fusion_{name}"] = _acc_at_k(fused, eval_targets)
```

Adapt names to the file's actual locals: `emb` / `emoji_head` / `enc` / the eval text tensor / the eval targets tensor / the existing `acc@k` helper (shown as `_acc_at_k` here). `_eval_rows_raw()` = the list of parsed eval JSON dicts (reuse whatever the section already iterates for targets, or add `[json.loads(l) for l in open(EVAL_JSONL)]` via the existing `_rows` helper). Each `_row_kw` needs the raw dict with a `kw` key.

- [ ] **Step 4: Render + JSON**

- HTML chart: add the `keywords` / `fusion_gate` / `fusion_gain` / `fusion_mix` lines to the same line chart that draws `EmojiHead` + CLDR baseline, with distinct labels (`Keywords`, `Fusion·Gate`, `Fusion·Gain`, `Fusion·Mix`). Only draw a fusion line when its key is present in `series`.
- `report.json`: under the existing `emoji` object, write `series` keys `keywords`, `fusion_gate`, `fusion_gain`, `fusion_mix` alongside the current `emojihead` / baseline arrays.
- In the "Model → Keyword vocab" section text, append a sentence: `report.json keywords_flex.ranked no longer feeds regen (kw vocab is derived directly from data/ii.json).`

- [ ] **Step 5: Gate**

```bash
uv run ruff check tools/report.py && uv run ruff format --check tools/report.py
uv run python tools/test_report.py
```

Expected: ruff clean; `tools/test_report.py` prints `ok` / `skip` lines, exit 0.

If `pt/enc.pt` etc. exist locally: `uv run python tools/report.py` and open the newest `report/*/report.html` — confirm the emoji chart shows up to 5 model lines (fusion lines only if `pt/fusion_*.pt` present) plus the CLDR baseline. If no `.pt` are present, this manual check is deferred to the Task 7 full run.

- [ ] **Step 6: Commit**

```bash
git add tools/report.py
git commit -m "feat(report): 5-way emoji acc@k overlay (EmojiHead/Keywords/Fusion x3)"
```

---

## Task 6: Web — `keywords.js`, `fusion.js`, wiring

**Files:**
- Create: `web/src/keywords.js`, `web/src/keywords.test.js`, `web/src/fusion.js`, `web/src/fusion.test.js`
- Modify: `web/src/hooks/useOnnx.js`, `web/src/App.jsx`, `web/package.json`, `package.json` (root)
- Delete: `web/src/flexrank.js`, `web/src/flexrank.test.js`, `web/src/flexrank.fixture.json`

**Interfaces:**
- Consumes: `web/public/kwproj.json` (Task 3); `meta.json` `fusion` block (Task 4); ONNX outputs `style_logits`, `emoji_logits`, `color` (Task 4); `normalize` from `web/src/model.js`.
- Produces:
  - `web/src/keywords.js`: `export function makeKeywordPredictor(kwprojJson, emojiCount): { predict(normText): Float32Array }` — `normText` is an already-`normalize`d string; Float32Array length `emojiCount`, `kw(e)` per spec §1.3-1.4 (uFuzzy per word of `normText.split(' ')`).
  - `web/src/fusion.js`: `export function makeFusion(metaFusion): { fuse(emojiLogits: Float32Array, kwArr: Float32Array): Float32Array }` — implements `gate` / `gain` / `mix` on raw logits (`z()` computed in JS).

No `web/src/tokenize.js` — word-splitting is `normalize(text, char2idx).split(' ')` and `normalize` already lives in `web/src/model.js`.

- [ ] **Step 1: Dependencies**

```bash
cd web && npm install @leeoniya/uFuzzy && npm uninstall flexsearch && cd ..
```

Then verify nothing else needs `flexsearch` (`grep -rn "flexsearch" tools/ web/src/ --include="*.ts" --include="*.js" --include="*.jsx"` → no matches). Root `package.json` `flexsearch` was removed in Task 3 Step 1; if it is still there, `bun remove flexsearch` now.

- [ ] **Step 2: `web/src/keywords.js` + failing test**

Create `web/src/keywords.test.js`:

```js
import { describe, expect, it } from 'vitest'
import { makeKeywordPredictor } from './keywords'

const proj = {
  pizza: [0],
  beaches: [1, 2],
  dog: [3],
}

describe('makeKeywordPredictor', () => {
  it('scores an exact keyword hit above zero, others zero', () => {
    const kp = makeKeywordPredictor({ proj }, 4)
    const v = kp.predict('pizza tonight')
    expect(v.length).toBe(4)
    expect(v[0]).toBeGreaterThan(0)
    expect(v[3]).toBe(0)
  })

  it('down-weights keys that fan out to many emoji', () => {
    const kp = makeKeywordPredictor({ proj }, 4)
    const v = kp.predict('a day at the beaches')
    expect(v[1]).toBeGreaterThan(0)
    expect(v[1]).toBeLessThan(1)
  })

  it('returns all zeros when nothing matches', () => {
    const kp = makeKeywordPredictor({ proj }, 4)
    expect(Array.from(kp.predict('qwerty xyzzy'))).toEqual([0, 0, 0, 0])
  })
})
```

Run: `cd web && npx vitest run src/keywords.test.js` → FAIL (module missing). `cd ..`

Create `web/src/keywords.js` (mirror of the `regen.ts` `kwVec` block, Task 3 Step 3):

```js
import uFuzzy from '@leeoniya/uFuzzy'

export function makeKeywordPredictor(kwprojJson, emojiCount) {
  const proj = kwprojJson.proj
  const keys = Object.keys(proj)
  const weight = new Map(keys.map((k) => [k, 1 / Math.log2(1 + proj[k].length)]))
  const uf = new uFuzzy({ intraIns: 1 })
  return {
    predict(normText) {
      const best = new Map()
      for (const word of normText.split(' ')) {
        if (word.length < 3) continue
        const idxs = uf.filter(keys, word)
        if (!idxs || !idxs.length) continue
        const info = uf.info(idxs, keys, word)
        for (let i = 0; i < info.idx.length; i++) {
          const k = keys[info.idx[i]]
          const sim = info.chars[i] / k.length
          if (sim >= 0.5 && sim > (best.get(k) ?? 0)) best.set(k, sim)
        }
      }
      const out = new Float32Array(emojiCount)
      for (const [k, sim] of best) {
        const v = sim * weight.get(k)
        for (const e of proj[k]) if (v > out[e]) out[e] = v
      }
      return out
    },
  }
}
```

Run: `cd web && npx vitest run src/keywords.test.js` → PASS. `cd ..`
(If the `beaches` case fails, `console.log(uf.info(uf.filter(['beaches'], 'beaches'), ['beaches'], 'beaches'))` to confirm the `chars` field name in the installed uFuzzy and adjust — same check as Task 3 Step 3's note.)

- [ ] **Step 3: `web/src/fusion.js` + failing test**

Create `web/src/fusion.test.js`:

```js
import { describe, expect, it } from 'vitest'
import { makeFusion } from './fusion'

const logits = Float32Array.from([2, 0, -1, 1])
const kw = Float32Array.from([0, 0.8, 0, 0])

function zscore(a) {
  const m = a.reduce((s, x) => s + x, 0) / a.length
  const v = a.reduce((s, x) => s + (x - m) ** 2, 0) / a.length
  return a.map((x) => (x - m) / (Math.sqrt(v) + 1e-6))
}

describe('makeFusion', () => {
  it('gain: fused = logits + softplus(beta) * kw', () => {
    const f = makeFusion({ variant: 'gain', beta: 0 })
    const out = f.fuse(logits, kw)
    const sp = Math.log1p(Math.exp(0))
    expect(out[1]).toBeCloseTo(logits[1] + sp * kw[1], 5)
    expect(out[0]).toBeCloseTo(logits[0], 5)
  })

  it('mix: fused = (1-g) z(logits) + g kw', () => {
    const f = makeFusion({ variant: 'mix', g: 0.25 })
    const z = zscore(Array.from(logits))
    const out = f.fuse(logits, kw)
    expect(out[1]).toBeCloseTo(0.75 * z[1] + 0.25 * kw[1], 4)
  })

  it('gate: a in (0,1), kw=0 entries equal a*z(logits)', () => {
    const f = makeFusion({
      variant: 'gate',
      bn_mean: [0, 0, 0, 0, 0, 0],
      bn_var: [1, 1, 1, 1, 1, 1],
      w: [0, 0, 0, 0, 0, 0],
      b: 0,
    })
    const z = zscore(Array.from(logits))
    const out = f.fuse(logits, kw)
    expect(out[0]).toBeCloseTo(0.5 * z[0], 4)
  })
})
```

Run: `cd web && npx vitest run src/fusion.test.js` → FAIL. `cd ..`

Create `web/src/fusion.js`:

```js
function zscore(a) {
  let m = 0
  for (const x of a) m += x
  m /= a.length
  let v = 0
  for (const x of a) v += (x - m) ** 2
  v = Math.sqrt(v / a.length) + 1e-6
  const out = new Float32Array(a.length)
  for (let i = 0; i < a.length; i++) out[i] = (a[i] - m) / v
  return out
}

const softplus = (x) => Math.log1p(Math.exp(-Math.abs(x))) + Math.max(x, 0)

export function makeFusion(mf) {
  if (mf.variant === 'gain') {
    const beta = softplus(mf.beta)
    return {
      fuse(logits, kw) {
        const out = new Float32Array(logits.length)
        for (let i = 0; i < logits.length; i++) out[i] = logits[i] + beta * kw[i]
        return out
      },
    }
  }
  if (mf.variant === 'mix') {
    const g = mf.g
    return {
      fuse(logits, kw) {
        const z = zscore(logits)
        const out = new Float32Array(logits.length)
        for (let i = 0; i < logits.length; i++) out[i] = (1 - g) * z[i] + g * kw[i]
        return out
      },
    }
  }
  const { bn_mean, bn_var, w, b } = mf
  return {
    fuse(logits, kw) {
      let mx = -Infinity
      let t1 = -Infinity
      let t2 = -Infinity
      let sum = 0
      let kmax = 0
      let kcount = 0
      let ksum = 0
      let smax = -Infinity
      for (const x of logits) if (x > smax) smax = x
      for (const x of logits) sum += Math.exp(x - smax)
      let ent = 0
      for (const x of logits) {
        const p = Math.exp(x - smax) / sum
        ent -= p * Math.log(p + 1e-9)
        if (x > t1) {
          t2 = t1
          t1 = x
        } else if (x > t2) t2 = x
        if (x > mx) mx = x
      }
      for (const x of kw) {
        if (x > kmax) kmax = x
        if (x > 0) kcount++
        ksum += x
      }
      const feat = [mx, t1 - t2, ent, kmax, kcount, ksum]
      let lin = b
      for (let i = 0; i < 6; i++) {
        const n = (feat[i] - bn_mean[i]) / Math.sqrt(bn_var[i] + 1e-5)
        lin += w[i] * n
      }
      const a = 1 / (1 + Math.exp(-lin))
      const z = zscore(logits)
      const out = new Float32Array(logits.length)
      for (let i = 0; i < logits.length; i++) out[i] = a * z[i] + (1 - a) * kw[i]
      return out
    },
  }
}
```

Run: `cd web && npx vitest run src/fusion.test.js` → PASS. `cd ..`

(Note the BN eval formula uses `eps = 1e-5`, PyTorch `BatchNorm1d`'s default — must match Task 4's export, which reads `running_var` as-is.)

- [ ] **Step 4: `web/src/hooks/useOnnx.js`**

- Add `normalize` to the `../model` import (currently imports `encode`, `decodeColorList`, `sigmoid`).
- Replace `import { makeFlexRanker } from '../flexrank'` with `import { makeKeywordPredictor } from '../keywords'` and `import { makeFusion } from '../fusion'`.
- Rename `flexRef` → `kwRef`; add `fusionRef`.
- In the load effect: replace the `flex.json` fetch with `fetch(BASE + 'kwproj.json').then((r) => r.json())`; after `setMeta(m)`, `kwRef.current = makeKeywordPredictor(kwproj, m.emojis.length)` and `fusionRef.current = makeFusion(m.fusion)`.
- `predict`: drop `flexTf` and the `flex_tf` input tensor. Keep `const ids = encode(text, m, char2idxRef.current)`. Run the session with `{ input: new ort.Tensor('int64', ids, [1, m.max_text_len]) }`. Then:

```js
    const emojiLogits = out.emoji_logits.data
    const normText = normalize(text, char2idxRef.current)
    const kwArr = kwRef.current.predict(normText)
    return {
      feeling: sigmoid(out.style_logits.data),
      emoji: sigmoid(emojiLogits),
      kw: kwArr,
      fusion: fusionRef.current.fuse(Float32Array.from(emojiLogits), kwArr),
      palettes: decodeColorList(out.color.data),
      ms,
    }
```

(`normalize(text, char2idx)` in `web/src/model.js` requires the `char2idx` map — `char2idxRef.current` is already built in the load effect. The regen side uses `tools/data/normalize.ts`; both collapse/lowercase/trim identically.)

- [ ] **Step 5: `web/src/App.jsx`**

Check `pickEmojiList` (line ~37): `mode === 'fusion' ? scores.fusion : mode === 'keywords' ? scores.kw : scores.emoji`. `scores.fusion` and `scores.kw` are now raw score arrays (higher = better), not sigmoid probabilities — confirm `pickEmojiList` only ranks by descending value / `argsort` and applies no probability threshold. If it thresholds at `>= 0.5`, change the `fusion` / `keywords` branches to take the top-N by value with no threshold (match how the model list is sliced by `emojiSlots`). Keep the `EMOJI_MODES` array and the masthead toggle unchanged.

- [ ] **Step 6: Delete flexrank web files**

```bash
git rm web/src/flexrank.js web/src/flexrank.test.js web/src/flexrank.fixture.json
grep -rn "flexrank\|flex.json\|flex_tf\|makeFlexRanker" web/src/
```

Expected: no matches.

- [ ] **Step 7: Gate**

```bash
cd web && npm test && npm run build && cd ..
```

Expected: vitest all green (including new `keywords.test.js` / `fusion.test.js`, minus the removed `flexrank.test.js`); `npm run build` writes `web/dist` with no errors.

- [ ] **Step 8: Commit**

```bash
git add web/src/keywords.js web/src/keywords.test.js web/src/fusion.js web/src/fusion.test.js web/src/hooks/useOnnx.js web/src/App.jsx web/package.json web/package-lock.json package.json bun.lock
git rm web/src/flexrank.js web/src/flexrank.test.js web/src/flexrank.fixture.json
git commit -m "feat(web): live uFuzzy keyword predictor + client-side fusion; drop flexrank"
```

---

## Task 7: `CLAUDE.md` + full-run verification

**Files:**
- Modify: `CLAUDE.md`

- [ ] **Step 1: Update `CLAUDE.md`**

Rewrite the stale passages (search for each phrase):

- The `## Project` paragraph on `KWHead` / `FusionHead` / `flex_tf` / `loss/fusion`: replace with — `EmojiHead` trains standalone (`lse_infonce`); the keyword signal is a non-learned uFuzzy lookup over `data/ii.json` projected to the emoji vocab (per `normalize(text).split(" ")` word), computed offline by `regen.ts` into a sparse `kw` row field and live in the browser; three detached learned combiners `FusionHeadGate` / `FusionHeadGain` / `FusionHeadMix` (in `model/model.py`) each map `(logit_m.detach(), kw)` → fused per-emoji scores, trained in parallel by `lse_infonce`; checkpoint keys on `MRR/fusion/val` = best of the three; stage 1 writes `fusion_gate.pt` / `fusion_gain.pt` / `fusion_mix.pt`.
- `model/train.py` bullet: `--heads … fusion` builds all three combiners; drop the `kw.pt` / `fusion.pt` / `FLEX_N` / `KWHead` references; `gan` stage now requires `fusion_{gate,gain,mix}.pt`.
- `model/export_onnx.py` bullet: two inputs → **one** (`input`); five outputs → **three** (`style_logits`, `emoji_logits`, `color`); `meta.json` gains a `fusion` block (winner variant, `FUSION_EXPORT_VARIANT`); drops `flex_kw` / `flex_n`; `regen.ts` writes `web/public/kwproj.json`.
- `tools/report.py` bullet: the emoji `acc@k` overlay is now `EmojiHead` / `Keywords` / `Fusion·Gate` / `Fusion·Gain` / `Fusion·Mix` on `data/eval.jsonl`; `keywords_flex.ranked` no longer feeds `regen`; `query_tokens` moved to `model/tokenize.py`.
- `regen.ts` bullet: `flex_tf` → `kw` (uFuzzy over `data/ii.json`, projected to vocab, per normalized word); `regen` is deterministic again (no report dependency); writes `web/public/kwproj.json`; `--no-kw` skips both. No bespoke tokenizer — `normalize` + `.split(" ")`.
- Remove mentions of `model/flexrank.py`, `tools/data/flexrank.ts`, `tools/data/kwvocab.ts`, `web/src/flexrank.js`, `web/src/flexrank.fixture.json`, `web/public/flex.json`, `FLEX_JSON`, `flexrank.fixture.json` conformance, `model/test_flexrank.py`, `web/src/flexrank.test.js`, `FLEX_FIXTURE_TEXTS`, the `flexsearch` dep.
- `model/pred.py` bullet: `fusion_top_labels` now needs `pt/fusion_gate.pt` + a `kw` field on the row (from `regen`), not `fusion.pt` / `kw.pt` / `flex.json`.
- `model/config.py` bullet: drop `DROPOUT_KW`; add `FUSION_GATE_PT` / `FUSION_GAIN_PT` / `FUSION_MIX_PT` / `KWPROJ_JSON` to the `files.py` path list; remove `FLEX_JSON` / `KW_PT` / `FUSION_PT`.
- Web bullet: 3-way masthead toggle unchanged (`fusion` / `model` / `keywords`) but fusion + keyword scoring is now fully client-side (`web/src/keywords.js` live uFuzzy + `web/src/fusion.js`); `flex.json` → `kwproj.json`; dep `flexsearch` → `@leeoniya/uFuzzy`.

- [ ] **Step 2: Non-training verification suite**

```bash
uv run ruff check . && uv run ruff format --check .
uv run python model/test_runmeta.py
uv run python model/test_tokenize.py
uv run python model/test_model_heads.py
uv run python model/test_train_cli.py
uv run python tools/test_report.py
cd web && npm test && npm run build && cd ..
```

Expected: all green.

- [ ] **Step 3: Commit**

```bash
git add CLAUDE.md
git commit -m "docs(CLAUDE): simple fusion — three FusionHeads, offline uFuzzy keyword payload"
```

- [ ] **Step 4: Deferred behavioural check (record, do not block)**

State in the handoff that the behavioural gate is a manual `train enc --local` run followed by `uv run python model/pred.py --pt pt` and `uv run python tools/report.py`, watching:
- `MRR/fusion_gate/val`, `MRR/fusion_gain/val`, `MRR/fusion_mix/val` vs `MRR/e/val` in TensorBoard — each fusion variant should land `>= MRR/e/val`.
- `MRR/fusion/val` (checkpoint key) = best of the three.
- `report/<ts>-<sha>/report.html` — the 5-way emoji overlay renders and one `Fusion·*` curve dominates.
- `model/export_onnx.py` runs after and writes `web/public/model.onnx` + `meta.json` (with a `fusion` block) + `config.json`.

---

## Self-Review

**1. Spec coverage**

| Spec section | Task |
|---|---|
| §1.1 Projection | Task 3 Step 3 |
| §1.2 no bespoke tokenizer; `model/tokenize.py` for report only | Task 1 (`model/tokenize.py`); Task 3 Step 3 + Task 6 Step 2 both use `normalize().split(" ")` |
| §1.3 uFuzzy index + per-word query | Task 3 Step 3, Task 6 Step 2 |
| §1.4 Aggregate `kw(e)` (`sim * w_k`) | Task 3 Step 3, Task 6 Step 2 |
| §1.5 Row `kw` field | Task 3 Step 3 |
| §1.6 `kwproj.json` | Task 3 Step 3 |
| §2 `FusionHeadGain/Mix/Gate` | Task 2 |
| §3.1 `nn.ModuleDict` heads | Task 4 Step 5 |
| §3.2 detached step + `loss_emoji` always added | Task 4 Step 5 |
| §3.3 checkpoint ladder (best-of-three `MRR/fusion/val`) | Task 4 Step 5 (monitor string unchanged; value = `torch.stack(frrs).max()`) |
| §3.4 `fusion_{gate,gain,mix}.pt` | Task 4 Step 5 |
| §3.5 gan required files | Task 4 Step 5 |
| §3.6 Modal byte plumbing | Task 4 Step 5 |
| §4.1 `data.py` `kw` plumbing | Task 4 Step 3 |
| §4.2 `config.py` drop `DROPOUT_KW` | Task 4 Step 2 |
| §5.1 5-way overlay | Task 5 |
| §5.2 keyword-vocab diagnostic note | Task 5 Step 4 |
| §5.3 `report.json` keys | Task 5 Step 4 |
| §6.1 ONNX one input / three outputs + `meta.fusion` | Task 4 Step 6 |
| §6.2 web modules + wiring + deps | Task 6 |
| §7 deletions + file inventory | Tasks 3, 4, 6 (`git rm` steps) |
| §8 verification | Task 7 Step 2 + per-task gates |
| §9 open risk (no fixture; keyword smoke test) | Task 6 Step 2 (`keywords.test.js`) |

No gaps.

**2. Placeholder scan**

Task 5 Steps 3-4 name locals abstractly (`_acc_at_k`, `eval_text`, `eval_targets`, `_eval_rows_raw`) because `tools/report.py`'s exact internals must be read first (Step 1). This is deliberate and bounded — the helper to convert ranked scores → `acc@k` and the eval text/target tensors already exist in that section; the step says to reuse them by their real names. Every other step carries literal code.

**3. Type consistency**

- `_row_kw(row)` accepts a dict (`row.get("kw")`) or a `record` (`row.kw`) — used with dicts in `pred.py` (Task 4 Step 7) and `report.py` (Task 5 Step 3), with `record` inside `EmojiDataset` (Task 4 Step 3). Matches the old `_row_tf` dual-mode signature.
- `FusionHead*.forward(logit_m, kw)` — call sites: `train.py` `head(lm, kw)` (Task 4 Step 5), `report.py` `head(logit_m.detach(), kw_dense)` (Task 5 Step 3), `pred.py` `gate(emoji_logits.detach(), kw_vec)` (Task 4 Step 7), tests (Task 2). All pass `[B, N]` / `[B, N]`.
- `FusionHeadGate.last_a` — set in Task 2, read in `train.py` as `self.fusion["gate"].last_a` (Task 4 Step 5) and asserted in Task 2 tests. Consistent (a non-persistent buffer, scalar).
- `makeFusion(metaFusion)` / `makeKeywordPredictor(kwprojJson, emojiCount)` — defined Task 6 Steps 2-3, called in `useOnnx.js` Task 6 Step 4 with `m.fusion` and `(kwproj, m.emojis.length)`. `makeKeywordPredictor.predict` takes an **already-`normalize`d** string; `useOnnx.js` calls `normalize(text, char2idx)` before `predict`. `regen.ts`'s `kwVec` calls `normalize(text)` (from `tools/data/normalize.ts`) — same transform. Consistent.
- uFuzzy usage — `new uFuzzy({ intraIns: 1 })`, `uf.filter(keys, word)`, `uf.info(idxs, keys, word)` reading `info.idx[i]` / `info.chars[i]`, `sim = info.chars[i] / keys[info.idx[i]].length`, floor `sim >= 0.5` — identical in `regen.ts` (Task 3 Step 3) and `web/src/keywords.js` (Task 6 Step 2). Both tasks carry the same "verify `uf.info` field names" note in case the installed uFuzzy differs.
- `meta.json` `fusion` block keys (`variant`, `bn_mean`, `bn_var`, `w`, `b` / `beta` / `g`) — written by `_fusion_meta` (Task 4 Step 6), read by `makeFusion` (Task 6 Step 3). Consistent, including BN `eps = 1e-5`.
- `FUSION_GATE_PT` / `FUSION_GAIN_PT` / `FUSION_MIX_PT` — defined `files.py` (Task 4 Step 1), consumed in `train.py` (Task 4 Step 5), `export_onnx.py` (Task 4 Step 6), `report.py` (Task 5 Step 2). `KWPROJ_JSON` — `files.py` (Task 4 Step 1) + `files.ts` (Task 3 Step 2); `regen.ts` writes via the `files.ts` constant, web fetches the literal `kwproj.json` path.

Fixes applied inline: renumbered Task 3 (tokenizer steps removed) and Task 6 (no `tokenize.js`) after the uFuzzy swap; cross-references updated.
