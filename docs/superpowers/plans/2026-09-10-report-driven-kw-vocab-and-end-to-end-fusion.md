# Report-driven keyword vocab + end-to-end FusionHead — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the `KWHead` keyword vocab the 1000 keywords a trained `EmojiHead` handles worst (selected from `data/ii.json` via `tools/report.py`, consumed by `tools/data/regen.ts`), and train the fusion path end-to-end from a single `loss/fusion` instead of three independent head losses.

**Architecture:** `tools/report.py` scores every qualifying `data/ii.json` keyword with the trained `EmojiHead` and writes a worst-first ranked list into `report.json`. `tools/data/regen.ts` reads the newest report's list, takes the worst 1000, and builds `web/public/flex.json` from them (deterministic `data/ii.json` fallback when no report exists). In `model/train.py`, when `fusion` is selected, `loss/emoji` and `loss/kw` stop being added to the total loss; only `loss/fusion` remains and backpropagates through the gate into `q_txt`→`EmojiHead`→encoder, `q_kw`→`KWHead`, and `EmojiEmbedding` with nothing detached. `FusionHead` bias init changes `4.0 → 0.0` and the mean gate value is logged.

**Tech Stack:** Python 3.13 (PyTorch, Lightning, Typer), `uv`; Bun/TypeScript for the data toolchain; plain-assert Python test scripts (`uv run python <path>`) and `bun test`.

**Spec:** `docs/superpowers/specs/2026-09-10-report-driven-kw-vocab-and-end-to-end-fusion-design.md` — the plan argues from the spec; executors read both.

## Global Constraints

- **No comments or docstrings in source.** Do not add them. Keep existing `# type: ignore` / `# noqa` / shebangs. (Markdown plan/spec text is exempt; `@_app.command()` docstrings already in the test files are exempt — match the file.)
- **Package management is `uv` only.** Never `pip install`. TS toolchain is `bun`.
- **`model/train.py` aborts on a dirty git tree** in every mode (`model/runmeta.py:require_clean_tree`) — a training run is not part of this plan; the behavioural check in Task 7 needs a clean tree (commit first).
- **`normalize` / `encode` must stay byte-identical** between `model/data.py` and `web/src/model.js` — this plan does not touch either.
- **Path constants:** add new data paths to `files.py` **and** `files.ts` first, then reference the constant — never hardcode a bare filename.
- **`ruff` is the Python linter/formatter:** `uv run ruff check .` and `uv run ruff format --check .` must stay clean.
- **Commit after every task.** `git add` only the files that task changed — never `git add -A` (a background job auto-commits unrelated "fix" commits to this branch).
- Verbatim values from the spec: candidate filter = `data/ii.json` key length `3..6` inclusive, key is a single `[a-z0-9]+` non-stopword token (`query_tokens(k) == [k]` / `queryTokens(k).length === 1 && queryTokens(k)[0] === k`), and `set(ii.json[k]) & set(EMOJIS)` non-empty. Vocab cap = `1000`. "missed" = `rank > 10`. `FusionHead` bias init = `0.0`.

---

## File map

| File | Change |
|---|---|
| `model/model.py` | `FusionHead.__init__` bias init `4.0 → 0.0` |
| `model/train.py` | `_step`: `loss/emoji` added only when `fusion` not selected; fusion block un-detaches `q_txt`/`q_kw`/`EmojiEmbedding`, drops `loss_kw` from the total, logs `gate/a/{split}` |
| `model/test_train_cli.py` | new assertion: fusion `_step` grad flow + zero bias + `gate/a` logged |
| `files.py` | add `II_JSON` |
| `files.ts` | add `II_JSON` |
| `tools/report.py` | `_ii_json`, `_flex_keyword_candidates`, `_section_keywords_flex`, `_keywords_flex_html`; wire into `build_report` + `_render_html` |
| `tools/test_report.py` | new assertions for `_flex_keyword_candidates` / `_section_keywords_flex` / `_keywords_flex_html` |
| `tools/data/kwvocab.ts` | **new** — `keywordVocabFromIiJson`, `keywordVocabFromReport` |
| `tools/data/kwvocab.test.ts` | **new** — `bun test` for both |
| `tools/data/flexrank.ts` | `buildFlexRanker(kwVocab: string[])` — drop CLDR `df==1` build, now sync |
| `tools/data/flexrank.test.ts` | rewrite for the new signature |
| `tools/data/regen.ts` | `useKw` block: pick vocab from report ↔ ii.json, call sync `buildFlexRanker`, report source in `kwLine` |
| `CLAUDE.md` | update the model paragraph, the `regen.ts` bullet, purity claims |

---

## Task 1: End-to-end FusionHead training

**Files:**
- Modify: `model/model.py` (`FusionHead.__init__`, ~line 126-132)
- Modify: `model/train.py` (`LitEncoder._step`, ~line 239-279)
- Test: `model/test_train_cli.py` (add one test fn + register it in `main`)

**Interfaces:**
- Consumes: nothing new.
- Produces: `LitEncoder(heads=("style","emoji","critic","fusion"))._step(batch, "train")` returns a scalar loss whose `.backward()` populates `.grad` on `m.kw.net[1].weight`, `m.emoji.net[1].weight`, `m.emoji_embed.embed.weight`, and encoder params. `FusionHead().net.bias` is all-zeros. `_step` logs `gate/a/train` and `gate/a/val`.

- [ ] **Step 1: Write the failing test**

Add to `model/test_train_cli.py` (after `test_litencoder_builds_only_selected_heads`):

```python
def test_fusion_step_end_to_end_grad():
    from model.data import EMOJIS, FLEX_N, MAX_TEXT_LEN, STYLES
    from model.model import FusionHead

    assert torch.count_nonzero(FusionHead().net.bias) == 0

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
    flex_tf = torch.zeros(b, FLEX_N)
    flex_tf[:, 0] = 1.0

    loss = m._step((text, emoji, style, colors, flex_tf), "train")
    loss.backward()

    assert m.kw.net[1].weight.grad is not None
    assert m.kw.net[1].weight.grad.abs().sum() > 0
    assert m.emoji.net[1].weight.grad is not None
    assert m.emoji.net[1].weight.grad.abs().sum() > 0
    assert m.emoji_embed.embed.weight.grad is not None
    assert next(m.enc.parameters()).grad is not None
    assert "gate/a/train" in logged
```

Register it in `main()`:

```python
    test_litencoder_builds_only_selected_heads()
    test_fusion_step_end_to_end_grad()
    test_colorcritic_forward_shape()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run python model/test_train_cli.py`
Expected: `AssertionError` — either `FusionHead().net.bias` is non-zero (currently `4.0`) or `m.kw.net[1].weight.grad.abs().sum()` is `0` (currently `q_kw` is detached in the fused path and `loss_kw` gives grad but through a different route — the assertion that pins the new behavior is the zero-bias line, which fails first).

- [ ] **Step 3: Change `FusionHead` bias init**

In `model/model.py`, `FusionHead.__init__`:

```python
        self.net = nn.Linear(TEXT_EMBED_SIZE + FLEX_N, 1)
        nn.init.zeros_(self.net.weight)
        nn.init.constant_(self.net.bias, 0.0)
```

- [ ] **Step 4: Rewrite the `_step` emoji + fusion blocks**

In `model/train.py:_step`, the `if "emoji" in self.heads:` block — change only the loss-accumulation line so it is skipped when `fusion` is selected:

```python
        if "emoji" in self.heads:
            q_txt = self.emoji(enc)
            emoji_logits = self.emoji_embed.score(q_txt)
            loss_emoji = lse_infonce(emoji_logits, emoji, INFONCE_TEMP)
            if "fusion" not in self.heads:
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

Replace the whole `if "fusion" in self.heads:` block with:

```python
        if "fusion" in self.heads:
            q_kw = self.kw(flex_tf)
            kw_logits = self.emoji_embed.score(q_kw)

            a = self.fusion(enc, flex_tf).unsqueeze(-1)
            q_fused = a * q_txt + (1 - a) * q_kw
            fusion_logits = self.emoji_embed.score(q_fused)
            loss_fusion = lse_infonce(fusion_logits, emoji, INFONCE_TEMP)
            loss = loss + loss_fusion
            self._log(f"loss/fusion/{split}", loss_fusion, bs)
            self._log(f"gate/a/{split}", a.mean(), bs)

            if n_e:
                krr = mrr(kw_logits[has_e], emoji[has_e]).mean()
                frr = mrr(fusion_logits[has_e], emoji[has_e]).mean()
            else:
                krr = torch.zeros((), device=emoji.device)
                frr = torch.zeros((), device=emoji.device)
            self._log(f"MRR/kw/{split}", krr, max(n_e, 1))
            self._log(f"MRR/fusion/{split}", frr, max(n_e, 1))
```

Changes vs. current: `loss_kw` and its `self._log("loss/kw/...")` are gone; `q_txt` / `q_kw` are no longer `.detach()`ed; `EmojiEmbedding.score` is used directly instead of the detached `w`/`b` matmul; `gate/a/{split}` is logged. `configure_optimizers` is unchanged (`emoji_embed` params already join when `"emoji" in heads`; `kw` params already join when `"fusion" in heads`).

- [ ] **Step 5: Run test to verify it passes**

Run: `uv run python model/test_train_cli.py`
Expected: `ok`

- [ ] **Step 6: Lint**

Run: `uv run ruff check model/ && uv run ruff format --check model/`
Expected: clean. If `format --check` complains, run `uv run ruff format model/model.py model/train.py model/test_train_cli.py` and re-run.

- [ ] **Step 7: Commit**

```bash
git add model/model.py model/train.py model/test_train_cli.py
git commit -m "feat(train): end-to-end FusionHead loss, drop aux emoji/kw losses

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01A7ANy3iuZSfpa6yPbfY7jP"
```

---

## Task 2: Report — ii.json keyword candidates + EmojiHead rank

**Files:**
- Modify: `files.py` (add `II_JSON` next to `KEYWORDS_JSON`, ~line 12)
- Modify: `tools/report.py` (add `II_JSON` to the `from files import (...)` block; add `_ii_json`, `_flex_keyword_candidates`, `_section_keywords_flex` near `_keyword_probe`, ~line 249)
- Test: `tools/test_report.py` (add one test fn + register it)

**Interfaces:**
- Consumes: `EMOJIS` (from `model.config`), `text_to_tensor`, `norm_text` (already imported in `report.py`), `_emoji_embed()`, `model.flexrank.query_tokens`.
- Produces:
  - `II_JSON: str` in `files` = `"data/ii.json"`.
  - `tools.report._flex_keyword_candidates() -> tuple[tuple[str, list[str]], ...]` — sorted `(keyword, [target emojis ∩ EMOJIS])` pairs passing the candidate filter. `@cache`.
  - `tools.report._section_keywords_flex(enc, emoji_head) -> dict` — `{}` when `enc`/`emoji_head`/`_emoji_embed()` is `None` or there are no candidates; otherwise `{"candidates": int, "missed": int, "ranked": [{"kw": str, "emojis": [str], "top5": [str], "rank": int}, ...]}` with `ranked` sorted by `rank` descending (worst first).

- [ ] **Step 1: Write the failing test**

Add to `tools/test_report.py` (after `test_kw_section_keys`):

```python
def test_flex_keyword_candidates_and_section():
    from model.flexrank import query_tokens
    from tools.report import (
        EMOJIS,
        _flex_keyword_candidates,
        _section_keywords_flex,
    )

    cands = _flex_keyword_candidates()
    if not cands:
        print("skip test_flex_keyword_candidates_and_section (no data/ii.json)")
        return
    vocab = set(EMOJIS)
    for kw, tgt in cands:
        assert 3 <= len(kw) <= 6, kw
        assert query_tokens(kw) == [kw], kw
        assert tgt and all(e in vocab for e in tgt), (kw, tgt)
    assert list(cands) == sorted(cands), "candidates not sorted"

    assert _section_keywords_flex(None, None) == {}
```

Register it in `main()`:

```python
    test_kw_section_keys()
    test_flex_keyword_candidates_and_section()
    print("ok")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run python tools/test_report.py`
Expected: `ImportError: cannot import name '_flex_keyword_candidates' from 'tools.report'`

- [ ] **Step 3: Add the `II_JSON` constant**

In `files.py`, after the `KEYWORDS_JSON = ...` line:

```python
KEYWORDS_JSON = f"{DATA_DIR}/keywords.json"
II_JSON = f"{DATA_DIR}/ii.json"
```

- [ ] **Step 4: Import it and add the helpers in `tools/report.py`**

Add `II_JSON` to the `from files import (...)` block (keep alphabetical-ish; put it after `FUSION_PT`):

```python
from files import (
    CLDR_BASELINE_JSON,
    CLDR_JSONL,
    COLORS_JSONL,
    DATA_JSONL,
    EMOJI_EMBED_PT,
    FLEX_JSON,
    FUSION_PT,
    II_JSON,
    KEYWORDS_JSON,
    KW_PT,
)
```

Add these three functions just above `def _keyword_probe(enc, head):`:

```python
@cache
def _ii_json() -> dict:
    p = Path(II_JSON)
    if not p.exists():
        return {}
    return json.loads(p.read_text(encoding="utf-8"))


@cache
def _flex_keyword_candidates() -> tuple:
    from model.flexrank import query_tokens

    vocab = set(EMOJIS)
    out = []
    for kw, emojis in _ii_json().items():
        if not 3 <= len(kw) <= 6:
            continue
        if query_tokens(kw) != [kw]:
            continue
        tgt = [e for e in emojis if e in vocab]
        if not tgt:
            continue
        out.append((kw, tgt))
    return tuple(sorted(out))


def _section_keywords_flex(enc, emoji_head) -> dict:
    emb = _emoji_embed()
    cands = _flex_keyword_candidates()
    if enc is None or emoji_head is None or emb is None or not cands:
        return {}
    idx = {e: i for i, e in enumerate(EMOJIS)}
    with torch.no_grad():
        texts = torch.stack([text_to_tensor(norm_text(kw)) for kw, _ in cands])
        order = emb.score(emoji_head(enc(texts))).argsort(dim=-1, descending=True)
    ranked = []
    for row, (kw, tgt) in enumerate(cands):
        pos = order[row].tolist()
        rank = min(pos.index(idx[e]) + 1 for e in tgt)
        ranked.append(
            {
                "kw": kw,
                "emojis": tgt,
                "top5": [EMOJIS[i] for i in pos[:5]],
                "rank": rank,
            }
        )
    ranked.sort(key=lambda r: r["rank"], reverse=True)
    return {
        "candidates": len(ranked),
        "missed": sum(1 for r in ranked if r["rank"] > 10),
        "ranked": ranked,
    }
```

- [ ] **Step 5: Run test to verify it passes**

Run: `uv run python tools/test_report.py`
Expected: `ok` (the new test runs against the real `data/ii.json`; it needs `bun run regen` to have produced `data/labels.json` — if `EMOJIS` import fails, run `bun run regen` first).

- [ ] **Step 6: Lint**

Run: `uv run ruff check tools/ files.py && uv run ruff format --check tools/report.py tools/test_report.py files.py`
Expected: clean.

- [ ] **Step 7: Commit**

```bash
git add files.py tools/report.py tools/test_report.py
git commit -m "feat(report): ii.json keyword candidates + EmojiHead rank

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01A7ANy3iuZSfpa6yPbfY7jP"
```

---

## Task 3: Report — HTML section + `build_report` wiring

**Files:**
- Modify: `tools/report.py` (`_keywords_flex_html`; `build_report` `want` default + `need_enc` + section call; `_render_html` body list)
- Test: `tools/test_report.py` (add one test fn + register it)

**Interfaces:**
- Consumes: `_section_keywords_flex` output shape from Task 2; `_esc`, `_fnum` (already in `report.py`).
- Produces: `tools.report._keywords_flex_html(d: dict) -> str`. For `d == {}` returns a string containing `Model — Keyword vocab` and `not available`. For a populated `d` returns a string containing an `<h2>Model — Keyword vocab</h2>`, a `.note` line with `d["candidates"]` and `d["missed"]`, and a `<table>` with a `<th>Keyword</th>` … `<th>Rank</th>` header row followed by one `<tr>` per `ranked` entry whose `rank > 10`. `build_report` adds `report["keywords_flex"]` when `"keywords_flex"` is in `want` (in the default set); `_render_html` renders it right after the emoji section.

- [ ] **Step 1: Write the failing test**

Add to `tools/test_report.py` (after `test_flex_keyword_candidates_and_section`):

```python
def test_keywords_flex_html():
    from tools.report import _keywords_flex_html

    empty = _keywords_flex_html({})
    assert "Model — Keyword vocab" in empty and "not available" in empty

    d = {
        "candidates": 3,
        "missed": 2,
        "ranked": [
            {"kw": "zzzz", "emojis": ["😀"], "top5": ["🍕", "🎉", "😀", "🐶", "🚀"], "rank": 42},
            {"kw": "yyyy", "emojis": ["🎉", "🥳"], "top5": ["🍕", "🎉"], "rank": 11},
            {"kw": "aaaa", "emojis": ["🍕"], "top5": ["🍕"], "rank": 1},
        ],
    }
    h = _keywords_flex_html(d)
    assert "<h2>Model — Keyword vocab</h2>" in h
    assert ">3<" in h or "3 candidates" in h or "3</" in h
    assert "<th>Keyword</th>" in h and "<th>Rank</th>" in h
    assert ">zzzz<" in h and ">yyyy<" in h
    assert ">aaaa<" not in h  # rank 1 is not "missed"
    assert h.index(">zzzz<") < h.index(">yyyy<")  # worst first
```

Register it in `main()`:

```python
    test_flex_keyword_candidates_and_section()
    test_keywords_flex_html()
    print("ok")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run python tools/test_report.py`
Expected: `ImportError: cannot import name '_keywords_flex_html'`

- [ ] **Step 3: Add `_keywords_flex_html`**

In `tools/report.py`, next to `_emoji_html` / `_cldr_html` (rendering section, ~line 799):

```python
def _keywords_flex_html(d) -> str:
    if not d:
        return (
            "<h2>Model — Keyword vocab</h2>"
            '<p class="note">enc.pt / emoji.pt / emoji_embed.pt not available.</p>'
        )
    head = (
        "<h2>Model — Keyword vocab</h2>"
        f'<p class="note">{_fnum(d["candidates"])} candidates &middot; '
        f'{_fnum(d["missed"])} missed (rank &gt; 10) &middot; vocab = worst 1000</p>'
    )
    misses = [r for r in d["ranked"] if r["rank"] > 10]
    if not misses:
        return head + '<p class="note">no misses.</p>'
    body = "".join(
        f"<tr><td>{_esc(r['kw'])}</td><td>{_esc(' '.join(r['emojis']))}</td>"
        f"<td>{_esc(' '.join(r['top5']))}</td><td>{r['rank']}</td></tr>"
        for r in misses
    )
    return (
        head
        + "<table><tr><th>Keyword</th><th>CLDR emojis</th>"
        "<th>Top 5 predictions</th><th>Rank</th></tr>"
        + body
        + "</table>"
    )
```

- [ ] **Step 4: Wire into `build_report`**

In `build_report`, extend the default `want` set:

```python
    want = {s.strip() for s in only.split(",") if s.strip()} or {
        "data",
        "labels",
        "emoji",
        "cldr",
        "cards",
        "keywords_flex",
    }
```

Extend `need_enc`:

```python
    need_enc = bool({"emoji", "cldr", "cards", "keywords_flex"} & want)
```

Add the section call right after the `report["emoji"] = _section_emoji(...)` line:

```python
    if "emoji" in want:
        report["emoji"] = _section_emoji(enc, emoji_head, eval_records)
    if "keywords_flex" in want:
        report["keywords_flex"] = _section_keywords_flex(enc, emoji_head)
```

- [ ] **Step 5: Wire into `_render_html`**

```python
    if "emoji" in report:
        body.append(_emoji_html(report["emoji"]))
    if "keywords_flex" in report:
        body.append(_keywords_flex_html(report["keywords_flex"]))
    if "cldr" in report:
        body.append(_cldr_html(report["cldr"]))
```

- [ ] **Step 6: Run test to verify it passes**

Run: `uv run python tools/test_report.py`
Expected: `ok`

- [ ] **Step 7: Lint**

Run: `uv run ruff check tools/ && uv run ruff format --check tools/report.py tools/test_report.py`
Expected: clean.

- [ ] **Step 8: Commit**

```bash
git add tools/report.py tools/test_report.py
git commit -m "feat(report): 'Model — Keyword vocab' section with missed-keyword table

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01A7ANy3iuZSfpa6yPbfY7jP"
```

---

## Task 4: `tools/data/kwvocab.ts` — vocab sources

**Files:**
- Modify: `files.ts` (add `II_JSON` next to `KEYWORDS_JSON`, ~line 14)
- Create: `tools/data/kwvocab.ts`
- Test: `tools/data/kwvocab.test.ts` (new)

**Interfaces:**
- Consumes: `queryTokens` from `tools/analysis/cldr-baseline.ts`; `II_JSON`, `REPORT_DIR` from `files.ts`; `report.json`'s `keywords_flex.ranked` shape from Task 2 (`{kw: string, rank: number}[]`).
- Produces (`tools/data/kwvocab.ts`):
  - `keywordVocabFromIiJson(emojiVocab: string[]): string[]` — `data/ii.json` keys passing the candidate filter (length `3..6`, `queryTokens(k).length === 1 && queryTokens(k)[0] === k`, `ii.json[k]` intersects `emojiVocab`), deduped, alphabetical, sliced to `1000`.
  - `keywordVocabFromReport(dir?: string): string[] | null` — the newest folder under `dir` (default `REPORT_DIR`) that contains a `report.json`; returns `report.keywords_flex.ranked.slice(0, 1000).map(r => r.kw).sort()`, or `null` when that dir/file is absent, unparseable, or lacks a non-empty `ranked`.

- [ ] **Step 1: Write the failing test**

Create `tools/data/kwvocab.test.ts`:

```ts
import { describe, expect, it } from "bun:test"
import { mkdirSync, mkdtempSync, rmSync, writeFileSync } from "node:fs"
import { tmpdir } from "node:os"
import { join } from "node:path"

import { keywordVocabFromIiJson, keywordVocabFromReport } from "./kwvocab.ts"

describe("keywordVocabFromIiJson", () => {
  const v = keywordVocabFromIiJson(["\u{1F600}", "\u{1F355}", "\u{1F389}"])
  it("keys are 3-6 char single lowercase alnum tokens", () => {
    for (const k of v) {
      expect(k).toMatch(/^[a-z0-9]+$/)
      expect(k.length).toBeGreaterThanOrEqual(3)
      expect(k.length).toBeLessThanOrEqual(6)
    }
  })
  it("deduped, alphabetical, capped at 1000", () => {
    expect(new Set(v).size).toBe(v.length)
    expect([...v].sort()).toEqual(v)
    expect(v.length).toBeLessThanOrEqual(1000)
  })
  it("empty emoji set -> empty vocab", () => {
    expect(keywordVocabFromIiJson([])).toEqual([])
  })
})

describe("keywordVocabFromReport", () => {
  it("null when the dir has no report.json", () => {
    const d = mkdtempSync(join(tmpdir(), "rep-"))
    try {
      expect(keywordVocabFromReport(d)).toBeNull()
    } finally {
      rmSync(d, { recursive: true, force: true })
    }
  })
  it("null when dir does not exist", () => {
    expect(keywordVocabFromReport(join(tmpdir(), "nope-does-not-exist"))).toBeNull()
  })
  it("newest folder's ranked, top-1000, alphabetical", () => {
    const d = mkdtempSync(join(tmpdir(), "rep-"))
    try {
      mkdirSync(join(d, "26-01-01-00-00-aaa"))
      mkdirSync(join(d, "26-02-02-00-00-bbb"))
      writeFileSync(
        join(d, "26-01-01-00-00-aaa", "report.json"),
        JSON.stringify({ keywords_flex: { ranked: [{ kw: "old", rank: 99 }] } }),
      )
      writeFileSync(
        join(d, "26-02-02-00-00-bbb", "report.json"),
        JSON.stringify({
          keywords_flex: {
            ranked: [
              { kw: "zebra", rank: 50 },
              { kw: "apple", rank: 40 },
            ],
          },
        }),
      )
      expect(keywordVocabFromReport(d)).toEqual(["apple", "zebra"])
    } finally {
      rmSync(d, { recursive: true, force: true })
    }
  })
  it("null when the newest report lacks keywords_flex", () => {
    const d = mkdtempSync(join(tmpdir(), "rep-"))
    try {
      mkdirSync(join(d, "26-03-03-00-00-ccc"))
      writeFileSync(join(d, "26-03-03-00-00-ccc", "report.json"), JSON.stringify({ data: {} }))
      expect(keywordVocabFromReport(d)).toBeNull()
    } finally {
      rmSync(d, { recursive: true, force: true })
    }
  })
})
```

- [ ] **Step 2: Run test to verify it fails**

Run: `bun test tools/data/kwvocab.test.ts`
Expected: FAIL — `Cannot find module './kwvocab.ts'`.

- [ ] **Step 3: Add `II_JSON` to `files.ts`**

After the `KEYWORDS_JSON` line:

```ts
export const KEYWORDS_JSON = `${DATA_DIR}/keywords.json`
export const II_JSON = `${DATA_DIR}/ii.json`
```

- [ ] **Step 4: Create `tools/data/kwvocab.ts`**

```ts
import { existsSync, readdirSync, readFileSync } from "node:fs"
import { join } from "node:path"

import { II_JSON, REPORT_DIR } from "../../files.ts"
import { queryTokens } from "../analysis/cldr-baseline.ts"

const CAP = 1000

function isCandidate(kw: string): boolean {
  if (kw.length < 3 || kw.length > 6) return false
  const qt = queryTokens(kw)
  return qt.length === 1 && qt[0] === kw
}

export function keywordVocabFromIiJson(emojiVocab: string[]): string[] {
  const emojis = new Set(emojiVocab)
  const ii = JSON.parse(readFileSync(II_JSON, "utf8")) as Record<string, string[]>
  const out: string[] = []
  for (const [kw, es] of Object.entries(ii)) {
    if (!isCandidate(kw)) continue
    if (!es.some((e) => emojis.has(e))) continue
    out.push(kw)
  }
  return [...new Set(out)].sort().slice(0, CAP)
}

export function keywordVocabFromReport(dir: string = REPORT_DIR): string[] | null {
  if (!existsSync(dir)) return null
  const folders = readdirSync(dir, { withFileTypes: true })
    .filter((e) => e.isDirectory())
    .map((e) => e.name)
    .sort()
  for (let i = folders.length - 1; i >= 0; i--) {
    const p = join(dir, folders[i], "report.json")
    if (!existsSync(p)) continue
    let ranked: { kw: string; rank: number }[] | undefined
    try {
      const parsed = JSON.parse(readFileSync(p, "utf8")) as {
        keywords_flex?: { ranked?: { kw: string; rank: number }[] }
      }
      ranked = parsed.keywords_flex?.ranked
    } catch {
      return null
    }
    if (!ranked || ranked.length === 0) return null
    return ranked
      .slice(0, CAP)
      .map((r) => r.kw)
      .sort()
  }
  return null
}
```

- [ ] **Step 5: Run test to verify it passes**

Run: `bun test tools/data/kwvocab.test.ts`
Expected: all pass. (`keywordVocabFromIiJson` reads the real `data/ii.json`; it is git-tracked so this works on a fresh checkout.)

- [ ] **Step 6: Commit**

```bash
git add files.ts tools/data/kwvocab.ts tools/data/kwvocab.test.ts
git commit -m "feat(regen): kwvocab.ts — report-driven + ii.json keyword vocab sources

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01A7ANy3iuZSfpa6yPbfY7jP"
```

---

## Task 5: `flexrank.ts` takes a vocab; `regen.ts` picks the source

**Files:**
- Modify: `tools/data/flexrank.ts` (whole file — `buildFlexRanker` signature + drop CLDR)
- Modify: `tools/data/flexrank.test.ts` (rewrite)
- Modify: `tools/data/regen.ts` (import from `kwvocab.ts`; `useKw` block ~line 346-376)

**Interfaces:**
- Consumes: `keywordVocabFromReport`, `keywordVocabFromIiJson` from Task 4; `emojis` (the freshly-computed vocab array already in scope in `regen.ts` at line ~338).
- Produces: `buildFlexRanker(kwVocab: string[])` — **synchronous** now — returns `{ kwVocab: string[]; tfVec: (text: string) => number[]; buildJson: () => { kw_vocab: string[] } }`, `kwVocab` echoing the input order. `FlexJson` type unchanged.

- [ ] **Step 1: Rewrite `tools/data/flexrank.test.ts`**

```ts
import { describe, expect, it } from "bun:test"

import { buildFlexRanker } from "./flexrank.ts"

const VOCAB = ["apple", "banana", "cherry", "grape", "lemon"]

describe("flex ranker", () => {
  it("kwVocab echoes the given vocab", () => {
    expect(buildFlexRanker(VOCAB).kwVocab).toEqual(VOCAB)
  })

  it("tfVec: exact hit = 1.0, miss = 0, dense length = |vocab|", () => {
    const r = buildFlexRanker(VOCAB)
    const v = r.tfVec("apple")
    expect(v.length).toBe(VOCAB.length)
    expect(v[0]).toBe(1.0)
    expect(r.tfVec("zzzznotawordzzzz").every((x) => x === 0)).toBe(true)
  })

  it("tfVec: fuzzy prefix match scores min/max length ratio", () => {
    expect(buildFlexRanker(["apple"]).tfVec("apples")[0]).toBeCloseTo(0.833, 3)
  })

  it("buildJson emits only kw_vocab", () => {
    const r = buildFlexRanker(VOCAB)
    expect(Object.keys(r.buildJson())).toEqual(["kw_vocab"])
    expect(r.buildJson().kw_vocab).toEqual(VOCAB)
  })
})
```

- [ ] **Step 2: Run test to verify it fails**

Run: `bun test tools/data/flexrank.test.ts`
Expected: FAIL — the current `buildFlexRanker(corpusTexts)` awaits `loadCldrAnnotations()` and derives its own vocab, so `kwVocab` will not equal `VOCAB` (and the call is `async`, returning a Promise).

- [ ] **Step 3: Rewrite `tools/data/flexrank.ts`**

```ts
import { FUZZY_MIN_LEN, queryTokens } from "../analysis/cldr-baseline.ts"
import { MIN_FUZZY_SCORE } from "./config"

export type FlexJson = { kw_vocab: string[] }

const r3 = (x: number) => Number(x.toFixed(3))

function overlap(w: string, k: string): number {
  if (w === k) return 1.0
  if (w.length < FUZZY_MIN_LEN || k.length < FUZZY_MIN_LEN) return 0.0
  if (!(w.startsWith(k) || k.startsWith(w))) return 0.0
  const r = Math.min(w.length, k.length) / Math.max(w.length, k.length)
  return r >= MIN_FUZZY_SCORE ? r : 0.0
}

export function buildFlexRanker(kwVocab: string[]): {
  kwVocab: string[]
  tfVec: (text: string) => number[]
  buildJson: () => FlexJson
} {
  const vocab = [...kwVocab]
  const tfVec = (text: string): number[] => {
    const q = queryTokens(text)
    return vocab.map((k) => {
      let s = 0
      for (const w of q) s += overlap(w, k)
      return r3(s)
    })
  }
  return { kwVocab: vocab, tfVec, buildJson: () => ({ kw_vocab: vocab }) }
}
```

(Dropped imports: `KW_MAX_LEN`, `KW_MIN_LEN`, `loadCldrAnnotations`, and the `df` / `corpusTokens` build.)

- [ ] **Step 4: Run test to verify it passes**

Run: `bun test tools/data/flexrank.test.ts`
Expected: all pass.

- [ ] **Step 5: Rewire `regen.ts`**

Add the import near the top imports (with the other `./` imports, ~line 4):

```ts
import { buildFlexRanker } from "./flexrank.ts"
import { keywordVocabFromIiJson, keywordVocabFromReport } from "./kwvocab.ts"
```

In the `if (useKw) {` block, replace:

```ts
    console.log("computing soft-TF fusion vectors...")
    const corpusTexts = (master as { text?: unknown }[])
      .map((r) => r.text)
      .filter((t): t is string => typeof t === "string")
    const ranker = await buildFlexRanker(corpusTexts)
```

with:

```ts
    console.log("computing soft-TF fusion vectors...")
    const reportVocab = keywordVocabFromReport()
    const kwVocab = reportVocab ?? keywordVocabFromIiJson(emojis)
    const kwSource = reportVocab ? "report" : "ii.json bootstrap"
    const ranker = buildFlexRanker(kwVocab)
```

And update the summary line at the end of that block:

```ts
    kwLine =
      `flex_tf               : N=${ranker.kwVocab.length}, `
      + `mean nz ${(nzSum / denom).toFixed(2)}, source=${kwSource}`
```

- [ ] **Step 6: Typecheck / run regen dry**

Run: `bun test tools/data/` (flexrank + kwvocab + any existing data tests)
Run: `grep -n "loadCldrAnnotations\|KW_MIN_LEN\|KW_MAX_LEN" tools/data/flexrank.ts` → expect no matches.
Run: `grep -n "corpusTexts" tools/data/regen.ts` → expect no matches.

- [ ] **Step 7: Commit**

```bash
git add tools/data/flexrank.ts tools/data/flexrank.test.ts tools/data/regen.ts
git commit -m "feat(regen): buildFlexRanker takes a vocab; regen picks report ↔ ii.json

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01A7ANy3iuZSfpa6yPbfY7jP"
```

---

## Task 6: Update `CLAUDE.md`

**Files:**
- Modify: `CLAUDE.md` (the `tools/report.py` bullet line 10; the model paragraph line 7; the `regen.ts` bullet line 18)

No test. Prose only.

- [ ] **Step 0: `tools/report.py` bullet (line 10)**

In the opening usage: `[--only data,labels,emoji,cldr,cards]` → `[--only data,labels,emoji,keywords_flex,cldr,cards]`.

After the **Model → Emojis** sentence in the Sections list, insert:

> **Model → Keyword vocab** (only with `pt/enc.pt` + `pt/emoji.pt` + `pt/emoji_embed.pt`) — every `data/ii.json` key that is 3–6 chars, a single `[a-z0-9]+` non-stopword token, and has ≥1 target emoji in `data/labels.json` is scored by `EmojiHead`; `rank` is the first position any target emoji reaches. A summary line (`N candidates · M missed (rank > 10) · vocab = worst 1000`) and a table of the `rank > 10` misses sorted worst-first (Keyword · CLDR emojis · Top 5 predictions · Rank). `tools/data/regen.ts` reads the full worst-first list from `report.json` to pick the `flex_tf` vocab;

In the `report.json` mirrors clause: `provenance` / `data` / `labels` / `emoji` / `cldr` / `cards` → add `keywords_flex`.

- [ ] **Step 1: Model paragraph (line 7) — the FusionHead sentence**

Find:

> `model/model.py:FusionHead` is now just a **gate** — `Linear(TEXT_EMBED_SIZE + FLEX_N → 1)` (weight zero-init, bias `+4.0`) → `sigmoid` → a per-row scalar `a ≈ 0.98` at init. The fused emoji logits are `EmojiEmbedding.score(a * q_txt.detach() + (1-a) * q_kw.detach())` with `E` / `bias` also detached in that path, so `loss/fusion` trains **only** the gate; the trunk, `EmojiHead`, `KWHead` and `EmojiEmbedding` see gradient only from `loss/emoji` + `loss/kw`. `loss/emoji`, `loss/kw`, `loss/fusion` are all the same `model/train.py:lse_infonce`.

Replace with:

> `model/model.py:FusionHead` is a **gate** — `Linear(TEXT_EMBED_SIZE + FLEX_N → 1)` (weight zero-init, bias `0.0`) → `sigmoid` → a per-row scalar `a ≈ 0.5` at init. The fused emoji logits are `EmojiEmbedding.score(a * q_txt + (1-a) * q_kw)` with **nothing detached**. When `fusion` is selected, `loss/emoji` and `loss/kw` are **not** added to the total loss — only `loss/fusion` trains, and it backprops through the gate into `q_txt` → `EmojiHead` → encoder, `q_kw` → `KWHead`, and `EmojiEmbedding`, so the two heads specialise instead of each solving the whole space. `loss/e/val` and `MRR/e/val` / `MRR/kw/val` are still logged (diagnostic only); `gate/a/{train,val}` logs the mean gate value. `loss/fusion` is `model/train.py:lse_infonce`. Without `fusion`, `--heads emoji` keeps its standalone `loss/emoji`.

Also update, earlier in the same paragraph:

> a shared module scored by both `EmojiHead` and `KWHead`, co-trained by `loss/emoji` + `loss/kw`.

→

> a shared module scored by both `EmojiHead` and `KWHead`, co-trained by `loss/fusion` (or `loss/emoji` + `loss/kw` when `fusion` is not selected).

- [ ] **Step 2: `regen.ts` bullet (line 18) — the `kw_vocab` derivation**

Find:

> `kw_vocab` = every CLDR keyword with `df == 1` (max IDF), a single `[a-z0-9]+` non-stopword token, `KW_MIN_LEN ≤ len ≤ KW_MAX_LEN` (`tools/data/config.ts`, currently `4 ≤ len ≤ 10`), that also occurs as a `queryTokens` token somewhere in the `data/data.jsonl` master (sorted alphabetically, used in full — **dynamic N ≈ 1388**). `tf_vec[j] = …` … Built by `tools/data/flexrank.ts:buildFlexRanker(corpusTexts)` (an inverted keyword→glyph index over `loadCldrAnnotations()`, no FlexSearch library despite the name); the three ranker surfaces stay byte-identical (…). Deterministic (static CLDR input).

Replace the `kw_vocab = …` clause and the "Built by … Deterministic (static CLDR input)." clause with:

> `kw_vocab` is chosen by **`tools/data/kwvocab.ts`**: if the newest `report/*/report.json` has a `keywords_flex.ranked` list, take its worst-first top **1000** keyword strings (the `data/ii.json` keywords a trained `EmojiHead` ranks worst — computed by `tools/report.py`), sorted alphabetically; otherwise fall back to a deterministic **bootstrap** — every `data/ii.json` key that is 3–6 chars, a single `[a-z0-9]+` non-stopword token (`queryTokens(k) == [k]`), and has ≥1 target emoji in the current `data/labels.json` vocab, alphabetical, capped at 1000. `regen`'s `flex_tf` summary line reports `source=report` or `source=ii.json bootstrap`. `tf_vec[j] = …` (unchanged). `tools/data/flexrank.ts:buildFlexRanker(kwVocab)` now just wraps a pre-selected vocab with the `tfVec` / `buildJson` closures (no CLDR load); the three ranker surfaces stay byte-identical (…). **`regen`'s `flex_tf` vocab is no longer pure** — it depends on the newest report (`--matrix` / `--analysis` / `--no-kw` never read a report and stay pure); the `flexrank.fixture.json` golden still locks whatever vocab was chosen.

Also update the earlier "**deterministically**" note in that bullet's first sentence to add: "(the `flex_tf` vocab excepted — see below)".

- [ ] **Step 3: Sanity-check the diff**

Run: `git diff CLAUDE.md` — confirm only the FusionHead sentence, the `co-trained by` clause, and the `kw_vocab` / `buildFlexRanker` / determinism clauses changed.

- [ ] **Step 4: Commit**

```bash
git add CLAUDE.md
git commit -m "docs: CLAUDE.md — report-driven kw vocab, end-to-end fusion

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01A7ANy3iuZSfpa6yPbfY7jP"
```

---

## Task 7: Full verification (no code)

Run every gate and the bootstrap path end to end. Do not commit anything except a `bun run regen` refresh of the tracked `web/src/flexrank.fixture.json` (and gitignored `flex.json`) if it changes.

- [ ] **Step 1: Python lint + format**

Run: `uv run ruff check . && uv run ruff format --check .`
Expected: no errors.

- [ ] **Step 2: Python test scripts**

Run: `uv run python model/test_runmeta.py && uv run python model/test_train_cli.py && uv run python tools/test_report.py`
Expected: three `ok` lines. (Requires `bun run regen` to have produced `data/labels.json` — run it first if `model.config` import fails.)

- [ ] **Step 3: Bun tests**

Run: `bun test tools/data/`
Expected: `kwvocab.test.ts`, `flexrank.test.ts`, and any existing data tests pass.

- [ ] **Step 4: Web tests unaffected**

Run: `cd web && npm test` (then `cd ..`)
Expected: pass (no `web/` code changed; `flex.json` / fixture regenerate in Step 5).

- [ ] **Step 5: Bootstrap regen**

Run: `bun run regen`
Expected: the `flex_tf` summary line reads `... source=ii.json bootstrap` (no `report.json` with `keywords_flex` exists yet) and `N=<= 1000`. `web/public/flex.json` and `web/src/flexrank.fixture.json` are rewritten. `git status` shows only `web/src/flexrank.fixture.json` (tracked) plus gitignored artifacts.

- [ ] **Step 6: Conformance goldens still line up**

Run: `uv run python model/test_flexrank.py`
Expected: `ok` (Python `FlexRanker` matches the freshly regenerated fixture).
Run: `cd web && npx vitest run src/flexrank.test.js && cd ..`
Expected: pass.

- [ ] **Step 7: Report round-trip (needs `pt/`)**

Only if `pt/enc.pt` + `pt/emoji.pt` + `pt/emoji_embed.pt` exist from a prior run:
Run: `uv run python tools/report.py --pt pt`
Expected: `report/<ts>-<sha>/report.html` written; open it and confirm a **"Model — Keyword vocab"** section with the summary line and a missed-keyword table; `report.json` has a `keywords_flex` object with `candidates` / `missed` / `ranked`.
Then run `bun run regen` again — expected `flex_tf` line now reads `source=report`.
If no `pt/` is available, note that this step is deferred to the next real training run and record it in the handoff.

- [ ] **Step 8: Commit the fixture refresh (if changed)**

```bash
git add web/src/flexrank.fixture.json
git commit -m "chore(regen): refresh flexrank fixture for ii.json bootstrap vocab

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01A7ANy3iuZSfpa6yPbfY7jP"
```

- [ ] **Step 9: Behavioural check (manual, out of band)**

Document for the user (not executed here): a full `train enc --local` on a clean tree, watching in TensorBoard that `MRR/kw/val` climbs off zero (gate not collapsed), `gate/a/train` settles below ~0.95, `MRR/fusion/val` is the checkpoint metric, and `MRR/e/val` may dip. Then `uv run python tools/report.py --pt pt`, then a second `bun run regen` to pick up `source=report`, then `train enc --local` again.

---

## Self-Review

**Spec coverage:**

| Spec section | Task |
|---|---|
| Feedback loop + bootstrap | Tasks 4, 5, 7 |
| Candidate set + rank metric | Task 2 |
| `report.py` — `_section_keywords_flex` + `report.json` key | Task 2 |
| `report.py` — HTML "Model — Keyword vocab" section, `rank > 10` table | Task 3 |
| `report.py` — `build_report` `want` / `need_enc` wiring | Task 3 |
| `flexrank.ts` — drop CLDR, `buildFlexRanker(kwVocab)` | Task 5 |
| `regen.ts` — `keywordVocabFromReport() ?? keywordVocabFromIiJson()`, source in summary | Task 5 |
| `regen.ts` — pure sub-commands untouched | Task 5 (change is inside `if (useKw)`; `--matrix` exits earlier, `--analysis` returns earlier, `--no-kw` skips the block) + Task 7 Step 5 |
| Training — drop `loss/emoji` + `loss/kw`, single `loss/fusion`, no detach | Task 1 |
| Training — `FusionHead` bias `4.0 → 0.0` | Task 1 |
| Training — `gate/a/{split}` logging | Task 1 |
| Training — `--heads emoji` alone unchanged; `fusion` still requires `emoji` | Task 1 (guard is `if "fusion" not in self.heads`) + existing `_parse_heads` (no change) |
| Training — `MRR/e/val` / `MRR/kw/val` still logged | Task 1 (both `self._log` lines kept) |
| Checkpoint metric unchanged (`MRR/fusion/val`) | no change — `model/train.py` `_ckpt_metric` logic untouched |
| `II_JSON` in `files.py` + `files.ts` | Tasks 2, 4 |
| `export_onnx` / `pred` / `web` unchanged | no task — verified by Task 7 Steps 4, 6 |
| `flexrank.fixture.json` regenerated | Task 7 Step 5 |
| Tests: `test_report.py`, `test_train_cli.py` | Tasks 2, 3, 1 |
| `CLAUDE.md` updates | Task 6 |

No gaps.

**Placeholder scan:** No "TBD"/"handle edge cases"/"similar to Task N". Every code step has the literal code. Task 7 has no code because it is a verification task; each step names the exact command and expected output.

**Type consistency:**
- `_flex_keyword_candidates` returns `tuple[(str, list[str]), ...]`; `_section_keywords_flex` iterates `for row, (kw, tgt) in enumerate(cands)` — matches. `ranked` entry keys `kw` / `emojis` / `top5` / `rank` are identical in Task 2 (producer), Task 3 test + `_keywords_flex_html` (Python consumer), and Task 4 `kwvocab.ts` (`{kw: string; rank: number}` — reads only `kw` and `rank`, both present). 
- `buildFlexRanker` return shape (`kwVocab` / `tfVec` / `buildJson`) identical in Task 5's `flexrank.ts`, `flexrank.test.ts`, and `regen.ts` (`ranker.kwVocab.length`, `ranker.tfVec`, `ranker.buildJson()`).
- `keywordVocabFromReport(dir?: string)` — called with no arg in `regen.ts`, with a dir in the test. Default `REPORT_DIR`. Consistent.
- `II_JSON` — `f"{DATA_DIR}/ii.json"` (Python) / `` `${DATA_DIR}/ii.json` `` (TS), same value `data/ii.json`.

---

## Execution Handoff

**Plan complete and saved to `docs/superpowers/plans/2026-09-10-report-driven-kw-vocab-and-end-to-end-fusion.md`. Two execution options:**

**1. Subagent-Driven (recommended)** — I dispatch a fresh subagent per task, review between tasks, fast iteration.

**2. Inline Execution** — Execute tasks in this session using executing-plans, batch execution with checkpoints.

**Which approach?**
