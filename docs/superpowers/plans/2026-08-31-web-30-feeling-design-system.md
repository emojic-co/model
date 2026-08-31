# Web 30-Feeling Design System Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Give all 30 open-set feelings a distinct gradient, readable text colour, font, one-shot text-entrance animation, and looping emoji animation in the web app, and trim the feeling bar to the top 5.

**Architecture:** A new pure module `web/src/feelings.js` maps each feeling to one of 9 emotional **clusters** (font stack + entrance motif + emoji motif + drift speed), with per-feeling overrides for font style and animation timing. Colours stay in `web/public/palette.json` (expanded 7 → 30). `web/src/styles.css` gains one `@keyframes` per motif (11 entrance + 11 emoji) selected by `data-entrance` / `data-emoji` attributes on the card and tuned by CSS custom properties from `feelings.js`. The one-shot entrance replays via a React `key` bound to a per-prediction `revision` counter.

**Tech Stack:** Vite 8 + React 19, `onnxruntime-web` (unchanged), vitest 4 (`environment: node`), Google Fonts.

**Spec:** `docs/superpowers/specs/2026-08-31-web-30-feeling-design-system-design.md`

## Global Constraints

- **No comments or docstrings in any source file.** Keep existing `type:ignore` / `noqa` / shebangs; add none. This applies to `.js`, `.jsx`, `.css`, `.json` (JSON has no comments anyway).
- **Do not touch Python / `uv` / the model.** No change to `main.py`, `model.pt`, `web/public/model.onnx`, `web/public/meta.json`, `web/public/config.json`.
- **Do not modify `web/src/model.js`, `normalize`, or `encode`** — they must stay byte-identical to training.
- **Colours live only in `web/public/palette.json`.** `feelings.js` holds font + animation mapping, never hex colours.
- **`Neutral` is the fallback** for any feeling missing from `palette.json` or `feelings.js`.
- **vitest env is `node`** (see `web/vite.config.js`) — every test must be a pure-function / JSON-data test. No DOM, no `@testing-library`, no jsdom.
- **Contrast floor:** WCAG contrast ratio ≥ 4.5 between `text_color` and *each* of `bg1`, `bg2` for every palette entry.
- **Run all web commands from `web/`.** `cd web && npm install` once before starting (fonts are runtime-only; no new npm deps).
- **Git:** work on branch `web-30-feeling-design-system` (already exists; the spec is committed there). Stage only the files each task names — the working tree may carry unrelated modifications to `web/public/meta.json` / `web/public/model.onnx`; never stage those.
- **Commit message trailer:** end every commit message body with these two lines:
  ```
  Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
  Claude-Session: https://claude.ai/code/session_012v6rNdy9RRaXLfAuoZmyiR
  ```

---

## File Structure

| File | Responsibility |
|---|---|
| `web/src/feelings.js` | **new** — `CLUSTERS`, `FEELINGS`, motif-name constants, `resolveFeeling(feeling)`, `topFeelings(scores, feelings, selected)`. Pure ESM, no React import. Replaces `web/src/fonts.js`. |
| `web/src/feelings.test.js` | **new** — coverage (every model feeling mapped), cluster/motif integrity, `Neutral` fallback, `topFeelings` ranking, palette shape + contrast. |
| `web/src/fonts.js` | **deleted** in Task 4 (folded into `feelings.js`). |
| `web/public/palette.json` | 7 → 30 `{bg1,bg2,text_color}` entries. |
| `web/src/styles.css` | Motif `@keyframes` + `data-entrance` / `data-emoji` rules, `--drift-sec` on the drift loop, extended reduced-motion block. Old infinite `text-*` / `emoji-*` per-feeling blocks removed. |
| `web/index.html` | Google Fonts `<link>` gains Baloo 2, Caveat, Poppins. |
| `web/src/components/Card.jsx` | Uses `resolveFeeling`; emits `data-cluster` / `data-entrance` / `data-emoji`, CSS var style, per-feeling text style; `<p>` keyed on `revision`; takes `feelingOptions` + `revision` props. |
| `web/src/App.jsx` | `revision` counter bumped per prediction; `feelingOptions = topFeelings(...)`; passes `feelingOptions` + `revision` to `Card`. |
| `web/src/hooks/useCardImage.js` | Font stack from `resolveFeeling` instead of `FEELING_FONTS`. |
| `web/src/components/FeelingBar.jsx` | **unchanged** — already renders whatever list it is handed. |
| `CLAUDE.md` | Replace the stale "web not yet migrated" bullet; note `feelings.js`. |

---

## Task 1: `feelings.js` module + logic tests

**Files:**
- Create: `web/src/feelings.js`
- Create: `web/src/feelings.test.js`

**Interfaces:**
- Consumes: `web/public/meta.json` (`.feelings` array) — for the test only.
- Produces:
  - `CLUSTERS: Record<string, {font: string, entrance: string, emoji: string, driftSec: number}>`
  - `FEELINGS: Record<string, {cluster: string, style?: object, dur?: {entrance?: number, emoji?: number}, entrance?: string, emoji?: string}>`
  - `ENTRANCE_MOTIFS: string[]`, `EMOJI_MOTIFS: string[]`, `MOTIF_DEFAULT_MS: {entrance: number, emoji: number}`
  - `resolveFeeling(feeling: string) => {cluster: string, font: string, entrance: string, emoji: string, style: object, vars: {'--entrance-dur': string, '--emoji-dur': string, '--drift-sec': string}}`
  - `topFeelings(feelingScores: number[] | null, feelings: string[], selected: string | null) => string[]` (length ≤ 6)

- [ ] **Step 1: Write the failing test** — create `web/src/feelings.test.js`:

```js
import { readFileSync } from 'node:fs'
import { fileURLToPath } from 'node:url'
import { describe, it, expect } from 'vitest'
import {
  CLUSTERS,
  FEELINGS,
  ENTRANCE_MOTIFS,
  EMOJI_MOTIFS,
  resolveFeeling,
  topFeelings,
} from './feelings'

const readJson = (rel) =>
  JSON.parse(readFileSync(fileURLToPath(new URL(rel, import.meta.url)), 'utf8'))
const meta = readJson('../public/meta.json')

describe('feelings coverage', () => {
  it('every model feeling has a FEELINGS entry', () => {
    for (const f of meta.feelings) expect(FEELINGS[f], f).toBeTruthy()
  })

  it('every FEELINGS entry points at a real cluster', () => {
    for (const [name, def] of Object.entries(FEELINGS)) {
      expect(CLUSTERS[def.cluster], name).toBeTruthy()
    }
  })

  it('every resolved motif name is known', () => {
    for (const f of meta.feelings) {
      const r = resolveFeeling(f)
      expect(ENTRANCE_MOTIFS, `${f} entrance`).toContain(r.entrance)
      expect(EMOJI_MOTIFS, `${f} emoji`).toContain(r.emoji)
    }
  })

  it('resolveFeeling returns css var strings', () => {
    const r = resolveFeeling('Happy')
    expect(r.vars['--entrance-dur']).toMatch(/^\d+ms$/)
    expect(r.vars['--emoji-dur']).toMatch(/^\d+ms$/)
    expect(r.vars['--drift-sec']).toMatch(/^\d+s$/)
  })

  it('unknown feeling resolves to the Neutral style', () => {
    expect(resolveFeeling('Nope')).toEqual(resolveFeeling('Neutral'))
  })
})

describe('topFeelings', () => {
  const feelings = ['A', 'B', 'C', 'D', 'E', 'F', 'G']
  const scores = [0.1, 0.9, 0.3, 0.7, 0.2, 0.5, 0.05]

  it('returns the 5 highest scoring, descending', () => {
    expect(topFeelings(scores, feelings, 'B')).toEqual(['B', 'D', 'F', 'C', 'E'])
  })

  it('appends the selected feeling when it is outside the top 5', () => {
    expect(topFeelings(scores, feelings, 'G')).toEqual(['B', 'D', 'F', 'C', 'E', 'G'])
  })

  it('returns [] when scores are missing', () => {
    expect(topFeelings(null, feelings, 'A')).toEqual([])
  })
})
```

- [ ] **Step 2: Run the test, verify it fails**

Run: `cd web && npx vitest run src/feelings.test.js`
Expected: FAIL — `Cannot find module './feelings'` (or `Failed to resolve import`).

- [ ] **Step 3: Create `web/src/feelings.js`** with exactly this content:

```js
export const CLUSTERS = {
  anger: { font: '"Anton", system-ui, sans-serif', entrance: 'slam', emoji: 'shake', driftSec: 10 },
  joy: { font: '"Fredoka", system-ui, sans-serif', entrance: 'pop', emoji: 'hop', driftSec: 12 },
  play: { font: '"Baloo 2", system-ui, sans-serif', entrance: 'spin', emoji: 'wobble', driftSec: 11 },
  calm: { font: '"Quicksand", system-ui, sans-serif', entrance: 'settle', emoji: 'breathe', driftSec: 22 },
  sad: { font: '"Playfair Display", Georgia, serif', entrance: 'drop', emoji: 'sink', driftSec: 20 },
  anxiety: { font: '"Shantell Sans", system-ui, cursive', entrance: 'jitter', emoji: 'tremor', driftSec: 9 },
  tender: { font: '"Caveat", "Segoe Script", cursive', entrance: 'bloom', emoji: 'heartbeat', driftSec: 16 },
  drive: { font: '"Poppins", system-ui, sans-serif', entrance: 'rise', emoji: 'lift', driftSec: 14 },
  reflective: { font: '"Inter", system-ui, sans-serif', entrance: 'fadeTilt', emoji: 'tilt', driftSec: 18 },
}

export const ENTRANCE_MOTIFS = ['slam', 'pop', 'spin', 'settle', 'drop', 'jitter', 'bloom', 'rise', 'fadeTilt', 'droop', 'shrinkBack']
export const EMOJI_MOTIFS = ['shake', 'hop', 'wobble', 'breathe', 'sink', 'tremor', 'heartbeat', 'lift', 'tilt', 'droop', 'shrinkBack']
export const MOTIF_DEFAULT_MS = { entrance: 650, emoji: 2400 }

export const FEELINGS = {
  Angry: { cluster: 'anger', style: { textTransform: 'uppercase', letterSpacing: '0.06em' }, dur: { entrance: 420, emoji: 450 } },
  Annoyed: { cluster: 'anger', style: { letterSpacing: '-0.01em' }, dur: { entrance: 500, emoji: 950 } },
  Frustrated: { cluster: 'anger', style: { textTransform: 'uppercase' }, dur: { entrance: 520, emoji: 600 } },
  Happy: { cluster: 'joy', style: { fontWeight: 600 }, dur: { entrance: 560, emoji: 900 } },
  Excited: { cluster: 'joy', style: { textTransform: 'uppercase', letterSpacing: '0.05em', fontWeight: 600 }, dur: { entrance: 460, emoji: 380 } },
  Amused: { cluster: 'joy', style: { fontWeight: 500 }, dur: { entrance: 620, emoji: 1400 } },
  Playful: { cluster: 'play', style: { fontWeight: 600 }, dur: { entrance: 600, emoji: 1100 } },
  Surprised: { cluster: 'play', style: { textTransform: 'uppercase', letterSpacing: '0.04em' }, dur: { entrance: 420, emoji: 2600 } },
  Calm: { cluster: 'calm', style: { fontWeight: 500 }, dur: { entrance: 900, emoji: 4200 } },
  Content: { cluster: 'calm', style: { fontWeight: 600 }, dur: { entrance: 850, emoji: 3800 } },
  Relieved: { cluster: 'calm', style: { fontWeight: 500, letterSpacing: '0.02em' }, dur: { entrance: 800, emoji: 3400 } },
  Sad: { cluster: 'sad', style: { fontStyle: 'italic' }, dur: { entrance: 1000, emoji: 3200 } },
  Disappointed: { cluster: 'sad', style: { fontStyle: 'italic', opacity: 0.92 }, dur: { entrance: 950, emoji: 3000 } },
  Lonely: { cluster: 'sad', style: { fontStyle: 'italic', letterSpacing: '0.06em', opacity: 0.88 }, dur: { entrance: 1100, emoji: 4000 } },
  Tired: { cluster: 'sad', entrance: 'droop', emoji: 'droop', style: { fontStyle: 'italic', letterSpacing: '0.04em', opacity: 0.9 }, dur: { entrance: 1150, emoji: 5200 } },
  Anxious: { cluster: 'anxiety', style: {}, dur: { entrance: 560, emoji: 220 } },
  Worried: { cluster: 'anxiety', style: { letterSpacing: '0.01em' }, dur: { entrance: 640, emoji: 420 } },
  Concerned: { cluster: 'anxiety', style: {}, dur: { entrance: 700, emoji: 600 } },
  Confused: { cluster: 'anxiety', style: { fontStyle: 'italic' }, dur: { entrance: 620, emoji: 900 } },
  Embarrassed: { cluster: 'anxiety', entrance: 'shrinkBack', emoji: 'shrinkBack', style: {}, dur: { entrance: 640, emoji: 3200 } },
  Love: { cluster: 'tender', style: { fontWeight: 700 }, dur: { entrance: 700, emoji: 1300 } },
  Caring: { cluster: 'tender', style: { fontWeight: 600 }, dur: { entrance: 750, emoji: 1600 } },
  Grateful: { cluster: 'tender', style: { fontWeight: 600 }, dur: { entrance: 780, emoji: 1800 } },
  Helpful: { cluster: 'tender', style: { fontWeight: 600 }, dur: { entrance: 720, emoji: 1500 } },
  Hopeful: { cluster: 'drive', style: { fontWeight: 500 }, dur: { entrance: 780, emoji: 3000 } },
  Proud: { cluster: 'drive', style: { textTransform: 'uppercase', letterSpacing: '0.05em', fontWeight: 600 }, dur: { entrance: 700, emoji: 2600 } },
  Determined: { cluster: 'drive', style: { textTransform: 'uppercase', fontWeight: 700 }, dur: { entrance: 560, emoji: 1400 } },
  Neutral: { cluster: 'reflective', style: {}, dur: { entrance: 700, emoji: 6000 } },
  Curious: { cluster: 'reflective', style: { fontWeight: 600 }, dur: { entrance: 650, emoji: 3200 } },
  Thoughtful: { cluster: 'reflective', style: { fontStyle: 'italic' }, dur: { entrance: 800, emoji: 4200 } },
}

export function resolveFeeling(feeling) {
  const f = FEELINGS[feeling] ?? FEELINGS.Neutral
  const c = CLUSTERS[f.cluster]
  return {
    cluster: f.cluster,
    font: c.font,
    entrance: f.entrance ?? c.entrance,
    emoji: f.emoji ?? c.emoji,
    style: f.style ?? {},
    vars: {
      '--entrance-dur': `${f.dur?.entrance ?? MOTIF_DEFAULT_MS.entrance}ms`,
      '--emoji-dur': `${f.dur?.emoji ?? MOTIF_DEFAULT_MS.emoji}ms`,
      '--drift-sec': `${c.driftSec}s`,
    },
  }
}

export function topFeelings(feelingScores, feelings, selected) {
  if (!feelingScores) return []
  const ranked = feelings
    .map((f, i) => ({ f, p: feelingScores[i] }))
    .sort((a, b) => b.p - a.p)
    .map((x) => x.f)
  const top = ranked.slice(0, 5)
  if (selected && !top.includes(selected)) top.push(selected)
  return top
}
```

- [ ] **Step 4: Run the test, verify it passes**

Run: `cd web && npx vitest run src/feelings.test.js`
Expected: PASS — all `feelings coverage` and `topFeelings` tests green.

- [ ] **Step 5: Run the full web suite, verify nothing else broke**

Run: `cd web && npm test`
Expected: PASS — `model.test.js`, `fit.test.js`, `feelings.test.js` all green.

- [ ] **Step 6: Commit**

```bash
git add web/src/feelings.js web/src/feelings.test.js
git commit -m "$(cat <<'EOF'
feat(web): feelings.js cluster map + resolveFeeling/topFeelings

Maps all 30 open-set feelings to 9 style clusters (font + entrance +
emoji motif + drift speed) with per-feeling overrides. topFeelings ranks
the feeling bar to the top 5 (+ the current override when off-list).

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_012v6rNdy9RRaXLfAuoZmyiR
EOF
)"
```

---

## Task 2: Expand `palette.json` to 30 feelings + contrast test

**Files:**
- Modify: `web/public/palette.json` (currently 7 entries → 30)
- Modify: `web/src/feelings.test.js` (add a `describe('palette', ...)` block)

**Interfaces:**
- Consumes: `web/public/meta.json` (`.feelings`), `web/public/palette.json`.
- Produces: nothing importable — a data file + tests.

- [ ] **Step 1: Write the failing test** — append this block to `web/src/feelings.test.js` (it reuses the `readJson` helper and `meta` from Task 1's header; add the `palette` line shown):

```js
const palette = readJson('../public/palette.json')

describe('palette', () => {
  const HEX = /^#[0-9a-f]{6}$/

  const luminance = (hex) => {
    const n = parseInt(hex.slice(1), 16)
    const lin = [(n >> 16) & 255, (n >> 8) & 255, n & 255].map((c) => {
      const s = c / 255
      return s <= 0.03928 ? s / 12.92 : ((s + 0.055) / 1.055) ** 2.4
    })
    return 0.2126 * lin[0] + 0.7152 * lin[1] + 0.0722 * lin[2]
  }

  const contrast = (a, b) => {
    const la = luminance(a)
    const lb = luminance(b)
    const hi = Math.max(la, lb)
    const lo = Math.min(la, lb)
    return (hi + 0.05) / (lo + 0.05)
  }

  it('every model feeling has a palette entry with a hex triplet', () => {
    for (const f of meta.feelings) {
      const p = palette[f]
      expect(p, f).toBeTruthy()
      expect(p.bg1, `${f}.bg1`).toMatch(HEX)
      expect(p.bg2, `${f}.bg2`).toMatch(HEX)
      expect(p.text_color, `${f}.text_color`).toMatch(HEX)
    }
  })

  it('text colour clears 4.5 contrast against both gradient stops', () => {
    for (const f of meta.feelings) {
      const p = palette[f]
      expect(contrast(p.text_color, p.bg1), `${f} text vs bg1`).toBeGreaterThanOrEqual(4.5)
      expect(contrast(p.text_color, p.bg2), `${f} text vs bg2`).toBeGreaterThanOrEqual(4.5)
    }
  })
})
```

- [ ] **Step 2: Run the test, verify it fails**

Run: `cd web && npx vitest run src/feelings.test.js -t palette`
Expected: FAIL — `Annoyed` (and 22 other feelings) missing from `palette.json`.

- [ ] **Step 3: Replace `web/public/palette.json`** with exactly this (30 entries; values already contrast-checked):

```json
{
  "Angry": { "bg1": "#b4000f", "bg2": "#880009", "text_color": "#fff0ed" },
  "Annoyed": { "bg1": "#a83224", "bg2": "#7d2016", "text_color": "#ffece7" },
  "Frustrated": { "bg1": "#b23a0b", "bg2": "#8a2606", "text_color": "#fff0e8" },
  "Happy": { "bg1": "#ffd571", "bg2": "#ffad22", "text_color": "#2f1100" },
  "Excited": { "bg1": "#ffc93c", "bg2": "#ff8a5c", "text_color": "#3a1500" },
  "Amused": { "bg1": "#ffe066", "bg2": "#ffd43b", "text_color": "#33260a" },
  "Playful": { "bg1": "#ff9ff3", "bg2": "#7ad7ff", "text_color": "#26142f" },
  "Surprised": { "bg1": "#a0f0ff", "bg2": "#67dcff", "text_color": "#0f333d" },
  "Calm": { "bg1": "#92efb5", "bg2": "#5bd9a4", "text_color": "#162f1e" },
  "Content": { "bg1": "#bfe8a0", "bg2": "#9fd97f", "text_color": "#21301a" },
  "Relieved": { "bg1": "#c6f0e4", "bg2": "#a3e5d2", "text_color": "#123028" },
  "Sad": { "bg1": "#4e73a7", "bg2": "#2d5492", "text_color": "#e2f1fd" },
  "Disappointed": { "bg1": "#4b6079", "bg2": "#354657", "text_color": "#e6ecf2" },
  "Lonely": { "bg1": "#3b4a63", "bg2": "#2a3547", "text_color": "#dbe2ec" },
  "Tired": { "bg1": "#645e79", "bg2": "#4a4559", "text_color": "#ece9f2" },
  "Anxious": { "bg1": "#4d6f8f", "bg2": "#4a4a66", "text_color": "#eef0f6" },
  "Worried": { "bg1": "#456864", "bg2": "#3b5350", "text_color": "#e9f1f0" },
  "Concerned": { "bg1": "#516480", "bg2": "#41526a", "text_color": "#e9edf3" },
  "Confused": { "bg1": "#5f5b83", "bg2": "#4b4869", "text_color": "#ecebf4" },
  "Embarrassed": { "bg1": "#ff9e8f", "bg2": "#ff8177", "text_color": "#3d160f" },
  "Love": { "bg1": "#ffb5c0", "bg2": "#ff929f", "text_color": "#39131b" },
  "Caring": { "bg1": "#e6b8e0", "bg2": "#d199d0", "text_color": "#331433" },
  "Grateful": { "bg1": "#ffd9a8", "bg2": "#ffc07d", "text_color": "#3a2205" },
  "Helpful": { "bg1": "#ffcaa9", "bg2": "#ffb38f", "text_color": "#3a1e0d" },
  "Hopeful": { "bg1": "#ffe1a0", "bg2": "#ffc98c", "text_color": "#3a2607" },
  "Proud": { "bg1": "#6a2fae", "bg2": "#4a1f88", "text_color": "#f2e9ff" },
  "Determined": { "bg1": "#bd3c00", "bg2": "#942d00", "text_color": "#fff0e6" },
  "Neutral": { "bg1": "#a8e2f4", "bg2": "#78c9f4", "text_color": "#282e36" },
  "Curious": { "bg1": "#c3d4f5", "bg2": "#9fb8ec", "text_color": "#1e2a44" },
  "Thoughtful": { "bg1": "#d8d4c8", "bg2": "#bdb8a8", "text_color": "#2b2822" }
}
```

- [ ] **Step 4: Run the test, verify it passes**

Run: `cd web && npx vitest run src/feelings.test.js`
Expected: PASS — `palette` block green (both hex-shape and contrast).
If any `text vs bg` assertion still fails: darken (for light-text cards) or lighten (for dark-text cards) the failing `bg` stop by ~8–12% while keeping its hue, until the ratio clears 4.5 with ~0.2 margin; re-run. Do not change `text_color`.

- [ ] **Step 5: Run the full web suite**

Run: `cd web && npm test`
Expected: PASS — all suites green.

- [ ] **Step 6: Commit**

```bash
git add web/public/palette.json web/src/feelings.test.js
git commit -m "$(cat <<'EOF'
feat(web): 30-feeling palette with contrast guard

Expands palette.json from 7 to 30 {bg1,bg2,text_color} entries, one per
open-set feeling, hue by cluster. New test asserts every model feeling
has an entry and text clears WCAG 4.5 against both gradient stops.

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_012v6rNdy9RRaXLfAuoZmyiR
EOF
)"
```

---

## Task 3: Animation motifs in CSS + font imports

**Files:**
- Modify: `web/src/styles.css` (replace the keyframe/animation section between the `.card` rule and `.copy-btn`)
- Modify: `web/index.html` (Google Fonts `<link href>`)

**Interfaces:**
- Consumes: CSS custom properties `--entrance-dur`, `--emoji-dur`, `--drift-sec` and attributes `data-feeling` / `data-entrance` / `data-emoji` on `.card` (set by Task 4). Selectors are inert until Task 4 emits those attributes — that is expected.
- Produces: `@keyframes` `entrance-*` (11) and `emoji-*` (11); the `drift` keyframe is kept.

- [ ] **Step 1: Update `web/index.html`** — replace the single Google Fonts `<link ... href=...>` line's `href` with (one line, no spaces):

```
https://fonts.googleapis.com/css2?family=Anton&family=Baloo+2:wght@500;600;700&family=Caveat:wght@600;700&family=Fredoka:wght@500;600&family=Inter:wght@400;600&family=Noto+Color+Emoji&family=Playfair+Display:ital,wght@1,600&family=Poppins:wght@500;600;700&family=Quicksand:wght@500;600&family=Shantell+Sans:wght@500;600&display=swap
```

- [ ] **Step 2: Edit `web/src/styles.css`** — delete everything from `@keyframes drift {` through the last `.card[data-feeling="Neutral"] .card-text { ... }` rule and the existing `@media (prefers-reduced-motion: reduce)` block, and replace that whole span with:

```css
@keyframes drift {
  0%, 100% { background-position: 0% 50%; }
  50% { background-position: 100% 50%; }
}

@keyframes entrance-slam {
  0% { opacity: 0; transform: scale(1.6); }
  60% { opacity: 1; transform: scale(0.94); }
  75% { transform: scale(1.03) translateX(-3px); }
  85% { transform: scale(1) translateX(3px); }
  100% { transform: scale(1) translateX(0); }
}
@keyframes entrance-pop {
  0% { opacity: 0; transform: scale(0.5) translateY(12px); }
  70% { opacity: 1; transform: scale(1.08) translateY(-4px); }
  100% { transform: scale(1) translateY(0); }
}
@keyframes entrance-spin {
  0% { opacity: 0; transform: rotate(-12deg) scale(0.6); }
  70% { opacity: 1; transform: rotate(4deg) scale(1.05); }
  100% { transform: rotate(0) scale(1); }
}
@keyframes entrance-settle {
  0% { opacity: 0; transform: scale(1.03); }
  100% { opacity: 1; transform: scale(1); }
}
@keyframes entrance-drop {
  0% { opacity: 0; transform: translateY(-40px); }
  70% { opacity: 1; transform: translateY(6px); }
  100% { transform: translateY(0); }
}
@keyframes entrance-jitter {
  0% { opacity: 0; transform: translate(0, 0); }
  20% { opacity: 0.6; transform: translate(-3px, 2px); }
  40% { transform: translate(3px, -2px); }
  60% { opacity: 1; transform: translate(-2px, 1px); }
  80% { transform: translate(2px, -1px); }
  100% { transform: translate(0, 0); }
}
@keyframes entrance-bloom {
  0% { opacity: 0; transform: scale(0.9); }
  100% { opacity: 1; transform: scale(1); }
}
@keyframes entrance-rise {
  0% { opacity: 0; transform: translateY(28px); }
  100% { opacity: 1; transform: translateY(0); }
}
@keyframes entrance-fadeTilt {
  0% { opacity: 0; transform: rotate(-2deg) translateY(6px); }
  100% { opacity: 1; transform: rotate(0) translateY(0); }
}
@keyframes entrance-droop {
  0% { opacity: 0; transform: translateY(-10px) scaleY(1.06); }
  60% { opacity: 1; transform: translateY(4px) scaleY(0.97); }
  100% { transform: translateY(0) scaleY(1); }
}
@keyframes entrance-shrinkBack {
  0% { opacity: 0; transform: scale(1.15); }
  40% { opacity: 1; transform: scale(0.9); }
  70% { transform: scale(0.97) rotate(-2deg); }
  100% { transform: scale(1) rotate(0); }
}

@keyframes emoji-shake {
  0%, 100% { transform: translateX(0) rotate(0); }
  20% { transform: translateX(-3px) rotate(-2deg); }
  40% { transform: translateX(3px) rotate(2deg); }
  60% { transform: translateX(-2px) rotate(-1deg); }
  80% { transform: translateX(2px) rotate(1deg); }
}
@keyframes emoji-hop {
  0%, 100% { transform: translateY(0); }
  50% { transform: translateY(-16px); }
}
@keyframes emoji-wobble {
  0%, 100% { transform: rotate(-8deg); }
  50% { transform: rotate(8deg); }
}
@keyframes emoji-breathe {
  0%, 100% { transform: scale(1); }
  50% { transform: scale(1.06); }
}
@keyframes emoji-sink {
  0%, 100% { transform: translateY(0) scale(1); }
  50% { transform: translateY(6px) scale(0.98); }
}
@keyframes emoji-tremor {
  0%, 100% { transform: translate(0, 0); }
  25% { transform: translate(-2px, 1px); }
  50% { transform: translate(2px, -1px); }
  75% { transform: translate(-1px, -2px); }
}
@keyframes emoji-heartbeat {
  0%, 100% { transform: scale(1); }
  15% { transform: scale(1.18); }
  30% { transform: scale(1); }
  45% { transform: scale(1.12); }
  60% { transform: scale(1); }
}
@keyframes emoji-lift {
  0%, 100% { transform: translateY(4px); }
  50% { transform: translateY(-10px); }
}
@keyframes emoji-tilt {
  0%, 100% { transform: rotate(-3deg); }
  50% { transform: rotate(3deg); }
}
@keyframes emoji-droop {
  0%, 100% { transform: translateY(0) rotate(0); }
  50% { transform: translateY(8px) rotate(4deg); }
}
@keyframes emoji-shrinkBack {
  0%, 100% { transform: scale(1) rotate(0); }
  50% { transform: scale(0.86) rotate(8deg); }
}

.card[data-feeling] {
  background-size: 200% 200%;
  animation: drift var(--drift-sec, 16s) ease-in-out infinite;
}

.card-text {
  animation-duration: var(--entrance-dur, 650ms);
  animation-timing-function: cubic-bezier(0.22, 1, 0.36, 1);
  animation-iteration-count: 1;
  animation-fill-mode: both;
}
.card[data-entrance="slam"] .card-text { animation-name: entrance-slam; animation-timing-function: cubic-bezier(0.36, 0.07, 0.19, 0.97); }
.card[data-entrance="pop"] .card-text { animation-name: entrance-pop; }
.card[data-entrance="spin"] .card-text { animation-name: entrance-spin; }
.card[data-entrance="settle"] .card-text { animation-name: entrance-settle; }
.card[data-entrance="drop"] .card-text { animation-name: entrance-drop; }
.card[data-entrance="jitter"] .card-text { animation-name: entrance-jitter; }
.card[data-entrance="bloom"] .card-text { animation-name: entrance-bloom; }
.card[data-entrance="rise"] .card-text { animation-name: entrance-rise; }
.card[data-entrance="fadeTilt"] .card-text { animation-name: entrance-fadeTilt; }
.card[data-entrance="droop"] .card-text { animation-name: entrance-droop; }
.card[data-entrance="shrinkBack"] .card-text { animation-name: entrance-shrinkBack; }

.card-emoji {
  animation-duration: var(--emoji-dur, 2400ms);
  animation-timing-function: ease-in-out;
  animation-iteration-count: infinite;
}
.card[data-emoji="shake"] .card-emoji { animation-name: emoji-shake; animation-timing-function: linear; }
.card[data-emoji="hop"] .card-emoji { animation-name: emoji-hop; }
.card[data-emoji="wobble"] .card-emoji { animation-name: emoji-wobble; }
.card[data-emoji="breathe"] .card-emoji { animation-name: emoji-breathe; }
.card[data-emoji="sink"] .card-emoji { animation-name: emoji-sink; }
.card[data-emoji="tremor"] .card-emoji { animation-name: emoji-tremor; animation-timing-function: linear; }
.card[data-emoji="heartbeat"] .card-emoji { animation-name: emoji-heartbeat; }
.card[data-emoji="lift"] .card-emoji { animation-name: emoji-lift; }
.card[data-emoji="tilt"] .card-emoji { animation-name: emoji-tilt; }
.card[data-emoji="droop"] .card-emoji { animation-name: emoji-droop; }
.card[data-emoji="shrinkBack"] .card-emoji { animation-name: emoji-shrinkBack; }

@media (prefers-reduced-motion: reduce) {
  .card[data-feeling] { animation: none; background-size: auto; }
  .card[data-feeling] .card-emoji,
  .card[data-feeling] .card-text { animation: none !important; }
}
```

- [ ] **Step 3: Verify the build compiles**

Run: `cd web && npm run build`
Expected: build succeeds, writes `web/dist`, no CSS parse error.

- [ ] **Step 4: Verify the test suite is still green**

Run: `cd web && npm test`
Expected: PASS — CSS changes touch no test.

- [ ] **Step 5: Commit**

```bash
git add web/src/styles.css web/index.html
git commit -m "$(cat <<'EOF'
feat(web): per-cluster entrance + emoji animation motifs

11 one-shot text-entrance keyframes and 11 looping emoji keyframes,
selected by data-entrance / data-emoji and tuned by CSS vars. Drift loop
speed is now per-cluster. Reduced-motion disables all of it. Adds Baloo
2, Caveat, Poppins to the font link.

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_012v6rNdy9RRaXLfAuoZmyiR
EOF
)"
```

---

## Task 4: Wire the app to `feelings.js`

**Files:**
- Modify: `web/src/components/Card.jsx`
- Modify: `web/src/App.jsx`
- Modify: `web/src/hooks/useCardImage.js`
- Delete: `web/src/fonts.js`
- Modify: `CLAUDE.md`

**Interfaces:**
- Consumes: `resolveFeeling`, `topFeelings` from `web/src/feelings.js` (Task 1).
- Produces: no new exports. `Card` prop shape becomes `{ text, emoji, feeling, feelingOptions, palette, revision, onPickFeeling, onCopy }` (the `feelings` prop is renamed to `feelingOptions`).

- [ ] **Step 1: Replace `web/src/components/Card.jsx`** with exactly:

```jsx
import { useFitText } from '../hooks/useFitText'
import { resolveFeeling } from '../feelings'
import { FeelingBar } from './FeelingBar'

export function Card({ text, emoji, feeling, feelingOptions, palette, revision, onPickFeeling, onCopy }) {
  const textRef = useFitText(text, { min: 32, max: 104, key: feeling })
  const pal = feeling && palette ? palette[feeling] ?? palette.Neutral : null
  const r = feeling ? resolveFeeling(feeling) : null
  const style =
    pal && r
      ? {
          backgroundImage: `linear-gradient(135deg, ${pal.bg1}, ${pal.bg2})`,
          color: pal.text_color,
          fontFamily: r.font,
          ...r.vars,
        }
      : undefined

  return (
    <div
      className="card"
      data-feeling={feeling || undefined}
      data-cluster={r?.cluster || undefined}
      data-entrance={r?.entrance || undefined}
      data-emoji={r?.emoji || undefined}
      style={style}
    >
      <span className="card-emoji">{emoji}</span>
      <div className="card-text-box" ref={textRef}>
        <p className="card-text" key={revision} style={r?.style}>
          {text}
        </p>
      </div>
      {feeling ? (
        <FeelingBar feelings={feelingOptions} active={feeling} onPick={onPickFeeling} />
      ) : (
        <span className="card-feeling-idle">—</span>
      )}
      <button className="copy-btn" type="button" aria-label="Copy card as image" onClick={onCopy}>
        copy
      </button>
    </div>
  )
}
```

- [ ] **Step 2: Edit `web/src/App.jsx`** — four changes:

  1. Update the import line:
     ```jsx
     import { argmax, normalize } from './model'
     import { topFeelings } from './feelings'
     ```
  2. Add `revision` state next to the other `useState` calls:
     ```jsx
     const [revision, setRevision] = useState(0)
     ```
  3. In the prediction `setTimeout` callback, bump `revision` right after `setScores(logits)`:
     ```jsx
     const logits = await predict(text)
     if (mine !== seq.current) return
     setScores(logits)
     setRevision((n) => n + 1)
     setOverride({ emoji: null, feeling: null })
     ```
  4. After the `const shownFeeling = override.feeling ?? predictedFeeling` line, add:
     ```jsx
     const feelingOptions = useMemo(
       () => topFeelings(scores?.feeling, meta?.feelings ?? [], shownFeeling),
       [scores, meta, shownFeeling],
     )
     ```
  5. In the `<Card ... />` JSX, remove `feelings={meta?.feelings ?? []}` and add `feelingOptions={feelingOptions}` and `revision={revision}`. Final `Card` element:
     ```jsx
     <Card
       text={text}
       emoji={shownEmoji ?? '🙂'}
       feeling={shownFeeling}
       feelingOptions={feelingOptions}
       revision={revision}
       palette={palette}
       onPickFeeling={(f) => setOverride((o) => ({ ...o, feeling: f }))}
       onCopy={copyCard}
     />
     ```

  (`useMemo` is already imported in `App.jsx`.)

- [ ] **Step 3: Edit `web/src/hooks/useCardImage.js`** — two changes:
  1. Replace `import { FEELING_FONTS } from '../fonts'` with `import { resolveFeeling } from '../feelings'`.
  2. In `render`, replace `const stack = FEELING_FONTS[feeling] ?? FEELING_FONTS.Neutral` with `const stack = resolveFeeling(feeling).font`.

- [ ] **Step 4: Delete the old font map**

```bash
git rm web/src/fonts.js
```

- [ ] **Step 5: Verify no dangling references**

Run: `cd web && grep -rn "FEELING_FONTS\|from '../fonts'\|from './fonts'" src`
Expected: no output.

- [ ] **Step 6: Verify build + tests**

Run: `cd web && npm run build && npm test`
Expected: build writes `web/dist` with no unresolved import; all three test suites green.

- [ ] **Step 7: Manual smoke test**

Run: `cd web && npm run dev`, open the printed URL, then:
- Type `omg this is the best day ever` — card gets a gradient + font, the text plays a one-shot entrance, the emoji loops; the emoji list shows 10, the feeling bar shows 5.
- Edit the text (add a word) — top emoji + feeling re-select automatically and the entrance replays once.
- Click through every name in the feeling bar — each switch changes gradient + font + entrance + emoji loop; contrast stays comfortable (no unreadable text).
- Pick a feeling, then keep clicking other bar entries — when the picked one is outside the top 5 the bar shows 6 and keeps your pick highlighted.
- Press Enter (and separately click `copy`) — a PNG lands on the clipboard (paste into any image target to confirm); the copied image uses the feeling's font + colours.
- Toggle OS "reduce motion" and reload — no entrance, no loops, card is legible and static.

- [ ] **Step 8: Update `CLAUDE.md`**
  1. Replace the bullet
     `- **Not yet migrated:** \`web/\` still reads \`data.jsonl\` and assumes the old closed 7-feeling / 300-emoji \`labels.json\`.`
     with
     `- **Web is on the open-set palette:** \`web/\` consumes the 30-feeling \`meta.json\` and \`web/public/palette.json\` (30 \`{bg1,bg2,text_color}\` entries); \`web/src/feelings.js\` maps each feeling to one of 9 style clusters (font + text-entrance motif + emoji motif + drift speed), with \`Neutral\` as the fallback for any feeling not listed.`
  2. In the `web/` bullet of the Project section, after "plus the hand-maintained `palette.json` (per-feeling hex colors)", append: " and `web/src/feelings.js` (per-feeling font + animation cluster)."

- [ ] **Step 9: Commit**

```bash
git add web/src/components/Card.jsx web/src/App.jsx web/src/hooks/useCardImage.js web/src/fonts.js CLAUDE.md
git commit -m "$(cat <<'EOF'
feat(web): drive card style + feeling bar from feelings.js

Card resolves font, per-feeling text style, cluster/entrance/emoji data
attributes and timing vars from feelings.js; the text node re-keys on a
per-prediction revision so the entrance replays. App ranks the feeling
bar to the top 5 (+ off-list override) via topFeelings. useCardImage
takes its font stack from feelings.js. Removes fonts.js.

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_012v6rNdy9RRaXLfAuoZmyiR
EOF
)"
```

---

## Self-Review

**1. Spec coverage**

| Spec requirement | Task |
|---|---|
| Every feeling in `labels.json`: gradient background | Task 2 (`palette.json` 30 entries) |
| Every feeling: readable text colour | Task 2 (contrast ≥ 4.5 test) |
| Every feeling: distinct font family + style | Task 1 (`CLUSTERS` fonts + `FEELINGS[*].style`), Task 3 (font imports), Task 4 (Card applies) |
| Every feeling: text entrance animation | Task 3 (`entrance-*` keyframes), Task 4 (`data-entrance` + `revision` key) |
| Every feeling: emoji animation | Task 3 (`emoji-*` keyframes), Task 4 (`data-emoji`) |
| Distinct style per feeling | Task 1 (cluster + per-feeling `style`/`dur` + 2 motif overrides) |
| Top 10 emojis left of card | already implemented in `EmojiList` — unchanged |
| Top 5 feelings at card bottom | Task 1 (`topFeelings`), Task 4 (App computes, Card passes) |
| Select emoji + feeling, Copy / Enter → image to clipboard | already implemented (`useCardImage`, `App` Enter handler) — Task 4 keeps it, swaps font source |
| Auto-select top emoji + feeling | already implemented (`setOverride` reset per prediction) — unchanged |
| Re-select on text change | already implemented + Task 4 adds entrance replay via `revision` |
| Video export "only if simple" | dropped per approved spec (Out of scope) |
| Noto Color Emoji for emojis | already first in `.card-emoji` stack + canvas `EMOJI_STACK`; canvas already awaits it — no change needed, covered by Task 4 smoke test |
| `web/src/model.js` byte-identical | untouched (Global Constraints) |

**2. Placeholder scan** — no `TBD` / `TODO` / "add error handling" / "similar to Task N" / prose-only code steps. Every code step carries full content.

**3. Type consistency** — `resolveFeeling` return shape (`cluster`, `font`, `entrance`, `emoji`, `style`, `vars`) is produced in Task 1 and consumed in Task 4 (`r.font`, `r.cluster`, `r.entrance`, `r.emoji`, `r.style`, `r.vars`) and Task 3 (the `--entrance-dur` / `--emoji-dur` / `--drift-sec` var names match `vars` keys). `topFeelings(feelingScores, feelings, selected)` signature is identical in Task 1 definition, Task 1 tests, and the Task 4 `App` call. Motif-name lists in `ENTRANCE_MOTIFS` / `EMOJI_MOTIFS` (Task 1) exactly match the `entrance-*` / `emoji-*` keyframes and the `data-entrance=` / `data-emoji=` selector values (Task 3). `Card` prop rename `feelings` → `feelingOptions` is applied in both `Card` (Task 4 Step 1) and its caller `App` (Task 4 Step 2.5); `FeelingBar`'s own `feelings` prop is unchanged and still fed by `Card`.

---

## Execution Handoff

**Plan complete and saved to `docs/superpowers/plans/2026-08-31-web-30-feeling-design-system.md`. Two execution options:**

**1. Subagent-Driven (recommended)** — I dispatch a fresh subagent per task, review between tasks, fast iteration.

**2. Inline Execution** — Execute tasks in this session using executing-plans, batch execution with checkpoints.

**Which approach?**
