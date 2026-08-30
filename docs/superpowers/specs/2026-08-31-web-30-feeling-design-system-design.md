# Web: 30-feeling design system

**Date:** 2026-08-31
**Status:** approved (design), pending implementation plan

## Problem

The model was migrated to an open-set palette: `labels.json` / `web/public/meta.json`
now carry **30 feelings** and **120 emojis**. The web app was built for the old closed
**7 feelings** (Happy, Calm, Sad, Angry, Anxious, Neutral, Love):

- `web/public/palette.json` has 7 colour entries; the other 23 feelings fall back to `Neutral`.
- `web/src/fonts.js` (`FEELING_FONTS`) has 7 font stacks; same fallback.
- `web/src/styles.css` keys per-feeling emoji/text animations off `[data-feeling="X"]` for the 7.
- `web/src/components/FeelingBar.jsx` renders **every** feeling it is handed. `App.jsx` hands it
  `meta.feelings` — all 30 — so the card foot is a 30-item wall instead of the intended top 5.

Goal: give **all 30** feelings a distinct, intentional style (gradient background, readable text
colour, font family + style, one-shot text entrance animation, looping emoji animation), and fix
the feeling bar to show the top 5.

Non-goals: video / animated export (clipboard video support is unreliable — dropped); any Python /
model / training change; changing `normalize` / `encode` / model I/O.

## Approach

**Cluster-driven data map + motif CSS.**

- New `web/src/feelings.js` is the single source of truth. It defines ~9 **clusters** (emotional
  families). Each cluster carries a font stack, a text-entrance motif name, an emoji-animation
  motif name, and a background-drift speed. Each of the 30 feelings maps to a cluster plus
  optional per-feeling overrides (font style, animation timing, and — for two feelings — a
  different motif).
- Colours stay in `web/public/palette.json`, expanded 7 → 30, honouring the existing convention
  ("colours never pass through Python; `web/src` fetches `palette.json` directly"). `Neutral`
  remains the fallback for any feeling not listed.
- `web/src/styles.css` gains one `@keyframes` per motif (11 entrance + 11 emoji) and one short
  rule per motif keyed off `data-entrance="…"` / `data-emoji="…"` on the card. Per-feeling timing
  and text style come in as CSS custom properties set from `feelings.js`.

Rejected alternatives:

- **30 fully-independent entries, no clusters** — the clustering choice done as duplication;
  `styles.css` balloons, "distinct" drifts into "arbitrary", shared motifs impossible to tune.
- **Keep pure `[data-feeling]` CSS selectors, add 23 more** — ~90 hand-written selectors
  (30 feelings × entrance + emoji + idle), no shared motif concept, unmaintainable.

## Data model — `web/src/feelings.js`

Pure ES module, no React import (must stay importable under the `node` test env, same as
`model.js`). Replaces `web/src/fonts.js`, which is deleted.

```js
export const CLUSTERS = {
  anger:      { font: '"Anton", system-ui, sans-serif',      entrance: 'slam',     emoji: 'shake',     driftSec: 10 },
  joy:        { font: '"Fredoka", system-ui, sans-serif',    entrance: 'pop',      emoji: 'hop',       driftSec: 12 },
  play:       { font: '"Baloo 2", system-ui, sans-serif',    entrance: 'spin',     emoji: 'wobble',    driftSec: 11 },
  calm:       { font: '"Quicksand", system-ui, sans-serif',  entrance: 'settle',   emoji: 'breathe',   driftSec: 22 },
  sad:        { font: '"Playfair Display", Georgia, serif',  entrance: 'drop',     emoji: 'sink',      driftSec: 20 },
  anxiety:    { font: '"Shantell Sans", system-ui, cursive', entrance: 'jitter',   emoji: 'tremor',    driftSec: 9  },
  tender:     { font: '"Caveat", "Segoe Script", cursive',   entrance: 'bloom',    emoji: 'heartbeat', driftSec: 16 },
  drive:      { font: '"Poppins", system-ui, sans-serif',    entrance: 'rise',     emoji: 'lift',      driftSec: 14 },
  reflective: { font: '"Inter", system-ui, sans-serif',      entrance: 'fadeTilt', emoji: 'tilt',      driftSec: 18 },
}
```

Per-feeling entries: `{ cluster, style?, dur?, entrance?, emoji? }`.

- `style` — object applied to `.card-text` inline: any of `textTransform`, `letterSpacing`,
  `fontWeight`, `fontStyle`, `opacity`.
- `dur` — `{ entrance?: ms, emoji?: ms }`, overriding motif defaults.
- `entrance` / `emoji` — override the cluster motif name. Used only by `Tired` (`droop`) and
  `Embarrassed` (`shrinkBack`), which share their cluster's font but need their own movement.

```js
export const FEELINGS = {
  // anger
  Angry:       { cluster: 'anger',   style: { textTransform: 'uppercase', letterSpacing: '0.06em' }, dur: { entrance: 420, emoji: 450 } },
  Annoyed:     { cluster: 'anger',   style: { letterSpacing: '-0.01em' },                            dur: { entrance: 500, emoji: 950 } },
  Frustrated:  { cluster: 'anger',   style: { textTransform: 'uppercase' },                          dur: { entrance: 520, emoji: 600 } },
  // joy
  Happy:       { cluster: 'joy',     style: { fontWeight: 600 },                                     dur: { entrance: 560, emoji: 900 } },
  Excited:     { cluster: 'joy',     style: { textTransform: 'uppercase', letterSpacing: '0.05em', fontWeight: 600 }, dur: { entrance: 460, emoji: 380 } },
  Amused:      { cluster: 'joy',     style: { fontWeight: 500 },                                     dur: { entrance: 620, emoji: 1400 } },
  // play
  Playful:     { cluster: 'play',    style: { fontWeight: 600 },                                     dur: { entrance: 600, emoji: 1100 } },
  Surprised:   { cluster: 'play',    style: { textTransform: 'uppercase', letterSpacing: '0.04em' }, dur: { entrance: 420, emoji: 2600 } },
  // calm
  Calm:        { cluster: 'calm',    style: { fontWeight: 500 },                                     dur: { entrance: 900, emoji: 4200 } },
  Content:     { cluster: 'calm',    style: { fontWeight: 600 },                                     dur: { entrance: 850, emoji: 3800 } },
  Relieved:    { cluster: 'calm',    style: { fontWeight: 500, letterSpacing: '0.02em' },            dur: { entrance: 800, emoji: 3400 } },
  // sad
  Sad:          { cluster: 'sad',    style: { fontStyle: 'italic' },                                 dur: { entrance: 1000, emoji: 3200 } },
  Disappointed: { cluster: 'sad',    style: { fontStyle: 'italic', opacity: 0.92 },                  dur: { entrance: 950, emoji: 3000 } },
  Lonely:       { cluster: 'sad',    style: { fontStyle: 'italic', letterSpacing: '0.06em', opacity: 0.88 }, dur: { entrance: 1100, emoji: 4000 } },
  Tired:        { cluster: 'sad',    entrance: 'droop', emoji: 'droop', style: { fontStyle: 'italic', letterSpacing: '0.04em', opacity: 0.9 }, dur: { entrance: 1150, emoji: 5200 } },
  // anxiety
  Anxious:     { cluster: 'anxiety', style: {},                                                     dur: { entrance: 560, emoji: 220 } },
  Worried:     { cluster: 'anxiety', style: { letterSpacing: '0.01em' },                            dur: { entrance: 640, emoji: 420 } },
  Concerned:   { cluster: 'anxiety', style: {},                                                     dur: { entrance: 700, emoji: 600 } },
  Confused:    { cluster: 'anxiety', style: { fontStyle: 'italic' },                                dur: { entrance: 620, emoji: 900 } },
  Embarrassed: { cluster: 'anxiety', entrance: 'shrinkBack', emoji: 'shrinkBack', style: {},        dur: { entrance: 640, emoji: 3200 } },
  // tender
  Love:        { cluster: 'tender',  style: { fontWeight: 700 },                                     dur: { entrance: 700, emoji: 1300 } },
  Caring:      { cluster: 'tender',  style: { fontWeight: 600 },                                     dur: { entrance: 750, emoji: 1600 } },
  Grateful:    { cluster: 'tender',  style: { fontWeight: 600 },                                     dur: { entrance: 780, emoji: 1800 } },
  Helpful:     { cluster: 'tender',  style: { fontWeight: 600 },                                     dur: { entrance: 720, emoji: 1500 } },
  // drive
  Hopeful:     { cluster: 'drive',   style: { fontWeight: 500 },                                     dur: { entrance: 780, emoji: 3000 } },
  Proud:       { cluster: 'drive',   style: { textTransform: 'uppercase', letterSpacing: '0.05em', fontWeight: 600 }, dur: { entrance: 700, emoji: 2600 } },
  Determined:  { cluster: 'drive',   style: { textTransform: 'uppercase', fontWeight: 700 },         dur: { entrance: 560, emoji: 1400 } },
  // reflective
  Neutral:     { cluster: 'reflective', style: {},                                                  dur: { entrance: 700, emoji: 6000 } },
  Curious:     { cluster: 'reflective', style: { fontWeight: 600 },                                  dur: { entrance: 650, emoji: 3200 } },
  Thoughtful:  { cluster: 'reflective', style: { fontStyle: 'italic' },                              dur: { entrance: 800, emoji: 4200 } },
}
```

Motif defaults (used when `dur` omits a field), and the canonical motif-name lists, are exported
so the test can assert coverage:

```js
export const ENTRANCE_MOTIFS = ['slam','pop','spin','settle','drop','jitter','bloom','rise','fadeTilt','droop','shrinkBack']
export const EMOJI_MOTIFS    = ['shake','hop','wobble','breathe','sink','tremor','heartbeat','lift','tilt','droop','shrinkBack']
export const MOTIF_DEFAULT_MS = { entrance: 650, emoji: 2400 }
```

### `resolveFeeling(feeling)`

```js
export function resolveFeeling(feeling) {
  const f = FEELINGS[feeling] ?? FEELINGS.Neutral
  const c = CLUSTERS[f.cluster]
  const entrance = f.entrance ?? c.entrance
  const emoji = f.emoji ?? c.emoji
  return {
    cluster: f.cluster,
    font: c.font,
    entrance,
    emoji,
    style: f.style ?? {},
    vars: {
      '--entrance-dur': `${f.dur?.entrance ?? MOTIF_DEFAULT_MS.entrance}ms`,
      '--emoji-dur': `${f.dur?.emoji ?? MOTIF_DEFAULT_MS.emoji}ms`,
      '--drift-sec': `${c.driftSec}s`,
    },
  }
}
```

### `topFeelings(feelingScores, feelings, selected)`

Pure ranking helper (kept here so it is unit-testable under the `node` env; no DOM).

```js
export function topFeelings(feelingScores, feelings, selected) {
  if (!feelingScores) return []
  const ranked = feelings
    .map((f, i) => ({ f, p: feelingScores[i] }))
    .sort((a, b) => b.p - a.p)
    .map((x) => x.f)
  const top = ranked.slice(0, 5)
  if (selected && !top.includes(selected)) top.push(selected)   // up to 6: top 5 + the override
  return top
}
```

## Colour system — `web/public/palette.json`

Expanded to 30 `{ bg1, bg2, text_color }` entries (lowercase 6-digit hex). Hue is set by the
cluster; lightness / saturation vary per feeling within the family. Two lightness regimes so both
gradient stops clear contrast against the text colour:

- **light card** (bg luminance high): near-black text.
- **dark card** (bg luminance low): near-white text.

Contrast target: WCAG ratio **≥ 4.5** between `text_color` and *each* of `bg1`, `bg2`. Enforced by
`feelings.test.js` (below). The values here are the starting point; any row the test flags gets
nudged darker/lighter during implementation until it passes, keeping the same hue.

| Feeling | bg1 | bg2 | text_color | note |
|---|---|---|---|---|
| Angry | `#b4000f` | `#880009` | `#fff0ed` | pure dark red (unchanged) |
| Annoyed | `#a83224` | `#7d2016` | `#ffece7` | brick / ember |
| Frustrated | `#b23a0b` | `#8a2606` | `#fff0e8` | hot red-orange |
| Happy | `#ffd571` | `#ffad22` | `#2f1100` | warm yellow-orange (unchanged) |
| Excited | `#ffc93c` | `#ff8a5c` | `#3a1500` | orange → coral, electric |
| Amused | `#ffe066` | `#ffd43b` | `#33260a` | sunny yellow |
| Playful | `#ff9ff3` | `#7ad7ff` | `#26142f` | candy pink → cyan |
| Surprised | `#a0f0ff` | `#67dcff` | `#0f333d` | bright cyan pop |
| Calm | `#92efb5` | `#5bd9a4` | `#162f1e` | soft green (unchanged) |
| Content | `#bfe8a0` | `#9fd97f` | `#21301a` | warm green |
| Relieved | `#c6f0e4` | `#a3e5d2` | `#123028` | pale mint |
| Sad | `#4e73a7` | `#2d5492` | `#e2f1fd` | muted blue (unchanged) |
| Disappointed | `#566d86` | `#3d5064` | `#e6ecf2` | grey-blue, flat |
| Lonely | `#3b4a63` | `#2a3547` | `#dbe2ec` | dark, cold |
| Tired | `#6a6480` | `#4f4a61` | `#ece9f2` | dim lavender-grey |
| Anxious | `#4d6f8f` | `#4a4a66` | `#eef0f6` | cool grey-blue (darkened) |
| Worried | `#4a6f6b` | `#3f5a57` | `#e9f1f0` | murky teal-grey |
| Concerned | `#556781` | `#45566d` | `#e9edf3` | slate |
| Confused | `#5f5b83` | `#4b4869` | `#ecebf4` | violet-grey |
| Embarrassed | `#ff9e8f` | `#ff8177` | `#3d160f` | flushed coral-red |
| Love | `#ffb5c0` | `#ff929f` | `#39131b` | classic pink (unchanged) |
| Caring | `#e6b8e0` | `#d199d0` | `#331433` | mauve-pink |
| Grateful | `#ffd9a8` | `#ffc07d` | `#3a2205` | peach / apricot |
| Helpful | `#ffcaa9` | `#ffb38f` | `#3a1e0d` | rosy gold |
| Hopeful | `#ffe1a0` | `#ffc98c` | `#3a2607` | dawn gold |
| Proud | `#6a2fae` | `#4a1f88` | `#f2e9ff` | royal purple |
| Determined | `#d64500` | `#ad3600` | `#fff0e6` | burnt orange |
| Neutral | `#a8e2f4` | `#78c9f4` | `#282e36` | light blue (unchanged) |
| Curious | `#c3d4f5` | `#9fb8ec` | `#1e2a44` | periwinkle |
| Thoughtful | `#d8d4c8` | `#bdb8a8` | `#2b2822` | warm grey / greige |

## Typography

9 shared font stacks (one per cluster). 3 new Google Font families are added to the single
`<link>` in `web/index.html`; the existing 6 stay:

- add: `Baloo+2:wght@500;600;700`, `Caveat:wght@600;700`, `Poppins:wght@500;600;700`
- keep: `Anton`, `Fredoka:wght@500;600`, `Inter:wght@400;600`, `Noto+Color+Emoji`,
  `Playfair+Display:ital,wght@1,600`, `Quicksand:wght@500;600`, `Shantell+Sans:wght@500;600`

Per-feeling `style` (weight / italic / letter-spacing / text-transform / opacity) differentiates
members inside a cluster — see the `FEELINGS` map. `.card` keeps `font-family` (so the emoji
element's own stack still overrides it); the rest of `style` is applied inline to `.card-text`.

## Animation system

Two independent parts:

1. **Text entrance** — a **one-shot** animation (`animation-fill-mode: both`, `iteration-count: 1`)
   that plays when a new prediction lands. Replaces the current infinite `text-*` loops, which are
   removed.
2. **Emoji animation** — an **infinite** idle loop, as today.
3. Background keeps the existing infinite `drift` gradient pan; its duration becomes
   `var(--drift-sec)` (per cluster) instead of the hard-coded `16s`.

### Motifs

| Cluster (feeling) | entrance | emoji |
|---|---|---|
| anger | `slam` — snap in from oversized + 2-frame shake settle | `shake` — fast horizontal tremor + micro-rotate |
| joy | `pop` — overshoot scale-up bounce | `hop` — vertical bounce |
| play | `spin` — rotate + scale in | `wobble` — rotate left/right |
| calm | `settle` — fade + gentle scale from 1.03 → 1 | `breathe` — slow scale 1 ↔ 1.05 |
| sad | `drop` — fall in from above, settle low | `sink` — slow downward drift + slight shrink |
| anxiety | `jitter` — fade in with positional tremor | `tremor` — small fast 2-axis jitter |
| tender | `bloom` — fade + scale from 0.9, ease-out | `heartbeat` — double-thump pulse |
| drive | `rise` — slide up from below, firm ease-out | `lift` — slow float upward ↔ rest |
| reflective | `fadeTilt` — fade + settle from −2° tilt | `tilt` — slow rotate −3° ↔ 3° |
| — Tired | `droop` — slow sag downward into place | `droop` — slow sag ↔ heavier sag |
| — Embarrassed | `shrinkBack` — recoil: quick shrink + small back-away, then partial return | `shrinkBack` — shrink + turn (rotate ~8°) ↔ rest |

`styles.css` structure:

```css
.card-text { animation-fill-mode: both; animation-iteration-count: 1; }
.card[data-entrance="slam"]  .card-text { animation: entrance-slam  var(--entrance-dur,650ms) cubic-bezier(.36,.07,.19,.97) both; }
/* …one line per entrance motif… */

.card-emoji { animation-iteration-count: infinite; }
.card[data-emoji="shake"] .card-emoji { animation: emoji-shake var(--emoji-dur,2400ms) linear infinite; }
/* …one line per emoji motif… */

.card[data-feeling] { background-size: 200% 200%; animation: drift var(--drift-sec,16s) ease-in-out infinite; }
```

### Entrance replay

App passes a `revision` counter that increments every time `scores` updates (the existing
debounced-prediction cycle in `App.jsx`). `Card` sets `key={revision}` on the `.card-text-box`
element so it remounts and the one-shot entrance animation restarts. Typing does not replay the
entrance on every keystroke — only when a fresh prediction resolves. On first mount (no scores)
`revision` is 0 and the neutral idle card shows with no entrance.

### Reduced motion

The existing `@media (prefers-reduced-motion: reduce)` block is extended to also neutralise
`.card[data-entrance] .card-text` and `.card[data-emoji] .card-emoji` (`animation: none`), leaving
elements in their resting state. Background drift already disabled there.

## Component / wiring changes

- **`web/src/feelings.js`** — new (above). **`web/src/fonts.js`** — deleted.
- **`web/src/components/Card.jsx`** — import `resolveFeeling` instead of `FEELING_FONTS`. Compute
  `const r = resolveFeeling(feeling)`. On `.card`: `data-feeling`, `data-cluster={r.cluster}`,
  `data-entrance={r.entrance}`, `data-emoji={r.emoji}`, `style={{ …gradient, color, fontFamily:
  r.font, ...r.vars }}`. On `.card-text`: `style={r.style}`. Wrap `.card-text-box` with
  `key={revision}`. Receive `feelingOptions` (array) + pass to `FeelingBar` as `feelings`.
- **`web/src/components/FeelingBar.jsx`** — unchanged markup; just renders whatever list it is
  given (now ≤ 6). Still marks `active`.
- **`web/src/App.jsx`** — add `revision` state, bump it in the `setScores` path of the prediction
  effect. Compute `feelingOptions = topFeelings(scores?.feeling, meta.feelings, shownFeeling)`
  (memoised). Pass `feelingOptions` and `revision` to `Card`. Remove `feelings={meta.feelings}`.
- **`web/src/hooks/useCardImage.js`** — import `resolveFeeling`; `const stack =
  resolveFeeling(feeling).font`. `ensureFonts` regex (`/"([^"]+)"/`) still extracts the first
  quoted family — works for the new stacks. The render is a still frame at rest state (no
  animation captured), which is the intended copied image. No video path.
- **`web/src/hooks/useFitText.js`** — no functional change; already re-keyed on `feeling` and
  re-fits on `document.fonts.ready`. Verify fit with the 3 new families.
- **`web/index.html`** — extend the Google Fonts `<link>` href (above).
- **`CLAUDE.md`** — replace the stale "**Not yet migrated:** web/ still reads data.jsonl and
  assumes the old closed 7-feeling / 300-emoji labels.json" bullet: the web app now consumes the
  30-feeling `meta.json` and a 30-entry `palette.json`; per-feeling font + animation live in
  `web/src/feelings.js` (clusters). Adjust the palette sentence in Conventions from "7" framing to
  "one `{bg1,bg2,text_color}` per feeling in `labels.json`, `Neutral` as fallback".

## Testing

`web/` test env is `node` (no jsdom) — tests are pure-function only.

**New `web/src/feelings.test.js`:**

1. Every feeling in `labels.json.feelings` has a `FEELINGS[feeling]` entry.
2. Every `FEELINGS` entry's `cluster` exists in `CLUSTERS`.
3. Every resolved `entrance` ∈ `ENTRANCE_MOTIFS`; every resolved `emoji` ∈ `EMOJI_MOTIFS`.
4. Every feeling in `labels.json.feelings` has a `palette.json` entry with `bg1` / `bg2` /
   `text_color` matching `/^#[0-9a-f]{6}$/`.
5. For every palette entry: WCAG contrast ratio (inline sRGB relative-luminance implementation)
   between `text_color` and each of `bg1`, `bg2` is **≥ 4.5**.
6. `resolveFeeling('NotAFeeling')` falls back to the `Neutral` resolution.
7. `topFeelings`: returns the 5 highest-scoring feelings in descending order; appends `selected`
   as a 6th when it is outside the top 5; returns `[]` for null scores; length ≤ 6.

**Unchanged:** `web/src/model.test.js`, `web/src/fit.test.js`.

**Manual smoke (`cd web && npm run dev`):** cycle through inputs that surface several clusters;
confirm each feeling shows its gradient + font + entrance + emoji loop, contrast is comfortable,
the feeling bar shows 5 (6 with an off-list override), changing text re-selects the top
emoji/feeling and replays the entrance, Enter / copy button writes a PNG to the clipboard.

No Python changed → no `ruff` run, no training run.

## Out of scope

- Animated / video export (dropped — unreliable clipboard support).
- Any change to the model, `normalize`/`encode`, `meta.json` / `config.json` generation.
- The emoji palette (120 emojis) — the top-10 list already scales; no per-emoji styling requested.
