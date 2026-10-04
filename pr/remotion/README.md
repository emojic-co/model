# pr/remotion

[Remotion](https://remotion.dev) **template** that films a teen-oriented showcase
of **emojify.ing**: nine real model cards, each replacing the previous one with a
slide (in from a side) or a flip (toward a side), then a cross-fade into an
`emojify.ing` outro card with the web app's emoji-rain easter egg. 1080×1350
(4:5), 30 fps, ~30 s — fits Play Store promo video and social feeds.

The cards are the **real** thing: `createPainter` and the clip timeline from
`web/src` (same painter as the web preview and GIF/MP4 export), with the original
animated Noto Lottie emoji. Nothing is typed.

Everything about a video lives in one JSON file in `configs/`. Each file becomes a
composition (id = file name): `en` (English) and `he` (Hebrew, RTL).

## Config (`configs/<id>.json`, types in `src/config.ts`)

| Field | What |
|---|---|
| `lang` | `en` / `he`; sets the card font and text direction. |
| `music` | Track in `public/mp3/`, credit line, volume and fades. |
| `timing` | Hold/transition/outro seconds. The video length follows from these. |
| `transitions` | One per card: `{type: "slide", from}` / `{type: "flip", to}` with `left/right/up/down`; the last must be `fade` (into the outro). |
| `cards` | `text`, `emoji`, `feeling` (style), and frozen `colors` (OKLab). |
| `outro` | Same shape as a card; `lang` should be `en` so `emojify.ing` is not mirrored. |

Cards are frozen in the JSON on purpose: the model's color output is noisy, so
nothing regenerates them.

## Tools

| Command | What |
|---|---|
| `npm run candidates -- <lang>` | Runs the real ONNX model over `texts/<lang>.txt`: top emojis that have an **original** Noto animation (with their model rank), top styles, and the best-scoring palettes out of many samples → `candidates/<lang>.json`. Never touches `configs/`. |
| `npm run validate` | Checks every config against the model: emoji in the top 5, style in the top 3, emoji is byte-identical to Google's Noto animation, text contrast ≥ the app's floor. Exits non-zero on failure. |
| `npm run sync-assets` | Copies the Lottie clips the configs use into `public/noto/`. |
| `npm run studio` | Preview / scrub. |
| `npm run render [id ...]` | Renders `out/<id>.mp4` (default: all configs). |

## Making a new video / language

1. Put candidate texts in `texts/<lang>.txt`, run `npm run candidates -- <lang>`.
2. Copy `configs/en.json` to `configs/<id>.json`; fill `cards` from the candidates
   (emoji + style from the lists, `colors` from a palette).
3. `npm run validate && npm run sync-assets`, then preview in the studio.

Music: "Doh De Oh" by Kevin MacLeod (incompetech.com), CC BY 4.0 — credit him on publish.

## Code

`src/Showcase.tsx` (timing, transitions, background, emoji rain), `src/CardClip.tsx`
(one real card on a canvas, drawn deterministically for a given time),
`src/fonts.ts` (Google Fonts), `webpack-override.mjs` (shims for the web modules'
Vite idioms).
