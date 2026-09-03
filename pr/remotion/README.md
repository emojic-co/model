# pr/remotion

[Remotion](https://remotion.dev) template that recreates the **emojify.ing** UI —
wordmark, input box, the generated card, footer — and films a typewriter demo:
a predefined phrase types itself out and the card (emoji + font + colors +
feeling label) transforms in real time, exactly as it does on the live site.
The emoji / color / style picker bars are omitted; only the card is shown.

The card is the **real** web component (`web/src/components/Card.jsx`,
`feelings.js`, `model.js`, `fit.js`, `styles.css` are imported directly), so it
can't drift from production.

## Files

| Path | What |
|---|---|
| `texts.txt` | Candidate phrases, one per line — short, catchy, chosen to exercise different emojis / styles / palettes. |
| `pregen.mjs` | Runs the **real** ONNX model (`web/public/model.onnx`) over **every prefix** of each phrase and records the argmax emoji, argmax style and palette 0 (`decodeColorList(...)[0]`, the site's default). Uses the exact `normalize`/`encode`/`decode` from `web/src/model.js`. Writes `data/<slug>.json` and `src/data.json`. |
| `src/Root.tsx` | One `<Composition>` per phrase in `data.json` (id = slug). 1080×1920, 30 fps, duration auto-fit to phrase length. |
| `src/Scene.tsx` | The composition: input + caret + counter + `<Card>` + footer. Frame → typed-prefix length (~11 cps after a 0.6 s beat, 2.6 s hold) → the pre-generated result for that prefix. |
| `src/fonts.ts` | The Google Fonts `<link>` from `web/index.html`, loaded via `delayRender`. |
| `render-all.mjs` | Bundles once, renders the chosen slugs to `out/<slug>.mp4`. |
| `data/`, `src/data.json` | Pre-generated per-prefix model output (checked in). |
| `out/` | Rendered `.mp4`s (gitignored). |

## Workflow

```sh
cd pr/remotion
npm install
npm run pregen                 # regenerate data/ + src/data.json from the current model
npm run studio                 # preview / scrub any phrase in the browser
npm run render                 # render the 3 default phrases to out/
node render-all.mjs <slug> ... # render specific phrases
```

Regenerate `web/public/model.onnx` (via `uv run python export_onnx.py`) → rerun
`npm run pregen` → rerender.

## Current renders

`pizza-s-finally-here` (🍕 Joyful), `rent-is-due-and-i-m-broke` (⏰ Anxious),
`quiet-night-just-me-and-tea` (🌙 Serene) — three different emojis, fonts and
palettes. Change the default list at the top of `render-all.mjs`.
