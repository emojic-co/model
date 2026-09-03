# pr/remotion

[Remotion](https://remotion.dev) template that films a typewriter demo of
**emojify.ing**: a predefined phrase types itself into an input field, a card
shows an "emojifying…" loading state, and then it crossfades to the real
model's card result for that phrase.

The result card is the **real** web component (`web/src/components/Card.jsx`,
`feelings.js`, `model.js`, `fit.js`, `styles.css` are imported directly), so it
can't drift from production. Only the input row, the card, and a "made with ❤️
by Gilad" line are shown — no wordmark, char counter, warnings or metadata.

## Timeline (per phrase)

1. **Hold** ~0.35 s — static frame, empty input field, caret blinking.
2. **Typewriter** (~7 cps) fills the phrase in **Fraunces**. No camera move —
   the frame is static throughout. Card area stays empty.
3. **When typing finishes**, the card fades in a **loading state**: a fast
   rotating emoji (30-emoji set, ~10/s) left of **"emojifying"** in
   **JetBrains Mono** with animated `.` `..` `...`. Shown for ~0.8 s.
4. **Slide** (~0.55 s) — the loading card slides out left while the real
   `<Card>` result pushes in from the right, both clipped to the card slot.
5. **Hold** ~2.4 s on the result.

The card does not update per keystroke — it's loading → crossfade → final only,
so `pregen.mjs` runs the model **once per full phrase**.

## Files

| Path | What |
|---|---|
| `texts.txt` | Candidate phrases, one per line — short, catchy, chosen to exercise different emojis / styles / palettes. |
| `pregen.mjs` | Runs the **real** ONNX model (`web/public/model.onnx`) on each full phrase and records the argmax emoji, argmax style and palette 0 (`decodeColorList(...)[0]`, the site's default). Uses the exact `encode`/`decode` from `web/src/model.js`. Writes `data/<slug>.json` and `src/data.json`. |
| `src/Root.tsx` | One `<Composition>` per phrase in `data.json` (id = slug). **1080×1350** (4:5), 30 fps, duration auto-fit to phrase length. |
| `src/Scene.tsx` | The composition: camera transform, input + caret, loading card, crossfade, footer. Timing constants exported here. |
| `src/scene.css` | Layout, gray-brown background, input pill, loading card. |
| `src/fonts.ts` | The Google Fonts `<link>` (from `web/index.html`, plus Fraunces), loaded via `delayRender`. |
| `render-all.mjs` | Bundles once, renders the chosen slugs to `out/<slug>.mp4`. |
| `data/`, `src/data.json` | Pre-generated model output, one record per phrase (checked in). |
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
