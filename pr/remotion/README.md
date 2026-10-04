# pr/remotion

[Remotion](https://remotion.dev) project that films a **showcase** of
**emojify.ing**: nine real model cards, each replacing the previous one with a
slide (in from a side) or a flip (toward a side), then a cross-fade into an
`emojify.ing` outro card with the web app's emoji-rain easter egg. The
transition list (type + direction) is `TRANSITIONS` in `src/Showcase.tsx`.
1080×1350 (4:5), 30 fps, ~29 s, with the Kevin MacLeod track "Doh De Oh" (CC BY 4.0, credit him on publish) — fits Play Store promo video and social feeds.

The cards are the **real** thing: `createPainter` and the clip timeline from
`web/src` (same painter as the web preview and GIF/MP4 export), with the animated
Noto Lottie emoji. Nothing is typed; each card plays its own entrance animation.

## Files

| Path | What |
|---|---|
| `showcase.txt` | The nine card texts + outro text. Format: `text \| palette (0-4) \| emoji \| style`, all but text optional. |
| `pregen.mjs` | Runs the **real** ONNX model (`web/public/model.onnx`) per line: highest-logit emoji that has a Lottie clone, argmax style, chosen palette. Writes `src/showcase.json` and copies the needed clips into `public/noto/`. |
| `src/Showcase.tsx` | The composition: timing, the five transitions, background, emoji rain. |
| `src/CardClip.tsx` | One real card on a canvas, drawn deterministically for a given time. |
| `src/fonts.ts` | Google Fonts for every card style. |
| `webpack-override.mjs` | Shims for the web modules' Vite idioms (`?raw`, `import.meta.env`). |
| `render.mjs` | Renders `out/showcase.mp4`. |

## Workflow

```sh
cd pr/remotion
npm install
npm run pregen   # after changing showcase.txt or the model
npm run studio   # preview / scrub
npm run render   # -> out/showcase.mp4
```

To pick a different card for a text, change its palette index (and optionally
emoji/style) in `showcase.txt` and rerun `pregen`.

Note: the model's palette output is not deterministic, so `pregen` gives different
colors on every run. `src/showcase.json` is the committed pick — rerun `pregen`
only when you want to re-roll the cards.
