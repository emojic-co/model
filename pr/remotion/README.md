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
| `src/showcase.json` | **The fixed cards** (text, emoji, style, colors), hand-picked and committed. Nothing regenerates it. |
| `showcase.txt` | Candidate texts for `candidates.mjs`. |
| `candidates.mjs` | Explores the real ONNX model: for each text, the top emojis that have an **original** Google Noto animation (byte-identical to Google's `lottie.json`, not one we generated) and the best-scoring palettes out of many samples. Writes `candidates.json`; never touches `showcase.json`. |
| `sync-assets.mjs` | Copies the Lottie clips used by `showcase.json` into `public/noto/`. |
| `src/Showcase.tsx` | The composition: timing, the five transitions, background, emoji rain. |
| `src/CardClip.tsx` | One real card on a canvas, drawn deterministically for a given time. |
| `src/fonts.ts` | Google Fonts for every card style. |
| `webpack-override.mjs` | Shims for the web modules' Vite idioms (`?raw`, `import.meta.env`). |
| `render.mjs` | Renders `out/showcase.mp4`. |

## Workflow

```sh
cd pr/remotion
npm install
npm run studio        # preview / scrub
npm run render        # -> out/showcase.mp4
```

To change a card, edit `src/showcase.json` (copy a pick from `candidates.json`
after `npm run candidates`), then `npm run sync-assets`. The model's color output
is noisy, so `candidates` gives different palettes each run — that is why the
chosen ones are frozen in `showcase.json`.
