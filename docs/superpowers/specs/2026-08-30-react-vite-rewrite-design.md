# React + Vite rewrite of the emojic web app

Date: 2026-08-30
Status: approved (design), pending implementation plan

## Goal

Replace the buildless vanilla-JS site in `docs/` with a React application
built by Vite, while keeping the deployed output a static, backend-free site
that GitHub Pages serves as-is. The build runs in CI (and locally); Pages
never runs a build.

Bundled into the rewrite are seven behavioural changes:

1. Larger card text that is fitted to the card: a font size that scales
   between a minimum tuned for ~48 characters and a maximum tuned for ~3
   characters.
2. Emojis rendered with the Noto Color Emoji font.
3. A clickable top-10 emoji list to the left of the card.
4. Clicking an emoji in that list sets the displayed emoji.
5. Clicking a feeling at the bottom of the card sets the displayed feeling.
6. The "type at least 3 characters…" hint element is removed; that text
   becomes the input's placeholder.
7. All seven feeling names are always visible at the card bottom, the
   predicted one emphasised.

## Non-goals

- No change to the model, training, `data.jsonl`, `labels.json`, or the
  ONNX export wrapper.
- No change to what the model predicts or how `normalize` / `encode`
  behave. The JS reimplementation stays byte-identical to `main.py`.
- No color head, no new model outputs. Colors remain a fixed lookup in
  `palette.json`.
- No server, no API, no COOP/COEP headers (onnxruntime-web stays
  single-threaded).

## Project layout

New top-level `web/` directory holds the Vite app. The old static-site
files under `docs/` are removed (`app.js`, `index.html`, `style.css`,
`palette.json`, `config.json`, `meta.json`, `model.onnx`, `.nojekyll`,
`vendor/` and its `README.md`). `docs/superpowers/` (specs) stays.

```
web/
  package.json
  vite.config.js
  index.html
  public/
    .nojekyll
    palette.json          hand-maintained (moved verbatim from docs/palette.json)
    model.onnx            written by main.py export_web
    meta.json             written by main.py export_web
    config.json           written by main.py export_web
  src/
    main.jsx
    App.jsx
    styles.css             ported from docs/style.css (feeling fonts + animations)
    model.js               normalize / encode / argmax / softmax
    model.test.js          vitest parity tests for model.js
    hooks/
      useOnnx.js           load meta/config/palette + create InferenceSession + predict
      useFitText.js        binary-search font size to fit a box
      useCardImage.js      canvas render of the card + clipboard write
    components/
      Card.jsx
      EmojiList.jsx
      FeelingBar.jsx
      Toast.jsx
```

### Dependencies (`web/package.json`)

Runtime: `react`, `react-dom`, `onnxruntime-web`.
Dev: `vite`, `@vitejs/plugin-react`, `vitest`, and — if the wasm copy step
is done with a plugin rather than hand-rolled config — `vite-plugin-static-copy`.

Versions are whatever `npm install` resolves at implementation time; the
onnxruntime-web major version is pinned in `package.json` once chosen and
recorded in the implementation plan. No ESLint/Prettier config is added
(the repo has none for JS today).

### Scripts

- `npm run dev` — Vite dev server. `public/` is served directly, so the
  model assets load without a build.
- `npm run build` — outputs `web/dist`.
- `npm run preview` — serves `web/dist` for local verification.
- `npm test` — `vitest run`.

All run from `web/` as CWD.

## Build configuration

`web/vite.config.js`:

- `base: './'` so asset URLs are relative and work under the Pages
  project path.
- `@vitejs/plugin-react`.
- A copy step (custom plugin or `vite-plugin-static-copy`, decided during
  implementation) that emits the onnxruntime-web single-threaded wasm
  backend files (`ort-wasm-simd.wasm` and its `.mjs` glue — exact file
  names confirmed against the installed package version) into `dist/` at
  a known path.

At runtime, before creating the session:

```
ort.env.wasm.numThreads = 1;
ort.env.wasm.wasmPaths = <the copied path, resolved from import.meta.env.BASE_URL>;
```

This keeps the backend single-threaded, so no cross-origin isolation
headers are required (matching today's `docs/vendor/` setup).

## onnxruntime-web

`import * as ort from 'onnxruntime-web'` in `useOnnx.js`. The committed
`docs/vendor/` blobs are deleted; the wasm backend now comes from the npm
package and is copied into the build output by Vite config.

Session creation and inference match the current `app.js`:

- Input tensor: `new ort.Tensor('int64', BigInt64Array, [1, max_text_len])`.
- Outputs read by name (`emoji_logits`, `feeling_logits`), argmax on each.

## Model asset pipeline

`main.py`'s `export_web` (and the `--export-only` path) writes
`model.onnx`, `meta.json`, and `config.json` to `web/public/` instead of
`docs/`. `palette.json` is moved into `web/public/` and stays
hand-maintained there.

`CLAUDE.md` is updated:

- The `docs/` bullet is rewritten to describe `web/` (Vite app, build
  step, `web/public/` assets, `npm` toolchain).
- The `docs/vendor/` bullet is removed.
- "Serve locally" guidance becomes `cd web && npm run dev`.
- The `deploy-pages.yml` bullet is updated for the new build step and
  `web/**` trigger path.
- The "no build step" statements in the Project and Conventions sections
  are corrected.

## Deploy workflow

`.github/workflows/deploy-pages.yml`:

- `on.push.paths`: `docs/**` → `web/**`; keep the workflow file path and
  `workflow_dispatch`.
- Job steps: `actions/checkout@v4` → `actions/setup-node@v4` (Node 20,
  `cache: npm`, `cache-dependency-path: web/package-lock.json`) →
  `npm ci` (working directory `web`) → `npm run build` (working directory
  `web`) → `actions/upload-pages-artifact@v3` with `path: web/dist` →
  `actions/deploy-pages@v4`.
- `permissions`, `concurrency`, and the `github-pages` environment are
  unchanged.

`web/public/.nojekyll` is included so the artifact keeps underscore-prefixed
asset files.

Repo Settings → Pages → Source stays "GitHub Actions".

## Application behaviour

### `model.js`

Pure functions, no React. Ported from `app.js` with identical logic:

- `normalize(text, char2idx)` — the exact regex chain from `app.js`:
  collapse whitespace, trim, lowercase, collapse 3+ repeats to 2, then
  drop chars not in the vocab. This must stay byte-identical to
  `main.py`'s `normalize`; changing it invalidates the committed
  `model.pt`.
- `encode(text, meta, char2idx)` — normalize, slice to `max_text_len`,
  map to indices, pad with `pad_idx`, return `BigInt64Array`.
- `argmax(arr)`, `softmax(arr)` — as today.

`model.test.js` (vitest) checks `normalize` and `encode` against fixed
input/expected vectors derived from the `meta.json` char set (whitespace
collapse, casing, repeat collapse, non-vocab drop, padding, truncation
at `max_text_len`).

### `useOnnx.js`

On mount: `fetch` `meta.json`, `config.json`, `palette.json` from
`import.meta.env.BASE_URL`; build `char2idx`; set `ort.env` wasm config;
`ort.InferenceSession.create('<base>model.onnx')`. Exposes
`{ meta, config, palette, ready, predict }` where
`predict(text) -> { feeling: number[], emoji: number[] }` returns raw
logits arrays.

### `App.jsx`

State:

- `text` — controlled input value.
- `scores` — `{ feeling: number[], emoji: number[] } | null` (raw logits
  from the last completed run).
- `override` — `{ emoji: string | null, feeling: string | null }`.

Inference effect:

- Debounce 100ms after the last `text` change.
- Skip (and clear `scores`, reset `override`) when
  `normalize(text).length < 3`.
- A `useRef` sequence counter is incremented per run; an async result is
  applied only if its sequence still matches (drops stale responses,
  same guard as today).
- **One-shot override:** when a run completes and sets `scores`, it also
  resets `override` to `{ emoji: null, feeling: null }`. So a click takes
  effect immediately and until the next completed prediction, which then
  wins.

Derived values:

- `feelingScores = scores && softmax(scores.feeling)`.
- `emojiScores = scores && softmax(scores.emoji)`.
- `predictedEmoji = scores ? meta.emojis[argmax(scores.emoji)] : null`.
- `predictedFeeling = scores ? meta.feelings[argmax(scores.feeling)] : null`.
- `shownEmoji = override.emoji ?? predictedEmoji`.
- `shownFeeling = override.feeling ?? predictedFeeling`.
- `pal = palette[shownFeeling] ?? palette.Neutral`.

Clearing the input (or dropping below 3 normalized chars) returns the
card to its idle state (`🙂`, `—`, no `data-feeling`).

Enter in the input triggers copy (as today).

### `components/Card.jsx`

Props: `text`, `emoji`, `feeling`, `feelings` (the 7), `palette`,
`onPickFeeling`, `onCopy`.

- Root `div` with `data-feeling={feeling || undefined}`; inline styles for
  the gradient background (`linear-gradient(135deg, bg1, bg2)`),
  `color: text_color`, and the per-feeling `fontFamily` (the
  `FEELING_FONTS` map ported from `app.js`).
- Emoji span: `font-family` set to the Noto Color Emoji stack; keeps the
  per-feeling animation classes from `styles.css`.
- Typed text `<p>`: font size from `useFitText` (see below); keeps the
  per-feeling text animation.
- `FeelingBar` at the bottom.
- Copy button (top-right, `position: absolute`), toast handled at App
  level.

### `useFitText.js`

`useFitText(text, { min: 32, max: 104 })` returns a font-size in px and a
`ref` to attach to the text element's container.

- In `useLayoutEffect` (on `text` change and on a `ResizeObserver`
  callback), binary-search the integer font size in `[min, max]` such
  that the text element's `scrollWidth <= clientWidth` and
  `scrollHeight <= clientHeight`, using the largest size that fits.
- The container is a fixed box inside the card (the vertical region
  between the emoji and the feeling bar); `word-break: break-word` and a
  line height come from `styles.css`.
- `min`/`max` are constants tuned so ~48 characters lands near `min` and
  ~3 characters reaches `max`; they are adjustable during implementation
  after eyeballing.

### `useCardImage.js`

Ported from `cardToBlob` / `copyCard` in `app.js`:

- 512×512 canvas, rounded clip, linear gradient fill, `text_color`.
- Emoji drawn with a Noto Color Emoji font string; the face is loaded via
  `FontFace` / `document.fonts.load` before drawing (as `ensureFont` does
  today, extended to the emoji face).
- Typed text drawn at the `useFitText` size scaled by `512 / 600`, word
  wrapped to at most 4 lines (`wrapLines` ported unchanged).
- Feeling drawn uppercased near the bottom.
- `navigator.clipboard.write([new ClipboardItem({ 'image/png': blob })])`;
  App shows a toast (`copied to clipboard ✓` / `copy failed` /
  `nothing to copy yet`).

### `components/EmojiList.jsx`

Props: `emojiScores` (softmax array or null), `emojis` (`meta.emojis`),
`onPick`.

- Renders an empty placeholder box (reserving its column width) when
  `emojiScores` is null, so the card doesn't shift when predictions arrive.
- Otherwise the top 10 by score, each a `<button>` showing the emoji
  glyph (Noto Color Emoji font) and a probability bar + percentage
  (ported `dbg-*` styles).
- Click → `onPick(emoji)` → sets `override.emoji`.
- Positioned to the left of the card via the `main` layout (see below).

### `components/FeelingBar.jsx`

Props: `feelings` (the 7), `active` (shown feeling), `onPick`.

- All seven names always rendered in a row at the card bottom.
- The `active` one is emphasised (weight / opacity / underline — visual
  detail during implementation).
- Click → `onPick(feeling)` → sets `override.feeling`.

### `components/Toast.jsx`

Ported from the `#toast` element + `toast()` helper: a fixed pill,
`role="status"`, auto-hides after 1600ms.

### Input and counter

- `<input>` `placeholder="type at least 3 characters…"`, `autoFocus`,
  `autocomplete="off"`, `maxLength={config.max_text_len}`.
- Character counter (`length / max`, `full` class at the limit) is kept.
- The `#hint` element and all hint logic are removed.

### Layout (`styles.css`)

- Ported from `docs/style.css`: the `:root` tokens, `#card` sizing
  (`--w: 600px`), all `@keyframes` and per-`data-feeling` animation rules,
  `prefers-reduced-motion` handling, `#toast`, and the `dbg-*` bar styles
  (reused by `EmojiList`).
- `main` becomes a horizontal flex row on wide viewports:
  `[EmojiList] [Card]`, centred. Below a breakpoint (~900px) it stacks,
  `EmojiList` under the `Card`.
- The old right-side `#debug` panel and the decorative `#feelings` swatch
  row are removed. The `model updated <date>` footer is kept.
- No comments in the ported CSS or any new source file (repo convention).

## Error handling

- Asset fetch / session creation failure in `useOnnx`: log to console,
  leave `ready` false; the card stays in its idle state and the input
  still echoes text. No user-facing error UI (matches today).
- Stale inference results: dropped by the sequence-counter guard.
- Clipboard write failure: `copy failed` toast.
- `palette[feeling]` miss: fall back to `palette.Neutral`.

## Testing

Automated:

- `npm test` — vitest on `model.js` (`normalize`, `encode`, `argmax`,
  `softmax`).

Manual verification checklist (run before calling the work done):

1. `cd web && npm install && npm run build && npm run preview`.
2. Load the preview: type text, confirm a prediction renders, the card
   gradient/font/animation match the feeling.
3. Card text visibly larger than before; long (~48 char) input shrinks to
   fit without overflow; short (~3 char) input is large.
4. Emojis render in color (Noto Color Emoji), on the card and in the left
   list.
5. Left list shows 10 emojis with bars; clicking one changes the card
   emoji; typing more re-predicts and replaces it (one-shot).
6. All 7 feelings visible at the card bottom; clicking one changes the
   card (colors/font) and the copied image; typing more replaces it.
7. Input placeholder reads "type at least 3 characters…"; no separate
   hint element; sub-3-char input shows the idle card.
8. Copy button and Enter both copy a PNG that reflects the shown emoji,
   feeling, and fitted text size.
9. Narrow viewport: list stacks below the card, nothing overflows
   horizontally.
10. `uv run ruff check .` and `uv run ruff format --check .` clean after
    the `main.py` export-path change.

CI verification: after merge to `main`, the `Deploy Pages` workflow runs
`npm ci && npm run build` and deploys `web/dist`.

## Risks

- **onnxruntime-web + Vite wasm resolution.** Getting `wasmPaths` and the
  copied backend files right is the main unknown. Mitigation: verify with
  `npm run preview` (not just `npm run dev`) and a real inference; pick
  the single-threaded backend explicitly.
- **`normalize` drift.** The JS must stay byte-identical to `main.py`.
  Mitigation: port the regex chain verbatim; vitest parity vectors.
- **Font-fit timing in React.** Measure in `useLayoutEffect` with a
  `ResizeObserver`; binary search over integer px to keep it cheap.
- **Google Fonts availability for Noto Color Emoji.** Acceptable: the
  current site already depends on Google Fonts, and "still buildless" was
  chosen over "no CDN". System emoji fonts are the fallback stack.

## Out of scope / follow-ups

- Self-hosting fonts (would satisfy a stricter "no CDN" bar).
- Any ESLint/Prettier setup for the JS tree.
- Removing `docs/` from git history.
