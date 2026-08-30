# React + Vite Web App Rewrite — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the buildless vanilla-JS site in `docs/` with a React app built by Vite, keeping the deployed output a static site GitHub Pages serves as-is, and fold in seven UI changes (bigger fitted card text, Noto Color Emoji, clickable top-10 emoji list, click-to-set emoji, click-to-set feeling, placeholder instead of hint, all 7 feelings visible).

**Architecture:** New top-level `web/` Vite project. `docs/` old site files are deleted; the model assets (`model.onnx`, `meta.json`, `config.json`, `palette.json`) move to `web/public/`. `train.py`'s `export_web` writes to `web/public/`. `deploy-pages.yml` gains an `npm ci && npm run build` step and uploads `web/dist`. onnxruntime-web comes from npm (wasm inlined in the `onnxruntime-web/wasm` bundle build — no vendored blobs, no CDN). Pure logic (`normalize`/`encode`/`argmax`/`softmax`, text-fit math) lives in framework-free modules with vitest coverage; React components stay thin.

**Tech Stack:** React 19, Vite 8, `@vitejs/plugin-react`, Vitest 4, onnxruntime-web 1.29, Google Fonts (feeling fonts + Noto Color Emoji). Python side: unchanged except `export_web`'s output path.

**Spec:** `docs/superpowers/specs/2026-08-30-react-vite-rewrite-design.md`

## Global Constraints

- **No comments or docstrings in any source file** (repo convention; keep `type: ignore` / `noqa` / shebangs only). This includes the new JS/JSX/CSS.
- **`normalize` must stay byte-identical to `train.py`'s `normalize`** — collapse whitespace, trim, lowercase, collapse 3+ char repeats to 2, drop chars not in the vocab. Changing it invalidates the committed `model.pt`.
- **Char index 0 is padding** (`pad_idx`); `meta.json` `chars` is indexed from 0, so `char2idx` maps `chars[i] -> i` and index 0 (`·`) is the pad char. Sequences are always length `meta.max_text_len`.
- **onnxruntime-web stays single-threaded** (`ort.env.wasm.numThreads = 1`) so no COOP/COEP headers are needed (GitHub Pages sends none).
- **Deployed output must be static** — the build runs in CI or locally, never on Pages.
- **Package manager for `web/` is npm.** All `web/` commands run with `web/` as CWD. Python stays on `uv`.
- The working tree already has pre-existing uncommitted edits from the user to `train.py`, `todo.txt`, `docs/app.js`, `docs/style.css`, `docs/meta.json`, `docs/model.onnx`, and an untracked `report/model/08-30-21:30.md`. These are expected. All ports in this plan are taken from the **current working-tree** contents of `docs/app.js` and `docs/style.css`. Do not revert the user's edits.
- Work happens on the existing branch `react-vite-rewrite`.
- Commit-message steps below show only the subject line. Append the session's required trailer to every commit:
  ```
  Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
  Claude-Session: https://claude.ai/code/session_0115gMvgKuqdoSRDmhF7pdqF
  ```

### Fixed data copied from the current site (used verbatim below)

`FEELING_FONTS` (per-feeling card font stack):

```js
const FEELING_FONTS = {
  Happy: '"Fredoka", system-ui, sans-serif',
  Calm: '"Quicksand", system-ui, sans-serif',
  Sad: '"Playfair Display", Georgia, serif',
  Angry: '"Anton", system-ui, sans-serif',
  Anxious: '"Shantell Sans", system-ui, cursive',
  Neutral: '"Inter", system-ui, sans-serif',
  Love: '"Fredoka", system-ui, sans-serif',
};
```

Emoji font stack (new): `'"Noto Color Emoji", "Apple Color Emoji", "Segoe UI Emoji", sans-serif'`

`meta.json` `chars` string (for tests): `"·abcdefghijklmnopqrstuvwxyz!?:()@$%&* "` — so `·`→0, `a`→1, `b`→2, `c`→3, … `z`→26, `!`→27, `?`→28, `:`→29, `(`→30, `)`→31, `@`→32, `$`→33, `%`→34, `&`→35, `*`→36, ` `→37. No digits in the vocab.

---

## Task 1: Scaffold `web/`, move model assets, remove the old `docs/` site

**Files:**
- Create: `web/package.json`, `web/vite.config.js`, `web/index.html`, `web/.gitignore`, `web/src/main.jsx`, `web/src/App.jsx`, `web/src/styles.css` (placeholder), `web/public/.nojekyll`
- Move (git mv): `docs/model.onnx` → `web/public/model.onnx`; `docs/meta.json` → `web/public/meta.json`; `docs/config.json` → `web/public/config.json`; `docs/palette.json` → `web/public/palette.json`
- Delete (git rm): `docs/index.html`, `docs/app.js`, `docs/style.css`, `docs/.nojekyll`, `docs/vendor/README.md`, `docs/vendor/ort.wasm.min.js`, `docs/vendor/ort-wasm-simd-threaded.mjs`, `docs/vendor/ort-wasm-simd-threaded.wasm`
- Modify: `.gitignore` (stale `docs/` comment block)

**Interfaces:**
- Produces: a `web/` Vite app that runs `npm run dev` and `npm run build`; model assets served from `web/public/` at `import.meta.env.BASE_URL + '<file>'`.

- [ ] **Step 1: Create `web/package.json`**

```json
{
  "name": "emojic-web",
  "private": true,
  "type": "module",
  "scripts": {
    "dev": "vite",
    "build": "vite build",
    "preview": "vite preview",
    "test": "vitest run"
  }
}
```

- [ ] **Step 2: Install deps (let npm resolve current versions)**

Run (CWD `web/`):

```bash
npm install react react-dom onnxruntime-web@1.29.0
npm install -D vite @vitejs/plugin-react vitest
```

Expected: `web/package.json` gains `dependencies` + `devDependencies`, `web/package-lock.json` and `web/node_modules/` created. `onnxruntime-web` is pinned to `1.29.0` (matches the wasm build this plan targets); the rest float on caret ranges.

- [ ] **Step 3: Create `web/vite.config.js`**

```js
import { defineConfig } from 'vitest/config'
import react from '@vitejs/plugin-react'

export default defineConfig({
  base: './',
  plugins: [react()],
  optimizeDeps: { exclude: ['onnxruntime-web'] },
  build: { chunkSizeWarningLimit: 30000 },
  test: { environment: 'node' },
})
```

- [ ] **Step 4: Create `web/index.html`**

```html
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>emojic</title>
  <link rel="preconnect" href="https://fonts.googleapis.com" />
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin />
  <link
    href="https://fonts.googleapis.com/css2?family=Anton&family=Fredoka:wght@500;600&family=Inter:wght@400;600&family=Noto+Color+Emoji&family=Playfair+Display:ital,wght@1,600&family=Quicksand:wght@500;600&family=Shantell+Sans:wght@500;600&display=swap"
    rel="stylesheet"
  />
</head>
<body>
  <div id="root"></div>
  <script type="module" src="/src/main.jsx"></script>
</body>
</html>
```

- [ ] **Step 5: Create `web/src/main.jsx`**

```jsx
import { createRoot } from 'react-dom/client'
import { App } from './App'
import './styles.css'

createRoot(document.getElementById('root')).render(<App />)
```

(No `StrictMode` — it would double-invoke the model-loading effect in dev.)

- [ ] **Step 6: Create `web/src/App.jsx` placeholder**

```jsx
export function App() {
  return <main><h1>emojic</h1></main>
}
```

- [ ] **Step 7: Create `web/src/styles.css` placeholder**

```css
body { margin: 0; font: 16px/1.5 system-ui, sans-serif; }
```

- [ ] **Step 8: Create `web/public/.nojekyll`**

Empty file.

- [ ] **Step 9: Create `web/.gitignore`**

```
node_modules/
dist/
```

- [ ] **Step 10: Move model assets into `web/public/`**

```bash
git mv docs/model.onnx web/public/model.onnx
git mv docs/meta.json web/public/meta.json
git mv docs/config.json web/public/config.json
git mv docs/palette.json web/public/palette.json
```

- [ ] **Step 11: Remove the old static site**

```bash
git rm docs/index.html docs/app.js docs/style.css docs/.nojekyll \
       docs/vendor/README.md docs/vendor/ort.wasm.min.js \
       docs/vendor/ort-wasm-simd-threaded.mjs docs/vendor/ort-wasm-simd-threaded.wasm
```

Expected: `docs/` now contains only `superpowers/`.

- [ ] **Step 12: Update the stale `docs/` comment in `.gitignore`**

Replace:

```
# The Pages site in docs/ is committed and served as-is (no build step).
# model.onnx / meta.json in it are regenerated by `uv run main.py`.
```

with:

```
# The Pages site is built from web/ by the Deploy Pages workflow (Vite).
# web/public/{model.onnx,meta.json,config.json} are regenerated by `uv run train.py`.
```

- [ ] **Step 13: Verify the dev server boots**

Run (CWD `web/`): `npm run dev`
Open the printed URL. Expected: page shows "emojic", no console errors. Stop the server.

- [ ] **Step 14: Verify a production build succeeds**

Run (CWD `web/`): `npm run build`
Expected: `web/dist/` created with `index.html` + `assets/`. A chunk-size warning for onnxruntime is acceptable. Then `npm run preview` and confirm the page loads.

- [ ] **Step 15: Commit**

```bash
git add web .gitignore
git add -u docs
git commit -m "web: scaffold Vite + React app, move model assets, remove old docs site"
```

---

## Task 2: `model.js` — normalize / encode / argmax / softmax (+ vitest)

**Files:**
- Create: `web/src/model.js`, `web/src/model.test.js`

**Interfaces:**
- Produces:
  - `normalize(text: string, char2idx: Map<string, number>) -> string`
  - `encode(text: string, meta: {max_text_len: number, pad_idx: number}, char2idx: Map<string, number>) -> BigInt64Array` (length `meta.max_text_len`)
  - `argmax(arr: number[] | Float32Array) -> number`
  - `softmax(arr: number[] | Float32Array) -> number[]`

- [ ] **Step 1: Write the failing tests**

`web/src/model.test.js`:

```js
import { describe, it, expect } from 'vitest'
import { normalize, encode, argmax, softmax } from './model'

const CHARS = '·abcdefghijklmnopqrstuvwxyz!?:()@$%&* '
const idx = new Map([...CHARS].map((c, i) => [c, i]))

describe('normalize', () => {
  it('lowercases and collapses whitespace', () => {
    expect(normalize('Hello   WORLD', idx)).toBe('hello world')
  })
  it('collapses 3+ char repeats to 2', () => {
    expect(normalize('soooo good', idx)).toBe('soo good')
  })
  it('drops chars outside the vocab (incl. digits and accents)', () => {
    expect(normalize('café #1!', idx)).toBe('caf !')
  })
  it('trims leading/trailing whitespace', () => {
    expect(normalize('  hi there  ', idx)).toBe('hi there')
  })
})

describe('encode', () => {
  const meta = { max_text_len: 5, pad_idx: 0 }
  it('maps chars to indices and pads to max_text_len', () => {
    expect(Array.from(encode('ab', meta, idx))).toEqual([1n, 2n, 0n, 0n, 0n])
  })
  it('truncates to max_text_len', () => {
    expect(Array.from(encode('abcdef', meta, idx))).toEqual([1n, 2n, 3n, 4n, 5n])
  })
  it('returns a BigInt64Array', () => {
    expect(encode('a', meta, idx)).toBeInstanceOf(BigInt64Array)
  })
})

describe('argmax / softmax', () => {
  it('argmax returns the index of the max', () => {
    expect(argmax([0.1, 0.9, 0.3])).toBe(1)
  })
  it('softmax sums to 1 and is monotonic', () => {
    const p = softmax([1, 2, 3])
    expect(p.reduce((a, b) => a + b, 0)).toBeCloseTo(1, 6)
    expect(p[2]).toBeGreaterThan(p[0])
  })
})
```

- [ ] **Step 2: Run tests, verify they fail**

Run (CWD `web/`): `npm test`
Expected: FAIL — `model.js` does not exist.

- [ ] **Step 3: Implement `web/src/model.js`**

```js
export function normalize(text, char2idx) {
  const t = text
    .replace(/\s+/g, ' ')
    .trim()
    .toLowerCase()
    .replace(/(.)\1{2,}/g, '$1$1')
  let out = ''
  for (const c of t) if (char2idx.has(c)) out += c
  return out
}

export function encode(text, meta, char2idx) {
  const norm = normalize(text, char2idx).slice(0, meta.max_text_len)
  const ids = new Array(meta.max_text_len).fill(meta.pad_idx)
  for (let i = 0; i < norm.length; i++) ids[i] = char2idx.get(norm[i])
  return BigInt64Array.from(ids, BigInt)
}

export function argmax(arr) {
  let best = 0
  for (let i = 1; i < arr.length; i++) if (arr[i] > arr[best]) best = i
  return best
}

export function softmax(arr) {
  let m = -Infinity
  for (const x of arr) if (x > m) m = x
  const exps = Array.from(arr, (x) => Math.exp(x - m))
  const sum = exps.reduce((a, b) => a + b, 0)
  return exps.map((e) => e / sum)
}
```

- [ ] **Step 4: Run tests, verify they pass**

Run (CWD `web/`): `npm test`
Expected: PASS (10 assertions).

- [ ] **Step 5: Commit**

```bash
git add web/src/model.js web/src/model.test.js
git commit -m "web: model.js normalize/encode/argmax/softmax with parity tests"
```

---

## Task 3: `useOnnx` hook + `App` inference wiring (debounce, gate, stale guard, one-shot override)

**Files:**
- Create: `web/src/hooks/useOnnx.js`
- Modify: `web/src/App.jsx` (replace placeholder)

**Interfaces:**
- Consumes: `encode`, `argmax`, `softmax`, `normalize` from `./model`.
- Produces (from `useOnnx()`): `{ meta, config, palette, ready: boolean, predict }` where
  `predict(text: string) -> Promise<{ feeling: number[], emoji: number[] }>` (raw logits, lengths `meta.feelings.length` and `meta.emojis.length`).
- Produces (App state contract for later tasks): `text: string`, `scores: {feeling, emoji} | null`, `override: {emoji: string|null, feeling: string|null}`; derived `shownEmoji`, `shownFeeling`, `emojiScores`, `feelingScores`, `cardData: {text, emoji, feeling, pal} | null`.

- [ ] **Step 1: Create `web/src/hooks/useOnnx.js`**

```js
import { useCallback, useEffect, useRef, useState } from 'react'
import * as ort from 'onnxruntime-web/wasm'
import { encode } from '../model'

const BASE = import.meta.env.BASE_URL

export function useOnnx() {
  const [meta, setMeta] = useState(null)
  const [config, setConfig] = useState(null)
  const [palette, setPalette] = useState(null)
  const [ready, setReady] = useState(false)
  const sessionRef = useRef(null)
  const char2idxRef = useRef(null)
  const metaRef = useRef(null)

  useEffect(() => {
    let cancelled = false
    ;(async () => {
      try {
        const [m, c, p] = await Promise.all([
          fetch(BASE + 'meta.json').then((r) => r.json()),
          fetch(BASE + 'config.json').then((r) => r.json()),
          fetch(BASE + 'palette.json').then((r) => r.json()),
        ])
        if (cancelled) return
        setMeta(m)
        setConfig(c)
        setPalette(p)
        metaRef.current = m
        char2idxRef.current = new Map([...m.chars].map((ch, i) => [ch, i]))
        ort.env.wasm.numThreads = 1
        const session = await ort.InferenceSession.create(BASE + 'model.onnx')
        if (cancelled) return
        sessionRef.current = session
        setReady(true)
      } catch (err) {
        console.error('emojic: model load failed', err)
      }
    })()
    return () => {
      cancelled = true
    }
  }, [])

  const predict = useCallback(async (text) => {
    const m = metaRef.current
    const ids = encode(text, m, char2idxRef.current)
    const tensor = new ort.Tensor('int64', ids, [1, m.max_text_len])
    const out = await sessionRef.current.run({ input: tensor })
    return {
      feeling: Array.from(out.feeling_logits.data),
      emoji: Array.from(out.emoji_logits.data),
    }
  }, [])

  return { meta, config, palette, ready, predict }
}
```

- [ ] **Step 2: Replace `web/src/App.jsx`**

```jsx
import { useEffect, useMemo, useRef, useState } from 'react'
import { useOnnx } from './hooks/useOnnx'
import { argmax, normalize, softmax } from './model'

const MIN_CHARS = 3
const DEBOUNCE_MS = 100

function formatDate(iso) {
  if (!iso) return '—'
  const d = new Date(iso)
  return Number.isNaN(d.getTime())
    ? iso
    : d.toLocaleString(undefined, {
        year: 'numeric',
        month: 'short',
        day: 'numeric',
        hour: 'numeric',
        minute: '2-digit',
        timeZoneName: 'short',
      })
}

export function App() {
  const { meta, config, palette, ready, predict } = useOnnx()
  const [text, setText] = useState('')
  const [scores, setScores] = useState(null)
  const [override, setOverride] = useState({ emoji: null, feeling: null })
  const seq = useRef(0)

  const char2idx = useMemo(
    () => (meta ? new Map([...meta.chars].map((c, i) => [c, i])) : null),
    [meta],
  )

  useEffect(() => {
    if (!ready || !char2idx) return
    if (normalize(text, char2idx).length < MIN_CHARS) {
      seq.current++
      setScores(null)
      setOverride({ emoji: null, feeling: null })
      return
    }
    const mine = ++seq.current
    const timer = setTimeout(async () => {
      const logits = await predict(text)
      if (mine !== seq.current) return
      setScores(logits)
      setOverride({ emoji: null, feeling: null })
    }, DEBOUNCE_MS)
    return () => clearTimeout(timer)
  }, [text, ready, char2idx, predict])

  const emojiScores = scores && softmax(scores.emoji)
  const feelingScores = scores && softmax(scores.feeling)
  const predictedEmoji = scores ? meta.emojis[argmax(scores.emoji)] : null
  const predictedFeeling = scores ? meta.feelings[argmax(scores.feeling)] : null
  const shownEmoji = override.emoji ?? predictedEmoji
  const shownFeeling = override.feeling ?? predictedFeeling

  const maxLen = config?.max_text_len ?? 0

  return (
    <main>
      <h1>emojic</h1>
      <div className="stage">
        <div className="card-col">
          <input
            className="input"
            type="text"
            autoComplete="off"
            autoFocus
            maxLength={maxLen || undefined}
            placeholder="type at least 3 characters…"
            value={text}
            onChange={(e) => setText(e.target.value)}
          />
          <div className={'counter' + (maxLen && text.length >= maxLen ? ' full' : '')}>
            {text.length}
            <span>/{maxLen}</span>
          </div>
          <div className="card" data-feeling={shownFeeling || undefined}>
            <span className="card-emoji">{shownEmoji ?? '🙂'}</span>
            <p className="card-text">{text}</p>
            <span className="card-feeling-idle">{shownFeeling ?? '—'}</span>
          </div>
        </div>
      </div>
      <footer className="footer">
        model updated <span>{formatDate(meta?.exported_at)}</span>
      </footer>
      {feelingScores ? null : null}
    </main>
  )
}
```

(The `feelingScores`/`emojiScores`/`palette`/`setOverride` bindings are wired to components in Tasks 6–8; keep them defined now so the contract is stable.)

- [ ] **Step 3: Verify inference end to end**

Run (CWD `web/`): `npm run dev`, open the URL.
- Type `i am so happy right now`. Expected: within ~150ms the card emoji changes from 🙂 to a predicted emoji and the idle text shows a feeling word; no console errors.
- Delete back to under 3 chars. Expected: card returns to 🙂 / `—`.
- Open DevTools Network tab, reload. Expected: `meta.json`, `config.json`, `palette.json`, `model.onnx` all 200. **No failed `.wasm` request.** If a `.wasm` request 404s, the bundle build is not embedding wasm on this version — fallback: `npm i -D vite-plugin-static-copy`, add a `viteStaticCopy` target copying `node_modules/onnxruntime-web/dist/ort-wasm-simd-threaded.{wasm,mjs}` to dist root, and set `ort.env.wasm.wasmPaths = BASE` in `useOnnx.js` before `InferenceSession.create`. Re-verify.

- [ ] **Step 4: Verify the build still works**

Run (CWD `web/`): `npm run build && npm run preview`. Open the preview URL, repeat the "i am so happy" check.

- [ ] **Step 5: Commit**

```bash
git add web/src/hooks/useOnnx.js web/src/App.jsx web/package.json web/package-lock.json
git commit -m "web: onnxruntime-web inference wired into App (debounce, gate, stale guard)"
```

---

## Task 4: Text-fit — `fit.js` (pure, tested) + `useFitText` hook

**Files:**
- Create: `web/src/fit.js`, `web/src/fit.test.js`, `web/src/hooks/useFitText.js`

**Interfaces:**
- Produces:
  - `wrapLines(measure: (s: string) => number, text: string, maxWidth: number, maxLines: number) -> string[]`
  - `fitCanvasFont({ text, maxWidth, maxHeight, min, max, lineHeight, widthAt }) -> number` where `widthAt(s: string, px: number) -> number`
  - `useFitText(text: string, opts?: { min?: number, max?: number }) -> [px: number, ref: RefObject<HTMLElement>]` — binds `ref` to a fixed-size box; sets `fontSize` on it so its content fits.

- [ ] **Step 1: Write the failing tests**

`web/src/fit.test.js`:

```js
import { describe, it, expect } from 'vitest'
import { wrapLines, fitCanvasFont } from './fit'

const widthAt = (s, px) => s.length * px * 0.5

describe('wrapLines', () => {
  it('wraps on width and caps at maxLines', () => {
    const lines = wrapLines((s) => s.length * 10, 'aa bb cc dd', 45, 2)
    expect(lines).toEqual(['aa bb', 'cc dd'])
  })
  it('keeps a single short line', () => {
    expect(wrapLines((s) => s.length, 'hi there', 100, 4)).toEqual(['hi there'])
  })
})

describe('fitCanvasFont', () => {
  it('returns max when the text easily fits', () => {
    const px = fitCanvasFont({
      text: 'hi', maxWidth: 500, maxHeight: 500,
      min: 20, max: 100, lineHeight: 1.3, widthAt,
    })
    expect(px).toBe(100)
  })
  it('shrinks long text below max', () => {
    const long = 'x'.repeat(48)
    const px = fitCanvasFont({
      text: long, maxWidth: 400, maxHeight: 260,
      min: 20, max: 100, lineHeight: 1.3, widthAt,
    })
    expect(px).toBeGreaterThanOrEqual(20)
    expect(px).toBeLessThan(100)
  })
  it('never returns below min', () => {
    const px = fitCanvasFont({
      text: 'y'.repeat(500), maxWidth: 50, maxHeight: 50,
      min: 20, max: 100, lineHeight: 1.3, widthAt,
    })
    expect(px).toBe(20)
  })
})
```

- [ ] **Step 2: Run tests, verify they fail**

Run (CWD `web/`): `npm test`
Expected: FAIL — `fit.js` missing.

- [ ] **Step 3: Implement `web/src/fit.js`**

```js
export function wrapLines(measure, text, maxWidth, maxLines) {
  const words = text.split(/\s+/).filter(Boolean)
  const lines = []
  let line = ''
  for (const w of words) {
    const next = line ? line + ' ' + w : w
    if (line && measure(next) > maxWidth) {
      lines.push(line)
      line = w
    } else {
      line = next
    }
  }
  if (line) lines.push(line)
  return lines.slice(0, maxLines)
}

export function fitCanvasFont({ text, maxWidth, maxHeight, min, max, lineHeight, widthAt }) {
  let lo = min
  let hi = max
  let best = min
  while (lo <= hi) {
    const px = (lo + hi) >> 1
    const measure = (s) => widthAt(s, px)
    const lines = wrapLines(measure, text, maxWidth, 999)
    const widest = lines.reduce((w, l) => Math.max(w, measure(l)), 0)
    const tall = lines.length * px * lineHeight
    if (widest <= maxWidth && tall <= maxHeight) {
      best = px
      lo = px + 1
    } else {
      hi = px - 1
    }
  }
  return best
}
```

- [ ] **Step 4: Implement `web/src/hooks/useFitText.js`**

```js
import { useLayoutEffect, useRef, useState } from 'react'

export function useFitText(text, { min = 32, max = 104 } = {}) {
  const ref = useRef(null)
  const [px, setPx] = useState(max)

  useLayoutEffect(() => {
    const el = ref.current
    if (!el) return
    const fit = () => {
      let lo = min
      let hi = max
      let best = min
      while (lo <= hi) {
        const mid = (lo + hi) >> 1
        el.style.fontSize = mid + 'px'
        if (el.scrollWidth <= el.clientWidth && el.scrollHeight <= el.clientHeight) {
          best = mid
          lo = mid + 1
        } else {
          hi = mid - 1
        }
      }
      el.style.fontSize = best + 'px'
      setPx(best)
    }
    fit()
    const ro = new ResizeObserver(fit)
    ro.observe(el)
    return () => ro.disconnect()
  }, [text, min, max])

  return [px, ref]
}
```

- [ ] **Step 5: Run tests, verify they pass**

Run (CWD `web/`): `npm test`
Expected: PASS (model.test.js + fit.test.js).

- [ ] **Step 6: Commit**

```bash
git add web/src/fit.js web/src/fit.test.js web/src/hooks/useFitText.js
git commit -m "web: text-fit helpers (pure fitCanvasFont + useFitText hook)"
```

---

## Task 5: `styles.css` — port the site styling and the new layout

**Files:**
- Modify: `web/src/styles.css` (replace placeholder with the full file below)

**Interfaces:**
- Consumes: class names used by `App.jsx` now (`stage`, `card-col`, `input`, `counter`, `card`, `card-emoji`, `card-text`, `card-feeling-idle`, `footer`) and by Tasks 6–8 (`card-text-box`, `feeling-bar`, `copy-btn`, `emoji-list`, `emoji-list-empty`, `emoji-list-glyph`, `bar-track`, `bar-fill`, `bar-pct`, `toast`).

- [ ] **Step 1: Replace `web/src/styles.css` with:**

```css
:root {
  color-scheme: light dark;
  --ink: #1a1a1a;
  --muted: #6b6b6b;
  --line: #d8d8d8;
  --w: 600px;
}

* {
  box-sizing: border-box;
}

body {
  margin: 0;
  min-height: 100vh;
  display: grid;
  place-items: center;
  padding: 32px;
  font: 16px/1.5 ui-sans-serif, system-ui, -apple-system, "Segoe UI", Roboto, sans-serif;
  background: #f4f4f5;
  color: var(--ink);
}

main {
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 20px;
}

h1 {
  margin: 0;
  font-size: 14px;
  font-weight: 600;
  letter-spacing: 0.24em;
  text-transform: uppercase;
  color: var(--muted);
}

.footer {
  font-size: 12px;
  letter-spacing: 0.08em;
  color: var(--muted);
}

.stage {
  display: flex;
  flex-direction: row;
  align-items: flex-start;
  gap: 28px;
}

.card-col {
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 20px;
}

.input {
  width: var(--w);
  max-width: 100%;
  padding: 12px 16px;
  font-size: 18px;
  border: 1px solid var(--line);
  border-radius: 10px;
  background: #fff;
  color: var(--ink);
  outline: none;
}

.input:focus {
  border-color: #888;
}

.counter {
  width: var(--w);
  max-width: 100%;
  margin-top: -14px;
  text-align: right;
  font-size: 12px;
  letter-spacing: 0.08em;
  color: var(--muted);
  font-variant-numeric: tabular-nums;
}

.counter.full {
  color: #c0392b;
}

.card {
  position: relative;
  width: var(--w);
  height: var(--w);
  max-width: 100%;
  border-radius: 20px;
  padding: 40px;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  gap: 20px;
  text-align: center;
  overflow: hidden;
  background: linear-gradient(135deg, #dedede, #bebebe);
  color: #161616;
  transition: background 220ms ease, color 220ms ease;
}

@keyframes drift {
  0%, 100% { background-position: 0% 50%; }
  50% { background-position: 100% 50%; }
}
@keyframes emoji-happy {
  0%, 100% { transform: translateY(0); }
  50% { transform: translateY(-16px); }
}
@keyframes emoji-calm {
  0%, 100% { transform: scale(1); }
  50% { transform: scale(1.05); }
}
@keyframes emoji-sad {
  0%, 100% { transform: translateY(0) scale(1); }
  50% { transform: translateY(4px) scale(0.99); }
}
@keyframes emoji-angry {
  0%, 100% { transform: translateX(0) rotate(0); }
  20% { transform: translateX(-3px) rotate(-1.5deg); }
  40% { transform: translateX(3px) rotate(1.5deg); }
  60% { transform: translateX(-2px) rotate(-1deg); }
  80% { transform: translateX(2px) rotate(1deg); }
}
@keyframes emoji-anxious {
  0%, 100% { transform: translate(0, 0); }
  25% { transform: translate(-2px, 1px); }
  50% { transform: translate(2px, -1px); }
  75% { transform: translate(-1px, -2px); }
}
@keyframes emoji-love {
  0%, 100% { transform: scale(1); }
  15% { transform: scale(1.15); }
  30% { transform: scale(1); }
  45% { transform: scale(1.12); }
  60% { transform: scale(1); }
}
@keyframes emoji-neutral {
  0%, 100% { transform: rotate(-0.6deg); }
  50% { transform: rotate(0.6deg); }
}
@keyframes text-happy {
  0%, 100% { transform: translateY(0); }
  50% { transform: translateY(-4px); }
}
@keyframes text-calm {
  0%, 100% { opacity: 1; }
  50% { opacity: 0.72; }
}
@keyframes text-sad {
  0%, 100% { transform: translateY(0); opacity: 1; }
  50% { transform: translateY(3px); opacity: 0.78; }
}
@keyframes text-angry {
  0%, 100% { transform: translateX(0); }
  25% { transform: translateX(-1.5px); }
  75% { transform: translateX(1.5px); }
}
@keyframes text-anxious {
  0%, 100% { transform: translate(0, 0); }
  25% { transform: translate(-1.5px, 1px); }
  50% { transform: translate(1.5px, -1px); }
  75% { transform: translate(-1px, -1.5px); }
}
@keyframes text-love {
  0%, 100% { transform: scale(1); }
  15% { transform: scale(1.04); }
  30% { transform: scale(1); }
  45% { transform: scale(1.03); }
  60% { transform: scale(1); }
}
@keyframes text-neutral {
  0%, 100% { opacity: 1; }
  50% { opacity: 0.9; }
}

.card[data-feeling] {
  background-size: 200% 200%;
  animation: drift 16s ease-in-out infinite;
}
.card[data-feeling="Happy"] .card-emoji { animation: emoji-happy 0.9s ease-in-out infinite; }
.card[data-feeling="Calm"] .card-emoji { animation: emoji-calm 4s ease-in-out infinite; }
.card[data-feeling="Sad"] .card-emoji { animation: emoji-sad 3s ease-in-out infinite; }
.card[data-feeling="Angry"] .card-emoji { animation: emoji-angry 0.45s linear infinite; }
.card[data-feeling="Anxious"] .card-emoji { animation: emoji-anxious 0.2s ease-in-out infinite; }
.card[data-feeling="Love"] .card-emoji { animation: emoji-love 1.4s ease-in-out infinite; }
.card[data-feeling="Neutral"] .card-emoji { animation: emoji-neutral 6s ease-in-out infinite; }
.card[data-feeling="Happy"] .card-text { animation: text-happy 0.9s ease-in-out infinite; }
.card[data-feeling="Calm"] .card-text { animation: text-calm 4s ease-in-out infinite; }
.card[data-feeling="Sad"] .card-text { animation: text-sad 3s ease-in-out infinite; }
.card[data-feeling="Angry"] .card-text { animation: text-angry 0.45s linear infinite; }
.card[data-feeling="Anxious"] .card-text { animation: text-anxious 0.2s ease-in-out infinite; }
.card[data-feeling="Love"] .card-text { animation: text-love 1.4s ease-in-out infinite; }
.card[data-feeling="Neutral"] .card-text { animation: text-neutral 6s ease-in-out infinite; }

@media (prefers-reduced-motion: reduce) {
  .card[data-feeling] { animation: none; background-size: auto; }
  .card[data-feeling] .card-emoji,
  .card[data-feeling] .card-text { animation: none; }
}

.copy-btn {
  position: absolute;
  top: 16px;
  right: 16px;
  padding: 6px 12px;
  font: inherit;
  font-size: 12px;
  letter-spacing: 0.14em;
  text-transform: uppercase;
  color: inherit;
  background: rgba(255, 255, 255, 0.18);
  border: 1px solid currentColor;
  border-radius: 999px;
  cursor: pointer;
  opacity: 0.55;
  transition: opacity 160ms ease;
  -webkit-backdrop-filter: blur(4px);
  backdrop-filter: blur(4px);
}

.copy-btn:hover,
.copy-btn:focus-visible {
  opacity: 1;
  outline: none;
}

.card-emoji {
  font-family: "Noto Color Emoji", "Apple Color Emoji", "Segoe UI Emoji", sans-serif;
  font-size: 140px;
  line-height: 1;
  flex: none;
}

.card-text-box {
  flex: 1 1 auto;
  min-height: 0;
  width: 100%;
  display: flex;
  align-items: center;
  justify-content: center;
  overflow: hidden;
}

.card-text {
  margin: 0;
  font-weight: 600;
  word-break: break-word;
  overflow-wrap: anywhere;
}

.card-feeling-idle {
  flex: none;
  font-size: 13px;
  letter-spacing: 0.28em;
  text-transform: uppercase;
  opacity: 0.85;
}

.feeling-bar {
  flex: none;
  display: flex;
  flex-wrap: wrap;
  justify-content: center;
  gap: 4px 10px;
}

.feeling-bar button {
  font: inherit;
  font-size: 11px;
  letter-spacing: 0.22em;
  text-transform: uppercase;
  color: inherit;
  background: none;
  border: none;
  padding: 2px 4px;
  cursor: pointer;
  opacity: 0.4;
  transition: opacity 160ms ease;
}

.feeling-bar button:hover,
.feeling-bar button:focus-visible {
  opacity: 0.8;
  outline: none;
}

.feeling-bar button.active {
  opacity: 1;
  font-weight: 700;
  text-decoration: underline;
  text-underline-offset: 3px;
}

.emoji-list {
  width: 244px;
  margin: 0;
  padding: 14px 16px;
  border: 1px solid var(--line);
  border-radius: 10px;
  background: #fff;
  list-style: none;
  display: flex;
  flex-direction: column;
  gap: 6px;
  font: 12px/1.4 ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
}

.emoji-list-empty {
  min-height: 360px;
}

.emoji-list li { display: block; }

.emoji-list button {
  width: 100%;
  display: grid;
  grid-template-columns: 1.8em 1fr 3.25em;
  align-items: center;
  gap: 8px;
  font: inherit;
  color: var(--ink);
  background: none;
  border: none;
  padding: 3px 4px;
  border-radius: 6px;
  cursor: pointer;
}

.emoji-list button:hover,
.emoji-list button:focus-visible {
  background: #f0f0f2;
  outline: none;
}

.emoji-list-glyph {
  font-family: "Noto Color Emoji", "Apple Color Emoji", "Segoe UI Emoji", sans-serif;
  font-size: 18px;
}

.bar-track {
  height: 6px;
  border-radius: 3px;
  background: var(--line);
  overflow: hidden;
}

.bar-fill {
  display: block;
  height: 100%;
  background: var(--muted);
  transition: width 160ms ease;
}

.bar-pct {
  text-align: right;
  color: var(--muted);
  font-variant-numeric: tabular-nums;
}

.toast {
  position: fixed;
  left: 50%;
  bottom: 32px;
  transform: translate(-50%, 8px);
  padding: 10px 18px;
  border-radius: 999px;
  font-size: 13px;
  letter-spacing: 0.08em;
  background: rgba(20, 20, 20, 0.9);
  color: #fafafa;
  opacity: 0;
  pointer-events: none;
  transition: opacity 180ms ease, transform 180ms ease;
}

.toast.show {
  opacity: 1;
  transform: translate(-50%, 0);
}

@media (max-width: 980px) {
  .stage {
    flex-direction: column;
    align-items: center;
  }
  .emoji-list {
    order: 2;
    width: var(--w);
    max-width: 100%;
  }
  .card-col {
    order: 1;
  }
}
```

- [ ] **Step 2: Verify**

Run (CWD `web/`): `npm run dev`. Type text. Expected: the card is 600×600 with a gradient, the emoji ~140px, per-feeling gradient/animation applies once a feeling is predicted, layout centered. Resize below 980px: nothing overflows horizontally.

- [ ] **Step 3: Commit**

```bash
git add web/src/styles.css
git commit -m "web: port site styling + stage/emoji-list/feeling-bar layout"
```

---

## Task 6: `Card` + `FeelingBar` components (fitted text, Noto emoji, feeling override)

**Files:**
- Create: `web/src/components/Card.jsx`, `web/src/components/FeelingBar.jsx`
- Modify: `web/src/App.jsx` (render `<Card>` instead of the inline card markup; wire `onPickFeeling`)

**Interfaces:**
- Consumes: `useFitText` from `../hooks/useFitText`.
- `Card` props: `{ text: string, emoji: string, feeling: string | null, feelings: string[], palette: object | null, onPickFeeling: (f: string) => void, onCopy: () => void }`.
- `FeelingBar` props: `{ feelings: string[], active: string, onPick: (f: string) => void }`.
- Produces: clicking a feeling calls `onPickFeeling(f)` → App sets `override.feeling` (one-shot; cleared on next completed prediction).

- [ ] **Step 1: Create `web/src/components/FeelingBar.jsx`**

```jsx
export function FeelingBar({ feelings, active, onPick }) {
  return (
    <div className="feeling-bar">
      {feelings.map((f) => (
        <button
          key={f}
          type="button"
          className={f === active ? 'active' : undefined}
          onClick={() => onPick(f)}
        >
          {f}
        </button>
      ))}
    </div>
  )
}
```

- [ ] **Step 2: Create `web/src/components/Card.jsx`**

```jsx
import { useFitText } from '../hooks/useFitText'
import { FeelingBar } from './FeelingBar'

const FEELING_FONTS = {
  Happy: '"Fredoka", system-ui, sans-serif',
  Calm: '"Quicksand", system-ui, sans-serif',
  Sad: '"Playfair Display", Georgia, serif',
  Angry: '"Anton", system-ui, sans-serif',
  Anxious: '"Shantell Sans", system-ui, cursive',
  Neutral: '"Inter", system-ui, sans-serif',
  Love: '"Fredoka", system-ui, sans-serif',
}

export function Card({ text, emoji, feeling, feelings, palette, onPickFeeling, onCopy }) {
  const [, textRef] = useFitText(text, { min: 32, max: 104 })
  const pal = feeling && palette ? palette[feeling] ?? palette.Neutral : null
  const style = pal
    ? {
        backgroundImage: `linear-gradient(135deg, ${pal.bg1}, ${pal.bg2})`,
        color: pal.text_color,
        fontFamily: FEELING_FONTS[feeling] ?? FEELING_FONTS.Neutral,
      }
    : undefined

  return (
    <div className="card" data-feeling={feeling || undefined} style={style}>
      <span className="card-emoji">{emoji}</span>
      <div className="card-text-box" ref={textRef}>
        <p className="card-text">{text}</p>
      </div>
      {feeling ? (
        <FeelingBar feelings={feelings} active={feeling} onPick={onPickFeeling} />
      ) : (
        <span className="card-feeling-idle">—</span>
      )}
      <button
        className="copy-btn"
        type="button"
        aria-label="Copy card as image"
        onClick={onCopy}
      >
        copy
      </button>
    </div>
  )
}
```

- [ ] **Step 3: Wire into `web/src/App.jsx`**

Add the import:

```jsx
import { Card } from './components/Card'
```

Replace the inline `<div className="card">…</div>` block with:

```jsx
<Card
  text={text}
  emoji={shownEmoji ?? '🙂'}
  feeling={shownFeeling}
  feelings={meta?.feelings ?? []}
  palette={palette}
  onPickFeeling={(f) => setOverride((o) => ({ ...o, feeling: f }))}
  onCopy={() => {}}
/>
```

(`onCopy` becomes real in Task 8.)

- [ ] **Step 4: Verify**

Run (CWD `web/`): `npm run dev`.
- Short input (`hi!`): card text is large (near 104px).
- Long input (~48 chars, e.g. `today was a long tiring day and i just want to rest`): text shrinks to fit inside the card with no clipping or overflow past the feeling row.
- Once a feeling is predicted, all 7 feeling names show at the card bottom, the predicted one emphasized.
- Click a different feeling: card gradient + font switch to it immediately. Type one more character: after the debounce the model's own feeling replaces it (one-shot).

- [ ] **Step 5: Commit**

```bash
git add web/src/components/Card.jsx web/src/components/FeelingBar.jsx web/src/App.jsx
git commit -m "web: Card + FeelingBar components with fitted text and feeling override"
```

---

## Task 7: `EmojiList` component (top-10, clickable, emoji override)

**Files:**
- Create: `web/src/components/EmojiList.jsx`
- Modify: `web/src/App.jsx` (render `<EmojiList>` in `.stage` before `.card-col`; wire `onPick`)

**Interfaces:**
- `EmojiList` props: `{ emojiScores: number[] | null, emojis: string[], onPick: (emoji: string) => void }`.
- Produces: clicking an emoji calls `onPick(emoji)` → App sets `override.emoji` (one-shot).

- [ ] **Step 1: Create `web/src/components/EmojiList.jsx`**

```jsx
export function EmojiList({ emojiScores, emojis, onPick }) {
  if (!emojiScores) {
    return <div className="emoji-list emoji-list-empty" aria-hidden="true" />
  }
  const top = emojiScores
    .map((p, i) => ({ emoji: emojis[i], p }))
    .sort((a, b) => b.p - a.p)
    .slice(0, 10)
  return (
    <ul className="emoji-list">
      {top.map(({ emoji, p }) => (
        <li key={emoji}>
          <button type="button" onClick={() => onPick(emoji)}>
            <span className="emoji-list-glyph">{emoji}</span>
            <span className="bar-track">
              <span className="bar-fill" style={{ width: `${(p * 100).toFixed(1)}%` }} />
            </span>
            <span className="bar-pct">{(p * 100).toFixed(1)}%</span>
          </button>
        </li>
      ))}
    </ul>
  )
}
```

- [ ] **Step 2: Wire into `web/src/App.jsx`**

Add the import:

```jsx
import { EmojiList } from './components/EmojiList'
```

Inside `.stage`, immediately before `<div className="card-col">`:

```jsx
<EmojiList
  emojiScores={emojiScores}
  emojis={meta?.emojis ?? []}
  onPick={(e) => setOverride((o) => ({ ...o, emoji: e }))}
/>
```

- [ ] **Step 3: Verify**

Run (CWD `web/`): `npm run dev`.
- Before typing 3 chars: an empty placeholder box sits left of the card; the card doesn't jump when predictions start.
- After typing: 10 emoji rows with probability bars, descending. Clicking one sets the card emoji immediately. Typing another character replaces it after the debounce (one-shot).
- Narrow viewport (<980px): the list moves below the card.

- [ ] **Step 4: Commit**

```bash
git add web/src/components/EmojiList.jsx web/src/App.jsx
git commit -m "web: clickable top-10 EmojiList with emoji override"
```

---

## Task 8: `useCardImage` + `Toast` + copy wiring (button + Enter)

**Files:**
- Create: `web/src/hooks/useCardImage.js`, `web/src/components/Toast.jsx`
- Modify: `web/src/App.jsx` (toast state, `showToast`, `copyCard`, `onKeyDown` on the input, pass `onCopy`)

**Interfaces:**
- Consumes: `wrapLines`, `fitCanvasFont` from `../fit`.
- `useCardImage(cardData: {text, emoji, feeling, pal} | null, showToast: (msg: string) => void) -> () => Promise<void>`.
- `Toast` props: `{ toast: { msg: string, n: number } }`.
- App adds: `showToast(msg)` bumps `{ msg, n }`; `copyCard = useCardImage(cardData, showToast)`; `cardData` is `shownEmoji && shownFeeling ? { text, emoji: shownEmoji, feeling: shownFeeling, pal: palette[shownFeeling] ?? palette.Neutral } : null`.

- [ ] **Step 1: Create `web/src/hooks/useCardImage.js`**

```js
import { useCallback } from 'react'
import { fitCanvasFont, wrapLines } from '../fit'

const S = 512
const EMOJI_STACK = '"Noto Color Emoji", "Apple Color Emoji", "Segoe UI Emoji", sans-serif'
const FEELING_FONTS = {
  Happy: '"Fredoka", system-ui, sans-serif',
  Calm: '"Quicksand", system-ui, sans-serif',
  Sad: '"Playfair Display", Georgia, serif',
  Angry: '"Anton", system-ui, sans-serif',
  Anxious: '"Shantell Sans", system-ui, cursive',
  Neutral: '"Inter", system-ui, sans-serif',
  Love: '"Fredoka", system-ui, sans-serif',
}

async function ensureFonts(stack) {
  if (!document.fonts) return
  const jobs = [document.fonts.load('400 120px "Noto Color Emoji"')]
  const name = stack.match(/"([^"]+)"/)?.[1]
  if (name) {
    jobs.push(document.fonts.load(`600 24px "${name}"`))
    jobs.push(document.fonts.load(`400 13px "${name}"`))
  }
  try {
    await Promise.all(jobs)
  } catch {}
}

async function render({ text, emoji, feeling, pal }) {
  const stack = FEELING_FONTS[feeling] ?? FEELING_FONTS.Neutral
  await ensureFonts(stack)

  const canvas = document.createElement('canvas')
  canvas.width = S
  canvas.height = S
  const ctx = canvas.getContext('2d')

  const r = Math.round((20 / 600) * S)
  ctx.beginPath()
  if (ctx.roundRect) ctx.roundRect(0, 0, S, S, r)
  else ctx.rect(0, 0, S, S)
  ctx.clip()

  const grad = ctx.createLinearGradient(0, 0, S, S)
  grad.addColorStop(0, pal.bg1)
  grad.addColorStop(1, pal.bg2)
  ctx.fillStyle = grad
  ctx.fillRect(0, 0, S, S)

  ctx.fillStyle = pal.text_color
  ctx.textAlign = 'center'
  ctx.textBaseline = 'middle'

  ctx.font = `120px ${EMOJI_STACK}`
  ctx.fillText(emoji, S / 2, S * 0.36)

  const lineHeight = 1.33
  const maxWidth = S - 96
  const widthAt = (str, px) => {
    ctx.font = `600 ${px}px ${stack}`
    return ctx.measureText(str).width
  }
  const fpx = fitCanvasFont({
    text,
    maxWidth,
    maxHeight: S * 0.34,
    min: Math.round((32 * S) / 600),
    max: Math.round((104 * S) / 600),
    lineHeight,
    widthAt,
  })

  ctx.font = `600 ${fpx}px ${stack}`
  const lines = wrapLines((str) => ctx.measureText(str).width, text, maxWidth, 4)
  let ty = S * 0.6 - ((lines.length - 1) * fpx * lineHeight) / 2
  for (const line of lines) {
    ctx.fillText(line, S / 2, ty)
    ty += fpx * lineHeight
  }

  ctx.font = `600 13px ${stack}`
  if ('letterSpacing' in ctx) ctx.letterSpacing = '3.5px'
  ctx.globalAlpha = 0.85
  ctx.fillText(feeling.toUpperCase(), S / 2, S * 0.84)
  ctx.globalAlpha = 1

  return new Promise((resolve, reject) => {
    canvas.toBlob((b) => (b ? resolve(b) : reject(new Error('toBlob failed'))), 'image/png')
  })
}

export function useCardImage(cardData, showToast) {
  return useCallback(async () => {
    if (!cardData) {
      showToast('nothing to copy yet')
      return
    }
    try {
      const blob = await render(cardData)
      await navigator.clipboard.write([new ClipboardItem({ 'image/png': blob })])
      showToast('copied to clipboard ✓')
    } catch (err) {
      console.error(err)
      showToast('copy failed')
    }
  }, [cardData, showToast])
}
```

- [ ] **Step 2: Create `web/src/components/Toast.jsx`**

```jsx
import { useEffect, useState } from 'react'

export function Toast({ toast }) {
  const [visible, setVisible] = useState(false)
  const [msg, setMsg] = useState('')

  useEffect(() => {
    if (!toast.n) return
    setMsg(toast.msg)
    setVisible(true)
    const t = setTimeout(() => setVisible(false), 1600)
    return () => clearTimeout(t)
  }, [toast])

  return (
    <div className={'toast' + (visible ? ' show' : '')} role="status" aria-live="polite">
      {msg}
    </div>
  )
}
```

- [ ] **Step 3: Wire into `web/src/App.jsx`**

Add imports:

```jsx
import { useCardImage } from './hooks/useCardImage'
import { Toast } from './components/Toast'
```

Add state + helpers (after `override` state):

```jsx
const [toast, setToast] = useState({ msg: '', n: 0 })
const showToast = (msg) => setToast((s) => ({ msg, n: s.n + 1 }))
```

After `shownEmoji` / `shownFeeling` are computed:

```jsx
const cardData =
  shownEmoji && shownFeeling && palette
    ? {
        text,
        emoji: shownEmoji,
        feeling: shownFeeling,
        pal: palette[shownFeeling] ?? palette.Neutral,
      }
    : null
const copyCard = useCardImage(cardData, showToast)
```

Set the input's `onKeyDown`:

```jsx
onKeyDown={(e) => {
  if (e.key === 'Enter') {
    e.preventDefault()
    copyCard()
  }
}}
```

Change the `<Card>` prop `onCopy={() => {}}` → `onCopy={copyCard}`.

Add `<Toast toast={toast} />` just before `</main>` and delete the leftover `{feelingScores ? null : null}` line.

- [ ] **Step 4: Verify**

Run (CWD `web/`): `npm run dev`.
- Type text, click **copy** → toast "copied to clipboard ✓"; paste into an image-accepting app → 512×512 PNG whose gradient, emoji (in color), fitted text size, and feeling label match the card.
- Press **Enter** in the input → same copy.
- Override the emoji, then copy → the PNG shows the overridden emoji.
- Clear the input, press Enter → toast "nothing to copy yet".

- [ ] **Step 5: Commit**

```bash
git add web/src/hooks/useCardImage.js web/src/components/Toast.jsx web/src/App.jsx
git commit -m "web: copy-card-as-PNG (button + Enter) with toast"
```

---

## Task 9: `train.py` — write `export_web` output to `web/public/`

**Files:**
- Modify: `train.py` (lines ~37, ~72-88, ~282)

**Interfaces:**
- Produces: `uv run train.py` writes `model.onnx` / `meta.json` / `config.json` into `web/public/` (not `docs/`).

- [ ] **Step 1: Change the constant**

`train.py` line 37: replace

```python
DOCS = Path("docs")
```

with

```python
WEB_PUBLIC = Path("web/public")
```

- [ ] **Step 2: Update `export_web`**

Replace the body's `DOCS` references:

```python
def export_web(model: nn.Module) -> None:
    WEB_PUBLIC.mkdir(parents=True, exist_ok=True)
    export_onnx(model, WEB_PUBLIC / "model.onnx")
    meta = {
        "chars": CHARS,
        "pad_idx": PAD_IDX,
        "max_text_len": MAX_TEXT_LEN,
        "emojis": EMOJIS,
        "feelings": FEELING,
        "exported_at": datetime.now(UTC).isoformat(timespec="minutes"),
    }
    (WEB_PUBLIC / "meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (WEB_PUBLIC / "config.json").write_text(
        json.dumps({"max_text_len": MAX_TEXT_LEN}, indent=2), encoding="utf-8"
    )
```

- [ ] **Step 3: Update the end-of-train print**

`train.py` ~line 282: replace `f"{MODEL_PT} and docs/ refreshed"` with `f"{MODEL_PT} and web/public/ refreshed"`.

- [ ] **Step 4: Lint**

Run: `uv run ruff check .` and `uv run ruff format --check .`
Expected: both clean.

- [ ] **Step 5: Commit**

```bash
git add train.py
git commit -m "train: export_web writes model assets to web/public/"
```

---

## Task 10: Deploy workflow build step + CLAUDE.md + final verification

**Files:**
- Modify: `.github/workflows/deploy-pages.yml`
- Modify: `CLAUDE.md`

**Interfaces:**
- Produces: pushing `web/**` to `main` builds the Vite app in CI and deploys `web/dist`.

- [ ] **Step 1: Replace `.github/workflows/deploy-pages.yml`**

```yaml
name: Deploy Pages

# Builds the Vite app in web/ and deploys the static output. The build runs
# here, not on Pages. Can also be run by hand from the Actions tab.
# Repo Settings -> Pages -> Source must be "GitHub Actions".
on:
  push:
    branches: [main]
    paths:
      - web/**
      - .github/workflows/deploy-pages.yml
  workflow_dispatch:

permissions:
  contents: read
  pages: write
  id-token: write

concurrency:
  group: pages
  cancel-in-progress: false

jobs:
  deploy:
    runs-on: ubuntu-latest
    environment:
      name: github-pages
      url: ${{ steps.deployment.outputs.page_url }}
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-node@v4
        with:
          node-version: 20
          cache: npm
          cache-dependency-path: web/package-lock.json
      - run: npm ci
        working-directory: web
      - run: npm run build
        working-directory: web
      - uses: actions/upload-pages-artifact@v3
        with:
          path: web/dist
      - id: deployment
        uses: actions/deploy-pages@v4
```

- [ ] **Step 2: Update `CLAUDE.md`**

Make these edits (keep wording consistent with the file's style):

1. **Project section**, first paragraph: the `docs/palette.json` sentence — change the path to `web/public/palette.json` and `docs/app.js` to `web/src` (the React app).
2. The `docs/` bullet (the long one) — replace with:

   > - `web/` — the web app: a Vite + React project. `npm run dev` for local dev, `npm run build` writes the static site to `web/dist`, which the Pages workflow deploys (the build runs in CI, never on Pages). `web/public/` holds the assets regenerated by `train.py`'s `export_web` (`model.onnx`, `meta.json`, `config.json`) plus the hand-maintained `palette.json` (per-feeling hex colors). `web/src/model.js` reimplements `normalize`/`encode` from `meta.json` and must stay byte-identical to `train.py`. Inference runs in the browser via `onnxruntime-web` (wasm backend from npm, single-threaded, no COOP/COEP). `train.py`'s `ExportWrapper` still replaces `pack_padded_sequence` with a `gather` so the LSTM traces to ONNX.

3. Remove the `docs/vendor/` bullet entirely.
4. **Environment & commands**: any line mentioning `uv run main.py` regenerating `docs/` — change `docs/` to `web/public/`. (Note: the actual entry point is `train.py`; leave the `main.py` name alone if the surrounding text already uses it, only fix the `docs/` path.)
5. Replace `Serve locally with any static server, e.g. \`uv run python -m http.server -d docs\`.` with `Run locally with \`cd web && npm install && npm run dev\`.`
6. The `deploy-pages.yml` bullet: change "on push to `main` touching `docs/**`" → "touching `web/**`", and "No build step." → "Runs `npm ci && npm run build` in `web/` and uploads `web/dist`."
7. **Conventions section**: the `docs/palette.json` / `docs/app.js` references — repoint to `web/public/palette.json` / `web/src`. The "No test suite" line — note that `web/` has a vitest suite (`npm test` in `web/`) covering `model.js` and `fit.js`, still no CI gate on it.

- [ ] **Step 3: Full production verification**

Run (CWD `web/`):

```bash
npm ci
npm test
npm run build
npm run preview
```

Open the preview URL and confirm the full checklist:
1. Type text → prediction renders; card gradient/font/animation match the feeling.
2. Card text is visibly larger than the old 16–32px; ~48-char input shrinks to fit with no overflow; ~3-char input is large (near 104px).
3. Emojis render in color (Noto Color Emoji) on the card and in the left list.
4. Left list: 10 emojis with bars; click changes the card emoji; typing more re-predicts and replaces it (one-shot).
5. All 7 feelings visible at the card bottom, predicted one emphasized; click changes the card + the copied image; typing more replaces it.
6. Input placeholder reads "type at least 3 characters…"; there is no separate hint element; sub-3-char input shows the idle 🙂 / — card.
7. Copy button and Enter both copy a PNG reflecting the shown emoji, feeling, and fitted text.
8. Narrow viewport (<980px): list stacks below the card; no horizontal scroll.
9. DevTools Network on reload: `meta.json`, `config.json`, `palette.json`, `model.onnx` all 200; no failed `.wasm`.

Run (repo root): `uv run ruff check .` and `uv run ruff format --check .` → clean.

- [ ] **Step 4: Commit**

```bash
git add .github/workflows/deploy-pages.yml CLAUDE.md
git commit -m "ci+docs: Vite build step in deploy-pages, update CLAUDE.md for web/"
```

- [ ] **Step 5: Final review of the whole branch**

```bash
git log --oneline main..HEAD
git diff --stat main..HEAD
```

Confirm: `docs/` contains only `superpowers/`; `web/` is the app; `train.py` + workflow + `CLAUDE.md` updated. Hand back to the requesting skill for code review / merge.

---

## Self-Review

**Spec coverage:**

| Spec item | Task |
|---|---|
| `web/` layout, deps, scripts | 1, 2–8 (files), 3 (deps installed) |
| `base: './'`, wasm handling | 1 (config), 3 (import + fallback) |
| Delete old `docs/` site, move assets | 1 |
| `model.js` parity + vitest | 2 |
| `useOnnx` (fetch assets, session, predict) | 3 |
| `App` state, debounce 100ms, <3 gate, stale seq guard, one-shot override | 3 |
| `useFitText` 32–104px + `fitCanvasFont` | 4 |
| `styles.css` port, keyframes, reduced-motion, responsive stage | 5 |
| `Card` (gradient, data-feeling, per-feeling font, Noto emoji, fitted text) | 6 |
| `FeelingBar` all 7 visible, active emphasis, click-to-set | 6 |
| `EmojiList` top-10, bars, placeholder box, click-to-set | 7 |
| `useCardImage` canvas port + Noto emoji FontFace + fitted size | 8 |
| `Toast` | 8 |
| Enter + button copy | 8 |
| Placeholder replaces hint; counter kept; no `#debug`/`#feelings`/`#hint` | 3 (placeholder/counter), 5 (dropped selectors) |
| `train.py` `export_web` → `web/public/` | 9 |
| `deploy-pages.yml` build step, `web/**` trigger | 10 |
| `CLAUDE.md` updates, vendor README removal | 1 (README rm), 10 (doc text) |
| Testing: vitest + manual checklist | 2, 4 (auto); 10 (full manual) |
| Risk: wasm path — verified in preview, documented fallback | 3 Step 3 |
| Risk: normalize drift — verbatim port + parity vectors | 2 |
| Risk: font-fit timing — useLayoutEffect + ResizeObserver | 4 |

No gaps.

**Placeholder scan:** No TBD/TODO/"handle edge cases"/"similar to Task N". Every code step has full code. The one conditional branch (wasm fallback in Task 3 Step 3) has concrete commands.

**Type consistency:** `predict` returns `{ feeling, emoji }` (raw logits) — consumed that way in App (Task 3). `override` is `{ emoji, feeling }` everywhere. `cardData` shape `{ text, emoji, feeling, pal }` defined in Task 8, consumed by `useCardImage`/`render` with the same keys. `useFitText` returns `[px, ref]`; Card uses `const [, textRef]`. `toast` is `{ msg, n }` in App and `Toast` props. `EmojiList` prop `emojiScores` matches App's `emojiScores` binding. `FeelingBar` prop `active` matches Card's `active={feeling}`. Consistent.
