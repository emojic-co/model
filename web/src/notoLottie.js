// Animated Noto emoji (Lottie, CC BY 4.0), vendored into public/noto/ by
// `bun run fetch-noto-lottie`. Emojis missing from index.json have no animated clone.
const BASE = `${import.meta.env.BASE_URL}noto/`

let indexPromise
export function loadNotoIndex() {
  indexPromise ??= fetch(`${BASE}index.json`)
    .then((r) => (r.ok ? r.json() : {}))
    .catch(() => ({}))
  return indexPromise
}

export const notoUrl = (stem) => `${BASE}${stem}.json`

export function prefersReducedMotion() {
  return typeof matchMedia === 'function' && matchMedia('(prefers-reduced-motion: reduce)').matches
}
