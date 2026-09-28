export function normalize(text, char2idx) {
  const t = text
    .toLowerCase()
    .replace(/\s+/g, ' ')
    .trim()
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

export const AB_RANGE = 0.4

function hexToRgb(hex) {
  const n = parseInt(hex.slice(1), 16)
  return [(n >> 16) & 255, (n >> 8) & 255, n & 255]
}

function srgbToLinear(c) {
  return c <= 0.04045 ? c / 12.92 : ((c + 0.055) / 1.055) ** 2.4
}

function linearToSrgb(c) {
  const v = c <= 0.0031308 ? 12.92 * c : 1.055 * Math.max(c, 0) ** (1 / 2.4) - 0.055
  return v * 255
}

export function srgbToOklab([r0, g0, b0]) {
  const r = srgbToLinear(r0 / 255)
  const g = srgbToLinear(g0 / 255)
  const b = srgbToLinear(b0 / 255)
  const l = Math.cbrt(0.4122214708 * r + 0.5363325363 * g + 0.0514459929 * b)
  const m = Math.cbrt(0.2119034982 * r + 0.6806995451 * g + 0.1073969566 * b)
  const s = Math.cbrt(0.0883024619 * r + 0.2817188376 * g + 0.6299787005 * b)
  return [
    0.2104542553 * l + 0.793617785 * m - 0.0040720468 * s,
    1.9779984951 * l - 2.428592205 * m + 0.4505937099 * s,
    0.0259040371 * l + 0.7827717662 * m - 0.808675766 * s,
  ]
}

export function oklabToSrgb([L, a, b]) {
  const l = (L + 0.3963377774 * a + 0.2158037573 * b) ** 3
  const m = (L - 0.1055613458 * a - 0.0638541728 * b) ** 3
  const s = (L - 0.0894841775 * a - 1.291485548 * b) ** 3
  return [
    linearToSrgb(4.0767416621 * l - 3.3077115913 * m + 0.2309699292 * s),
    linearToSrgb(-1.2684380046 * l + 2.6097574011 * m - 0.3413193965 * s),
    linearToSrgb(-0.0041960863 * l - 0.7034186147 * m + 1.707614701 * s),
  ]
}

export function hexToOklab(hex) {
  return srgbToOklab(hexToRgb(hex))
}

export function toCssOklab([L, a, b]) {
  return `oklab(${(L * 100).toFixed(2)}% ${a.toFixed(4)} ${b.toFixed(4)})`
}

// hero-patterns (see patterns.js) only accepts hex fill colors -- it strips
// everything before a literal "#" and inlines the rest unescaped into a
// pre-encoded SVG data URI, so an oklab() string corrupts the encoding.
export function toHexColor([L, a, b]) {
  const clamp = (v) => Math.max(0, Math.min(255, Math.round(v))).toString(16).padStart(2, '0')
  return '#' + oklabToSrgb([L, a, b]).map(clamp).join('')
}

function decodeColor([n0, n1, n2], abRange) {
  return [(n0 + 1) / 2, n1 * abRange, n2 * abRange]
}

export function decodeColors(color, abRange = AB_RANGE) {
  return {
    bg1: decodeColor([color[0], color[1], color[2]], abRange),
    bg2: decodeColor([color[3], color[4], color[5]], abRange),
    text_color: decodeColor([color[6], color[7], color[8]], abRange),
  }
}

export function decodeColorList(flat, abRange = AB_RANGE) {
  const arr = Array.from(flat)
  const out = []
  for (let i = 0; i + 9 <= arr.length; i += 9) out.push(decodeColors(arr.slice(i, i + 9), abRange))
  return out
}

function relLuminance([r, g, b]) {
  return (
    0.2126 * srgbToLinear(r / 255) +
    0.7152 * srgbToLinear(g / 255) +
    0.0722 * srgbToLinear(b / 255)
  )
}

export function contrastRatio(oklabA, oklabB) {
  const la = relLuminance(oklabToSrgb(oklabA))
  const lb = relLuminance(oklabToSrgb(oklabB))
  return (Math.max(la, lb) + 0.05) / (Math.min(la, lb) + 0.05)
}

export const CONTRAST_MIN = 3
export const BLACK = [0, 0, 0]
export const WHITE = [1, 0, 0]

function minMargin(fg, bg1, bg2) {
  return Math.min(contrastRatio(fg, bg1), contrastRatio(fg, bg2))
}

export function fixContrast(palette, minContrast = CONTRAST_MIN) {
  const { bg1, bg2, text_color } = palette
  const ok = (fg) => minMargin(fg, bg1, bg2) >= minContrast
  if (ok(text_color)) return palette

  const [L0, a, b] = text_color
  const STEP = 0.02
  let best = null
  let bestCost = Infinity
  for (const dir of [-1, 1]) {
    for (let L = L0 + dir * STEP; L >= 0 && L <= 1; L += dir * STEP) {
      const cand = [L, a, b]
      if (ok(cand)) {
        if (Math.abs(L - L0) < bestCost) {
          best = cand
          bestCost = Math.abs(L - L0)
        }
        break
      }
    }
  }
  if (!best) {
    best = minMargin(BLACK, bg1, bg2) >= minMargin(WHITE, bg1, bg2) ? BLACK : WHITE
  }
  return { bg1, bg2, text_color: best }
}

export function mixColors(a, b, t = 0.5) {
  const [L1, a1, b1] = a
  const [L2, a2, b2] = b
  return [L1 + (L2 - L1) * t, a1 + (a2 - a1) * t, b1 + (b2 - b1) * t]
}

export function patternTint(bg1, bg2) {
  const [L, a, b] = mixColors(bg1, bg2)
  return [Math.max(0.94, L), a, b]
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

export function sigmoid(arr) {
  return Array.from(arr, (x) => 1 / (1 + Math.exp(-x)))
}
