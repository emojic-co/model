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

function linearToSrgb(c) {
  const v = c <= 0.0031308 ? 12.92 * c : 1.055 * c ** (1 / 2.4) - 0.055
  return Math.min(255, Math.max(0, Math.round(v * 255)))
}

export function oklabToHex(L, a, b) {
  const l_ = L + 0.3963377774 * a + 0.2158037573 * b
  const m_ = L - 0.1055613458 * a - 0.0638541728 * b
  const s_ = L - 0.0894841775 * a - 1.291485548 * b
  const l = l_ ** 3
  const m = m_ ** 3
  const s = s_ ** 3
  const r = 4.0767416621 * l - 3.3077115913 * m + 0.2309699292 * s
  const g = -1.2684380046 * l + 2.6097574011 * m - 0.3413193965 * s
  const bl = -0.0041960863 * l - 0.7034186147 * m + 1.707614701 * s
  const hex = (n) => n.toString(16).padStart(2, '0')
  return '#' + hex(linearToSrgb(r)) + hex(linearToSrgb(g)) + hex(linearToSrgb(bl))
}

export function decodeColors(color) {
  return {
    bg1: oklabToHex(color[0], color[1], color[2]),
    bg2: oklabToHex(color[3], color[4], color[5]),
    text_color: oklabToHex(color[6], color[7], color[8]),
  }
}

const HUE_ANGLES = [12, -12, 24, -24]
const L_NUDGES = [0.03, -0.03, 0.05, -0.05]

export function paletteVariants(color, n = 4) {
  const out = [decodeColors(color)]
  for (let k = 0; k < n; k++) {
    const rad = (HUE_ANGLES[k % HUE_ANGLES.length] * Math.PI) / 180
    const cos = Math.cos(rad)
    const sin = Math.sin(rad)
    const dL = L_NUDGES[k % L_NUDGES.length]
    const v = new Array(9)
    for (let i = 0; i < 9; i += 3) {
      const a = color[i + 1]
      const b = color[i + 2]
      v[i] = Math.min(1, Math.max(0, color[i] + dL))
      v[i + 1] = a * cos - b * sin
      v[i + 2] = a * sin + b * cos
    }
    out.push(decodeColors(v))
  }
  return out
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
