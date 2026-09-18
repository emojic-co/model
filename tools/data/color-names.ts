export type HSL = { h: number; s: number; l: number }
export type NamedColor = { name: string; hex: string }
export type ColorTerm = { text: string; bg: [string, string]; fg: string }

function clamp01(x: number): number {
  return Math.min(1, Math.max(0, x))
}

export function hexToHsl(hex: string): HSL {
  const n = parseInt(hex.replace(/^#/, ""), 16)
  const r = ((n >> 16) & 255) / 255
  const g = ((n >> 8) & 255) / 255
  const b = (n & 255) / 255
  const max = Math.max(r, g, b)
  const min = Math.min(r, g, b)
  const l = (max + min) / 2
  const d = max - min
  if (d === 0) return { h: 0, s: 0, l }
  const s = d / (1 - Math.abs(2 * l - 1))
  let h: number
  switch (max) {
    case r:
      h = ((g - b) / d) % 6
      break
    case g:
      h = (b - r) / d + 2
      break
    default:
      h = (r - g) / d + 4
  }
  h *= 60
  if (h < 0) h += 360
  return { h, s, l }
}

export function hslToHex({ h, s, l }: HSL): string {
  const c = (1 - Math.abs(2 * l - 1)) * s
  const hp = (((h % 360) + 360) % 360) / 60
  const x = c * (1 - Math.abs((hp % 2) - 1))
  const [r1, g1, b1] =
    hp < 1 ? [c, x, 0]
    : hp < 2 ? [x, c, 0]
    : hp < 3 ? [0, c, x]
    : hp < 4 ? [0, x, c]
    : hp < 5 ? [x, 0, c]
    : [c, 0, x]
  const m = l - c / 2
  const toHex = (v: number) => Math.round(clamp01(v + m) * 255).toString(16).padStart(2, "0")
  return `#${toHex(r1)}${toHex(g1)}${toHex(b1)}`
}

export function derivePalette(baseHex: string): { bg: [string, string]; fg: string } {
  const base = hexToHsl(baseHex)
  const bg2: HSL = {
    h: base.h,
    s: clamp01(base.s - (base.s >= 0.15 ? 0.15 : 0)),
    l: clamp01(base.l + (base.l < 0.5 ? 0.12 : -0.12)),
  }
  const avgL = (base.l + bg2.l) / 2
  const fg: HSL = { h: base.h, s: clamp01(base.s * 0.3), l: avgL < 0.5 ? 0.95 : 0.12 }
  return { bg: [baseHex, hslToHex(bg2)], fg: hslToHex(fg) }
}

export function buildColorTerms(names: NamedColor[]): ColorTerm[] {
  const seen = new Set<string>()
  const out: ColorTerm[] = []
  for (const { name, hex } of names) {
    const text = name.toLowerCase().trim()
    if (!text || seen.has(text)) continue
    seen.add(text)
    out.push({ text, ...derivePalette(hex) })
  }
  return out
}
