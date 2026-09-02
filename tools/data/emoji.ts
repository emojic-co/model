const seg = new Intl.Segmenter("en", { granularity: "grapheme" })

export function splitEmojis(raw: string): string[] {
  const out: string[] = []
  for (const tok of (raw ?? "").trim().split(/\s+/)) {
    if (!tok) continue
    for (const { segment } of seg.segment(tok)) {
      const s = segment.trim()
      if (s && [...s].some((c) => (c.codePointAt(0) ?? 0) > 0x2000)) out.push(s)
    }
  }
  return out
}

const FACE_RANGES: [number, number][] = [
  [0x1f600, 0x1f64f],
  [0x1f910, 0x1f92f],
  [0x1f970, 0x1f97f],
  [0x1fae0, 0x1fae8],
  [0x2639, 0x263b],
]

export function isFaceEmoji(emoji: string): boolean {
  for (const ch of emoji) {
    const cp = ch.codePointAt(0) ?? 0
    for (const [lo, hi] of FACE_RANGES) if (cp >= lo && cp <= hi) return true
  }
  return false
}

import GROUPS from "./emoji-groups.json" with { type: "json" }

const GROUP_MAP = GROUPS as Record<string, string>

function stripModifiers(s: string): string {
  return [...s]
    .filter((c) => {
      const cp = c.codePointAt(0) ?? 0
      return cp !== 0xfe0f && !(cp >= 0x1f3fb && cp <= 0x1f3ff)
    })
    .join("")
}

const FALLBACK_RANGES: [number, number, string][] = [
  [0x1f600, 0x1f64f, "Smileys & Emotion"],
  [0x1f910, 0x1f92f, "Smileys & Emotion"],
  [0x1f970, 0x1f97f, "Smileys & Emotion"],
  [0x1fae0, 0x1faef, "Smileys & Emotion"],
  [0x1f400, 0x1f43f, "Animals & Nature"],
  [0x1f980, 0x1f9ae, "Animals & Nature"],
  [0x1f950, 0x1f96f, "Food & Drink"],
  [0x1f680, 0x1f6ff, "Travel & Places"],
  [0x1f1e6, 0x1f1ff, "Flags"],
  [0x2600, 0x27bf, "Symbols"],
  [0x2b00, 0x2bff, "Symbols"],
  [0x2300, 0x23ff, "Objects"],
]

export function coarseEmojiGroup(emoji: string): string {
  const hit = GROUP_MAP[emoji] ?? GROUP_MAP[stripModifiers(emoji)]
  if (hit) return hit
  const cp = emoji.codePointAt(0) ?? 0
  for (const [lo, hi, g] of FALLBACK_RANGES) if (cp >= lo && cp <= hi) return g
  return "Other"
}
