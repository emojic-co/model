import { isFaceEmoji } from "./emoji.ts"
import { normalize } from "./normalize.ts"

export const MAX_TEXT_LEN = 42

export function dedupe<T extends { text: string }>(
  rows: T[],
): { kept: T[]; duplicate: number; degenerate: number } {
  const seen = new Set<string>()
  const kept: T[] = []
  let duplicate = 0
  let degenerate = 0
  for (const r of rows) {
    const n = normalize(r.text)
    if (!n || n.length > MAX_TEXT_LEN) {
      degenerate++
      continue
    }
    if (seen.has(n)) {
      duplicate++
      continue
    }
    seen.add(n)
    kept.push(r)
  }
  return { kept, duplicate, degenerate }
}

export function stableHash(s: string): number {
  let h = 2166136261 >>> 0
  for (let i = 0; i < s.length; i++) {
    h ^= s.charCodeAt(i)
    h = Math.imul(h, 16777619) >>> 0
  }
  return h >>> 0
}

export function freqBucket(count: number): number {
  return Math.floor(Math.log2(count + 1))
}

export function sortPool<T extends { text: string; _emoji?: string }>(
  rows: T[],
): T[] {
  const freq = new Map<string, number>()
  for (const r of rows) {
    const e = r._emoji ?? ""
    freq.set(e, (freq.get(e) ?? 0) + 1)
  }
  return rows
    .map((r) => {
      const e = r._emoji ?? ""
      return {
        r,
        face: e && isFaceEmoji(e) ? 1 : 0,
        bucket: freqBucket(freq.get(e) ?? 0),
        hash: stableHash(normalize(r.text)),
      }
    })
    .sort((a, b) => a.face - b.face || a.bucket - b.bucket || a.hash - b.hash)
    .map((x) => x.r)
}
