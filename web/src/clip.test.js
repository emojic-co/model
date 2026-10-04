import { describe, expect, it } from 'vitest'
import { existsSync, readFileSync, writeFileSync } from 'node:fs'
import { fileURLToPath } from 'node:url'
import { CLIP, clipFor, createTimeline, springScale } from './clip'

describe('clipFor', () => {
  const base = { entranceMs: 1000, startDelayMs: 250, passMs: 1500 } // 2750 ms of content
  it('stops when both the emoji and the shimmer are done', () => {
    expect(clipFor({ ...base, loopMs: 1000 })).toEqual({ durationMs: 2750, loops: 1 })
    expect(clipFor({ ...base, loopMs: 4000 }, { ...CLIP, maxClipMs: 6000 })).toEqual({ durationMs: 4000, loops: 1 })
  })
  it('cuts a long emoji at maxClipMs but never the content', () => {
    expect(clipFor({ ...base, loopMs: 4000 }, { ...CLIP, maxClipMs: 3500 })).toEqual({ durationMs: 3500, loops: 1 })
    expect(clipFor({ ...base, loopMs: 4000 }, { ...CLIP, maxClipMs: 1000 })).toEqual({ durationMs: 2750, loops: 1 })
  })
  it('keeps an exported GIF within 6 s even for the longest entrance, shimmer and emoji loop', () => {
    for (const loopMs of [0, 700, 2500, 8067]) {
      const { durationMs } = clipFor({ entranceMs: 2000, startDelayMs: 250, passMs: 3000, loopMs })
      expect(durationMs, String(loopMs)).toBeLessThanOrEqual(6000)
    }
  })
  it('spring-only emoji uses just text + shimmer', () => {
    expect(clipFor({ ...base, loopMs: 0 })).toEqual({ durationMs: 2750, loops: 0 })
  })
})

describe('springScale', () => {
  it('starts shrunk, ends at rest', () => {
    expect(springScale(0)).toBeCloseTo(CLIP.spring.from, 5)
    expect(springScale(CLIP.spring.durationMs)).toBe(1)
  })
})

describe('createTimeline', () => {
  it('builds a loop whose shimmer starts after entrance + startDelay', () => {
    const tl = createTimeline({ motif: 'slam', feeling: 'Angry', cluster: 'anger', unitCount: 5, loopMs: 1500 })
    expect(tl.durationMs).toBeGreaterThanOrEqual(1500)
    expect(tl.shimmerStartMs).toBeCloseTo(tl.text.totalMs + tl.shimmer.startDelayMs)
    expect(tl.shimmer.pose(tl.shimmerStartMs - 1)).toBeNull()
    expect(tl.shimmer.pose(tl.shimmerStartMs + 1)).not.toBeNull()
    expect(tl.shimmerStartMs + tl.shimmer.passMs).toBeLessThanOrEqual(tl.durationMs + 1e-6)
  })
})

const CASES = [
  { motif: 'slam', feeling: 'Angry', cluster: 'anger', unitCount: 5, loopMs: 1500 },
  { motif: 'settle', feeling: 'Serene', cluster: 'reflective', unitCount: 12, loopMs: 2400 },
  { motif: 'jitter', feeling: 'Sarcastic', cluster: 'drive', unitCount: 9, loopMs: 0 },
]
const TIMES = [0, 250, 700, 900, 1100, 1300, 1500, 1900, 2600, 3300]
const fixturePath = fileURLToPath(new URL('./clipFixture.json', import.meta.url))

function build() {
  return CASES.map((c) => {
    const tl = createTimeline(c)
    return {
      ...c,
      durationMs: tl.durationMs,
      loops: tl.loops,
      samples: TIMES.filter((t) => t <= tl.durationMs).map((t) => ({
        t,
        unit0: tl.text.pose(0, t),
        unitLast: tl.text.pose(c.unitCount - 1, t),
        shimmer: tl.shimmer.pose(t),
        spring: c.loopMs ? null : springScale(t),
      })),
    }
  })
}

describe('golden fixture (shared with Android ClipTest)', () => {
  it('matches clipFixture.json (UPDATE_CLIP_FIXTURE=1 regenerates)', () => {
    const fresh = build()
    if (process.env.UPDATE_CLIP_FIXTURE || !existsSync(fixturePath)) writeFileSync(fixturePath, JSON.stringify(fresh, null, 2))
    expect(JSON.parse(JSON.stringify(fresh))).toEqual(JSON.parse(readFileSync(fixturePath, 'utf8')))
  })
})
