import { describe, it, expect } from 'vitest'
import { SHIMMERS as S, resolveShimmer, shimmerBackground, shimmerKeyframes } from './shimmer'
import { CLUSTERS, FEELINGS } from './feelings'

describe('shimmer spec', () => {
  it('defines an effect for every cluster and only known styles', () => {
    for (const c of Object.keys(CLUSTERS)) expect(S.clusters[c], c).toBeDefined()
    for (const s of Object.keys(S.styles)) expect(FEELINGS[s], s).toBeDefined()
  })
  it('effects are well formed', () => {
    for (const [name, e] of Object.entries(S.clusters)) {
      expect(['sweep', 'radial', 'flash'], name).toContain(e.kind)
      expect(e.easing, name).toHaveLength(4)
      expect(e.stops[0].color, name).toMatch(/^#[0-9A-F]{8}$/)
      const at = e.keyframes.map((k) => k.at)
      expect(at[0], name).toBe(0)
      expect(at.at(-1), name).toBe(1)
      expect([...at].sort((a, b) => a - b), name).toEqual(at)
    }
  })
  it('never runs a pass longer than 3 s', () => {
    for (const name of Object.keys(S.clusters)) {
      expect(resolveShimmer(S, name, undefined).durationMs, name).toBeLessThanOrEqual(3000)
      for (const f of Object.keys(S.styles)) expect(resolveShimmer(S, name, f).durationMs, `${name}/${f}`).toBeLessThanOrEqual(3000)
    }
  })
  it('applies per-style overrides', () => {
    expect(resolveShimmer(S, 'anger', 'Furious').durationMs).toBe(480)
    expect(resolveShimmer(S, 'anger', 'Irritated').durationMs).toBe(650)
  })
})

describe('shimmerKeyframes / background', () => {
  it('plays one pass of durationMs and ends hidden', () => {
    const e = resolveShimmer(S, 'drive', 'Hopeful')
    const { frames, durationMs } = shimmerKeyframes(e)
    expect(durationMs).toBe(e.durationMs)
    expect(frames[0].offset).toBe(0)
    expect(frames.at(-1)).toMatchObject({ offset: 1, opacity: 0 })
  })
  it('builds css gradients per kind', () => {
    expect(shimmerBackground(resolveShimmer(S, 'drive', 'Hopeful'))).toContain('linear-gradient(105deg')
    expect(shimmerBackground(resolveShimmer(S, 'tender', 'Tender'))).toContain('radial-gradient(circle')
    expect(shimmerBackground(resolveShimmer(S, 'anxiety', 'Tense'))).toContain('linear-gradient(180deg')
  })
})
