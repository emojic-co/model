import { describe, expect, it } from 'vitest'
import { lineUnits, shimmerTimeline, textTimeline } from './cardAnim'

describe('lineUnits', () => {
  it('splits words into graphemes with offsets and skips spaces', () => {
    expect(lineUnits('ab c')).toEqual([[0, 1], [1, 2], [3, 4]])
  })
})

describe('textTimeline', () => {
  it('starts at the first keyframe and ends at the final pose', () => {
    const { pose, totalMs } = textTimeline('slam', 'Angry', 3)
    expect(pose(0, 0).opacity).toBe(0)
    expect(pose(2, totalMs)).toMatchObject({ opacity: 1, scale: 1, x: 0, y: 0 })
  })
})

describe('shimmerTimeline', () => {
  it('draws during the pass and nothing before start or after the pass', () => {
    const calls = []
    const ctx = new Proxy({}, { get: (_, k) => (k === 'createLinearGradient' ? () => ({ addColorStop() {} }) : () => calls.push(k)) , set: () => true })
    const sh = shimmerTimeline('joy', 'Joyful')
    sh.draw(ctx, 512, -1)
    expect(calls).toEqual([])
    sh.draw(ctx, 512, 500)
    expect(calls).toContain('fillRect')
    calls.length = 0
    sh.draw(ctx, 512, 2000)
    expect(calls).toEqual([])
  })
})
