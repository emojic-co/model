import { describe, expect, it } from 'vitest'
import { frameCount } from './cardGif'
import { CLIP } from './clip'

describe('frameCount', () => {
  it('samples ceil(L * fps / 1000) frames', () => {
    expect(frameCount(3000, CLIP.gifFps)).toBe(36)
    expect(frameCount(3000, CLIP.mp4Fps)).toBe(60)
    expect(frameCount(1, 12)).toBe(1)
  })
})
