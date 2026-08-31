import { readFileSync } from 'node:fs'
import { fileURLToPath } from 'node:url'
import { describe, it, expect } from 'vitest'
import {
  CLUSTERS,
  FEELINGS,
  ENTRANCE_MOTIFS,
  EMOJI_MOTIFS,
  resolveFeeling,
  topFeelings,
} from './feelings'

const readJson = (rel) =>
  JSON.parse(readFileSync(fileURLToPath(new URL(rel, import.meta.url)), 'utf8'))
const meta = readJson('../public/meta.json')
const css = readFileSync(fileURLToPath(new URL('./styles.css', import.meta.url)), 'utf8')

describe('feelings coverage', () => {
  it('every model feeling has a FEELINGS entry', () => {
    for (const f of meta.feelings) expect(FEELINGS[f], f).toBeTruthy()
  })

  it('FEELINGS keys are exactly the model feelings', () => {
    expect([...Object.keys(FEELINGS)].sort()).toEqual([...meta.feelings].sort())
  })

  it('every FEELINGS entry points at a real cluster', () => {
    for (const [name, def] of Object.entries(FEELINGS)) {
      expect(CLUSTERS[def.cluster], name).toBeTruthy()
    }
  })

  it('every resolved motif name is known', () => {
    for (const f of meta.feelings) {
      const r = resolveFeeling(f)
      expect(ENTRANCE_MOTIFS, `${f} entrance`).toContain(r.entrance)
      expect(EMOJI_MOTIFS, `${f} emoji`).toContain(r.emoji)
    }
  })

  it('resolveFeeling returns css var strings', () => {
    const r = resolveFeeling('Happy')
    expect(r.vars['--entrance-dur']).toMatch(/^\d+ms$/)
    expect(r.vars['--emoji-dur']).toMatch(/^\d+ms$/)
    expect(r.vars['--drift-sec']).toMatch(/^\d+s$/)
  })

  it('unknown feeling resolves to the Neutral style', () => {
    expect(resolveFeeling('Nope')).toEqual(resolveFeeling('Neutral'))
  })

  it('every feeling resolves to a non-empty font stack', () => {
    for (const f of meta.feelings) {
      expect(resolveFeeling(f).font, f).toMatch(/^".+", .+/)
    }
  })

  it('each feeling gets its own font family', () => {
    const families = meta.feelings.map((f) => resolveFeeling(f).font.match(/^"([^"]+)"/)[1])
    expect(new Set(families).size).toBe(meta.feelings.length)
  })
})

describe('topFeelings', () => {
  const feelings = ['A', 'B', 'C', 'D', 'E', 'F', 'G']
  const scores = [0.1, 0.9, 0.3, 0.7, 0.2, 0.5, 0.05]

  it('returns the 5 highest scoring, descending', () => {
    expect(topFeelings(scores, feelings, 'B')).toEqual(['B', 'D', 'F', 'C', 'E'])
  })

  it('replaces the 5th with the selected feeling when it is outside the top 5', () => {
    expect(topFeelings(scores, feelings, 'G')).toEqual(['B', 'D', 'F', 'C', 'G'])
  })

  it('returns [] when scores are missing', () => {
    expect(topFeelings(null, feelings, 'A')).toEqual([])
  })
})

describe('css motif contract', () => {
  it('every entrance motif has a keyframe and a selector', () => {
    for (const m of ENTRANCE_MOTIFS) {
      expect(css, m).toContain(`@keyframes entrance-${m}`)
      expect(css, m).toContain(`[data-entrance="${m}"]`)
    }
  })
  it('every emoji motif has a keyframe and a selector', () => {
    for (const m of EMOJI_MOTIFS) {
      expect(css, m).toContain(`@keyframes emoji-${m}`)
      expect(css, m).toContain(`[data-emoji="${m}"]`)
    }
  })
  it('no keyframe references an unknown motif', () => {
    const names = [...css.matchAll(/@keyframes (entrance|emoji)-([a-zA-Z]+)/g)].map((x) => `${x[1]}:${x[2]}`)
    const known = new Set([
      ...ENTRANCE_MOTIFS.map((m) => `entrance:${m}`),
      ...EMOJI_MOTIFS.map((m) => `emoji:${m}`),
    ])
    for (const n of names) expect(known.has(n), n).toBe(true)
  })
})
