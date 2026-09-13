import {
  jigsaw,
  fallingTriangles,
  graphPaper,
  happyIntersection,
  randomShapes,
  brickWall,
  plus,
  bubbles,
  floatingCogs,
  ticTacToe,
  stripes,
  circuitBoard,
  endlessClouds,
  polkaDots,
} from 'hero-patterns'
import { softmax } from './model'

export const PATTERN_FNS = {
  Joyful: bubbles,
  Excited: polkaDots,
  Hopeful: happyIntersection,
  Serene: endlessClouds,
  Tender: bubbles,
  Playful: ticTacToe,
  Whimsical: floatingCogs,
  Awed: circuitBoard,
  Earnest: plus,
  Determined: brickWall,
  Proud: stripes,
  Wistful: endlessClouds,
  Melancholy: endlessClouds,
  Anxious: circuitBoard,
  Tense: fallingTriangles,
  Furious: stripes,
  Irritated: brickWall,
  Disgusted: randomShapes,
  Startled: fallingTriangles,
  Sarcastic: jigsaw,
  Deadpan: graphPaper,
  Neutral: graphPaper,
}

const MAX_OPACITY = 0.25
const BLEND_THRESHOLD = 0.6

export function patternLayers(feeling, feelingLogits, styles, fill) {
  const primaryFn = PATTERN_FNS[feeling] ?? PATTERN_FNS.Neutral
  if (!feelingLogits || !styles?.length) return [primaryFn(fill, MAX_OPACITY)]

  const probs = softmax(feelingLogits)
  const ranked = styles
    .map((s, i) => ({ style: s, p: probs[i] }))
    .sort((a, b) => b.p - a.p)
  const top = ranked[0]
  if (feeling !== top.style) return [primaryFn(fill, MAX_OPACITY)]

  const second = ranked[1]
  const layers = second && top.p > 0 && second.p / top.p >= BLEND_THRESHOLD ? [top, second] : [top]
  const total = layers.reduce((sum, l) => sum + l.p, 0) || 1
  return layers.map((l) => (PATTERN_FNS[l.style] ?? primaryFn)(fill, (l.p / total) * MAX_OPACITY))
}
