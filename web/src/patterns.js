import {
  architect,
  autumn,
  circuitBoard,
  endlessClouds,
  eyes,
  fallingTriangles,
  formalInvitation,
  fourPointStars,
  glamorous,
  graphPaper,
  happyIntersection,
  linesInMotion,
  lips,
  melt,
  polkaDots,
  randomShapes,
  steelBeams,
  ticTacToe,
  topography,
  volcanoLamp,
  zigZag,
} from 'hero-patterns'
import { softmax } from './model'

export const PATTERN_FNS = {
  Joyful: happyIntersection,
  Excited: glamorous,
  Hopeful: endlessClouds,
  Serene: topography,
  Tender: lips,
  Playful: ticTacToe,
  Whimsical: randomShapes,
  Awed: fourPointStars,
  Earnest: architect,
  Determined: steelBeams,
  Proud: formalInvitation,
  Wistful: autumn,
  Melancholy: fallingTriangles,
  Anxious: zigZag,
  Tense: linesInMotion,
  Furious: volcanoLamp,
  Irritated: polkaDots,
  Disgusted: melt,
  Startled: circuitBoard,
  Sarcastic: eyes,
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
