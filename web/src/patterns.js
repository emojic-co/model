import {
  anchorsAway,
  brickWall,
  bubbles,
  circuitBoard,
  diagonalStripes,
  dominos,
  endlessClouds,
  fallingTriangles,
  floatingCogs,
  fourPointStars,
  glamorous,
  graphPaper,
  hideout,
  overlappingCircles,
  skulls,
  squaresInSquares,
  ticTacToe,
  topography,
  volcanoLamp,
  wiggle,
  zigZag
} from 'hero-patterns'
import { softmax } from './model'

export const PATTERN_FNS = {
  Anxious: zigZag,
  Awed: fourPointStars,
  Deadpan: hideout,
  Determined: anchorsAway,
  Disgusted: wiggle,
  Earnest: brickWall,
  Excited: glamorous,
  Furious: volcanoLamp,
  Hopeful: endlessClouds,
  Irritated: skulls,
  Joyful: dominos,
  Melancholy: fallingTriangles,
  Neutral: graphPaper,
  Playful: ticTacToe,
  Proud: overlappingCircles,
  Sarcastic: diagonalStripes,
  Serene: topography,
  Startled: circuitBoard,
  Tender: bubbles,
  Tense: squaresInSquares,
  Whimsical: floatingCogs,
  Wistful: endlessClouds,
}

const MAX_OPACITY = 0.25
const BLEND_THRESHOLD = 0.6
const REFERENCE_PX = 600

function tileLayer(fn, fill, opacity) {
  const image = fn(fill, opacity)
  const encoded = image.match(/^url\((['"]?)data:image\/svg\+xml,(.*)\1\)$/)?.[2]
  const tag = encoded ? decodeURIComponent(encoded).match(/<svg[^>]*>/)?.[0] : undefined
  const w = tag?.match(/width="([\d.]+)"/)?.[1]
  const h = tag?.match(/height="([\d.]+)"/)?.[1]
  return {
    image,
    w: w ? Number(w) / REFERENCE_PX : 1,
    h: h ? Number(h) / REFERENCE_PX : 1,
  }
}

export function patternLayers(feeling, feelingLogits, styles, fill) {
  const primaryFn = PATTERN_FNS[feeling] ?? PATTERN_FNS.Neutral
  if (!feelingLogits || !styles?.length) return [tileLayer(primaryFn, fill, MAX_OPACITY)]

  const probs = softmax(feelingLogits)
  const ranked = styles
    .map((s, i) => ({ style: s, p: probs[i] }))
    .sort((a, b) => b.p - a.p)
  const top = ranked[0]
  if (feeling !== top.style) return [tileLayer(primaryFn, fill, MAX_OPACITY)]

  const second = ranked[1]
  const layers = second && top.p > 0 && second.p / top.p >= BLEND_THRESHOLD ? [top, second] : [top]
  const total = layers.reduce((sum, l) => sum + l.p, 0) || 1
  return layers.map((l) => tileLayer(PATTERN_FNS[l.style] ?? primaryFn, fill, (l.p / total) * MAX_OPACITY))
}

export function patternSizeCss(layers) {
  return layers.map((l) => `${l.w * 100}% ${l.h * 100}%`)
}
