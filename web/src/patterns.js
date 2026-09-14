import {
  anchorsAway,
  brickWall,
  bubbles,
  circuitBoard,
  diagonalStripes,
  endlessClouds,
  fallingTriangles,
  floatingCogs,
  fourPointStars,
  glamorous,
  hideout,
  overlappingCircles,
  skulls,
  squaresInSquares,
  stripes,
  ticTacToe,
  topography,
  volcanoLamp,
  wiggle,
  zigZag
} from 'hero-patterns'
import { softmax } from './model'

export const PATTERN_FNS = {
  Anxious: { pattern: zigZag, scale: 2 },
  Awed: { pattern: fourPointStars, scale: 2 },
  Deadpan: { pattern: hideout, scale: 2 },
  Determined: { pattern: anchorsAway, scale: 2 },
  Disgusted: { pattern: wiggle, scale: 2 },
  Earnest: { pattern: brickWall, scale: 2 },
  Excited: { pattern: glamorous, scale: 2 },
  Furious: { pattern: volcanoLamp, scale: 2 },
  Hopeful: { pattern: endlessClouds, scale: 2 },
  Irritated: { pattern: skulls, scale: 2 },
  Joyful: { pattern: stripes, scale: 2 },
  Melancholy: { pattern: fallingTriangles, scale: 2 },
  Neutral: { pattern: hideout, scale: 2 },
  Playful: { pattern: ticTacToe, scale: 2 },
  Proud: { pattern: overlappingCircles, scale: 2 },
  Sarcastic: { pattern: diagonalStripes, scale: 2 },
  Serene: { pattern: topography, scale: 2 },
  Startled: { pattern: circuitBoard, scale: 2 },
  Tender: { pattern: bubbles, scale: 2 },
  Tense: { pattern: squaresInSquares, scale: 2 },
  Whimsical: { pattern: floatingCogs, scale: 2 },
  Wistful: { pattern: endlessClouds, scale: 2 },
}

const MAX_OPACITY = 0.25
const BLEND_THRESHOLD = 0.6
const REFERENCE_PX = 600

function tileLayer({ pattern, scale }, fill, opacity) {
  const image = pattern(fill, opacity)
  const encoded = image.match(/^url\((['"]?)data:image\/svg\+xml,(.*)\1\)$/)?.[2]
  const tag = encoded ? decodeURIComponent(encoded).match(/<svg[^>]*>/)?.[0] : undefined
  const w = tag?.match(/width="([\d.]+)"/)?.[1]
  const h = tag?.match(/height="([\d.]+)"/)?.[1]
  return {
    image,
    w: (w ? Number(w) / REFERENCE_PX : 1) * scale,
    h: (h ? Number(h) / REFERENCE_PX : 1) * scale,
  }
}

export function patternLayers(feeling, feelingLogits, styles, fill) {
  const primaryEntry = PATTERN_FNS[feeling] ?? PATTERN_FNS.Neutral
  if (!feelingLogits || !styles?.length) return [tileLayer(primaryEntry, fill, MAX_OPACITY)]

  const probs = softmax(feelingLogits)
  const ranked = styles
    .map((s, i) => ({ style: s, p: probs[i] }))
    .sort((a, b) => b.p - a.p)
  const top = ranked[0]
  if (feeling !== top.style) return [tileLayer(primaryEntry, fill, MAX_OPACITY)]

  const second = ranked[1]
  const layers = second && top.p > 0 && second.p / top.p >= BLEND_THRESHOLD ? [top, second] : [top]
  const total = layers.reduce((sum, l) => sum + l.p, 0) || 1
  return layers.map((l) => tileLayer(PATTERN_FNS[l.style] ?? primaryEntry, fill, (l.p / total) * MAX_OPACITY))
}

export function patternSizeCss(layers) {
  return layers.map((l) => `${l.w * 100}% ${l.h * 100}%`)
}
