// https://www.fffuel.co/dddoodle/
import {
  anchorsAway,
  aztec,
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
  hideout,
  jigsaw,
  overlappingCircles,
  skulls,
  squaresInSquares,
  stripes,
  volcanoLamp,
  wiggle,
  zigZag
} from 'hero-patterns'

export const PATTERN_FNS = {
  Anxious: { pattern: zigZag, scale: 2 },
  Awed: { pattern: fourPointStars, scale: 2 },
  Deadpan: { pattern: hideout, scale: 2 },
  Determined: { pattern: anchorsAway, scale: 2 },
  Disgusted: { pattern: wiggle, scale: 2 },
  Earnest: { pattern: brickWall, scale: 2 },
  Excited: { pattern: glamorous, scale: 2 },
  Furious: { pattern: volcanoLamp, scale: 2 },
  Hopeful: { pattern: aztec, scale: 2 },
  Irritated: { pattern: skulls, scale: 2 },
  Joyful: { pattern: stripes, scale: 2 },
  Melancholy: { pattern: fallingTriangles, scale: 2 },
  Neutral: { pattern: hideout, scale: 2 },
  Playful: { pattern: dominos, scale: 2 },
  Proud: { pattern: overlappingCircles, scale: 2 },
  Sarcastic: { pattern: diagonalStripes, scale: 2 },
  Serene: { pattern: jigsaw, scale: 2 },
  Startled: { pattern: circuitBoard, scale: 2 },
  Tender: { pattern: bubbles, scale: 2 },
  Tense: { pattern: squaresInSquares, scale: 2 },
  Whimsical: { pattern: floatingCogs, scale: 2 },
  Wistful: { pattern: endlessClouds, scale: 2 },
}

export const PATTERN_NAMES = new Map([
  [anchorsAway, 'anchors-away'],
  [aztec, 'aztec'],
  [brickWall, 'brick-wall'],
  [bubbles, 'bubbles'],
  [circuitBoard, 'circuit-board'],
  [diagonalStripes, 'diagonal-stripes'],
  [dominos, 'dominos'],
  [endlessClouds, 'endless-clouds'],
  [fallingTriangles, 'falling-triangles'],
  [floatingCogs, 'floating-cogs'],
  [fourPointStars, 'four-point-stars'],
  [glamorous, 'glamorous'],
  [hideout, 'hideout'],
  [jigsaw, 'jigsaw'],
  [overlappingCircles, 'overlapping-circles'],
  [skulls, 'skulls'],
  [squaresInSquares, 'squares-in-squares'],
  [stripes, 'stripes'],
  [volcanoLamp, 'volcano-lamp'],
  [wiggle, 'wiggle'],
  [zigZag, 'zig-zag'],
])

export function patternName(feeling) {
  const entry = PATTERN_FNS[feeling] ?? PATTERN_FNS.Neutral
  return PATTERN_NAMES.get(entry.pattern)
}

export const MAX_OPACITY = 0.14
export const REFERENCE_PX = 600

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

export function patternLayers(feeling, fill) {
  const entry = PATTERN_FNS[feeling] ?? PATTERN_FNS.Neutral
  return [tileLayer(entry, fill, MAX_OPACITY)]
}

export function patternSizeCss(layers) {
  return layers.map((l) => `${l.w * 100}% ${l.h * 100}%`)
}
