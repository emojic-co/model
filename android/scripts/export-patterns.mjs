import { writeFileSync, mkdirSync } from 'node:fs'
import { fileURLToPath } from 'node:url'
import path from 'node:path'

const scriptDir = path.dirname(fileURLToPath(import.meta.url))
const repoRoot = path.resolve(scriptDir, '..', '..')
const heroPatternsPath = path.join(repoRoot, 'web', 'node_modules', 'hero-patterns', 'dist', 'hero-patterns.cjs.js')

const {
  anchorsAway, brickWall, bubbles, circuitBoard, diagonalStripes, endlessClouds,
  fallingTriangles, floatingCogs, fourPointStars, glamorous, hideout,
  overlappingCircles, skulls, squaresInSquares, stripes, ticTacToe, topography,
  volcanoLamp, wiggle, zigZag,
} = await import(`file://${heroPatternsPath}`)

const CLUSTER_PATTERNS = {
  anger: volcanoLamp,
  joy: stripes,
  play: ticTacToe,
  calm: topography,
  sad: fallingTriangles,
  anxiety: zigZag,
  tender: bubbles,
  drive: anchorsAway,
  reflective: hideout,
}

const outDir = path.join(repoRoot, 'android', 'app', 'src', 'main', 'assets', 'patterns')
mkdirSync(outDir, { recursive: true })

for (const [cluster, pattern] of Object.entries(CLUSTER_PATTERNS)) {
  const dataUrl = pattern('#ffffff', 1)
  const encoded = dataUrl.match(/^url\((['"]?)data:image\/svg\+xml,(.*)\1\)$/)[2]
  const svg = decodeURIComponent(encoded)
  writeFileSync(path.join(outDir, `${cluster}.svg`), svg)
}

console.log('exported', Object.keys(CLUSTER_PATTERNS).length, 'pattern SVGs')
