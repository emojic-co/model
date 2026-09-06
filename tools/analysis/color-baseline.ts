import {
  COLOR_BASELINE_JSON,
  EVAL_JSONL,
  TRAIN_JSONL,
} from "../../files.ts"
import { readJsonl } from "../data/io.ts"

const LIN_TO_LMS = [
  [0.4122214708, 0.5363325363, 0.0514459929],
  [0.2119034982, 0.6806995451, 0.1073969566],
  [0.0883024619, 0.2817188376, 0.6299787005],
]
const LMS_TO_LAB = [
  [0.2104542553, 0.793617785, -0.0040720468],
  [1.9779984951, -2.428592205, 0.4505937099],
  [0.0259040371, 0.7827717662, -0.808675766],
]

function srgbToLinear(c: number): number {
  return c <= 0.04045 ? c / 12.92 : ((c + 0.055) / 1.055) ** 2.4
}

function matVec(m: number[][], v: number[]): number[] {
  return m.map((row) => row[0] * v[0] + row[1] * v[1] + row[2] * v[2])
}

export function hexToRgb(hex: string): [number, number, number] {
  const h = hex.replace(/^#/, "")
  return [
    parseInt(h.slice(0, 2), 16),
    parseInt(h.slice(2, 4), 16),
    parseInt(h.slice(4, 6), 16),
  ]
}

export function srgbByteToOklab(
  rgb: [number, number, number],
): [number, number, number] {
  const lin = rgb.map((b) => srgbToLinear(b / 255))
  const lms = matVec(LIN_TO_LMS, lin).map((x) => Math.cbrt(x))
  const [l, a, b] = matVec(LMS_TO_LAB, lms)
  return [l, a, b]
}

export function paletteToOklab(bg: string[], fg: string): number[] {
  return [bg[0], bg[1], fg].flatMap((hex) => srgbByteToOklab(hexToRgb(hex)))
}

export function meanVec(vs: number[][]): number[] {
  const out = new Array(vs[0].length).fill(0)
  for (const v of vs) for (let i = 0; i < v.length; i++) out[i] += v[i]
  return out.map((x) => x / vs.length)
}

export function deltaE(a: number[], b: number[]): number {
  let sum = 0
  for (let i = 0; i < a.length; i += 3) {
    const dl = a[i] - b[i]
    const da = a[i + 1] - b[i + 1]
    const db = a[i + 2] - b[i + 2]
    sum += Math.sqrt(dl * dl + da * da + db * db)
  }
  return sum / (a.length / 3)
}

export function mse(a: number[], b: number[]): number {
  let sum = 0
  for (let i = 0; i < a.length; i++) sum += (a[i] - b[i]) ** 2
  return sum / a.length
}

export type OklabRow = { styles: string[]; oklab: number[] }

export function styleMeans(rows: OklabRow[]): Map<string, number[]> {
  const groups = new Map<string, number[][]>()
  for (const r of rows) {
    for (const s of r.styles) {
      const g = groups.get(s) ?? []
      g.push(r.oklab)
      groups.set(s, g)
    }
  }
  return new Map([...groups].map(([s, vs]) => [s, meanVec(vs)]))
}

export function predictStyleMean(
  styles: string[],
  means: Map<string, number[]>,
  global: number[],
): number[] {
  const hit = styles.map((s) => means.get(s)).filter((v): v is number[] => !!v)
  return hit.length ? meanVec(hit) : global
}

export function summarize(
  preds: number[][],
  targets: number[][],
): { dE: number; mse: number } {
  const n = preds.length || 1
  let de = 0
  let ms = 0
  for (let i = 0; i < preds.length; i++) {
    de += deltaE(preds[i], targets[i])
    ms += mse(preds[i], targets[i])
  }
  return { dE: de / n, mse: ms / n }
}

type RawRow = { styles?: unknown; bg?: unknown; fg?: unknown }

function toOklabRow(r: RawRow): OklabRow | null {
  const { bg, fg } = r
  if (!Array.isArray(bg) || bg.length < 2 || typeof fg !== "string") return null
  const styles = Array.isArray(r.styles)
    ? r.styles.filter((s): s is string => typeof s === "string")
    : []
  return { styles, oklab: paletteToOklab(bg.slice(0, 2) as string[], fg) }
}

export async function runColorBaseline() {
  const train = (await readJsonl<RawRow>(TRAIN_JSONL))
    .map(toOklabRow)
    .filter((r): r is OklabRow => !!r)
  const held = (await readJsonl<RawRow>(EVAL_JSONL))
    .map(toOklabRow)
    .filter((r): r is OklabRow => !!r)

  const global = meanVec(train.map((r) => r.oklab))
  const sMeans = styleMeans(train)
  const targets = held.map((r) => r.oklab)

  return {
    generated: new Date().toISOString(),
    train_rows: train.length,
    eval_rows: held.length,
    methods: {
      global_mean: summarize(held.map(() => global), targets),
      style_mean: summarize(
        held.map((r) => predictStyleMean(r.styles, sMeans, global)),
        targets,
      ),
    },
  }
}

if (import.meta.main) {
  const b = await runColorBaseline()
  if (process.argv.includes("--json")) {
    console.log(JSON.stringify(b, null, 2))
  } else {
    console.log(
      `color baseline: ${b.train_rows} train / ${b.eval_rows} eval rows `
      + `with a palette, OKLab mean dE (lower is better)\n`,
    )
    for (const [name, s] of Object.entries(b.methods)) {
      console.log(
        `  ${name.padEnd(12)} dE ${s.dE.toFixed(4)}  mse ${s.mse.toFixed(4)}`,
      )
    }
    console.log(`\n-> ${COLOR_BASELINE_JSON}`)
  }
}
