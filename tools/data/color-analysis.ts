import { readJsonl } from "./io.ts"
import { shuffleSeeded } from "./regen.ts"

const DATA = "./data.jsonl"

const SIM_SPAN = 0.5
const DEFAULT_SAMPLE = 5000
const DEFAULT_SEED = 42
const DEFAULT_EXAMPLES = 5

const ANCHOR_SETS: { name: string; anchors: Record<string, string> }[] = [
  {
    name: "Muted anchors",
    anchors: {
      red: "#c0392b",
      orange: "#e07a3f",
      yellow: "#e8d44d",
      green: "#4c9a52",
      blue: "#4a6fd1",
      purple: "#7a5aa8",
      pink: "#d98cc0",
      brown: "#6b4a35",
      black: "#1c1c1c",
      white: "#f2f2f2",
      gray: "#8a8a8a",
    },
  },
  {
    name: "Vivid anchors",
    anchors: {
      red: "#e00000",
      orange: "#f57c00",
      yellow: "#ffd400",
      green: "#1faa33",
      blue: "#1a53d0",
      purple: "#8e24aa",
      pink: "#ec4899",
      brown: "#7b4a24",
      black: "#000000",
      white: "#ffffff",
      gray: "#808080",
    },
  },
]

export function hexToRgb(hex: string): [number, number, number] | null {
  const m = /^#?([0-9a-f]{6})$/i.exec((hex ?? "").trim())
  if (!m) return null
  const n = parseInt(m[1], 16)
  return [(n >> 16) & 255, (n >> 8) & 255, n & 255]
}

function srgbToLinear(c: number): number {
  return c <= 0.04045 ? c / 12.92 : ((c + 0.055) / 1.055) ** 2.4
}

export function oklab(rgb: number[]): [number, number, number] {
  const [r, g, b] = rgb.map((v) => srgbToLinear(v / 255))
  const lm = 0.4122214708 * r + 0.5363325363 * g + 0.0514459929 * b
  const mm = 0.2119034982 * r + 0.6806995451 * g + 0.1073969566 * b
  const sm = 0.0883024619 * r + 0.2817188376 * g + 0.6299787005 * b
  const l = Math.cbrt(lm)
  const m = Math.cbrt(mm)
  const s = Math.cbrt(sm)
  return [
    0.2104542553 * l + 0.793617785 * m - 0.0040720468 * s,
    1.9779984951 * l - 2.428592205 * m + 0.4505937099 * s,
    0.0259040371 * l + 0.7827717662 * m - 0.808675766 * s,
  ]
}

export function deltaE(a: number[], b: number[]): number {
  return Math.hypot(a[0] - b[0], a[1] - b[1], a[2] - b[2])
}

export function colorSimilarity(rgb1: number[], rgb2: number[]): number {
  const dist = deltaE(oklab(rgb1), oklab(rgb2))
  const sim = Math.max(0, 100 * (1 - dist / SIM_SPAN))
  return Math.round(sim * 100) / 100
}

export function mean(xs: number[]): number {
  return xs.length ? xs.reduce((a, b) => a + b, 0) / xs.length : 0
}

export function meanStd(xs: number[]): { mean: number; std: number } {
  if (!xs.length) return { mean: 0, std: 0 }
  const m = mean(xs)
  const v = mean(xs.map((x) => (x - m) ** 2))
  return { mean: m, std: Math.sqrt(v) }
}

function argInt(flag: string): number | undefined {
  const i = process.argv.indexOf(flag)
  if (i < 0 || i + 1 >= process.argv.length) return undefined
  const n = Number(process.argv[i + 1])
  return Number.isFinite(n) ? n : undefined
}

function argStr(flag: string): string | undefined {
  const i = process.argv.indexOf(flag)
  if (i < 0 || i + 1 >= process.argv.length) return undefined
  return process.argv[i + 1].trim() || undefined
}

function stamp(): { header: string; file: string } {
  const d = new Date()
  const p = (n: number) => String(n).padStart(2, "0")
  return {
    header: `${d.getFullYear()}-${p(d.getMonth() + 1)}-${p(d.getDate())} ${p(d.getHours())}:${p(d.getMinutes())}`,
    file: `${p(d.getMonth() + 1)}-${p(d.getDate())}-${p(d.getHours())}-${p(d.getMinutes())}`,
  }
}

type Bg = [[number, number, number], [number, number, number]]
type Sample = { color: string | null; bg: Bg; text: string }
type Scored = { text: string; bg: Bg; meanSim: number; maxSim: number }

function parseBg(raw: unknown): Bg | null {
  if (!Array.isArray(raw) || raw.length < 2) return null
  const a = hexToRgb(typeof raw[0] === "string" ? raw[0] : "")
  const b = hexToRgb(typeof raw[1] === "string" ? raw[1] : "")
  return a && b ? [a, b] : null
}

function scoreRows(anchor: number[], rows: Sample[]): Scored[] {
  return rows.map((r) => {
    const s0 = colorSimilarity(anchor, r.bg[0])
    const s1 = colorSimilarity(anchor, r.bg[1])
    return { text: r.text, bg: r.bg, meanSim: (s0 + s1) / 2, maxSim: Math.max(s0, s1) }
  })
}

function bar(value: number, peak: number): string {
  return peak > 0 ? "#".repeat(Math.round((30 * value) / peak)) : ""
}

function hist(values: number[]): string[] {
  if (!values.length) return ["_(no rows)_"]
  const by = new Map<number, number>()
  for (const v of values) {
    const b = Math.min(19, Math.floor(v / 5))
    by.set(b, (by.get(b) ?? 0) + 1)
  }
  const peak = Math.max(...by.values())
  const out = ["| range | rows | |", "| --- | ---: | :-- |"]
  for (let b = 0; b <= 19; b++) {
    const n = by.get(b) ?? 0
    out.push(`| ${b * 5}–${b * 5 + 4} | ${n} | ${n ? "#".repeat(Math.round((30 * n) / peak)) : ""} |`)
  }
  return out
}

if (import.meta.main) {
  const file = argStr("--file") ?? DATA
  const sampleSize = argInt("--sample") ?? DEFAULT_SAMPLE
  const seed = argInt("--seed") ?? DEFAULT_SEED
  const examples = argInt("--examples") ?? DEFAULT_EXAMPLES

  const raw = await readJsonl<Record<string, unknown>>(file)
  const withBg: Sample[] = []
  let badBg = 0
  for (const r of raw) {
    const bg = parseBg(r.bg)
    if (!bg) {
      badBg++
      continue
    }
    withBg.push({
      color: typeof r.color === "string" ? r.color : null,
      bg,
      text: typeof r.text === "string" ? r.text : "",
    })
  }
  const tagged = withBg.filter((r) => r.color)
  const byColor = new Map<string, Sample[]>()
  for (const r of tagged) {
    const list = byColor.get(r.color as string) ?? []
    list.push(r)
    byColor.set(r.color as string, list)
  }

  const known = new Set(ANCHOR_SETS.flatMap((s) => Object.keys(s.anchors)))
  const unknownColors = [...byColor.keys()].filter((c) => !known.has(c))

  const { header } = stamp()
  const doc: string[] = [
    `# color analysis — ${header}`,
    "",
    `source: \`${file}\` · ${raw.length} rows · ${withBg.length} with a usable \`bg\``
    + `${badBg ? ` · ${badBg} dropped (bad bg)` : ""}`,
    "",
    `\`color\`-tagged rows: ${tagged.length} across ${byColor.size} names`
    + `${unknownColors.length ? ` · not in any anchor set: ${unknownColors.join(", ")}` : ""}`,
    "",
    `baseline: one random sample per color from every \`bg\` row minus that color's own`
    + ` (seed ${seed}, ${sampleSize === 0 ? "whole pool" : `${sampleSize} rows/color`}),`
    + ` scored against the same anchor`,
    "",
    `\`sim(a,b) = max(0, 100·(1 − ΔE_OK(a,b) / ${SIM_SPAN}))\`; per row`
    + ` \`mean\`/\`max\` over the two \`bg\` stops`,
    "",
  ]

  for (const { name, anchors } of ANCHOR_SETS) {
    const names = Object.keys(anchors).filter((c) => byColor.has(c))
    doc.push(`## ${name}`, "")
    if (!names.length) {
      doc.push("_(no tagged rows for any color in this set)_", "")
      continue
    }
    doc.push(
      "| color | n | anchor | sim(gen) mean | sim(rand) mean | Δmean | sim(gen) max | sim(rand) max | Δmax |",
      "| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    )

    const rowStats: {
      color: string
      genMean: number
      genMax: number
      randMean: number
      randMax: number
      scored: Scored[]
    }[] = []

    names.forEach((color, i) => {
      const anchor = hexToRgb(anchors[color]) as number[]
      const genRows = byColor.get(color) ?? []
      const gen = scoreRows(anchor, genRows)

      const pool = withBg.filter((r) => r.color !== color)
      const sample =
        sampleSize === 0
          ? pool
          : shuffleSeeded(pool, seed + i).slice(0, Math.min(sampleSize, pool.length))
      const rand = scoreRows(anchor, sample)

      const genMean = mean(gen.map((s) => s.meanSim))
      const genMax = mean(gen.map((s) => s.maxSim))
      const randMean = mean(rand.map((s) => s.meanSim))
      const randMax = mean(rand.map((s) => s.maxSim))
      rowStats.push({ color, genMean, genMax, randMean, randMax, scored: gen })

      doc.push(
        `| ${color} | ${genRows.length} | \`${anchors[color]}\` `
        + `| ${genMean.toFixed(1)} | ${randMean.toFixed(1)} | ${(genMean - randMean >= 0 ? "+" : "") + (genMean - randMean).toFixed(1)} `
        + `| ${genMax.toFixed(1)} | ${randMax.toFixed(1)} | ${(genMax - randMax >= 0 ? "+" : "") + (genMax - randMax).toFixed(1)} |`,
      )
    })

    const macroGenMean = mean(rowStats.map((r) => r.genMean))
    const macroRandMean = mean(rowStats.map((r) => r.randMean))
    const macroGenMax = mean(rowStats.map((r) => r.genMax))
    const macroRandMax = mean(rowStats.map((r) => r.randMax))
    const beats = rowStats.filter((r) => r.genMean > r.randMean).length
    doc.push(
      `| **macro** | ${rowStats.reduce((a, r) => a + r.scored.length, 0)} | `
      + `| **${macroGenMean.toFixed(1)}** | **${macroRandMean.toFixed(1)}** | **${(macroGenMean - macroRandMean >= 0 ? "+" : "") + (macroGenMean - macroRandMean).toFixed(1)}** `
      + `| **${macroGenMax.toFixed(1)}** | **${macroRandMax.toFixed(1)}** | **${(macroGenMax - macroRandMax >= 0 ? "+" : "") + (macroGenMax - macroRandMax).toFixed(1)}** |`,
    )
    doc.push("")
    doc.push(
      `generated beats random on mean sim: **${beats} / ${rowStats.length}** colors`,
      "",
    )

    const peak = Math.max(...rowStats.map((r) => r.genMean), 1)
    doc.push("per-color generated mean sim:", "")
    for (const r of [...rowStats].sort((a, b) => a.genMean - b.genMean)) {
      doc.push(`- \`${r.color.padEnd(7)}\` ${r.genMean.toFixed(1).padStart(5)} ${bar(r.genMean, peak)}`)
    }
    doc.push("")

    doc.push(
      `### ${name} — worst ${examples} generated rows per color`,
      "",
    )
    for (const r of rowStats) {
      doc.push(`**${r.color}** (n=${r.scored.length})`, "")
      const worst = [...r.scored].sort((a, b) => a.meanSim - b.meanSim).slice(0, examples)
      for (const s of worst) {
        const hexes = s.bg
          .map(
            (c) =>
              "#" + c.map((v) => v.toString(16).padStart(2, "0")).join(""),
          )
          .join(" ")
        doc.push(
          `- ${s.meanSim.toFixed(1)} mean / ${s.maxSim.toFixed(1)} max · ${hexes} · ${JSON.stringify(s.text)}`,
        )
      }
      doc.push("")
    }

    const allGen = rowStats.flatMap((r) => r.scored.map((s) => s.meanSim))
    doc.push(`### ${name} — generated mean-sim distribution`, "", ...hist(allGen), "")
  }

  console.log(doc.join("\n") + "\n")
  process.exit(0)
}
