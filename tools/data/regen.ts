import { cac } from "cac"

import { splitEmojis } from "./emoji.ts"
import { normalize } from "./normalize.ts"
import { STYLE_SET } from "./styles.ts"

const MIN_COUNT = 50
const MAX_COUNT = 1000
const EVAL_SIZE = 2000

const MIN_MAX_RATIO = 10
const MAX_MAX_RATIO = 30
const MATRIX_MIN = [50, 75, 100, 125, 150, 200, 250]
const MATRIX_MAX = [500, 600, 750, 1000, 1250, 1500, 2000, 3000, 4000, 5000]

export type Palette = { bg: string[]; fg: string }
export type Row = {
  text: string
  emojis: string
  styles: string[]
  bg?: string[]
  fg?: string
  extra?: Record<string, unknown>
}

const BASE_FIELDS = new Set(["text", "emojis", "styles", "bg", "fg"])

type Acc = {
  text: string
  emojis: Set<string>
  styles: Set<string>
  palette: Palette | undefined
  extra: Record<string, unknown>
}

function rowPalette(row: Record<string, unknown>): Palette | undefined {
  const { bg, fg } = row
  if (Array.isArray(bg) && bg.length >= 2 && typeof fg === "string") {
    return { bg: (bg as string[]).slice(0, 2), fg }
  }
  return undefined
}

export function collapse(rows: unknown[]): Row[] {
  const acc = new Map<string, Acc>()
  for (const raw of rows) {
    const row = (raw ?? {}) as Record<string, unknown>
    const text = typeof row.text === "string" ? row.text : ""
    const key = normalize(text)
    if (!key) continue
    let a = acc.get(key)
    if (!a) {
      a = {
        text,
        emojis: new Set(),
        styles: new Set(),
        palette: undefined,
        extra: {},
      }
      acc.set(key, a)
    }
    if (typeof row.emojis === "string") {
      for (const e of splitEmojis(row.emojis)) a.emojis.add(e)
    }
    if (Array.isArray(row.styles)) {
      for (const s of row.styles) {
        if (typeof s === "string" && STYLE_SET.has(s)) a.styles.add(s)
      }
    }
    const p = rowPalette(row)
    if (p) a.palette = p
    for (const [k, v] of Object.entries(row)) {
      if (BASE_FIELDS.has(k) || v === undefined || k in a.extra) continue
      a.extra[k] = v
    }
  }

  const out: Row[] = []
  for (const [, a] of acc) {
    const rec: Row = {
      text: a.text,
      emojis: [...a.emojis].join(" "),
      styles: [...a.styles],
    }
    if (a.palette) {
      rec.bg = a.palette.bg
      rec.fg = a.palette.fg
    }
    if (Object.keys(a.extra).length) rec.extra = a.extra
    out.push(rec)
  }
  return out
}

function mulberry32(seed: number): () => number {
  let a = seed >>> 0
  return () => {
    a = (a + 0x6d2b79f5) | 0
    let t = Math.imul(a ^ (a >>> 15), 1 | a)
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296
  }
}

export function shuffle<T>(rows: T[], seed = SEED): T[] {
  const rand = mulberry32(seed)
  const out = [...rows]
  for (let i = out.length - 1; i > 0; i--) {
    const j = Math.floor(rand() * (i + 1))
      ;[out[i], out[j]] = [out[j], out[i]]
  }
  return out
}

export function greedyCap(
  records: Row[],
  maxCount: number,
): { kept: Row[]; counts: Map<string, number>; dropped: number } {
  const counts = new Map<string, number>()
  const kept: Row[] = []
  for (const r of records) {
    const es = [...new Set(splitEmojis(r.emojis))]
    if (es.some((e) => (counts.get(e) ?? 0) >= maxCount)) continue
    for (const e of es) counts.set(e, (counts.get(e) ?? 0) + 1)
    kept.push(r)
  }
  return { kept, counts, dropped: records.length - kept.length }
}

export function emojiVocab(counts: Map<string, number>, minCount: number): string[] {
  return [...counts.entries()]
    .filter(([, c]) => c >= minCount)
    .sort((a, b) => b[1] - a[1])
    .map(([e]) => e)
}

import { existsSync } from "node:fs"

import {
  CLDR_BASELINE_JSON as BASELINE,
  CLDR_JSONL as CLDR,
  DATA_JSONL as DATA,
  EVAL_JSONL as EVAL,
  LABELS_JSON as LABELS,
  REGEN_MD,
  TRAIN_JSONL as TRAIN,
} from "../../files.ts"
import { runBaseline } from "../analysis/cldr-baseline.ts"
import { SEED, STYLES } from "./config"
import { readJsonl, writeFileAtomic } from "./io.ts"

export function toLine(r: Row): string {
  const base =
    r.bg && r.fg
      ? { text: r.text, emojis: r.emojis, styles: r.styles, bg: r.bg, fg: r.fg }
      : { text: r.text, emojis: r.emojis, styles: r.styles }
  return JSON.stringify(r.extra ? { ...base, ...r.extra } : base)
}

function printMatrix(records: Row[], useCldr: boolean): void {
  const shuffled = shuffle(records)
  const inWindow = (min: number, max: number) => {
    const r = max / min
    return r >= MIN_MAX_RATIO && r <= MAX_MAX_RATIO
  }

  const caps = MATRIX_MAX.map((max) => {
    if (!MATRIX_MIN.some((min) => inWindow(min, max))) {
      return { max, samples: 0, counts: null as Map<string, number> | null }
    }
    const { kept, counts } = greedyCap(shuffled, max)
    return { max, samples: kept.length, counts }
  })

  const k = (n: number) => `${Math.round(n / 1000)}k`
  const grid = MATRIX_MIN.map((min) => ({
    min,
    cells: caps.map((c) =>
      c.counts && inWindow(min, c.max)
        ? `${k(c.samples)}/${emojiVocab(c.counts, min).length}`
        : "·",
    ),
  }))

  const colW =
    Math.max(
      ...MATRIX_MAX.map((m) => String(m).length),
      ...grid.flatMap((r) => r.cells.map((s) => s.length)),
    ) + 2
  const headW =
    Math.max("min\\max".length, ...MATRIX_MIN.map((m) => String(m).length)) + 2

  console.log(
    `\nmatrix: kept-rows / emoji-vocab  (cldr: ${useCldr ? "included" : "excluded"}, distinct texts: ${records.length}, max/min ratio ${MIN_MAX_RATIO}-${MAX_MAX_RATIO})\n`,
  )
  console.log(
    "min\\max".padStart(headW) + MATRIX_MAX.map((m) => String(m).padStart(colW)).join(""),
  )
  for (const row of grid) {
    console.log(
      String(row.min).padStart(headW) + row.cells.map((s) => s.padStart(colW)).join(""),
    )
  }
}

function byCodePoint(a: string, b: string): number {
  return (a.codePointAt(0) ?? 0) - (b.codePointAt(0) ?? 0) || (a < b ? -1 : a > b ? 1 : 0)
}

function section(title: string, emojis: string[]): string {
  return `## ${title}\n\n${emojis.length ? emojis.join(" ") : "_(none)_"}\n`
}

async function writeAnalysis(minCount: number, maxCount: number): Promise<void> {
  const master = await readJsonl<unknown>(DATA)
  const { counts } = greedyCap(shuffle(collapse(master)), maxCount)
  const trainVocab = emojiVocab(counts, minCount)
  const trainSet = new Set(trainVocab)

  const cldrSet = new Set<string>()
  for (const r of await readJsonl<{ emojis?: unknown }>(CLDR)) {
    if (typeof r.emojis === "string") for (const e of splitEmojis(r.emojis)) cldrSet.add(e)
  }

  const both = trainVocab.filter((e) => cldrSet.has(e))
  const trainOnly = trainVocab.filter((e) => !cldrSet.has(e))
  const cldrOnly = [...cldrSet].filter((e) => !trainSet.has(e)).sort(byCodePoint)

  const md =
    [
      "# regen analysis",
      "",
      `_generated ${new Date().toISOString().slice(0, 10)} · `
      + `min-count ${minCount} · max-count ${maxCount} · `
      + `train vocab simulated from ${DATA} (no cldr merge), CLDR set from ${CLDR}_`,
      "",
      "| section | count |",
      "| --- | --- |",
      `| in train set & in CLDR | ${both.length} |`,
      `| in train set, not in CLDR | ${trainOnly.length} |`,
      `| in CLDR, not in train set | ${cldrOnly.length} |`,
      "",
      section("Emojis in train set and in CLDR", both),
      section("Emojis in train set but not in CLDR", trainOnly),
      section("Emojis in CLDR but not in train set", cldrOnly),
    ].join("\n") + "\n"

  await writeFileAtomic(REGEN_MD, md)
  console.log(
    `-> ${REGEN_MD} : train∩cldr ${both.length}, `
    + `train∖cldr ${trainOnly.length}, cldr∖train ${cldrOnly.length}`,
  )
}

const cli = cac("regen")
cli.usage("[options]")
cli
  .option("--min-count <n>", `min kept-records for an emoji to enter labels.json (default ${MIN_COUNT})`)
  .option("--max-count <n>", `cap on kept-records per emoji (default ${MAX_COUNT})`)
  .option("--n <n>", "eval.jsonl row count (default 1500)")
  .option("--no-cldr", "ignore data/cldr.jsonl; build from data/data.jsonl only")
  .option(
    "--matrix",
    "dry sweep: print kept-rows / emoji-vocab for a grid of min/max-count, write nothing",
  )
  .option(
    "--analysis",
    `write ${REGEN_MD} (train-vocab vs CLDR emoji coverage) and nothing else`,
  )
cli.help()

if (import.meta.main) {
  const { options } = cli.parse(process.argv, { run: false })
  if (options.help) process.exit(0)
  const minCount = Number(options.minCount ?? MIN_COUNT)
  const maxCount = Number(options.maxCount ?? MAX_COUNT)
  const n = Number(options.n ?? EVAL_SIZE)

  if (options.analysis) {
    if (!existsSync(CLDR)) {
      console.error(`${CLDR} is missing; run \`bun run build-cldr\` first`)
      process.exit(1)
    }
    await writeAnalysis(minCount, maxCount)
    process.exit(0)
  }

  const useCldr = options.cldr !== false
  if (useCldr && !existsSync(CLDR)) {
    console.error(
      `${CLDR} is missing; run \`bun run build-cldr\` first or pass --no-cldr`,
    )
    process.exit(1)
  }

  const master = await readJsonl<unknown>(DATA)
  const cldr = useCldr ? await readJsonl<unknown>(CLDR) : []
  const raw = [...master, ...cldr]
  const records = collapse(raw)
  const merged = records.length
  const dupKeys = raw.length - merged

  if (options.matrix) {
    printMatrix(records, useCldr)
    process.exit(0)
  }

  const { kept, counts, dropped } = greedyCap(shuffle(records), maxCount)
  const emojis = emojiVocab(counts, minCount)
  const ranked = [...counts.entries()].sort((a, b) => b[1] - a[1])
  const belowMin = ranked.length - emojis.length

  const split = shuffle(kept, SEED + 1)
  const held = split.slice(0, n)
  const rest = split.slice(n)

  await writeFileAtomic(EVAL, held.map(toLine).join("\n") + "\n")
  await writeFileAtomic(TRAIN, rest.map(toLine).join("\n") + "\n")

  const labels = {
    styles: [...STYLES],
    emojis,
  }
  await writeFileAtomic(LABELS, JSON.stringify(labels, null, 2) + "\n")

  let baselineLine = `-> ${BASELINE} : failed (skipped)`
  try {
    const baseline = await runBaseline()
    await writeFileAtomic(BASELINE, JSON.stringify(baseline, null, 2) + "\n")
    const best = Object.entries(baseline.methods).sort(
      (a, b) =>
        b[1].acc_at_k.at(-1)! - a[1].acc_at_k.at(-1)! || b[1].mrr - a[1].mrr,
    )[0]
    baselineLine =
      `-> ${BASELINE} : best "${best[0]}" `
      + `acc@10 ${(100 * best[1].acc_at_k.at(-1)!).toFixed(1)}`
  } catch (err) {
    console.error(`cldr baseline failed: ${(err as Error).message}`)
  }

  const keptRanked = ranked.filter(([, c]) => c >= minCount)
  const fmt = (es: [string, number][]) => es.map(([e, c]) => `${e} ${c}`).join(", ")

  console.log("\n--- regen ---")
  console.log(`master lines read     : ${master.length}`)
  console.log(
    `cldr lines read       : ${useCldr ? cldr.length : "skipped (--no-cldr)"}`,
  )
  console.log(`distinct texts        : ${merged} (collapsed away ${dupKeys})`)
  console.log(`min-count / max-count : ${minCount} / ${maxCount}`)
  console.log(`greedy kept           : ${kept.length} rows (dropped ${dropped} over max-count)`)
  console.log(`emoji vocab (>= ${minCount})  : ${emojis.length} (below min-count: ${belowMin})`)
  console.log(`  most frequent       : ${fmt(keptRanked.slice(0, 5))}`)
  console.log(`  least frequent kept : ${fmt(keptRanked.slice(-5))}`)
  console.log(`-> ${EVAL}       : ${held.length}`)
  console.log(`-> ${TRAIN}      : ${rest.length}`)
  console.log(
    `-> ${LABELS}    : ${labels.styles.length} styles, ${labels.emojis.length} emojis`,
  )
  console.log(baselineLine)
  process.exit(0)
}
