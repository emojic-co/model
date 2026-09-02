import { existsSync } from "node:fs"

import cliProgress from "cli-progress"

import { annotate, annotateBatchCount } from "./annotate.ts"
import { appendJsonl, readJsonl, writeFileAtomic } from "./io.ts"

const DATA = "./data.jsonl"
const TRAIN = "./train.jsonl"

type PoolRow = { text: string; bg?: string[]; fg?: string }

function argInt(flag: string): number | undefined {
  const i = process.argv.indexOf(flag)
  if (i < 0 || i + 1 >= process.argv.length) return undefined
  const n = Number(process.argv[i + 1])
  return Number.isFinite(n) ? n : undefined
}

function seededSampleIdx(n: number, k: number, seed: number): number[] {
  let s = seed >>> 0
  const rand = () => {
    s = (s * 1664525 + 1013904223) >>> 0
    return s / 2 ** 32
  }
  const idx = Array.from({ length: n }, (_, i) => i)
  const take = Math.min(k, n)
  for (let i = 0; i < take; i++) {
    const j = i + Math.floor(rand() * (n - i))
    ;[idx[i], idx[j]] = [idx[j], idx[i]]
  }
  return idx.slice(0, take).sort((a, b) => a - b)
}

if (import.meta.main) {
  if (!existsSync(DATA)) {
    throw new Error(`${DATA} not found - run snapshot.ts first`)
  }

  const fresh = process.argv.includes("--fresh")
  const keep = process.argv.includes("--keep")
  const offset = argInt("--offset") ?? 0
  const limit = argInt("--limit")
  const sampleN = argInt("--sample")

  const pool = await readJsonl<PoolRow>(DATA)
  const usable = (i: number) =>
    typeof pool[i]?.text === "string" && pool[i].text.trim().length > 0

  let picked: number[]
  if (sampleN !== undefined) {
    picked = seededSampleIdx(pool.length, sampleN, argInt("--seed") ?? 42)
    console.log(`re-annotating a random sample of ${picked.length} from ${DATA}`)
  } else {
    const end = limit === undefined ? pool.length : offset + limit
    picked = Array.from({ length: pool.length }, (_, i) => i).slice(offset, end)
    console.log(
      `re-annotating ${picked.length} rows from ${DATA}`
      + ` (offset ${offset}${limit === undefined ? "" : `, limit ${limit}`})`,
    )
  }
  picked = picked.filter(usable)

  const bar = new cliProgress.SingleBar(
    {
      format:
        "annotating |{bar}| {percentage}% | {value}/{total} batches | ETA: {eta}s",
    },
    cliProgress.Presets.shades_classic,
  )
  bar.start(annotateBatchCount(picked.length), 0)
  const labels = await annotate(
    picked.map((i) => pool[i].text),
    { colors: false, onBatchDone: () => bar.increment() },
  )
  bar.stop()

  const rows: string[] = []
  const consumed = new Set<number>()
  let noPalette = 0
  for (let p = 0; p < picked.length; p++) {
    const label = labels.get(p)
    if (!label) continue
    const src = pool[picked[p]]
    const bg =
      Array.isArray(src.bg) && src.bg.length >= 2 ? src.bg.slice(0, 2) : undefined
    const fg = typeof src.fg === "string" ? src.fg : undefined
    if (!bg || !fg) {
      noPalette++
      continue
    }
    rows.push(
      JSON.stringify({
        text: src.text,
        emojis: label.emojis.join(" "),
        styles: label.styles,
        bg,
        fg,
      }),
    )
    consumed.add(picked[p])
  }

  if (fresh) {
    await writeFileAtomic(TRAIN, rows.length ? rows.join("\n") + "\n" : "")
  } else {
    await appendJsonl(TRAIN, rows)
  }

  let poolLeft = pool.length
  if (!keep && consumed.size) {
    const remaining = pool.filter((_, i) => !consumed.has(i))
    poolLeft = remaining.length
    await writeFileAtomic(
      DATA,
      remaining.length ? remaining.map((r) => JSON.stringify(r)).join("\n") + "\n" : "",
      true,
    )
  }

  console.log("\n--- summary ---")
  console.log(`picked             : ${picked.length}`)
  console.log(`annotated          : ${labels.size}`)
  console.log(`dropped no palette : ${noPalette}`)
  console.log(`${fresh ? "wrote " : "appended"} -> ${TRAIN}: ${rows.length}`)
  console.log(
    keep
      ? `${DATA} left untouched (--keep): ${pool.length}`
      : `drained from ${DATA}: ${consumed.size} (${pool.length} -> ${poolLeft})`,
  )
  process.exit(0)
}
