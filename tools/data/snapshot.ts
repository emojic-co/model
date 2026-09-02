import { existsSync } from "node:fs"
import { copyFile, mkdir, writeFile } from "node:fs/promises"

import { readJsonl } from "./io.ts"
import { dedupe, sortPool } from "./pool.ts"

const TRAIN = "./train.jsonl"
const EVAL = "./eval.jsonl"
const DATA = "./data.jsonl"
const ARCHIVE_DIR = "report/multi-label"

function stamp(): string {
  const d = new Date()
  const p = (n: number) => String(n).padStart(2, "0")
  return `${p(d.getMonth() + 1)}-${p(d.getDate())}-${p(d.getHours())}:${p(d.getMinutes())}`
}

if (import.meta.main) {
  if (!existsSync(TRAIN)) throw new Error(`${TRAIN} not found`)
  await mkdir(ARCHIVE_DIR, { recursive: true })
  const s = stamp()

  if (existsSync(DATA)) {
    const dest = `${ARCHIVE_DIR}/${s}.legacy-data.jsonl`
    await copyFile(DATA, dest)
    console.log(`archived ${DATA} -> ${dest}`)
  }
  for (const [src, tag] of [
    [TRAIN, "train"],
    [EVAL, "eval"],
  ] as const) {
    if (existsSync(src)) {
      const dest = `${ARCHIVE_DIR}/${s}.${tag}-before.jsonl`
      await copyFile(src, dest)
      console.log(`snapshot ${src} -> ${dest}`)
    }
  }

  const raw = await readJsonl<Record<string, unknown>>(TRAIN)
  const rows = raw
    .filter((r) => typeof r.text === "string" && (r.text as string).trim())
    .map((r) => ({
      text: r.text as string,
      bg: Array.isArray(r.bg) ? (r.bg as string[]) : undefined,
      fg: typeof r.fg === "string" ? (r.fg as string) : undefined,
      _emoji: typeof r.emoji === "string" ? (r.emoji as string) : undefined,
    }))

  const { kept, duplicate, degenerate } = dedupe(rows)
  const sorted = sortPool(kept)

  const out = sorted
    .map((r) => JSON.stringify({ text: r.text, bg: r.bg, fg: r.fg }))
    .join("\n")
  await writeFile(DATA, out + "\n")

  console.log("\n--- snapshot ---")
  console.log(`train rows read   : ${raw.length}`)
  console.log(`duplicate (norm)  : ${duplicate}`)
  console.log(`degenerate        : ${degenerate}`)
  console.log(`pool -> ${DATA}   : ${sorted.length}`)
  process.exit(0)
}
