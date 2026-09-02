import { existsSync } from "node:fs"
import { copyFile, mkdir } from "node:fs/promises"

import { readJsonl, writeFileAtomic } from "./io.ts"

const TRAIN = "./train.jsonl"
const EVAL = "./eval.jsonl"
const ARCHIVE_DIR = "report/multi-label"

function argInt(flag: string, fallback: number): number {
  const i = process.argv.indexOf(flag)
  if (i < 0 || i + 1 >= process.argv.length) return fallback
  const n = Number(process.argv[i + 1])
  return Number.isFinite(n) ? n : fallback
}

function shuffle<T>(rows: T[], seed: number): T[] {
  let s = seed >>> 0
  const rand = () => {
    s = (s * 1664525 + 1013904223) >>> 0
    return s / 2 ** 32
  }
  const out = [...rows]
  for (let i = out.length - 1; i > 0; i--) {
    const j = Math.floor(rand() * (i + 1))
    ;[out[i], out[j]] = [out[j], out[i]]
  }
  return out
}

function stamp(): string {
  const d = new Date()
  const p = (n: number) => String(n).padStart(2, "0")
  return `${p(d.getMonth() + 1)}-${p(d.getDate())}-${p(d.getHours())}:${p(d.getMinutes())}`
}

if (import.meta.main) {
  if (!existsSync(TRAIN)) throw new Error(`${TRAIN} not found`)
  const n = argInt("--n", 1500)
  const seed = argInt("--seed", 42)

  await mkdir(ARCHIVE_DIR, { recursive: true })
  if (existsSync(EVAL)) {
    const dest = `${ARCHIVE_DIR}/${stamp()}.eval-pre-split.jsonl`
    await copyFile(EVAL, dest)
    console.log(`archived ${EVAL} -> ${dest}`)
  }

  const all = await readJsonl<Record<string, unknown>>(TRAIN)
  const shuffled = shuffle(all, seed)
  const held = shuffled.slice(0, n)
  const rest = shuffled.slice(n)

  await writeFileAtomic(EVAL, held.map((r) => JSON.stringify(r)).join("\n") + "\n")
  await writeFileAtomic(TRAIN, rest.map((r) => JSON.stringify(r)).join("\n") + "\n")

  console.log("\n--- split ---")
  console.log(`train rows read : ${all.length}`)
  console.log(`-> ${EVAL}      : ${held.length}`)
  console.log(`-> ${TRAIN}     : ${rest.length}`)
  process.exit(0)
}
