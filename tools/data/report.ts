import { readdir, stat } from "node:fs/promises"

import { REPORT_DIR } from "../../files.ts"

export type Miss = {
  keyword: string
  target: string
  rank: number | null
  top5: string[]
  emoji_freq: number
  pair_freq: number
}

type Probe = { n?: number; total?: number; acc_at_k?: number[]; misses?: Miss[] }

export type Report = {
  emoji?: { keywords?: Probe }
  cldr?: Probe
}

export async function latestReport(dir = REPORT_DIR): Promise<string> {
  const dirs = (await readdir(dir)).sort()
  let latest = ""
  let mtime = 0
  for (const d of dirs) {
    const p = `${dir}/${d}/report.json`
    try {
      const m = (await stat(p)).mtimeMs
      if (m >= mtime) {
        mtime = m
        latest = p
      }
    } catch {
      // no report.json in this dir
    }
  }
  if (!latest) throw new Error(`no */report.json under ${dir}`)
  return latest
}
