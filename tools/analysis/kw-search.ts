import uFuzzy from "@leeoniya/ufuzzy"
import cliProgress from "cli-progress"

import { KWPROJ_JSON, LABELS_JSON } from "../../files.ts"
import { readJsonl } from "../data/io.ts"
import { queryTokens } from "../data/tokenize.ts"

const PRIMARY_BONUS = 0.15
const KS = [1, 5, 10]

export type Proj = Record<string, number[]>
export type Row = { text: string; targets: number[] }
export type Scored = { text: string; kw: [number, number][] }

export function makeSearch(proj: Proj) {
  const keys = Object.keys(proj)
  const weight = new Map(keys.map((k) => [k, 1 / Math.log2(1 + proj[k].length)]))
  const uf = new uFuzzy({ intraIns: 1 })
  function* matches(word: string): Generator<[string, number]> {
    if (proj[word]) yield [word, 1.0]
    if (word.length < 3) return
    const idxs = uf.filter(keys, word)
    if (!idxs || !idxs.length) return
    const info = uf.info(idxs, keys, word)
    for (let i = 0; i < info.idx.length; i++) {
      const k = keys[info.idx[i]]
      if (k === word) continue
      const sim = info.chars[i] / k.length
      if (sim >= 0.5) yield [k, sim]
    }
  }
  return {
    predict(text: string): [number, number][] {
      const acc = new Map<number, number>()
      for (const word of queryTokens(text)) {
        for (const [k, strength] of matches(word)) {
          const base = strength * weight.get(k)!
          const pl = proj[k]
          for (let j = 0; j < pl.length; j++) {
            const v = base + (j === 0 ? PRIMARY_BONUS : 0)
            if (v > (acc.get(pl[j]) ?? 0)) acc.set(pl[j], v)
          }
        }
      }
      return [...acc.entries()]
        .map(([i, v]) => [i, Number(v.toFixed(3))] as [number, number])
        .sort((a, b) => a[0] - b[0])
    },
  }
}

export function rankByScore(kw: [number, number][], vocabSize: number): number[] {
  const scores = new Float64Array(vocabSize)
  for (const [i, v] of kw) scores[i] = v
  return [...scores.keys()].sort((a, b) => scores[b] - scores[a] || a - b)
}

export function hitAtK(ranked: number[], targets: number[], k: number): boolean {
  const top = new Set(ranked.slice(0, k))
  return targets.some((t) => top.has(t))
}

export function rowTargets(emojis: string, idx: Map<string, number>): number[] {
  const out: number[] = []
  for (const e of String(emojis ?? "").split(/\s+/)) {
    const i = idx.get(e)
    if (i !== undefined && !out.includes(i)) out.push(i)
  }
  return out
}

export function accAtK(
  rows: Row[],
  scored: Scored[],
  vocabSize: number,
  ks: number[] = KS,
): Record<number, number> {
  const n = rows.length || 1
  const ranked = scored.map((s) => rankByScore(s.kw, vocabSize))
  const out: Record<number, number> = {}
  for (const k of ks) {
    out[k] = rows.filter((r, i) => hitAtK(ranked[i], r.targets, k)).length / n
  }
  return out
}

async function main() {
  const args = process.argv.slice(2)
  const asJson = args.includes("--json")
  const file = args.find((a) => !a.startsWith("--"))
  if (!file) {
    console.error("usage: bun run tools/analysis/kw-search.ts <data.jsonl> [--json]")
    process.exit(1)
  }

  const emojis: string[] = JSON.parse(await Bun.file(LABELS_JSON).text()).emojis
  const idx = new Map(emojis.map((e, i) => [e, i]))
  const proj: Proj = JSON.parse(await Bun.file(KWPROJ_JSON).text()).proj
  const search = makeSearch(proj)

  const records = await readJsonl<{ text?: string; emojis?: string }>(file)
  const rows: Row[] = []
  for (const r of records) {
    const text = String(r.text ?? "").trim()
    const targets = rowTargets(r.emojis ?? "", idx)
    if (text && targets.length) rows.push({ text, targets })
  }

  const bar = new cliProgress.SingleBar(
    {
      format: "kw-search |{bar}| {percentage}% | {value}/{total} rows | ETA: {eta}s",
      stream: process.stderr,
    },
    cliProgress.Presets.shades_classic,
  )
  bar.start(rows.length, 0)
  const scored: Scored[] = rows.map((r) => {
    const s = { text: r.text, kw: search.predict(r.text) }
    bar.increment()
    return s
  })
  bar.stop()
  const acc = accAtK(rows, scored, emojis.length)

  if (asJson) {
    const out = { n: rows.length, total: records.length, acc_at_k: acc, rows: scored }
    console.log(JSON.stringify(out))
  } else {
    console.log(
      `kw-search: ${rows.length}/${records.length} rows scored (>=1 in-vocab target), `
      + `vocab=${emojis.length}`,
    )
    for (const k of KS) console.log(`  acc@${k}: ${(100 * acc[k]).toFixed(1)}%`)
  }
}

if (import.meta.main) {
  await main()
}
