import uFuzzy from "@leeoniya/ufuzzy"
import { readFileSync } from "node:fs"
import { mkdir } from "node:fs/promises"

import {
  DATA_JSONL,
  EMOJI_POPULARITY_JSON,
  II_JSON,
  STEP1_DIR,
  STEP1_EVAL_JSONL,
  STEP1_EXACT_EVAL_JSONL,
  STEP1_FUZZY_EVAL_JSONL,
  STEP1_II_JSON,
  STEP1_KWPROJ_JSON,
  STEP1_LABELS_JSON,
  STEP1_TRAIN_JSONL,
} from "../../files.ts"
import { SEED, STYLES } from "./config"
import { coarseEmojiGroup, splitEmojis } from "./emoji.ts"
import { fuzzVariants } from "./fuzz_keywords.ts"
import { writeFileAtomic } from "./io.ts"
import { normalize } from "./normalize.ts"
import { collapse, type Row, shuffle, toLine } from "./regen.ts"
import { queryTokens } from "./tokenize.ts"

const TARGET_VOCAB = 150
const MAX_TARGETS = 3
const KEY_MIN = 3
const KEY_MAX = 12
const GROUP_FLOOR = 12
const FLOOR_GROUPS = [
  "Animals & Nature",
  "Food & Drink",
  "Travel & Places",
  "Objects",
  "Symbols",
  "Smileys & Emotion",
]
const COMMON_FLAGS = [
  "🇺🇸", "🇬🇧", "🇫🇷", "🇩🇪", "🇯🇵", "🇮🇹", "🇪🇸", "🇧🇷", "🇮🇳", "🇨🇳",
  "🇨🇦", "🇷🇺", "🇰🇷", "🇲🇽", "🇳🇱", "🇸🇪", "🇨🇭", "🇦🇺", "🇵🇹", "🇵🇱",
]
const PRIMARY_BONUS = 0.15
const HOLDOUT_FRAC = 0.2
const FUZZY_TRAIN_PER_KW = 2
const SHORT_MAX_LEN = 32
const SHORT_EVAL_FRAC = 0.15
const DUMMY_STYLE = "Deadpan"
const DUMMY_BG = ["#101014", "#1c1c22"]
const DUMMY_FG = "#f4f4f5"

type Curated = { kw: string; targets: string[]; group: string }

function stripVar(e: string): string {
  return [...e].filter((c) => c.codePointAt(0) !== 0xfe0f).join("")
}

function loadPopularity(): Map<string, number> {
  const raw = JSON.parse(readFileSync(EMOJI_POPULARITY_JSON, "utf8")) as Record<string, number>
  const m = new Map<string, number>()
  for (const [e, s] of Object.entries(raw)) {
    m.set(e, Math.max(m.get(e) ?? 0, s))
    m.set(stripVar(e), Math.max(m.get(stripVar(e)) ?? 0, s))
  }
  return m
}

function pop(m: Map<string, number>, e: string): number {
  return m.get(e) ?? m.get(stripVar(e)) ?? 0
}

function isSingleToken(k: string): boolean {
  if (k !== k.toLowerCase()) return false
  if (k.length < KEY_MIN || k.length > KEY_MAX) return false
  if (!/^[a-z]+$/.test(k)) return false
  const t = queryTokens(k)
  return t.length === 1 && t[0] === k
}

function curate(): { curated: Curated[]; emojiList: string[] } {
  const ii = JSON.parse(readFileSync(II_JSON, "utf8")) as Record<string, string[]>
  const popMap = loadPopularity()

  const candidates: Curated[] = []
  for (const [kw, rawTargets] of Object.entries(ii)) {
    if (!isSingleToken(kw)) continue
    const targets = [...new Set(rawTargets)]
    if (targets.length < 1 || targets.length > MAX_TARGETS) continue
    const groups = new Set(targets.map(coarseEmojiGroup))
    if (groups.size !== 1) continue
    const group = [...groups][0]
    if (group === "Other" || group === "Component") continue
    const sorted = [...targets].sort(
      (a, b) => pop(popMap, b) - pop(popMap, a) || (a < b ? -1 : 1),
    )
    candidates.push({ kw, targets: sorted, group })
  }

  candidates.sort(
    (a, b) =>
      a.targets.length - b.targets.length ||
      Math.max(...b.targets.map((e) => pop(popMap, e))) -
        Math.max(...a.targets.map((e) => pop(popMap, e))) ||
      (a.kw < b.kw ? -1 : 1),
  )

  const vocab = new Set<string>()
  const emojiList: string[] = []
  const add = (e: string) => {
    if (!vocab.has(e)) {
      vocab.add(e)
      emojiList.push(e)
    }
  }
  const curated: Curated[] = []
  const groupCount = new Map<string, number>()
  const take = (c: Curated) => {
    c.targets.forEach(add)
    curated.push(c)
    groupCount.set(c.group, (groupCount.get(c.group) ?? 0) + 1)
  }

  const remaining: Curated[] = []
  for (const c of candidates) {
    if (vocab.size >= TARGET_VOCAB) {
      remaining.push(c)
      continue
    }
    take(c)
  }
  for (const g of FLOOR_GROUPS) {
    for (const c of remaining) {
      if ((groupCount.get(g) ?? 0) >= GROUP_FLOOR) break
      if (c.group === g && !curated.includes(c)) take(c)
    }
  }

  for (const f of COMMON_FLAGS) if (coarseEmojiGroup(f) === "Flags") add(f)

  return { curated, emojiList }
}

function buildProj(
  curated: Curated[],
  emojiIdx: Map<string, number>,
): Record<string, number[]> {
  const proj: Record<string, number[]> = {}
  for (const c of curated) {
    const idxs = c.targets
      .map((e) => emojiIdx.get(e))
      .filter((i): i is number => i !== undefined)
    if (idxs.length) proj[c.kw] = idxs
  }
  return proj
}

function makeKwVec(proj: Record<string, number[]>) {
  const keys = Object.keys(proj)
  const weight = new Map(keys.map((k) => [k, 1 / Math.log2(1 + proj[k].length)]))
  const uf = new uFuzzy({ intraIns: 1 })
  const matches = function* (word: string): Generator<[string, number]> {
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
  return (text: string): [number, number][] => {
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
      .filter(([, v]) => v > 0)
      .sort((a, b) => a[0] - b[0])
  }
}

function synthRow(text: string, emojis: string[], base?: string): Row {
  const r: Row = {
    text,
    emojis: emojis.join(" "),
    styles: [DUMMY_STYLE],
    bg: DUMMY_BG,
    fg: DUMMY_FG,
  }
  if (base) r.extra = { base }
  return r
}

async function shortSlice(vocab: Set<string>): Promise<Row[]> {
  const master = await Bun.file(DATA_JSONL).text()
  const rows = collapse(
    master
      .split("\n")
      .filter((l) => l.trim())
      .map((l) => JSON.parse(l)),
  )
  const out: Row[] = []
  for (const r of rows) {
    if (!r.bg || !r.fg || !r.styles?.length) continue
    if (normalize(r.text).length > SHORT_MAX_LEN) continue
    const es = [...new Set(splitEmojis(r.emojis))].filter((e) => vocab.has(e))
    if (!es.length) continue
    out.push({ text: r.text, emojis: es.join(" "), styles: r.styles, bg: r.bg, fg: r.fg })
  }
  return out
}

async function main(): Promise<void> {
  await mkdir(STEP1_DIR, { recursive: true })

  const { curated, emojiList } = curate()
  const emojiIdx = new Map(emojiList.map((e, i) => [e, i]))
  const vocab = new Set(emojiList)
  const proj = buildProj(curated, emojiIdx)
  const kwVec = makeKwVec(proj)

  const withGroup = shuffle(curated, SEED)
  const byGroup = new Map<string, Curated[]>()
  for (const c of withGroup) {
    const g = byGroup.get(c.group) ?? []
    g.push(c)
    byGroup.set(c.group, g)
  }
  const holdout = new Set<string>()
  for (const [, list] of byGroup) {
    const n = list.length >= 3 ? Math.max(1, Math.round(HOLDOUT_FRAC * list.length)) : 0
    for (let i = 0; i < n; i++) holdout.add(list[i].kw)
  }

  const exactAll = curated.map((c) => synthRow(c.kw, c.targets))
  const exactTrain = exactAll.filter((r) => !holdout.has(r.text))
  const exactEval = exactAll.filter((r) => holdout.has(r.text))

  const fuzzyTrain: Row[] = []
  const fuzzyEval: Row[] = []
  for (const c of curated) {
    const variants = fuzzVariants(c.kw)
    if (holdout.has(c.kw)) {
      for (const v of variants) fuzzyEval.push(synthRow(v, c.targets, c.kw))
    } else {
      variants.forEach((v, i) =>
        (i < FUZZY_TRAIN_PER_KW ? fuzzyTrain : fuzzyEval).push(synthRow(v, c.targets, c.kw)),
      )
    }
  }

  const short = shuffle(await shortSlice(vocab), SEED + 2)
  const nEval = Math.round(short.length * SHORT_EVAL_FRAC)
  const shortEval = short.slice(0, nEval)
  const shortTrain = short.slice(nEval)

  const train = shuffle([...shortTrain, ...exactTrain, ...fuzzyTrain], SEED + 3)
  const setKw = (rows: Row[]) => rows.map((r) => ({ ...r, kw: kwVec(r.text) }))

  const iiOut: Record<string, string[]> = {}
  for (const c of curated) iiOut[c.kw] = c.targets

  await writeFileAtomic(
    STEP1_LABELS_JSON,
    JSON.stringify({ styles: [...STYLES], emojis: emojiList }, null, 2) + "\n",
  )
  await writeFileAtomic(STEP1_II_JSON, JSON.stringify(iiOut) + "\n")
  await writeFileAtomic(STEP1_KWPROJ_JSON, JSON.stringify({ proj }) + "\n")
  await writeFileAtomic(STEP1_TRAIN_JSONL, setKw(train).map(toLine).join("\n") + "\n")
  await writeFileAtomic(STEP1_EVAL_JSONL, setKw(shortEval).map(toLine).join("\n") + "\n")
  await writeFileAtomic(
    STEP1_EXACT_EVAL_JSONL,
    setKw(exactEval).map(toLine).join("\n") + "\n",
  )
  await writeFileAtomic(
    STEP1_FUZZY_EVAL_JSONL,
    setKw(fuzzyEval).map(toLine).join("\n") + "\n",
  )

  console.log("\n--- regen-step1 ---")
  console.log(`curated keywords      : ${curated.length} (holdout ${holdout.size})`)
  console.log(`emoji vocab           : ${emojiList.length}`)
  console.log(`short-text slice      : ${short.length} (train ${shortTrain.length}, eval ${shortEval.length})`)
  console.log(`exact rows            : train ${exactTrain.length}, eval ${exactEval.length}`)
  console.log(`fuzzy rows            : train ${fuzzyTrain.length}, eval ${fuzzyEval.length}`)
  console.log(`-> ${STEP1_TRAIN_JSONL} : ${train.length}`)
  console.log(`-> ${STEP1_EVAL_JSONL} : ${shortEval.length}`)
  console.log(`-> ${STEP1_EXACT_EVAL_JSONL} : ${exactEval.length}`)
  console.log(`-> ${STEP1_FUZZY_EVAL_JSONL} : ${fuzzyEval.length}`)
  console.log(`-> ${STEP1_LABELS_JSON} / ${STEP1_II_JSON} / ${STEP1_KWPROJ_JSON}`)
}

if (import.meta.main) await main()
