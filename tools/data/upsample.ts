import { readFile } from "node:fs/promises"

import { generateText } from "ai"
import { cac } from "cac"
import cliProgress from "cli-progress"
import PQueue from "p-queue"
import { parse as parseYaml } from "yaml"

import {
  DATA_JSONL as DATA,
  GOALS_YML,
  GROUP_JSON,
  LABELS_JSON,
  TRAIN_JSONL,
} from "../../files.ts"
import { MODEL, annotate, annotateBatchCount, lastFills } from "./annotate.ts"
import { SEED } from "./config"
import { splitEmojis } from "./emoji.ts"
import { appendJsonl, readJsonl } from "./io.ts"

const TEXTS_PER_EMOJI = 50
const COLOR_BATCH = 50
const MOTIVATIONAL_BATCH = 50
const MOTIVATIONAL_COUNT = 1000
const LINKEDIN_BATCH = 50
const LINKEDIN_COUNT = 1000
const TOP_BATCH = 50
const TOP_COUNT = 100
const BALANCE_FRACTION = 0.1
const MIN_LEN = 4
const MAX_LEN = 42
const GEN_CONCURRENCY = 30

const COLORS = ["red", "green", "blue", "dark", "bright"]

const VOICES = [
  "a teenager",
  "a college student",
  "a new parent",
  "a retiree",
  "a shift worker",
  "a freelancer",
  "someone in their 30s",
  "a grandparent",
  "an office worker",
  "a nurse",
  "a tradesperson",
  "a student athlete",
  "a sibling",
  "a neighbor",
  "a coworker",
  "a classmate",
  "a teammate",
  "a close friend",
]

function pickVoice(): string {
  return VOICES[Math.floor(Math.random() * VOICES.length)]
}

export function languageName(code: string): string {
  try {
    return new Intl.DisplayNames(["en"], { type: "language" }).of(code) ?? code
  } catch {
    return code
  }
}

export function langLine(lang?: string): string[] {
  return lang
    ? [
      `Write every message in ${languageName(lang)}, using natural, native`,
      `wording and phrasing - not a translation from English.`,
    ]
    : []
}

export function langSuffix(lang?: string): string {
  return lang ? ` (lang: ${lang})` : ""
}

export function parseLang(raw: unknown): string | undefined {
  if (raw === undefined) return undefined
  const code = String(raw).trim().toLowerCase()
  return code || undefined
}

export function countEmojis(rows: { emojis?: string }[]): Map<string, number> {
  const counts = new Map<string, number>()
  for (const row of rows) {
    if (typeof row.emojis !== "string") continue
    for (const e of new Set(splitEmojis(row.emojis))) {
      counts.set(e, (counts.get(e) ?? 0) + 1)
    }
  }
  return counts
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

function seededShuffle<T>(arr: T[], seed: number): T[] {
  const rand = mulberry32(seed)
  const out = [...arr]
  for (let i = out.length - 1; i > 0; i--) {
    const j = Math.floor(rand() * (i + 1))
      ;[out[i], out[j]] = [out[j], out[i]]
  }
  return out
}

export function batchSizes(total: number, per: number): number[] {
  const out: number[] = []
  for (let left = total; left > 0; left -= per) out.push(Math.min(per, left))
  return out
}

export function colorBatchPlan(
  colors: string[],
  per: number,
  batchSize: number,
): { color: string; n: number }[] {
  return colors.flatMap((color) =>
    batchSizes(per, batchSize).map((n) => ({ color, n })),
  )
}

const stripVariation = (e: string): string => e.replace(/️/g, "")

export type GroupDeficit = { name: string; needed: number; missing: string[] }

export function groupDeficits(
  groups: Record<string, string[]>,
  vocabEmojis: string[],
  targets: Record<string, number>,
): GroupDeficit[] {
  const vocab = new Set(vocabEmojis.map(stripVariation))
  const out: GroupDeficit[] = []
  for (const [name, members] of Object.entries(groups)) {
    const target = targets[name]
    if (target === undefined || !members.length) continue
    const covered = members.filter((e) => vocab.has(stripVariation(e))).length
    const needed = Math.max(0, Math.ceil(target * members.length - covered - 1e-9))
    if (needed <= 0) continue
    const missing = members.filter((e) => !vocab.has(stripVariation(e)))
    out.push({ name, needed: Math.min(needed, missing.length), missing })
  }
  return out
}

export function rankMissingByFreq(
  missing: string[],
  counts: Map<string, number>,
  seed = SEED,
): string[] {
  const norm = new Map<string, number>()
  for (const [k, v] of counts) {
    const s = stripVariation(k)
    norm.set(s, (norm.get(s) ?? 0) + v)
  }
  const freq = (e: string) => norm.get(stripVariation(e)) ?? 0
  return seededShuffle(missing, seed)
    .map((e, i) => ({ e, i, f: freq(e) }))
    .sort((a, b) => b.f - a.f || a.i - b.i)
    .map((x) => x.e)
}

export function lowestFreqEmojis(
  counts: Map<string, number>,
  fraction: number,
): string[] {
  const n = Math.max(1, Math.round(counts.size * fraction))
  return [...counts.keys()]
    .map((k, i) => ({ k, i, c: counts.get(k) ?? 0 }))
    .sort((a, b) => a.c - b.c || a.i - b.i)
    .slice(0, n)
    .map((x) => x.k)
}

function genPrompt(voice: string, emoji: string, per: number, lang?: string): string {
  return [
    `Write ${per} short text messages as if sent by ${voice}, one per line.`,
    `Each message between ${MIN_LEN} and ${MAX_LEN} characters.`,
    ...langLine(lang),
    `Every message must read naturally as one a person would send together with`,
    `the emoji ${emoji} - its subject, activity, place, or mood fits that emoji.`,
    `Do not put any emoji in the output, and never name or describe the emoji.`,
    `Vary sender, tone, and intent: updates, questions, complaints, plans,`,
    `reactions, reminders, small talk. Sound real and specific.`,
    `No numbering, no bullets, no quotes, no commentary.`,
  ].join("\n")
}

function genColorPrompt(voice: string, color: string, per: number, lang?: string): string {
  return [
    `Write ${per} short text messages as if sent by ${voice}, one per line.`,
    `Each message between ${MIN_LEN} and ${MAX_LEN} characters.`,
    ...langLine(lang),
    `Every message must evoke the colour "${color}" - its light and mood, or`,
    `concrete things that are almost always that colour.`,
    `Most messages must NOT contain the word "${color}": name objects, places,`,
    `weather, plants, food, or materials instead. A few may use "${color}"`,
    `naturally, but never write about colour itself (no "the colour ${color} is`,
    `beautiful", no "I love ${color}").`,
    `E.g. for red: "the roses are blooming in the garden"; for blue: "the sky`,
    `is clear and the ocean is calm".`,
    `Do not put any emoji in the output.`,
    `Vary sender, tone, and intent: updates, questions, complaints, plans,`,
    `reactions, reminders, small talk. Sound real and specific.`,
    `No numbering, no bullets, no quotes, no commentary.`,
  ].join("\n")
}

function genMotivationalPrompt(voice: string, per: number, lang?: string): string {
  return [
    `Write ${per} short inspirational, encouraging, or motivational messages,`,
    `one per line, as if sent by ${voice} to encourage someone else.`,
    `Each message between ${MIN_LEN} and ${MAX_LEN} characters.`,
    ...langLine(lang),
    `Vary tone and occasion: cheering someone on, comfort after a setback,`,
    `a pep talk before something hard, praise for effort, a reminder to keep going.`,
    `Sound warm and specific, not generic greeting-card fluff.`,
    `Do not put any emoji in the output.`,
    `No numbering, no bullets, no quotes, no commentary.`,
  ].join("\n")
}

function genLinkedinPrompt(voice: string, per: number, lang?: string): string {
  return [
    `Write ${per} short, funny, relatable workplace or day-in-the-life`,
    `messages as if sent by ${voice}, one per line.`,
    `Each message between ${MIN_LEN} and ${MAX_LEN} characters.`,
    ...langLine(lang),
    `Capture a specific everyday work moment - meetings, commute, coffee,`,
    `deadlines, standups, a good or bad day at the job - with dry wit or`,
    `light complaint. Never generic motivational quotes, slogans, or hashtags.`,
    `Sound like something a real person would actually say, not an ad.`,
    `Do not put any emoji in the output.`,
    `No numbering, no bullets, no quotes, no commentary.`,
  ].join("\n")
}

function genTopPrompt(voice: string, per: number, lang?: string): string {
  return [
    `Write ${per} short text messages as if sent by ${voice}, one per line.`,
    `Each message between ${MIN_LEN} and ${MAX_LEN} characters.`,
    ...langLine(lang),
    `Cover everyday life broadly - plans, feelings, food, weather, work,`,
    `family, friends, hobbies, travel, complaints, celebrations, small talk -`,
    `varying subject and mood from message to message.`,
    `Do not put any emoji in the output.`,
    `Vary sender, tone, and intent: updates, questions, complaints, plans,`,
    `reactions, reminders, small talk. Sound real and specific.`,
    `No numbering, no bullets, no quotes, no commentary.`,
  ].join("\n")
}

function cleanLines(text: string): string[] {
  return text
    .split("\n")
    .map((l) =>
      l
        .replace(/^\s*(?:[-*•]|\d+[.)])\s+/, "")
        .replace(/^["'`]+|["'`]+$/g, "")
        .trim(),
    )
    .filter((l) => l && !l.startsWith("```"))
}

async function genBatch(
  voice: string,
  emoji: string,
  per: number,
  lang?: string,
): Promise<string[]> {
  const { text } = await generateText({
    model: MODEL,
    prompt: genPrompt(voice, emoji, per, lang),
  })
  return cleanLines(text)
}

async function genColorBatch(
  voice: string,
  color: string,
  per: number,
  lang?: string,
): Promise<string[]> {
  const { text } = await generateText({
    model: MODEL,
    prompt: genColorPrompt(voice, color, per, lang),
  })
  return cleanLines(text)
}

async function genMotivationalBatch(
  voice: string,
  per: number,
  lang?: string,
): Promise<string[]> {
  const { text } = await generateText({
    model: MODEL,
    prompt: genMotivationalPrompt(voice, per, lang),
  })
  return cleanLines(text)
}

async function genLinkedinBatch(
  voice: string,
  per: number,
  lang?: string,
): Promise<string[]> {
  const { text } = await generateText({
    model: MODEL,
    prompt: genLinkedinPrompt(voice, per, lang),
  })
  return cleanLines(text)
}

async function genTopBatch(
  voice: string,
  per: number,
  lang?: string,
): Promise<string[]> {
  const { text } = await generateText({
    model: MODEL,
    prompt: genTopPrompt(voice, per, lang),
  })
  return cleanLines(text)
}

async function loadGroups(): Promise<Record<string, string[]>> {
  return JSON.parse(await readFile(GROUP_JSON, "utf8")) as Record<string, string[]>
}

async function loadLabels(): Promise<{ styles: string[]; emojis: string[] }> {
  return JSON.parse(await readFile(LABELS_JSON, "utf8")) as {
    styles: string[]
    emojis: string[]
  }
}

async function loadGoalCoverage(): Promise<Record<string, number>> {
  const doc = parseYaml(await readFile(GOALS_YML, "utf8")) as {
    goals?: { vocabulary?: { coverage?: Record<string, number> } }
  }
  return doc?.goals?.vocabulary?.coverage ?? {}
}

type Cand = {
  text: string
  target?: string
  color?: string
  group?: string
  lang?: string
}

async function generateForEmojis(
  targets: string[],
  per: number,
  lang?: string,
): Promise<Cand[]> {
  const cands: Cand[] = []
  const genBar = new cliProgress.SingleBar(
    {
      format: "generating |{bar}| {percentage}% | {value}/{total} emojis | ETA: {eta}s",
    },
    cliProgress.Presets.shades_classic,
  )
  genBar.start(targets.length, 0)
  const genQ = new PQueue({ concurrency: GEN_CONCURRENCY })
  genQ.addAll(
    targets.map((emoji) => async () => {
      try {
        for (const t of await genBatch(pickVoice(), emoji, per, lang)) {
          cands.push({ text: t, target: emoji, lang })
        }
      } catch (err) {
        console.warn(`\n  gen (${emoji}) failed: ${err}`)
      }
      genBar.increment()
    }),
  )
  await genQ.onIdle()
  genBar.stop()
  return cands
}

async function generateForColors(per: number, lang?: string): Promise<Cand[]> {
  const colorPlan = colorBatchPlan(COLORS, per, COLOR_BATCH)
  console.log(
    `colors mode -> ${per} texts per colour for ${COLORS.join(", ")} `
    + `-> ${COLORS.length * per} texts in ${colorPlan.length} batches of up to ${COLOR_BATCH}${langSuffix(lang)}`,
  )
  const cands: Cand[] = []
  const genBar = new cliProgress.SingleBar(
    {
      format: "generating |{bar}| {percentage}% | {value}/{total} batches | ETA: {eta}s",
    },
    cliProgress.Presets.shades_classic,
  )
  genBar.start(colorPlan.length, 0)
  const genQ = new PQueue({ concurrency: GEN_CONCURRENCY })
  genQ.addAll(
    colorPlan.map(({ color, n }) => async () => {
      try {
        for (const t of await genColorBatch(pickVoice(), color, n, lang)) {
          cands.push({ text: t, color, lang })
        }
      } catch (err) {
        console.warn(`\n  gen (${color}) failed: ${err}`)
      }
      genBar.increment()
    }),
  )
  await genQ.onIdle()
  genBar.stop()
  return cands
}

async function generateForMotivational(count: number, lang?: string): Promise<Cand[]> {
  const sizes = batchSizes(count, MOTIVATIONAL_BATCH)
  console.log(
    `motivational mode -> ${count} texts in ${sizes.length} batches of up to ${MOTIVATIONAL_BATCH}${langSuffix(lang)}`,
  )
  const cands: Cand[] = []
  const genBar = new cliProgress.SingleBar(
    {
      format: "generating |{bar}| {percentage}% | {value}/{total} batches | ETA: {eta}s",
    },
    cliProgress.Presets.shades_classic,
  )
  genBar.start(sizes.length, 0)
  const genQ = new PQueue({ concurrency: GEN_CONCURRENCY })
  genQ.addAll(
    sizes.map((n) => async () => {
      try {
        for (const t of await genMotivationalBatch(pickVoice(), n, lang)) {
          cands.push({ text: t, lang })
        }
      } catch (err) {
        console.warn(`\n  gen (motivational) failed: ${err}`)
      }
      genBar.increment()
    }),
  )
  await genQ.onIdle()
  genBar.stop()
  return cands
}

async function generateForLinkedin(count: number, lang?: string): Promise<Cand[]> {
  const sizes = batchSizes(count, LINKEDIN_BATCH)
  console.log(
    `linkedin mode -> ${count} texts in ${sizes.length} batches of up to ${LINKEDIN_BATCH}${langSuffix(lang)}`,
  )
  const cands: Cand[] = []
  const genBar = new cliProgress.SingleBar(
    {
      format: "generating |{bar}| {percentage}% | {value}/{total} batches | ETA: {eta}s",
    },
    cliProgress.Presets.shades_classic,
  )
  genBar.start(sizes.length, 0)
  const genQ = new PQueue({ concurrency: GEN_CONCURRENCY })
  genQ.addAll(
    sizes.map((n) => async () => {
      try {
        for (const t of await genLinkedinBatch(pickVoice(), n, lang)) {
          cands.push({ text: t, lang })
        }
      } catch (err) {
        console.warn(`\n  gen (linkedin) failed: ${err}`)
      }
      genBar.increment()
    }),
  )
  await genQ.onIdle()
  genBar.stop()
  return cands
}

async function generateForTop(count: number, lang?: string): Promise<Cand[]> {
  const sizes = batchSizes(count, TOP_BATCH)
  console.log(
    `top mode -> ${count} texts in ${sizes.length} batches of up to ${TOP_BATCH}${langSuffix(lang)}`,
  )
  const cands: Cand[] = []
  const genBar = new cliProgress.SingleBar(
    {
      format: "generating |{bar}| {percentage}% | {value}/{total} batches | ETA: {eta}s",
    },
    cliProgress.Presets.shades_classic,
  )
  genBar.start(sizes.length, 0)
  const genQ = new PQueue({ concurrency: GEN_CONCURRENCY })
  genQ.addAll(
    sizes.map((n) => async () => {
      try {
        for (const t of await genTopBatch(pickVoice(), n, lang)) {
          cands.push({ text: t, lang })
        }
      } catch (err) {
        console.warn(`\n  gen (top) failed: ${err}`)
      }
      genBar.increment()
    }),
  )
  await genQ.onIdle()
  genBar.stop()
  return cands
}

async function annotateAndAppend(cands: Cand[], src: string): Promise<void> {
  console.log(`\n${cands.length} texts generated, annotating`)
  const annBar = new cliProgress.SingleBar(
    {
      format:
        "annotating |{bar}| {percentage}% | {value}/{total} batches | ETA: {eta}s",
    },
    cliProgress.Presets.shades_classic,
  )
  annBar.start(annotateBatchCount(cands.length), 0)

  const today = new Date().toISOString().slice(0, 10)
  const lines: string[] = []
  let noLabel = 0
  let noPalette = 0
  let hitTarget = 0
  let missTarget = 0

  await annotate(cands.map((c) => c.text), {
    colors: true,
    fillPalette: true,
    onBatchDone: () => annBar.increment(),
    onBatch: async (batch, got) => {
      const batchLines: string[] = []
      for (const { id } of batch) {
        const label = got.get(id)
        if (!label) {
          noLabel++
          continue
        }
        if (!label.bg || !label.fg) {
          noPalette++
          continue
        }
        const target = cands[id].target
        let emojis: string
        if (target) {
          if (label.emojis.includes(target)) hitTarget++
          else missTarget++
          emojis = [target, ...label.emojis.filter((e) => e !== target)].join(" ")
        } else {
          emojis = label.emojis.join(" ")
        }
        const row: Record<string, unknown> = {
          text: cands[id].text,
          emojis,
          styles: label.styles,
          bg: label.bg,
          fg: label.fg,
        }
        if (cands[id].lang) row.lang = cands[id].lang
        if (cands[id].color) row.color = cands[id].color
        if (cands[id].group) row.group = cands[id].group
        row.meta = { date: today, src }
        batchLines.push(JSON.stringify(row))
      }
      lines.push(...batchLines)
      if (batchLines.length) await appendJsonl(DATA, batchLines)
    },
  })
  annBar.stop()

  console.log("\n--- summary ---")
  console.log(`generated            : ${cands.length}`)
  console.log(`appended -> data     : ${lines.length}`)
  console.log(`dropped no label     : ${noLabel}`)
  console.log(`filled palette       : ${lastFills.palette}`)
  console.log(`dropped no palette   : ${noPalette}`)
  if (cands.some((c) => c.target)) {
    console.log(
      `target hit / miss    : ${hitTarget} / ${missTarget} (target injected either way)`,
    )
  }
}

function parsePer(raw: unknown): number {
  const per = Number(raw ?? TEXTS_PER_EMOJI)
  if (!(per >= 1)) {
    console.error(`--per must be >= 1, got ${JSON.stringify(raw)}`)
    process.exit(1)
  }
  return per
}

function parseCount(raw: unknown, def: number): number {
  const count = Number(raw ?? def)
  if (!(count >= 1)) {
    console.error(`--count must be >= 1, got ${JSON.stringify(raw)}`)
    process.exit(1)
  }
  return count
}

const cli = cac("upsample")

cli.option(
  "--lang <code>",
  "generate text in this language (ISO 639-1 code, e.g. 'he' for Hebrew); omit for English",
)

cli
  .command(
    "",
    `generate generic texts across everyday topics and annotate them (default mode, ${TOP_COUNT} texts unless --count is given)`,
  )
  .option("--count <n>", `messages to generate (default ${TOP_COUNT})`)
  .option("--dry", "report what would be upsampled, then exit without generating, annotating, or appending")
  .action(async (options) => {
    const count = parseCount(options.count, TOP_COUNT)
    const lang = parseLang(options.lang)
    if (options.dry) {
      console.log("\n--- dry run: nothing generated, annotated, or appended ---")
      console.log(`mode                 : top`)
      console.log(`would generate       : ${count} texts${langSuffix(lang)}`)
      return
    }
    const cands = await generateForTop(count, lang)
    await annotateAndAppend(cands, "top")
  })

cli
  .command(
    "emojis <...list>",
    "upsample exactly the given emoji (space- or comma-separated)",
  )
  .option("--per <n>", `texts to generate per emoji (default ${TEXTS_PER_EMOJI})`)
  .option("--dry", "report what would be upsampled, then exit without generating, annotating, or appending")
  .action(async (list: string[], options) => {
    const only = list.join(" ").trim()
    const targets = [...new Set(splitEmojis(only))]
    if (!targets.length) {
      console.error(`no recognizable emoji in: ${JSON.stringify(only)}`)
      process.exit(1)
    }
    const per = parsePer(options.per)
    const lang = parseLang(options.lang)
    console.log(`targeting ${targets.length} emoji -> ${targets.join(" ")}`)
    if (options.dry) {
      console.log("\n--- dry run: nothing generated, annotated, or appended ---")
      console.log(`mode                 : emojis`)
      console.log(`would generate       : ~${targets.length * per} texts (${per}/emoji)${langSuffix(lang)}`)
      return
    }
    const cands = await generateForEmojis(targets, per, lang)
    await annotateAndAppend(cands, "emoji-target")
  })

cli
  .command("colors", `upsample texts for each of ${COLORS.join(", ")}`)
  .option("--per <n>", `texts to generate per colour (default ${TEXTS_PER_EMOJI})`)
  .option("--dry", "report what would be upsampled, then exit without generating, annotating, or appending")
  .action(async (options) => {
    const per = parsePer(options.per)
    const lang = parseLang(options.lang)
    if (options.dry) {
      const colorPlan = colorBatchPlan(COLORS, per, COLOR_BATCH)
      console.log("\n--- dry run: nothing generated, annotated, or appended ---")
      console.log(`mode                 : colors`)
      console.log(
        `would generate       : ${colorPlan.reduce((s, b) => s + b.n, 0)} texts over ${COLORS.length} colours${langSuffix(lang)}`,
      )
      return
    }
    const cands = await generateForColors(per, lang)
    await annotateAndAppend(cands, "colors")
  })

cli
  .command(
    "motivational",
    "generate inspirational/encouraging/motivational messages",
  )
  .option("--count <n>", `messages to generate (default ${MOTIVATIONAL_COUNT})`)
  .option("--dry", "report what would be upsampled, then exit without generating, annotating, or appending")
  .action(async (options) => {
    const count = parseCount(options.count, MOTIVATIONAL_COUNT)
    const lang = parseLang(options.lang)
    if (options.dry) {
      console.log("\n--- dry run: nothing generated, annotated, or appended ---")
      console.log(`mode                 : motivational`)
      console.log(`would generate       : ${count} texts${langSuffix(lang)}`)
      return
    }
    const cands = await generateForMotivational(count, lang)
    await annotateAndAppend(cands, "motivational")
  })

cli
  .command(
    "linkedin",
    "generate funny/relatable workplace content for LinkedIn share cards",
  )
  .option("--count <n>", `messages to generate (default ${LINKEDIN_COUNT})`)
  .option("--dry", "report what would be upsampled, then exit without generating, annotating, or appending")
  .action(async (options) => {
    const count = parseCount(options.count, LINKEDIN_COUNT)
    const lang = parseLang(options.lang)
    if (options.dry) {
      console.log("\n--- dry run: nothing generated, annotated, or appended ---")
      console.log(`mode                 : linkedin`)
      console.log(`would generate       : ${count} texts${langSuffix(lang)}`)
      return
    }
    const cands = await generateForLinkedin(count, lang)
    await annotateAndAppend(cands, "linkedin")
  })

cli
  .command(
    "groups",
    "upsample under-covered emoji groups toward their goals.yml vocabulary.coverage target",
  )
  .option("--per <n>", `texts to generate per emoji (default ${TEXTS_PER_EMOJI})`)
  .option("--dry", "report what would be upsampled, then exit without generating, annotating, or appending")
  .action(async (options) => {
    const per = parsePer(options.per)
    const lang = parseLang(options.lang)
    const [rows, groups, labels, coverage] = await Promise.all([
      readJsonl<{ emojis?: string }>(DATA),
      loadGroups(),
      loadLabels(),
      loadGoalCoverage(),
    ])
    const counts = countEmojis(rows)
    const deficits = groupDeficits(groups, labels.emojis, coverage)
    if (!deficits.length) {
      console.log("no under-covered groups - every group meets its goals.yml target")
      return
    }
    const plan = deficits.map((d) => ({
      name: d.name,
      needed: d.needed,
      targets: rankMissingByFreq(d.missing, counts).slice(0, d.needed),
    }))
    const groupOf = new Map<string, string>()
    for (const g of plan) for (const e of g.targets) groupOf.set(e, g.name)
    const targets = [...groupOf.keys()]
    console.log(`${plan.length} group(s) under-covered -> ${targets.length} emoji to upsample`)
    for (const g of plan) {
      console.log(`  ${g.name.padEnd(24)}: +${g.needed} -> ${g.targets.join(" ")}`)
    }
    if (options.dry) {
      console.log("\n--- dry run: nothing generated, annotated, or appended ---")
      console.log(`mode                 : groups`)
      console.log(`would generate       : ~${targets.length * per} texts (${per}/emoji)${langSuffix(lang)}`)
      return
    }
    const cands = await generateForEmojis(targets, per, lang)
    for (const c of cands) if (c.target) c.group = groupOf.get(c.target)
    await annotateAndAppend(cands, "group")
  })

cli
  .command(
    "balance",
    `upsample the lowest ${BALANCE_FRACTION * 100}% of emoji by frequency in ${TRAIN_JSONL}`,
  )
  .option("--per <n>", `texts to generate per emoji (default ${TEXTS_PER_EMOJI})`)
  .option("--dry", "report what would be upsampled, then exit without generating, annotating, or appending")
  .action(async (options) => {
    const per = parsePer(options.per)
    const lang = parseLang(options.lang)
    const rows = await readJsonl<{ emojis?: string }>(TRAIN_JSONL)
    const counts = countEmojis(rows)
    if (!counts.size) {
      console.error(`no emoji found in ${TRAIN_JSONL}`)
      process.exit(1)
    }
    const targets = lowestFreqEmojis(counts, BALANCE_FRACTION)
    console.log(
      `${counts.size} distinct emoji in ${TRAIN_JSONL} -> targeting lowest `
      + `${BALANCE_FRACTION * 100}% (${targets.length}) `
      + `(${counts.get(targets[0])}..${counts.get(targets.at(-1) ?? "")} rows each)`,
    )
    console.log(targets.join(" "))
    if (options.dry) {
      console.log("\n--- dry run: nothing generated, annotated, or appended ---")
      console.log(`mode                 : balance`)
      console.log(`would generate       : ~${targets.length * per} texts (${per}/emoji)${langSuffix(lang)}`)
      return
    }
    const cands = await generateForEmojis(targets, per, lang)
    await annotateAndAppend(cands, "balance")
  })

cli.help()

if (import.meta.main) {
  cli.parse()
}
