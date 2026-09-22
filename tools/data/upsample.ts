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
import {
  ANNOTATE_BATCH_SIZE,
  ANNOTATE_CONCURRENCY,
  MODEL,
  annotateBatch,
  formatDrops,
  formatUsage,
} from "./annotate.ts"
import type { Drops, Fills, Usage } from "./annotate.ts"
import { langForText } from "../../web/src/scriptFonts.js"
import { SEED } from "./config"
import { splitEmojis } from "./emoji.ts"
import { appendJsonl, readJsonl } from "./io.ts"
import { LANGS, LANG_SET } from "./langs.ts"

const TEXTS_PER_EMOJI = 50
const COLOR_BATCH = 50
const MOTIVATIONAL_BATCH = 50
const MOTIVATIONAL_COUNT = 1000
const LINKEDIN_BATCH = 50
const LINKEDIN_COUNT = 1000
const SARCASM_BATCH = 50
const SARCASM_COUNT = 1000
const TOPIC_BATCH = 50
const DEFAULT_COUNT = 2500
const BALANCE_FRACTION = 0.1
const MIN_LEN = 4
const MAX_LEN = 42
const GEN_CONCURRENCY = 30
const CREATIVE_TEMPERATURE = 1.15

const COLORS = ["red", "green", "blue", "dark", "bright"]

export const TOPICS = [
  "food & cooking",
  "chores & housework",
  "commute & getting around",
  "weather & seasons",
  "pets & animals",
  "phones, apps & devices",
  "shopping & money",
  "tv, film & streaming",
  "video games",
  "sport & exercise",
  "health, sleep & the body",
  "school, class & studying",
  "work & the office",
  "music & gigs",
  "clothes & how things look",
  "events, parties & celebrations",
  "religious holidays & traditions (christian, jewish, muslim, hindu, buddhist, and others)",
  "hobbies, making & fixing things",
  "outdoors, parks & nature",
  "plans, scheduling & logistics",
  "eating out & food delivery",
  "cars, bikes & public transport",
  "news & things happening",
  "family, friends & relationships",
  "random small talk",
]

export function topicForBatch(i: number): string {
  return TOPICS[i % TOPICS.length]
}

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

export function resolveLangs(lang?: string): string[] {
  return lang ? [lang] : [...LANGS]
}

export function langsSuffix(langs: string[]): string {
  return langs.length > 1 ? ` (langs: ${langs.join(", ")})` : langSuffix(langs[0])
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

export function countLangs(rows: { text?: string; lang?: unknown }[]): Map<string, number> {
  const counts = new Map<string, number>()
  for (const row of rows) {
    if (typeof row.text !== "string") continue
    const lang = typeof row.lang === "string" && LANG_SET.has(row.lang)
      ? row.lang
      : langForText(row.text)
    counts.set(lang, (counts.get(lang) ?? 0) + 1)
  }
  return counts
}

export function leastFrequentLang(counts: Map<string, number>): string {
  return [...LANGS].sort(
    (a, b) => (counts.get(a) ?? 0) - (counts.get(b) ?? 0) || LANGS.indexOf(a) - LANGS.indexOf(b),
  )[0]
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

function genPrompt(
  voice: string,
  emoji: string,
  per: number,
  lang?: string,
  maxLen: number = MAX_LEN,
): string {
  return [
    `Write ${per} short WhatsApp messages as if sent by ${voice}, one per line.`,
    `Each message between ${MIN_LEN} and ${maxLen} characters.`,
    ...langLine(lang),
    `Every message must read naturally as one a person would send together with`,
    `the emoji ${emoji} - its subject, activity, place, or mood fits that emoji.`,
    `Do not put any emoji in the output. Mostly avoid naming or describing the`,
    `emoji directly, but a few messages may reference it naturally.`,
    `Vary sender, tone, and intent: updates, questions, complaints, plans,`,
    `reactions, reminders, small talk. Sound real and specific.`,
    `No numbering, no bullets, no quotes, no commentary.`,
  ].join("\n")
}

function genColorPrompt(
  voice: string,
  color: string,
  per: number,
  lang?: string,
  maxLen: number = MAX_LEN,
): string {
  return [
    `Write ${per} short WhatsApp messages as if sent by ${voice}, one per line.`,
    `Each message between ${MIN_LEN} and ${maxLen} characters.`,
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

function genMotivationalPrompt(
  voice: string,
  per: number,
  lang?: string,
  maxLen: number = MAX_LEN,
): string {
  return [
    `Write ${per} short inspirational, encouraging, or motivational WhatsApp`,
    `messages, one per line, as if sent by ${voice} to encourage someone else.`,
    `Each message between ${MIN_LEN} and ${maxLen} characters.`,
    ...langLine(lang),
    `Vary tone and occasion: cheering someone on, comfort after a setback,`,
    `a pep talk before something hard, praise for effort, a reminder to keep going.`,
    `Sound warm and specific, not generic greeting-card fluff.`,
    `Do not put any emoji in the output.`,
    `No numbering, no bullets, no quotes, no commentary.`,
  ].join("\n")
}

function genLinkedinPrompt(
  voice: string,
  per: number,
  lang?: string,
  maxLen: number = MAX_LEN,
): string {
  return [
    `Write ${per} short, funny, relatable workplace or day-in-the-life`,
    `WhatsApp messages as if sent by ${voice}, one per line.`,
    `Each message between ${MIN_LEN} and ${maxLen} characters.`,
    ...langLine(lang),
    `Capture a specific everyday work moment - meetings, commute, coffee,`,
    `deadlines, standups, a good or bad day at the job - with dry wit or`,
    `light complaint. Never generic motivational quotes, slogans, or hashtags.`,
    `Sound like something a real person would actually say, not an ad.`,
    `Do not put any emoji in the output.`,
    `No numbering, no bullets, no quotes, no commentary.`,
  ].join("\n")
}

function genSarcasmPrompt(
  voice: string,
  per: number,
  lang?: string,
  maxLen: number = MAX_LEN,
): string {
  return [
    `Write ${per} short sarcastic WhatsApp messages as if sent by ${voice}, one per line.`,
    `Each message between ${MIN_LEN} and ${maxLen} characters.`,
    ...langLine(lang),
    `Each message must be dripping with sarcasm or irony - saying the opposite`,
    `of what is meant, mock enthusiasm, deadpan exaggeration, or a backhanded`,
    `remark - about everyday annoyances: work, chores, weather, traffic, plans`,
    `falling through, technology, waiting around.`,
    `Sound like something a real person would actually text, not a meme caption.`,
    `Do not put any emoji in the output.`,
    `No numbering, no bullets, no quotes, no commentary.`,
  ].join("\n")
}

function genTopicPrompt(
  topic: string,
  voice: string,
  per: number,
  lang?: string,
  maxLen?: number,
): string {
  return [
    `Write ${per} short WhatsApp messages as if sent by ${voice}, one per line.`,
    ...(maxLen !== undefined ? [`Each message between ${MIN_LEN} and ${maxLen} characters.`] : []),
    ...langLine(lang),
    `Every message is about ${topic}.`,
    "No numbering, no bullets, no quotes, no emoji, no commentary.",
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
  maxLen?: number,
): Promise<string[]> {
  const { text } = await generateText({
    model: MODEL,
    temperature: CREATIVE_TEMPERATURE,
    prompt: genPrompt(voice, emoji, per, lang, maxLen),
  })
  return cleanLines(text)
}

async function genColorBatch(
  voice: string,
  color: string,
  per: number,
  lang?: string,
  maxLen?: number,
): Promise<string[]> {
  const { text } = await generateText({
    model: MODEL,
    temperature: CREATIVE_TEMPERATURE,
    prompt: genColorPrompt(voice, color, per, lang, maxLen),
  })
  return cleanLines(text)
}

async function genMotivationalBatch(
  voice: string,
  per: number,
  lang?: string,
  maxLen?: number,
): Promise<string[]> {
  const { text } = await generateText({
    model: MODEL,
    temperature: CREATIVE_TEMPERATURE,
    prompt: genMotivationalPrompt(voice, per, lang, maxLen),
  })
  return cleanLines(text)
}

async function genLinkedinBatch(
  voice: string,
  per: number,
  lang?: string,
  maxLen?: number,
): Promise<string[]> {
  const { text } = await generateText({
    model: MODEL,
    temperature: CREATIVE_TEMPERATURE,
    prompt: genLinkedinPrompt(voice, per, lang, maxLen),
  })
  return cleanLines(text)
}

async function genSarcasmBatch(
  voice: string,
  per: number,
  lang?: string,
  maxLen?: number,
): Promise<string[]> {
  const { text } = await generateText({
    model: MODEL,
    temperature: CREATIVE_TEMPERATURE,
    prompt: genSarcasmPrompt(voice, per, lang, maxLen),
  })
  return cleanLines(text)
}

async function genTopicBatch(
  topic: string,
  voice: string,
  per: number,
  lang?: string,
  maxLen?: number,
): Promise<string[]> {
  const { text } = await generateText({
    model: MODEL,
    temperature: CREATIVE_TEMPERATURE,
    prompt: genTopicPrompt(topic, voice, per, lang, maxLen),
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
  sarcastic?: boolean
}

export type Sink = { push(cand: Cand): void }

export class Batcher<T> {
  private buf: T[] = []
  constructor(private size: number) {}

  push(item: T): T[] | null {
    this.buf.push(item)
    if (this.buf.length < this.size) return null
    const out = this.buf
    this.buf = []
    return out
  }

  flush(): T[] | null {
    if (!this.buf.length) return null
    const out = this.buf
    this.buf = []
    return out
  }
}

export class StreamingAnnotator {
  private batcher = new Batcher<Cand>(ANNOTATE_BATCH_SIZE)
  private queue = new PQueue({ concurrency: ANNOTATE_CONCURRENCY })
  private usage: Usage = { calls: 0, input: 0, output: 0, total: 0 }
  private drops: Drops = { batch: 0, missingId: 0, noStyle: 0, noPalette: 0 }
  private fills: Fills = { palette: 0 }
  private today = new Date().toISOString().slice(0, 10)
  private bar: cliProgress.SingleBar
  generated = 0
  appended = 0
  noLabel = 0
  noPalette = 0
  hitTarget = 0
  missTarget = 0

  constructor(
    private src: string,
    multibar: cliProgress.MultiBar,
    totalHint: number,
  ) {
    this.bar = multibar.create(
      Math.max(1, Math.ceil(totalHint / ANNOTATE_BATCH_SIZE)),
      0,
      {},
      {
        format:
          "annotating  |{bar}| {percentage}% | {value}/{total} batches | ETA: {eta}s",
      },
    )
  }

  push(cand: Cand): void {
    this.generated++
    this.bar.setTotal(
      Math.max(this.bar.getTotal(), Math.ceil(this.generated / ANNOTATE_BATCH_SIZE)),
    )
    const batch = this.batcher.push(cand)
    if (batch) this.dispatch(batch)
  }

  private dispatch(batch: Cand[]): void {
    this.queue.add(() => this.annotateChunk(batch))
  }

  private async annotateChunk(batch: Cand[]): Promise<void> {
    const items = batch.map((c, id) => ({
      id,
      text: c.text,
      ...(c.sarcastic ? { style_hint: "sarcastic" } : {}),
    }))
    const got = await annotateBatch(items, true, true, this.usage, this.drops, this.fills)
    const lines: string[] = []
    for (const { id } of items) {
      const label = got.get(id)
      if (!label) {
        this.noLabel++
        continue
      }
      if (!label.bg || !label.fg) {
        this.noPalette++
        continue
      }
      const cand = batch[id]
      let emojis: string
      if (cand.target) {
        if (label.emojis.includes(cand.target)) this.hitTarget++
        else this.missTarget++
        emojis = [cand.target, ...label.emojis.filter((e) => e !== cand.target)].join(" ")
      } else {
        emojis = label.emojis.join(" ")
      }
      const row: Record<string, unknown> = {
        text: cand.text,
        emojis,
        styles: label.styles,
        bg: label.bg,
        fg: label.fg,
      }
      if (cand.lang) row.lang = cand.lang
      if (cand.color) row.color = cand.color
      if (cand.group) row.group = cand.group
      row.meta = { date: this.today, src: this.src }
      lines.push(JSON.stringify(row))
    }
    if (lines.length) await appendJsonl(DATA, lines)
    this.appended += lines.length
    this.bar.increment()
  }

  async finish(): Promise<void> {
    const last = this.batcher.flush()
    if (last) this.dispatch(last)
    await this.queue.onIdle()
    this.bar.stop()

    console.log("\n--- summary ---")
    console.log(`generated            : ${this.generated}`)
    console.log(`appended -> data     : ${this.appended}`)
    console.log(`dropped no label     : ${this.noLabel}`)
    console.log(`filled palette       : ${this.fills.palette}`)
    console.log(`dropped no palette   : ${this.noPalette}`)
    if (this.hitTarget + this.missTarget > 0) {
      console.log(
        `target hit / miss    : ${this.hitTarget} / ${this.missTarget} (target injected either way)`,
      )
    }
    console.log(`\n${formatUsage(this.usage)}`)
    console.log(formatDrops(this.drops))
  }
}

async function generateForEmojis(
  targets: string[],
  per: number,
  sink: Sink,
  multibar: cliProgress.MultiBar,
  lang?: string,
  maxLen?: number,
): Promise<void> {
  const genBar = multibar.create(targets.length, 0, {}, {
    format: "generating  |{bar}| {percentage}% | {value}/{total} emojis | ETA: {eta}s",
  })
  const genQ = new PQueue({ concurrency: GEN_CONCURRENCY })
  genQ.addAll(
    targets.map((emoji) => async () => {
      try {
        for (const t of await genBatch(pickVoice(), emoji, per, lang, maxLen)) {
          sink.push({ text: t, target: emoji, lang })
        }
      } catch (err) {
        console.warn(`\n  gen (${emoji}) failed: ${err}`)
      }
      genBar.increment()
    }),
  )
  await genQ.onIdle()
  genBar.stop()
}

async function generateForColors(
  per: number,
  sink: Sink,
  multibar: cliProgress.MultiBar,
  lang?: string,
  maxLen?: number,
): Promise<void> {
  const colorPlan = colorBatchPlan(COLORS, per, COLOR_BATCH)
  console.log(
    `colors mode -> ${per} texts per colour for ${COLORS.join(", ")} `
    + `-> ${COLORS.length * per} texts in ${colorPlan.length} batches of up to ${COLOR_BATCH}${langSuffix(lang)}`,
  )
  const genBar = multibar.create(colorPlan.length, 0, {}, {
    format: "generating  |{bar}| {percentage}% | {value}/{total} batches | ETA: {eta}s",
  })
  const genQ = new PQueue({ concurrency: GEN_CONCURRENCY })
  genQ.addAll(
    colorPlan.map(({ color, n }) => async () => {
      try {
        for (const t of await genColorBatch(pickVoice(), color, n, lang, maxLen)) {
          sink.push({ text: t, color, lang })
        }
      } catch (err) {
        console.warn(`\n  gen (${color}) failed: ${err}`)
      }
      genBar.increment()
    }),
  )
  await genQ.onIdle()
  genBar.stop()
}

async function generateForMotivational(
  count: number,
  sink: Sink,
  multibar: cliProgress.MultiBar,
  lang?: string,
  maxLen?: number,
): Promise<void> {
  const sizes = batchSizes(count, MOTIVATIONAL_BATCH)
  console.log(
    `motivational mode -> ${count} texts in ${sizes.length} batches of up to ${MOTIVATIONAL_BATCH}${langSuffix(lang)}`,
  )
  const genBar = multibar.create(sizes.length, 0, {}, {
    format: "generating  |{bar}| {percentage}% | {value}/{total} batches | ETA: {eta}s",
  })
  const genQ = new PQueue({ concurrency: GEN_CONCURRENCY })
  genQ.addAll(
    sizes.map((n) => async () => {
      try {
        for (const t of await genMotivationalBatch(pickVoice(), n, lang, maxLen)) {
          sink.push({ text: t, lang })
        }
      } catch (err) {
        console.warn(`\n  gen (motivational) failed: ${err}`)
      }
      genBar.increment()
    }),
  )
  await genQ.onIdle()
  genBar.stop()
}

async function generateForLinkedin(
  count: number,
  sink: Sink,
  multibar: cliProgress.MultiBar,
  lang?: string,
  maxLen?: number,
): Promise<void> {
  const sizes = batchSizes(count, LINKEDIN_BATCH)
  console.log(
    `linkedin mode -> ${count} texts in ${sizes.length} batches of up to ${LINKEDIN_BATCH}${langSuffix(lang)}`,
  )
  const genBar = multibar.create(sizes.length, 0, {}, {
    format: "generating  |{bar}| {percentage}% | {value}/{total} batches | ETA: {eta}s",
  })
  const genQ = new PQueue({ concurrency: GEN_CONCURRENCY })
  genQ.addAll(
    sizes.map((n) => async () => {
      try {
        for (const t of await genLinkedinBatch(pickVoice(), n, lang, maxLen)) {
          sink.push({ text: t, lang })
        }
      } catch (err) {
        console.warn(`\n  gen (linkedin) failed: ${err}`)
      }
      genBar.increment()
    }),
  )
  await genQ.onIdle()
  genBar.stop()
}

async function generateForSarcasm(
  count: number,
  sink: Sink,
  multibar: cliProgress.MultiBar,
  lang?: string,
  maxLen?: number,
): Promise<void> {
  const sizes = batchSizes(count, SARCASM_BATCH)
  console.log(
    `sarcasm mode -> ${count} texts in ${sizes.length} batches of up to ${SARCASM_BATCH}${langSuffix(lang)}`,
  )
  const genBar = multibar.create(sizes.length, 0, {}, {
    format: "generating  |{bar}| {percentage}% | {value}/{total} batches | ETA: {eta}s",
  })
  const genQ = new PQueue({ concurrency: GEN_CONCURRENCY })
  genQ.addAll(
    sizes.map((n) => async () => {
      try {
        for (const t of await genSarcasmBatch(pickVoice(), n, lang, maxLen)) {
          sink.push({ text: t, lang, sarcastic: true })
        }
      } catch (err) {
        console.warn(`\n  gen (sarcasm) failed: ${err}`)
      }
      genBar.increment()
    }),
  )
  await genQ.onIdle()
  genBar.stop()
}

async function generateForTopics(
  count: number,
  sink: Sink,
  multibar: cliProgress.MultiBar,
  lang?: string,
  maxLen?: number,
): Promise<void> {
  const sizes = batchSizes(count, TOPIC_BATCH)
  console.log(
    `topics mode -> ${count} texts in ${sizes.length} batches of up to ${TOPIC_BATCH}${langSuffix(lang)}`,
  )
  const genBar = multibar.create(sizes.length, 0, {}, {
    format: "generating  |{bar}| {percentage}% | {value}/{total} batches | ETA: {eta}s",
  })
  const genQ = new PQueue({ concurrency: GEN_CONCURRENCY })
  genQ.addAll(
    sizes.map((n, i) => async () => {
      const topic = topicForBatch(i)
      try {
        for (const t of await genTopicBatch(topic, pickVoice(), n, lang, maxLen)) {
          sink.push({ text: t, lang })
        }
      } catch (err) {
        console.warn(`\n  gen (${topic}) failed: ${err}`)
      }
      genBar.increment()
    }),
  )
  await genQ.onIdle()
  genBar.stop()
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

function parseMaxLen(raw: unknown): number {
  const maxLen = Number(raw ?? MAX_LEN)
  if (!(maxLen >= MIN_LEN)) {
    console.error(`--max-len must be >= ${MIN_LEN}, got ${JSON.stringify(raw)}`)
    process.exit(1)
  }
  return maxLen
}

function printSamplePrompt(prompt: string): void {
  console.log(`\n--- sample prompt ---\n${prompt}`)
}

const cli = cac("upsample")

cli.option(
  "--lang <code>",
  `generate text in this language (ISO 639-1 code, e.g. 'he' for Hebrew); `
  + `omit to upsample in every language (${LANGS.join(", ")})`,
)

cli.option(
  "--max-len <n>",
  `maximum characters per generated message (default ${MAX_LEN})`,
)

cli
  .command(
    "",
    `generate generic texts across rotating everyday topics and annotate them `
    + `(default mode, same as the old \`bun run train\`; ${DEFAULT_COUNT} texts unless --count `
    + `is given; a single untagged pass unless --lang names one language - unlike other modes, `
    + `omitting --lang here does not fan out to every language)`,
  )
  .option("--count <n>", `messages to generate (default ${DEFAULT_COUNT})`)
  .option("--dry", "report what would be upsampled, then exit without generating, annotating, or appending")
  .action(async (options) => {
    const count = parseCount(options.count, DEFAULT_COUNT)
    const lang = parseLang(options.lang)
    const maxLen = options.maxLen !== undefined ? parseMaxLen(options.maxLen) : undefined
    if (options.dry) {
      console.log("\n--- dry run: nothing generated, annotated, or appended ---")
      console.log(`mode                 : topics`)
      console.log(`would generate       : ${count} texts${langSuffix(lang)}`)
      printSamplePrompt(
        genTopicPrompt(topicForBatch(0), pickVoice(), Math.min(count, TOPIC_BATCH), lang, maxLen),
      )
      return
    }
    const multibar = new cliProgress.MultiBar(
      { clearOnComplete: false, hideCursor: true },
      cliProgress.Presets.shades_classic,
    )
    const sink = new StreamingAnnotator("topics", multibar, count)
    await generateForTopics(count, sink, multibar, lang, maxLen)
    await sink.finish()
    multibar.stop()
  })

cli
  .command(
    "lang",
    `upsample the least-represented language in ${DATA} with rotating-topic texts `
    + `(picks among ${LANGS.join(", ")}; ignores --lang)`,
  )
  .option("--count <n>", `messages to generate (default ${DEFAULT_COUNT})`)
  .option("--dry", "report what would be upsampled, then exit without generating, annotating, or appending")
  .action(async (options) => {
    const count = parseCount(options.count, DEFAULT_COUNT)
    const maxLen = options.maxLen !== undefined ? parseMaxLen(options.maxLen) : undefined
    const rows = await readJsonl<{ text?: string; lang?: unknown }>(DATA)
    const counts = countLangs(rows)
    const lang = leastFrequentLang(counts)
    console.log(
      LANGS.map((l) => `${l}: ${counts.get(l) ?? 0}`).join(", ")
      + ` -> upsampling least represented: ${lang}`,
    )
    if (options.dry) {
      console.log("\n--- dry run: nothing generated, annotated, or appended ---")
      console.log(`mode                 : lang`)
      console.log(`would generate       : ${count} texts (lang: ${lang})`)
      printSamplePrompt(
        genTopicPrompt(topicForBatch(0), pickVoice(), Math.min(count, TOPIC_BATCH), lang, maxLen),
      )
      return
    }
    const multibar = new cliProgress.MultiBar(
      { clearOnComplete: false, hideCursor: true },
      cliProgress.Presets.shades_classic,
    )
    const sink = new StreamingAnnotator("lang", multibar, count)
    await generateForTopics(count, sink, multibar, lang, maxLen)
    await sink.finish()
    multibar.stop()
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
    const langs = resolveLangs(parseLang(options.lang))
    const maxLen = parseMaxLen(options.maxLen)
    console.log(`targeting ${targets.length} emoji -> ${targets.join(" ")}`)
    if (options.dry) {
      console.log("\n--- dry run: nothing generated, annotated, or appended ---")
      console.log(`mode                 : emojis`)
      console.log(
        `would generate       : ~${targets.length * per * langs.length} texts `
        + `(${per}/emoji x ${langs.length} lang(s))${langsSuffix(langs)}`,
      )
      printSamplePrompt(genPrompt(pickVoice(), targets[0], per, langs[0], maxLen))
      return
    }
    const multibar = new cliProgress.MultiBar(
      { clearOnComplete: false, hideCursor: true },
      cliProgress.Presets.shades_classic,
    )
    const sink = new StreamingAnnotator("emoji-target", multibar, targets.length * per * langs.length)
    for (const lang of langs) await generateForEmojis(targets, per, sink, multibar, lang, maxLen)
    await sink.finish()
    multibar.stop()
  })

cli
  .command("colors", `upsample texts for each of ${COLORS.join(", ")}`)
  .option("--per <n>", `texts to generate per colour (default ${TEXTS_PER_EMOJI})`)
  .option("--dry", "report what would be upsampled, then exit without generating, annotating, or appending")
  .action(async (options) => {
    const per = parsePer(options.per)
    const langs = resolveLangs(parseLang(options.lang))
    const maxLen = parseMaxLen(options.maxLen)
    if (options.dry) {
      const colorPlan = colorBatchPlan(COLORS, per, COLOR_BATCH)
      console.log("\n--- dry run: nothing generated, annotated, or appended ---")
      console.log(`mode                 : colors`)
      console.log(
        `would generate       : ${colorPlan.reduce((s, b) => s + b.n, 0) * langs.length} texts `
        + `over ${COLORS.length} colours x ${langs.length} lang(s)${langsSuffix(langs)}`,
      )
      printSamplePrompt(genColorPrompt(pickVoice(), COLORS[0], colorPlan[0].n, langs[0], maxLen))
      return
    }
    const multibar = new cliProgress.MultiBar(
      { clearOnComplete: false, hideCursor: true },
      cliProgress.Presets.shades_classic,
    )
    const sink = new StreamingAnnotator("colors", multibar, COLORS.length * per * langs.length)
    for (const lang of langs) await generateForColors(per, sink, multibar, lang, maxLen)
    await sink.finish()
    multibar.stop()
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
    const langs = resolveLangs(parseLang(options.lang))
    const maxLen = parseMaxLen(options.maxLen)
    if (options.dry) {
      console.log("\n--- dry run: nothing generated, annotated, or appended ---")
      console.log(`mode                 : motivational`)
      console.log(`would generate       : ${count} texts x ${langs.length} lang(s)${langsSuffix(langs)}`)
      printSamplePrompt(
        genMotivationalPrompt(pickVoice(), Math.min(count, MOTIVATIONAL_BATCH), langs[0], maxLen),
      )
      return
    }
    const multibar = new cliProgress.MultiBar(
      { clearOnComplete: false, hideCursor: true },
      cliProgress.Presets.shades_classic,
    )
    const sink = new StreamingAnnotator("motivational", multibar, count * langs.length)
    for (const lang of langs) await generateForMotivational(count, sink, multibar, lang, maxLen)
    await sink.finish()
    multibar.stop()
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
    const langs = resolveLangs(parseLang(options.lang))
    const maxLen = parseMaxLen(options.maxLen)
    if (options.dry) {
      console.log("\n--- dry run: nothing generated, annotated, or appended ---")
      console.log(`mode                 : linkedin`)
      console.log(`would generate       : ${count} texts x ${langs.length} lang(s)${langsSuffix(langs)}`)
      printSamplePrompt(
        genLinkedinPrompt(pickVoice(), Math.min(count, LINKEDIN_BATCH), langs[0], maxLen),
      )
      return
    }
    const multibar = new cliProgress.MultiBar(
      { clearOnComplete: false, hideCursor: true },
      cliProgress.Presets.shades_classic,
    )
    const sink = new StreamingAnnotator("linkedin", multibar, count * langs.length)
    for (const lang of langs) await generateForLinkedin(count, sink, multibar, lang, maxLen)
    await sink.finish()
    multibar.stop()
  })

cli
  .command(
    "sarcasm",
    "generate sarcastic/ironic messages",
  )
  .option("--count <n>", `messages to generate (default ${SARCASM_COUNT})`)
  .option("--dry", "report what would be upsampled, then exit without generating, annotating, or appending")
  .action(async (options) => {
    const count = parseCount(options.count, SARCASM_COUNT)
    const langs = resolveLangs(parseLang(options.lang))
    const maxLen = parseMaxLen(options.maxLen)
    if (options.dry) {
      console.log("\n--- dry run: nothing generated, annotated, or appended ---")
      console.log(`mode                 : sarcasm`)
      console.log(`would generate       : ${count} texts x ${langs.length} lang(s)${langsSuffix(langs)}`)
      printSamplePrompt(
        genSarcasmPrompt(pickVoice(), Math.min(count, SARCASM_BATCH), langs[0], maxLen),
      )
      return
    }
    const multibar = new cliProgress.MultiBar(
      { clearOnComplete: false, hideCursor: true },
      cliProgress.Presets.shades_classic,
    )
    const sink = new StreamingAnnotator("sarcasm", multibar, count * langs.length)
    for (const lang of langs) await generateForSarcasm(count, sink, multibar, lang, maxLen)
    await sink.finish()
    multibar.stop()
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
    const langs = resolveLangs(parseLang(options.lang))
    const maxLen = parseMaxLen(options.maxLen)
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
      console.log(
        `would generate       : ~${targets.length * per * langs.length} texts `
        + `(${per}/emoji x ${langs.length} lang(s))${langsSuffix(langs)}`,
      )
      printSamplePrompt(genPrompt(pickVoice(), targets[0], per, langs[0], maxLen))
      return
    }
    const multibar = new cliProgress.MultiBar(
      { clearOnComplete: false, hideCursor: true },
      cliProgress.Presets.shades_classic,
    )
    const sink = new StreamingAnnotator("group", multibar, targets.length * per * langs.length)
    const groupedSink: Sink = {
      push: (c) => sink.push(c.target ? { ...c, group: groupOf.get(c.target) } : c),
    }
    for (const lang of langs) await generateForEmojis(targets, per, groupedSink, multibar, lang, maxLen)
    await sink.finish()
    multibar.stop()
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
    const langs = resolveLangs(parseLang(options.lang))
    const maxLen = parseMaxLen(options.maxLen)
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
      console.log(
        `would generate       : ~${targets.length * per * langs.length} texts `
        + `(${per}/emoji x ${langs.length} lang(s))${langsSuffix(langs)}`,
      )
      printSamplePrompt(genPrompt(pickVoice(), targets[0], per, langs[0], maxLen))
      return
    }
    const multibar = new cliProgress.MultiBar(
      { clearOnComplete: false, hideCursor: true },
      cliProgress.Presets.shades_classic,
    )
    const sink = new StreamingAnnotator("balance", multibar, targets.length * per * langs.length)
    for (const lang of langs) await generateForEmojis(targets, per, sink, multibar, lang, maxLen)
    await sink.finish()
    multibar.stop()
  })

cli.help()

if (import.meta.main) {
  cli.parse()
}
