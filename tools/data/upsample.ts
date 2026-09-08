import { readdirSync } from "node:fs"
import { readFile } from "node:fs/promises"
import { join } from "node:path"

import { generateText } from "ai"
import { cac } from "cac"
import cliProgress from "cli-progress"
import PQueue from "p-queue"

import { DATA_JSONL as DATA, REPORT_DIR } from "../../files.ts"
import { MODEL, annotate, annotateBatchCount, lastFills } from "./annotate.ts"
import { splitEmojis } from "./emoji.ts"
import { appendJsonl, readJsonl } from "./io.ts"
import { normalize } from "./normalize.ts"

const MIN_RANK = 600
const MAX_RANK = 800
const RARE_MIN_FREQ = 10
const RARE_MAX_COUNT = 100
const TEXTS_PER_EMOJI = 50
const NEG_COUNT = 1000
const SHORT_COUNT = 2000
const SINGLE_EMOJI_COUNT = 5000
const KEYWORDS_PER = 50
const COLOR_PER = 1000
const COLOR_BATCH = 50
const MIN_LEN = 4
const MAX_LEN = 42
const GEN_CONCURRENCY = 20

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

export function singleEmojiTexts(
  rows: { text?: unknown; emojis?: unknown }[],
  count: number,
): string[] {
  const perText = new Map<string, number>()
  for (const row of rows) {
    if (typeof row.text !== "string") continue
    const key = normalize(row.text)
    if (!key) continue
    perText.set(key, (perText.get(key) ?? 0) + 1)
  }
  const out: string[] = []
  for (const row of rows) {
    if (out.length >= count) break
    if (typeof row.text !== "string" || typeof row.emojis !== "string") continue
    const key = normalize(row.text)
    if (!key || perText.get(key) !== 1) continue
    if (new Set(splitEmojis(row.emojis)).size > 1) continue
    out.push(row.text)
  }
  return out
}

export function parseKeywords(s: string): string[] {
  const out: string[] = []
  const seen = new Set<string>()
  for (const raw of s.split(",")) {
    const kw = raw.trim()
    if (!kw || seen.has(kw)) continue
    seen.add(kw)
    out.push(kw)
  }
  return out
}

export function rankWindow(
  counts: Map<string, number>,
  minRank: number,
  maxRank: number,
): string[] {
  return [...counts.keys()]
    .map((k, i) => ({ k, i, c: counts.get(k) ?? 0 }))
    .sort((a, b) => b.c - a.c || a.i - b.i)
    .slice(Math.max(0, minRank - 1), maxRank)
    .map((x) => x.k)
}

export function rareEmojis(
  counts: Map<string, number>,
  minFreq: number,
  maxCount: number,
): string[] {
  return [...counts.keys()]
    .map((k, i) => ({ k, i, c: counts.get(k) ?? 0 }))
    .filter((x) => x.c >= minFreq)
    .sort((a, b) => a.c - b.c || a.i - b.i)
    .slice(0, Math.max(0, maxCount))
    .map((x) => x.k)
}

export function batchSizes(total: number, per: number): number[] {
  const out: number[] = []
  for (let left = total; left > 0; left -= per) out.push(Math.min(per, left))
  return out
}

export function weightedMedian(pairs: [number, number][]): number {
  const total = pairs.reduce((s, [, c]) => s + c, 0)
  if (total <= 0) throw new Error("weightedMedian: empty distribution")
  let acc = 0
  for (const [len, c] of [...pairs].sort((a, b) => a[0] - b[0])) {
    acc += c
    if (acc * 2 >= total) return len
  }
  return pairs[pairs.length - 1][0]
}

async function latestReportMedianLen(): Promise<number> {
  let dirs: string[] = []
  try {
    dirs = readdirSync(REPORT_DIR, { withFileTypes: true })
      .filter((d) => d.isDirectory())
      .map((d) => d.name)
      .sort()
  } catch {
    dirs = []
  }
  for (const name of dirs.reverse()) {
    try {
      const report = JSON.parse(
        await readFile(join(REPORT_DIR, name, "report.json"), "utf8"),
      ) as { data?: { length_distribution?: [number, number][] } }
      const dist = report.data?.length_distribution
      if (dist?.length) return weightedMedian(dist)
    } catch {
      continue
    }
  }
  throw new Error(
    `no ${REPORT_DIR}/*/report.json with data.length_distribution - run tools/report.py first`,
  )
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

function genPrompt(voice: string, emoji: string, per: number): string {
  return [
    `Write ${per} short text messages as if sent by ${voice}, one per line.`,
    `Each message between ${MIN_LEN} and ${MAX_LEN} characters.`,
    `Every message must read naturally as one a person would send together with`,
    `the emoji ${emoji} - its subject, activity, place, or mood fits that emoji.`,
    `Do not put any emoji in the output, and never name or describe the emoji.`,
    `Vary sender, tone, and intent: updates, questions, complaints, plans,`,
    `reactions, reminders, small talk. Sound real and specific.`,
    `No numbering, no bullets, no quotes, no commentary.`,
  ].join("\n")
}

function genKeywordPrompt(voice: string, keyword: string, per: number): string {
  return [
    `Write ${per} short text messages as if sent by ${voice}, one per line.`,
    `Each message between ${MIN_LEN} and ${MAX_LEN} characters.`,
    `Every message must naturally use the word or phrase "${keyword}".`,
    `Do not put any emoji in the output.`,
    `Vary sender, tone, and intent: updates, questions, complaints, plans,`,
    `reactions, reminders, small talk. Sound real and specific.`,
    `No numbering, no bullets, no quotes, no commentary.`,
  ].join("\n")
}

function genColorPrompt(voice: string, color: string, per: number): string {
  return [
    `Write ${per} short text messages as if sent by ${voice}, one per line.`,
    `Each message between ${MIN_LEN} and ${MAX_LEN} characters.`,
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

function genNegationPrompt(voice: string, per: number): string {
  return [
    `Write ${per} short text messages as if sent by ${voice}, one per line.`,
    `Each message between ${MIN_LEN} and ${MAX_LEN} characters.`,
    `Every message must be built on negation: a negated opinion ("not good",`,
    `"don't like it"), a refusal ("no thanks", "never again"), a denial, or a`,
    `correction ("it isn't a dog it's a cat", "that's not what I meant").`,
    `Use negation words like not, n't, no, never, none, nothing, without.`,
    `Do not put any emoji in the output.`,
    `Vary sender, tone, and intent: complaints, corrections, refusals, denials,`,
    `disappointed reactions, small talk. Sound real and specific.`,
    `No numbering, no bullets, no quotes, no commentary.`,
  ].join("\n")
}

function genShortPrompt(voice: string, per: number, maxLen: number): string {
  return [
    `Write ${per} short text messages as if sent by ${voice}, one per line.`,
    `Each message between ${MIN_LEN} and ${maxLen} characters - keep them brief.`,
    `Do not put any emoji in the output.`,
    `Vary sender, tone, and intent: updates, questions, complaints, plans,`,
    `reactions, reminders, small talk. Cover positive, negative, and flat moods.`,
    `Sound real and specific; do not lean on any single persona or sentence shape.`,
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
): Promise<string[]> {
  const { text } = await generateText({
    model: MODEL,
    prompt: genPrompt(voice, emoji, per),
  })
  return cleanLines(text)
}

async function genNegationBatch(voice: string, per: number): Promise<string[]> {
  const { text } = await generateText({
    model: MODEL,
    prompt: genNegationPrompt(voice, per),
  })
  return cleanLines(text)
}

async function genShortBatch(
  voice: string,
  per: number,
  maxLen: number,
): Promise<string[]> {
  const { text } = await generateText({
    model: MODEL,
    prompt: genShortPrompt(voice, per, maxLen),
  })
  return cleanLines(text)
}

async function genColorBatch(
  voice: string,
  color: string,
  per: number,
): Promise<string[]> {
  const { text } = await generateText({
    model: MODEL,
    prompt: genColorPrompt(voice, color, per),
  })
  return cleanLines(text)
}

async function genKeywordBatch(
  voice: string,
  keyword: string,
  per: number,
): Promise<string[]> {
  const { text } = await generateText({
    model: MODEL,
    prompt: genKeywordPrompt(voice, keyword, per),
  })
  return cleanLines(text)
}

const cli = cac("upsample")
cli.usage("[options]")
cli
  .option("--emojis <list>", "target exactly these emoji instead of a rank window")
  .option("--min-rank <n>", `lowest (most frequent) rank to target (default ${MIN_RANK})`)
  .option("--max-rank <n>", `highest (least frequent) rank to target (default ${MAX_RANK})`)
  .option("--rare", "target the rarest emoji in data.jsonl first, by record count (not standalone)")
  .option("--min-freq <n>", `with --rare, skip emoji with fewer than this many records (default ${RARE_MIN_FREQ})`)
  .option("--max-count <n>", `with --rare, how many of the rarest emoji to target - a target count, not a freq cap (default ${RARE_MAX_COUNT})`)
  .option("--iter <n>", "with --rare, repeat the whole select/generate/annotate/append cycle this many times, recomputing the rarest set each pass (default 1)")
  .option("--keywords <list>", "standalone: generate texts using each comma-separated keyword, one keyword at a time (ignores emoji targeting)")
  .option("--per <n>", `texts to generate per target emoji / keyword / batch (default ${TEXTS_PER_EMOJI}, ${KEYWORDS_PER} with --keywords, ${COLOR_PER} per colour with --colors, ${TEXTS_PER_EMOJI} with --rare)`)
  .option("--negation", "standalone: generate negation-heavy texts (ignores emoji targeting)")
  .option("--short", "standalone: generate short texts capped at the last report's median length (ignores emoji targeting)")
  .option("--single-emoji", "standalone: re-annotate corpus rows that carry at most one emoji (ignores emoji targeting)")
  .option("--colors", `standalone: generate texts related to each of ${COLORS.join(", ")} (ignores emoji targeting)`)
  .option("--count <n>", `cap on texts for --negation (default ${NEG_COUNT}) / --short (default ${SHORT_COUNT}) / --single-emoji (default ${SINGLE_EMOJI_COUNT})`)
  .option("--dry", "report what would be upsampled, then exit without generating, annotating, or appending")
cli.help()

if (import.meta.main) {
  const { options } = cli.parse(process.argv, { run: false })
  if (options.help) process.exit(0)
  const only = options.emojis ? String(options.emojis).trim() || undefined : undefined
  const minRank = Number(options.minRank ?? MIN_RANK)
  const maxRank = Number(options.maxRank ?? MAX_RANK)
  const negation = Boolean(options.negation)
  const short = Boolean(options.short)
  const singleEmoji = Boolean(options.singleEmoji)
  const colors = Boolean(options.colors)
  const rare = Boolean(options.rare)
  const dry = Boolean(options.dry)
  const minFreq = Number(options.minFreq ?? RARE_MIN_FREQ)
  const maxCount = Number(options.maxCount ?? RARE_MAX_COUNT)
  const iterRaw = Number(options.iter ?? 1)
  const iters =
    rare && !dry && Number.isFinite(iterRaw) && iterRaw >= 1
      ? Math.floor(iterRaw)
      : 1
  const kw = options.keywords != null
  const kwList = kw ? parseKeywords(String(options.keywords)) : []
  const per = Number(
    options.per
    ?? (kw ? KEYWORDS_PER : colors ? COLOR_PER : TEXTS_PER_EMOJI),
  )
  const count = Number(
    options.count
    ?? (singleEmoji ? SINGLE_EMOJI_COUNT : short ? SHORT_COUNT : NEG_COUNT),
  )

  if ([negation, short, singleEmoji, colors, kw].filter(Boolean).length > 1) {
    console.error(
      "--negation, --short, --single-emoji, --colors and --keywords are mutually exclusive",
    )
    process.exit(1)
  }
  const standalone = negation || short || singleEmoji || colors || kw
  if (
    rare
    && (standalone || only || options.minRank != null || options.maxRank != null)
  ) {
    console.error(
      "--rare cannot be combined with a standalone mode / --emojis / --min-rank / --max-rank",
    )
    process.exit(1)
  }
  if (
    !rare
    && (options.minFreq != null
      || options.maxCount != null
      || options.iter != null)
  ) {
    console.warn("--min-freq / --max-count / --iter only apply with --rare")
  }
  if (rare && options.iter != null && !(Number.isFinite(iterRaw) && iterRaw >= 1)) {
    console.error(`--iter must be a number >= 1, got ${JSON.stringify(options.iter)}`)
    process.exit(1)
  }
  if (rare && !(minFreq >= 0)) {
    console.error(`--min-freq must be >= 0, got ${JSON.stringify(options.minFreq)}`)
    process.exit(1)
  }
  if (rare && !(maxCount >= 1)) {
    console.error(`--max-count must be >= 1, got ${JSON.stringify(options.maxCount)}`)
    process.exit(1)
  }
  if (dry && rare && options.iter != null && iterRaw > 1) {
    console.warn(
      "--dry reports only the first iteration (later passes depend on appended rows)",
    )
  }
  const standaloneName = negation
    ? "negation"
    : short
      ? "short"
      : singleEmoji
        ? "single-emoji"
        : colors
          ? "colors"
          : "keywords"
  if (standalone && (only || options.minRank != null || options.maxRank != null)) {
    console.warn(
      `--${standaloneName} ignores --emojis / --min-rank / --max-rank`,
    )
  }
  if (singleEmoji && options.per != null) {
    console.warn("--single-emoji ignores --per")
  }
  if ((!standalone || colors || kw) && options.count != null) {
    console.warn("--count only applies with --negation / --short / --single-emoji")
  }
  if (!(per >= 1)) {
    console.error(`--per must be >= 1, got ${JSON.stringify(options.per)}`)
    process.exit(1)
  }

  for (let iterIdx = 0; iterIdx < iters; iterIdx++) {
    let targets: string[]
    let negBatches: number[] = []
    let shortBatches: number[] = []
    let shortMedianLen = 0
    let singleTexts: string[] = []
    let colorPlan: { color: string; n: number }[] = []
    if (singleEmoji) {
      if (!(count >= 1)) {
        console.error(`--count must be >= 1, got ${JSON.stringify(options.count)}`)
        process.exit(1)
      }
      const rows = await readJsonl<{ text?: unknown; emojis?: unknown }>(DATA)
      singleTexts = singleEmojiTexts(rows, count)
      targets = []
      console.log(
        `single-emoji mode -> ${rows.length} master rows -> `
        + `${singleTexts.length} unique rows (<=1 emoji) to re-annotate `
        + `(cap ${count})`,
      )
      if (!singleTexts.length) {
        console.error("no rows matched --single-emoji")
        process.exit(1)
      }
    } else if (negation) {
      if (!(count >= 1)) {
        console.error(`--count must be >= 1, got ${JSON.stringify(options.count)}`)
        process.exit(1)
      }
      targets = []
      negBatches = batchSizes(count, per)
      console.log(
        `negation mode -> generating ${count} texts in ${negBatches.length} `
        + `batches of up to ${per}`,
      )
    } else if (short) {
      if (!(count >= 1)) {
        console.error(`--count must be >= 1, got ${JSON.stringify(options.count)}`)
        process.exit(1)
      }
      targets = []
      shortMedianLen = await latestReportMedianLen()
      shortBatches = batchSizes(count, per)
      console.log(
        `short mode -> generating ${count} texts (<= ${shortMedianLen} chars, `
        + `last report's median) in ${shortBatches.length} batches of up to ${per}`,
      )
    } else if (colors) {
      targets = []
      colorPlan = colorBatchPlan(COLORS, per, COLOR_BATCH)
      console.log(
        `colors mode -> ${per} texts per colour for ${COLORS.join(", ")} `
        + `-> ${COLORS.length * per} texts in ${colorPlan.length} batches of up to ${COLOR_BATCH}`,
      )
    } else if (kw) {
      targets = []
      if (!kwList.length) {
        console.error(`--keywords had no usable keyword: ${JSON.stringify(options.keywords)}`)
        process.exit(1)
      }
      console.log(
        `keywords mode -> ${per} texts each for ${kwList.length} keywords `
        + `-> ${kwList.join(", ")}`,
      )
    } else if (rare) {
      const rows = await readJsonl<{ emojis?: string }>(DATA)
      const counts = countEmojis(rows)
      const eligible = [...counts.values()].filter((c) => c >= minFreq).length
      targets = rareEmojis(counts, minFreq, maxCount)
      if (!targets.length) {
        console.error(`no emoji in data has >= ${minFreq} records to upsample`)
        process.exit(1)
      }
      console.log(
        `${counts.size} distinct emoji in data -> ${eligible} with >= ${minFreq} `
        + `records -> targeting ${targets.length} rarest `
        + `(${counts.get(targets[0])}..${counts.get(targets.at(-1) ?? "")} rows each)`
        + (iters > 1 ? ` [iter ${iterIdx + 1}/${iters}]` : ""),
      )
    } else if (only) {
      targets = [...new Set(splitEmojis(only))]
      if (!targets.length) {
        console.error(`--emojis had no recognizable emoji: ${JSON.stringify(only)}`)
        process.exit(1)
      }
      console.log(`targeting ${targets.length} emoji -> ${targets.join(" ")}`)
    } else {
      const rows = await readJsonl<{ emojis?: string }>(DATA)
      const counts = countEmojis(rows)
      targets = rankWindow(counts, minRank, maxRank)
      console.log(
        `${counts.size} distinct emoji in data -> targeting ${targets.length} ranked `
        + `${minRank}-${maxRank} (${counts.get(targets[0])}..`
        + `${counts.get(targets.at(-1) ?? "")} rows each)`,
      )
    }

    const mode = negation
      ? "negation"
      : short
        ? "short"
        : singleEmoji
          ? "single-emoji"
          : colors
            ? "colors"
            : kw
              ? "keywords"
              : rare
                ? "rare"
                : "emoji-target"

    if (dry) {
      console.log("\n--- dry run: nothing generated, annotated, or appended ---")
      console.log(`mode                 : ${mode}`)
      if (!standalone) {
        console.log(`targets (${targets.length})`.padEnd(21) + `: ${targets.join(" ")}`)
        console.log(`would generate       : ~${targets.length * per} texts (${per}/emoji)`)
      } else if (negation) {
        console.log(
          `would generate       : ${count} texts in ${negBatches.length} batches of up to ${per}`,
        )
      } else if (short) {
        console.log(
          `median length        : ${shortMedianLen} (last report)`,
        )
        console.log(
          `would generate       : ${count} texts in ${shortBatches.length} batches of up to ${per}`,
        )
      } else if (singleEmoji) {
        console.log(`would re-annotate    : ${singleTexts.length} corpus rows`)
      } else if (colors) {
        console.log(
          `would generate       : ${colorPlan.reduce((s, b) => s + b.n, 0)} texts over ${COLORS.length} colours`,
        )
      } else if (kw) {
        console.log(
          `would generate       : ${kwList.length} keywords x ${per} = ${kwList.length * per} texts`,
        )
      }
      break
    }

    const cands: { text: string; target?: string; color?: string; keyword?: string }[] = []
    if (singleEmoji) {
      for (const t of singleTexts) cands.push({ text: t })
      console.log(`${cands.length} corpus rows selected, annotating`)
    } else {
      const genUnit = negation || short || colors
        ? "batches"
        : kw
          ? "keywords"
          : "emojis"
      const genBar = new cliProgress.SingleBar(
        {
          format:
            `generating |{bar}| {percentage}% | {value}/{total} ${genUnit} | ETA: {eta}s`,
        },
        cliProgress.Presets.shades_classic,
      )

      const genQ = new PQueue({ concurrency: GEN_CONCURRENCY })
      if (negation) {
        genBar.start(negBatches.length, 0)
        genQ.addAll(
          negBatches.map((n) => async () => {
            try {
              for (const t of await genNegationBatch(pickVoice(), n)) {
                cands.push({ text: t })
              }
            } catch (err) {
              console.warn(`\n  gen (negation) failed: ${err}`)
            }
            genBar.increment()
          }),
        )
      } else if (short) {
        genBar.start(shortBatches.length, 0)
        genQ.addAll(
          shortBatches.map((n) => async () => {
            try {
              for (const t of await genShortBatch(pickVoice(), n, shortMedianLen)) {
                cands.push({ text: t })
              }
            } catch (err) {
              console.warn(`\n  gen (short) failed: ${err}`)
            }
            genBar.increment()
          }),
        )
      } else if (kw) {
        genBar.start(kwList.length, 0)
        genQ.addAll(
          kwList.map((keyword) => async () => {
            try {
              for (const t of await genKeywordBatch(pickVoice(), keyword, per)) {
                cands.push({ text: t, keyword })
              }
            } catch (err) {
              console.warn(`\n  gen (${keyword}) failed: ${err}`)
            }
            genBar.increment()
          }),
        )
      } else if (colors) {
        genBar.start(colorPlan.length, 0)
        genQ.addAll(
          colorPlan.map(({ color, n }) => async () => {
            try {
              for (const t of await genColorBatch(pickVoice(), color, n)) {
                cands.push({ text: t, color })
              }
            } catch (err) {
              console.warn(`\n  gen (${color}) failed: ${err}`)
            }
            genBar.increment()
          }),
        )
      } else {
        genBar.start(targets.length, 0)
        genQ.addAll(
          targets.map((emoji) => async () => {
            try {
              for (const t of await genBatch(pickVoice(), emoji, per)) {
                cands.push({ text: t, target: emoji })
              }
            } catch (err) {
              console.warn(`\n  gen (${emoji}) failed: ${err}`)
            }
            genBar.increment()
          }),
        )
      }
      await genQ.onIdle()
      genBar.stop()

      console.log(`\n${cands.length} texts generated, annotating`)
    }

    const annBar = new cliProgress.SingleBar(
      {
        format:
          "annotating |{bar}| {percentage}% | {value}/{total} batches | ETA: {eta}s",
      },
      cliProgress.Presets.shades_classic,
    )
    annBar.start(annotateBatchCount(cands.length), 0)
    const labels = await annotate(
      cands.map((c) => c.text),
      { colors: true, fillPalette: true, onBatchDone: () => annBar.increment() },
    )
    annBar.stop()

    const today = new Date().toISOString().slice(0, 10)
    const lines: string[] = []
    let noLabel = 0
    let noPalette = 0
    let noEmoji = 0
    let hitTarget = 0
    let missTarget = 0
    for (let i = 0; i < cands.length; i++) {
      const label = labels.get(i)
      if (!label) {
        noLabel++
        continue
      }
      if (!label.bg || !label.fg) {
        noPalette++
        continue
      }
      if (singleEmoji && !label.emojis.length) {
        noEmoji++
        continue
      }
      const target = cands[i].target
      let emojis: string
      if (target) {
        if (label.emojis.includes(target)) hitTarget++
        else missTarget++
        emojis = [target, ...label.emojis.filter((e) => e !== target)].join(" ")
      } else {
        emojis = label.emojis.join(" ")
      }
      const meta: Record<string, unknown> = { date: today }
      if (singleEmoji) meta["single-emoji"] = true
      if (short) meta.src = "short"
      if (colors) meta.src = "colors"
      if (kw) meta.src = "keywords"
      if (rare) meta.src = "rare"
      const row: Record<string, unknown> = {
        text: cands[i].text,
        emojis,
        styles: label.styles,
        bg: label.bg,
        fg: label.fg,
      }
      if (negation) row.neg = "true"
      if (colors) row.color = cands[i].color
      if (kw) row.keyword = cands[i].keyword
      row.meta = meta
      lines.push(JSON.stringify(row))
    }
    await appendJsonl(DATA, lines)

    console.log("\n--- summary ---")
    console.log(
      `mode                 : ${mode}`
      + (iters > 1 ? ` (iter ${iterIdx + 1}/${iters})` : ""),
    )
    if (!standalone) console.log(`targets              : ${targets.length}`)
    if (kw) console.log(`keywords             : ${kwList.length}`)
    if (colors) {
      const perColor = new Map<string, number>()
      for (const l of lines) {
        const c = (JSON.parse(l) as { color?: string }).color
        if (c) perColor.set(c, (perColor.get(c) ?? 0) + 1)
      }
      console.log(
        `per colour appended  : `
        + COLORS.map((c) => `${c} ${perColor.get(c) ?? 0}`).join(", "),
      )
    }
    console.log(`${(singleEmoji ? "selected" : "generated").padEnd(21)}: ${cands.length}`)
    console.log(`appended -> data     : ${lines.length}`)
    console.log(`dropped no label     : ${noLabel}`)
    console.log(`filled palette       : ${lastFills.palette}`)
    console.log(`dropped no palette   : ${noPalette}`)
    if (singleEmoji) console.log(`dropped no emoji     : ${noEmoji}`)
    if (!standalone) {
      console.log(
        `target hit / miss    : ${hitTarget} / ${missTarget} `
        + `(target injected either way)`,
      )
    }
  }
  process.exit(0)
}
