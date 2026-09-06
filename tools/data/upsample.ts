import { readFile } from "node:fs/promises"

import { generateText } from "ai"
import { cac } from "cac"
import cliProgress from "cli-progress"
import PQueue from "p-queue"

import { DATA_JSONL as DATA } from "../../files.ts"
import { MODEL, annotate, annotateBatchCount, lastFills } from "./annotate.ts"
import { splitEmojis } from "./emoji.ts"
import { appendJsonl, readJsonl } from "./io.ts"
import { normalize } from "./normalize.ts"
import { type Miss, type Report, latestReport } from "./report.ts"

const MIN_RANK = 200
const MAX_RANK = 400
const FAIL_RANK = 5
const TEXTS_PER_EMOJI = 40
const NEG_COUNT = 1000
const SINGLE_EMOJI_COUNT = 5000
const CLDR_PER = 10
const CLDR_KEYWORDS = 500
const MIN_LEN = 4
const MAX_LEN = 42
const GEN_CONCURRENCY = 20

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

export function failingEmojis(misses: Miss[], maxRank: number): string[] {
  const out: string[] = []
  const seen = new Set<string>()
  for (const m of misses) {
    if (!m.target || seen.has(m.target)) continue
    if (m.rank == null || m.rank > maxRank) {
      seen.add(m.target)
      out.push(m.target)
    }
  }
  return out
}

export function missedCldrKeywords(
  misses: Miss[],
): { keyword: string; targets: string[] }[] {
  const map = new Map<string, string[]>()
  for (const m of misses) {
    if (!m.keyword || !m.target) continue
    const targets = map.get(m.keyword)
    if (targets) {
      if (!targets.includes(m.target)) targets.push(m.target)
    } else {
      map.set(m.keyword, [m.target])
    }
  }
  return [...map].map(([keyword, targets]) => ({ keyword, targets }))
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

export function batchSizes(total: number, per: number): number[] {
  const out: number[] = []
  for (let left = total; left > 0; left -= per) out.push(Math.min(per, left))
  return out
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

function genCldrPrompt(
  voice: string,
  keyword: string,
  targets: string[],
  per: number,
): string {
  return [
    `Write ${per} short text messages as if sent by ${voice}, one per line.`,
    `Each message between ${MIN_LEN} and ${MAX_LEN} characters.`,
    `Every message must naturally use or clearly evoke "${keyword}", in a`,
    `context that fits at least one of these emoji: ${targets.join(" ")}.`,
    `Do not put any emoji in the output, and never name or describe the emoji.`,
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

async function genCldrBatch(
  voice: string,
  keyword: string,
  targets: string[],
  per: number,
): Promise<string[]> {
  const { text } = await generateText({
    model: MODEL,
    prompt: genCldrPrompt(voice, keyword, targets, per),
  })
  return cleanLines(text)
}

const cli = cac("upsample")
cli.usage("[options]")
cli
  .option("--emojis <list>", "target exactly these emoji instead of a rank window")
  .option("--min-rank <n>", `lowest (most frequent) rank to target (default ${MIN_RANK})`)
  .option("--max-rank <n>", `highest (least frequent) rank to target (default ${MAX_RANK})`)
  .option("--report", "target emoji failing the latest report's keywords.json keyword probe")
  .option("--cldr", "standalone: generate texts for the latest report's missed CLDR keywords (ignores emoji targeting)")
  .option("--per <n>", `texts to generate per target emoji / keyword / batch (default ${TEXTS_PER_EMOJI}, ${CLDR_PER} with --cldr)`)
  .option("--negation", "standalone: generate negation-heavy texts (ignores emoji targeting)")
  .option("--single-emoji", "standalone: re-annotate corpus rows that carry at most one emoji (ignores emoji targeting)")
  .option("--count <n>", `cap on texts for --negation (default ${NEG_COUNT}) / --single-emoji (default ${SINGLE_EMOJI_COUNT}) / missed keywords for --cldr (default ${CLDR_KEYWORDS})`)
cli.help()

if (import.meta.main) {
  const { options } = cli.parse(process.argv, { run: false })
  if (options.help) process.exit(0)
  const only = options.emojis ? String(options.emojis).trim() || undefined : undefined
  const minRank = Number(options.minRank ?? MIN_RANK)
  const maxRank = Number(options.maxRank ?? MAX_RANK)
  const negation = Boolean(options.negation)
  const singleEmoji = Boolean(options.singleEmoji)
  const cldr = Boolean(options.cldr)
  const per = Number(options.per ?? (cldr ? CLDR_PER : TEXTS_PER_EMOJI))
  const count = Number(
    options.count
      ?? (singleEmoji ? SINGLE_EMOJI_COUNT : cldr ? CLDR_KEYWORDS : NEG_COUNT),
  )

  if ([negation, singleEmoji, cldr].filter(Boolean).length > 1) {
    console.error("--negation, --single-emoji and --cldr are mutually exclusive")
    process.exit(1)
  }
  const standalone = negation || singleEmoji || cldr
  const standaloneName = negation ? "negation" : singleEmoji ? "single-emoji" : "cldr"
  if (standalone && (options.report || only || options.minRank != null || options.maxRank != null)) {
    console.warn(
      `--${standaloneName} ignores --report / --emojis / --min-rank / --max-rank`,
    )
  }
  if (singleEmoji && options.per != null) {
    console.warn("--single-emoji ignores --per")
  }
  if (!standalone && options.count != null) {
    console.warn("--count only applies with --negation / --single-emoji / --cldr")
  }
  if (!(per >= 1)) {
    console.error(`--per must be >= 1, got ${JSON.stringify(options.per)}`)
    process.exit(1)
  }

  let targets: string[]
  let negBatches: number[] = []
  let singleTexts: string[] = []
  let cldrKws: { keyword: string; targets: string[] }[] = []
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
  } else if (cldr) {
    if (!(count >= 1)) {
      console.error(`--count must be >= 1, got ${JSON.stringify(options.count)}`)
      process.exit(1)
    }
    const reportPath = await latestReport()
    const report = JSON.parse(await readFile(reportPath, "utf8")) as Report
    const misses = report.cldr?.misses
    if (!misses) throw new Error(`${reportPath}: no cldr.misses`)
    cldrKws = missedCldrKeywords(misses).slice(0, count)
    targets = []
    console.log(
      `${reportPath}: ${misses.length} CLDR miss rows -> `
      + `${cldrKws.length} missed keywords (cap ${count}), `
      + `${per} texts each`,
    )
    if (!cldrKws.length) {
      console.error("no missed CLDR keywords in the latest report")
      process.exit(1)
    }
  } else if (options.report) {
    const reportPath = await latestReport()
    const report = JSON.parse(await readFile(reportPath, "utf8")) as Report
    const keywords = report.emoji?.keywords
    const misses = keywords?.misses ?? []
    if (!keywords) throw new Error(`${reportPath}: no emoji.keywords`)
    targets = failingEmojis(misses, FAIL_RANK)
    console.log(
      `${reportPath}: ${keywords.n ?? "?"} words probed, ${misses.length} missed `
      + `-> targeting ${targets.length} failing emoji `
      + `(target not in top ${FAIL_RANK}) -> ${targets.join(" ")}`,
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

  const cands: { text: string; target?: string }[] = []
  if (singleEmoji) {
    for (const t of singleTexts) cands.push({ text: t })
    console.log(`${cands.length} corpus rows selected, annotating`)
  } else {
    const genUnit = negation ? "batches" : cldr ? "keywords" : "emojis"
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
    } else if (cldr) {
      genBar.start(cldrKws.length, 0)
      genQ.addAll(
        cldrKws.map(({ keyword, targets }) => async () => {
          try {
            for (const t of await genCldrBatch(pickVoice(), keyword, targets, per)) {
              cands.push({ text: t })
            }
          } catch (err) {
            console.warn(`\n  gen (${keyword}) failed: ${err}`)
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
    if (cldr) meta.src = "cldr"
    const row: Record<string, unknown> = {
      text: cands[i].text,
      emojis,
      styles: label.styles,
      bg: label.bg,
      fg: label.fg,
    }
    if (negation) row.neg = "true"
    row.meta = meta
    lines.push(JSON.stringify(row))
  }
  await appendJsonl(DATA, lines)

  const mode = negation
    ? "negation"
    : singleEmoji
      ? "single-emoji"
      : cldr
        ? "cldr"
        : "emoji-target"
  console.log("\n--- summary ---")
  console.log(`mode                 : ${mode}`)
  if (!standalone) console.log(`targets              : ${targets.length}`)
  if (cldr) console.log(`keywords             : ${cldrKws.length}`)
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
  process.exit(0)
}
