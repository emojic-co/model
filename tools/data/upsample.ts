import { readFile } from "node:fs/promises"

import { generateText } from "ai"
import { cac } from "cac"
import cliProgress from "cli-progress"
import PQueue from "p-queue"

import { MODEL, annotate, annotateBatchCount } from "./annotate.ts"
import { splitEmojis } from "./emoji.ts"
import { appendJsonl, readJsonl } from "./io.ts"
import { type Report, type Word, latestReport } from "./report.ts"

const DATA = "./data.jsonl"

const MIN_RANK = 200
const MAX_RANK = 400
const FAIL_RANK = 5
const TEXTS_PER_EMOJI = 40
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

export function failingEmojis(words: Word[], maxRank: number): string[] {
  const out: string[] = []
  const seen = new Set<string>()
  for (const w of words) {
    const emoji = w.expected?.[0]
    if (!emoji || seen.has(emoji)) continue
    if (w.rank === null || w.rank > maxRank) {
      seen.add(emoji)
      out.push(emoji)
    }
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

async function genBatch(
  voice: string,
  emoji: string,
  per: number,
): Promise<string[]> {
  const { text } = await generateText({
    model: MODEL,
    prompt: genPrompt(voice, emoji, per),
  })
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

const cli = cac("upsample")
cli.usage("[options]")
cli
  .option("--emojis <list>", "target exactly these emoji instead of a rank window")
  .option("--min-rank <n>", `lowest (most frequent) rank to target (default ${MIN_RANK})`)
  .option("--max-rank <n>", `highest (least frequent) rank to target (default ${MAX_RANK})`)
  .option("--report", "target emoji failing the latest report's words.json keyword probe")
  .option("--per <n>", `texts to generate per target emoji (default ${TEXTS_PER_EMOJI})`)
cli.help()

if (import.meta.main) {
  const { options } = cli.parse(process.argv, { run: false })
  if (options.help) process.exit(0)
  const only = options.emojis ? String(options.emojis).trim() || undefined : undefined
  const minRank = Number(options.minRank ?? MIN_RANK)
  const maxRank = Number(options.maxRank ?? MAX_RANK)
  const per = Number(options.per ?? TEXTS_PER_EMOJI)

  let targets: string[]
  if (options.report) {
    const reportPath = await latestReport()
    const report = JSON.parse(await readFile(reportPath, "utf8")) as Report
    const words = report.emoji?.keywords?.words ?? []
    if (!words.length) throw new Error(`${reportPath}: no emoji.keywords.words`)
    targets = failingEmojis(words, FAIL_RANK)
    console.log(
      `${reportPath}: ${words.length} words -> targeting ${targets.length} `
      + `failing emoji (target not in top ${FAIL_RANK}) -> ${targets.join(" ")}`,
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

  const genBar = new cliProgress.SingleBar(
    {
      format:
        "generating |{bar}| {percentage}% | {value}/{total} emojis | ETA: {eta}s",
    },
    cliProgress.Presets.shades_classic,
  )
  genBar.start(targets.length, 0)

  const cands: { text: string; target: string }[] = []
  const genQ = new PQueue({ concurrency: GEN_CONCURRENCY })
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
  await genQ.onIdle()
  genBar.stop()

  console.log(`\n${cands.length} texts generated, annotating`)

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
    { colors: true, onBatchDone: () => annBar.increment() },
  )
  annBar.stop()

  const lines: string[] = []
  let noLabel = 0
  let noPalette = 0
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
    const hit = label.emojis.includes(cands[i].target)
    if (hit) hitTarget++
    else missTarget++
    const emojis = [
      cands[i].target,
      ...label.emojis.filter((e) => e !== cands[i].target),
    ]
    lines.push(
      JSON.stringify({
        text: cands[i].text,
        emojis: emojis.join(" "),
        styles: label.styles,
        bg: label.bg,
        fg: label.fg,
      }),
    )
  }
  await appendJsonl(DATA, lines)

  console.log("\n--- summary ---")
  console.log(`targets              : ${targets.length}`)
  console.log(`generated            : ${cands.length}`)
  console.log(`appended -> data     : ${lines.length}`)
  console.log(`dropped no label     : ${noLabel}`)
  console.log(`dropped no palette   : ${noPalette}`)
  console.log(
    `target hit / miss    : ${hitTarget} / ${missTarget} `
    + `(target injected either way)`,
  )
  process.exit(0)
}
