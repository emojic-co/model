import { readFile } from "node:fs/promises"

import { generateText } from "ai"
import { cac } from "cac"
import cliProgress from "cli-progress"
import PQueue from "p-queue"

import { MODEL, annotate, annotateBatchCount } from "./annotate.ts"
import { appendJsonl } from "./io.ts"
import { type Report, type Word, latestReport } from "./report.ts"

const DATA = "./data.jsonl"

const FAIL_RANK = 5
const TEXTS_PER_WORD = 10
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

export function pickFailed(
  words: Word[],
  maxRank: number,
): { word: string; emoji: string }[] {
  const out: { word: string; emoji: string }[] = []
  for (const w of words) {
    const emoji = w.expected?.[0]
    if (!emoji) continue
    if (w.rank === null || w.rank > maxRank) out.push({ word: w.keyword, emoji })
  }
  return out
}

function genPrompt(voice: string, word: string, emoji: string, per: number): string {
  return [
    `Write ${per} short text messages as if sent by ${voice}, one per line.`,
    `Each message between ${MIN_LEN} and ${MAX_LEN} characters.`,
    `Every message must contain the word "${word}" and read naturally as one a`,
    `person would send together with the emoji ${emoji} - its subject, activity,`,
    `place, or mood fits both the word and that emoji.`,
    `Do not put any emoji in the output, and never name or describe the emoji.`,
    `Vary sender, tone, and intent: updates, questions, complaints, plans,`,
    `reactions, reminders, small talk. Sound real and specific.`,
    `No numbering, no bullets, no quotes, no commentary.`,
  ].join("\n")
}

async function genBatch(
  voice: string,
  word: string,
  emoji: string,
  per: number,
): Promise<string[]> {
  const { text } = await generateText({
    model: MODEL,
    prompt: genPrompt(voice, word, emoji, per),
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

const cli = cac("upsample-emoji-test")
cli.usage("[options]")
cli
  .option("--report <path>", "report.json to read (default: latest under ./report)")
  .option("--rank <n>", `treat a word as failed if its emoji ranks worse than this (default ${FAIL_RANK})`)
  .option("--per <n>", `texts to generate per failed word (default ${TEXTS_PER_WORD})`)
cli.help()

if (import.meta.main) {
  const { options } = cli.parse(process.argv, { run: false })
  if (options.help) process.exit(0)
  const reportPath =
    (options.report ? String(options.report).trim() : "") || (await latestReport())
  const maxRank = Number(options.rank ?? FAIL_RANK)
  const per = Number(options.per ?? TEXTS_PER_WORD)

  const report = JSON.parse(await readFile(reportPath, "utf8")) as Report
  const words = report.emoji?.keywords?.words ?? []
  if (!words.length) throw new Error(`${reportPath}: no emoji.keywords.words`)
  const targets = pickFailed(words, maxRank)
  console.log(
    `${reportPath}: ${words.length} words -> `
    + `${targets.length} failed (target not in top ${maxRank})`,
  )
  if (!targets.length) process.exit(0)

  const genBar = new cliProgress.SingleBar(
    {
      format:
        "generating |{bar}| {percentage}% | {value}/{total} words | ETA: {eta}s",
    },
    cliProgress.Presets.shades_classic,
  )
  genBar.start(targets.length, 0)

  const cands: { text: string; target: string }[] = []
  const genQ = new PQueue({ concurrency: GEN_CONCURRENCY })
  genQ.addAll(
    targets.map(({ word, emoji }) => async () => {
      try {
        for (const t of await genBatch(pickVoice(), word, emoji, per)) {
          cands.push({ text: t, target: emoji })
        }
      } catch (err) {
        console.warn(`\n  gen (${word} ${emoji}) failed: ${err}`)
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
  console.log(`report               : ${reportPath}`)
  console.log(`failed words         : ${targets.length}`)
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
