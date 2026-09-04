import { generateText } from "ai"
import { cac } from "cac"
import cliProgress from "cli-progress"
import PQueue from "p-queue"

import { MODEL, annotate, annotateBatchCount } from "./annotate.ts"
import { appendJsonl } from "./io.ts"

const DATA = "./data.jsonl"

const COLORS = [
  "red",
  "orange",
  "yellow",
  "green",
  "blue",
  "purple",
  "pink",
  "brown",
  "black",
  "white",
  "gray",
]

const BATCH_SIZE = 20
const BATCHES_PER_COLOR = 5
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

const cli = cac("upsample-colors")
cli.usage("[--colors NAME ...]")
cli.option(
  "--colors <names>",
  "restrict to these color names, space-separated"
    + ` (default: all ${COLORS.length}; known: ${COLORS.join(" ")})`,
)
cli.help()

export function parseColors(argv: string[], list: string[]): string[] {
  const i = argv.indexOf("--colors")
  if (i < 0) return [...list]

  const set = new Set(list)
  const picked: string[] = []
  const unknown: string[] = []
  for (let j = i + 1; j < argv.length; j++) {
    if (argv[j].startsWith("--")) break
    const c = argv[j].trim().toLowerCase()
    if (!c || picked.includes(c)) continue
    if (set.has(c)) picked.push(c)
    else unknown.push(c)
  }
  if (unknown.length) throw new Error(`unknown color(s): ${unknown.join(", ")}`)
  if (!picked.length) throw new Error("--colors given with no known color names")
  return picked
}

function genPrompt(voice: string, color: string, per: number): string {
  return [
    `Write ${per} short text messages as if sent by ${voice}, one per line.`,
    `Each message between ${MIN_LEN} and ${MAX_LEN} characters.`,
    `Every message should evoke the color ${color} indirectly - through an`,
    `object, place, food, plant, animal, weather, light, or mood strongly tied`,
    `to it - the way "the smell of roses at dawn" evokes red or "a dark and`,
    `stormy night" evokes black.`,
    `Never write the word "${color}", any shade or synonym of it, or name any`,
    `colour at all. Do not put any emoji in the output.`,
    `Vary sender, tone, and intent: updates, questions, complaints, plans,`,
    `reactions, reminders, small talk. Sound real and specific.`,
    `No numbering, no bullets, no quotes, no commentary.`,
  ].join("\n")
}

async function genBatch(
  voice: string,
  color: string,
  per: number,
): Promise<string[]> {
  const { text } = await generateText({
    model: MODEL,
    prompt: genPrompt(voice, color, per),
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

if (import.meta.main) {
  cli.parse(process.argv, { run: false })
  if (cli.options.help) process.exit(0)

  const targets = parseColors(process.argv, COLORS)
  console.log(
    `targeting ${targets.length} color(s) -> ${targets.join(" ")} `
    + `(${BATCHES_PER_COLOR} x ${BATCH_SIZE} texts each)`,
  )

  const jobs = targets.flatMap((color) =>
    Array.from({ length: BATCHES_PER_COLOR }, () => color),
  )

  const genBar = new cliProgress.SingleBar(
    {
      format:
        "generating |{bar}| {percentage}% | {value}/{total} batches | ETA: {eta}s",
    },
    cliProgress.Presets.shades_classic,
  )
  genBar.start(jobs.length, 0)

  const cands: { text: string; color: string }[] = []
  const genQ = new PQueue({ concurrency: GEN_CONCURRENCY })
  genQ.addAll(
    jobs.map((color) => async () => {
      try {
        for (const t of await genBatch(pickVoice(), color, BATCH_SIZE)) {
          cands.push({ text: t, color })
        }
      } catch (err) {
        console.warn(`\n  gen (${color}) failed: ${err}`)
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
  const perColor = new Map<string, number>(targets.map((c) => [c, 0]))
  let noLabel = 0
  let noPalette = 0
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
    const { color } = cands[i]
    perColor.set(color, (perColor.get(color) ?? 0) + 1)
    lines.push(
      JSON.stringify({
        text: cands[i].text,
        emojis: label.emojis.join(" "),
        styles: label.styles,
        bg: label.bg,
        fg: label.fg,
        color,
      }),
    )
  }
  await appendJsonl(DATA, lines)

  console.log("\n--- summary ---")
  console.log(`colors               : ${targets.length}`)
  console.log(`generated            : ${cands.length}`)
  console.log(`appended -> data     : ${lines.length}`)
  console.log(`dropped no label     : ${noLabel}`)
  console.log(`dropped no palette   : ${noPalette}`)
  for (const color of targets) {
    console.log(`  ${color.padEnd(8)}: ${perColor.get(color) ?? 0}`)
  }
  process.exit(0)
}
