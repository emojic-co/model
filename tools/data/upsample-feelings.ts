import { generateText } from "ai"
import cliProgress from "cli-progress"
import PQueue from "p-queue"

import { MODEL, annotate, annotateBatchCount } from "./annotate.ts"
import { appendJsonl, readJsonl } from "./io.ts"
import { rowMeta } from "./meta.ts"
import { rarest } from "./rarest.ts"

const TRAIN = "./train.jsonl"
const LABELS = "./labels.json"

const FEELINGS_PER_RUN = 3
const BATCHES_PER_FEELING = 20
const GEN_BATCH_SIZE = 100
const MAX_RAW_LEN = 50
const GEN_CONCURRENCY = 20

const VOICES = [
  "a teenager", "a college student", "a new parent", "a retiree",
  "a shift worker", "a freelancer", "someone in their 30s", "a grandparent",
  "an office worker", "a nurse", "a tradesperson", "a student athlete",
  "a girl", "a boy",
  "a father", "a mother", "a sibling", "a cousin", "a friend", "a neighbor",
  "a coworker", "a classmate", "a teammate", "a mentor", "a mentee",
]

function pickVoice(): string {
  return VOICES[Math.floor(Math.random() * VOICES.length)]
}

export function countPaletteFeelings(
  rows: { emoji: string; feeling: string }[],
  feelings: string[],
  emojiPalette: Set<string>,
): Map<string, number> {
  const counts = new Map<string, number>(feelings.map((f) => [f, 0]))
  for (const r of rows) {
    if (counts.has(r.feeling) && emojiPalette.has(r.emoji)) {
      counts.set(r.feeling, counts.get(r.feeling)! + 1)
    }
  }
  return counts
}

async function genBatch(voice: string, feeling: string): Promise<string[]> {
  const { text } = await generateText({
    model: MODEL,
    prompt: [
      `Write ${GEN_BATCH_SIZE} short WhatsApp-style messages as if sent by ${voice}.`,
      `Every message must unmistakably convey the feeling "${feeling}" -- someone reading it cold, with no context, should name that feeling. Convey it through what is said and how, never by naming the feeling.`,
      `One message per line. No numbering, no bullets, no quotes, no emoji, no commentary.`,
      `Each message at most ${MAX_RAW_LEN} characters.`,
      `Vary the wording, situation and sentence shape, but keep every line firmly in the target feeling.`,
      `Sound real and specific.`,
    ].join("\n"),
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
  const labels = JSON.parse(await Bun.file(LABELS).text()) as {
    feelings: string[]
    emojis: string[]
  }
  const feelings = labels.feelings
  const emojiPalette = new Set(labels.emojis)

  const rows = await readJsonl<{ emoji: string; feeling: string }>(TRAIN)
  const counts = countPaletteFeelings(rows, feelings, emojiPalette)
  const targets = rarest(feelings, counts, FEELINGS_PER_RUN)
  console.log(
    `feeling coverage: ${feelings.map((f) => `${f}=${counts.get(f)}`).join(" ")}`,
  )
  console.log(`upsampling rarest: ${targets.join(", ")}`)

  const batchCount = targets.length * BATCHES_PER_FEELING
  const genBar = new cliProgress.SingleBar(
    {
      format:
        "generating |{bar}| {percentage}% | {value}/{total} batches | ETA: {eta}s",
    },
    cliProgress.Presets.shades_classic,
  )
  genBar.start(batchCount, 0)

  const cands: { text: string; target: string; voice: string }[] = []
  const genQ = new PQueue({ concurrency: GEN_CONCURRENCY })
  genQ.addAll(
    Array.from({ length: batchCount }, (_, i) => async () => {
      const feeling = targets[i % targets.length]
      const voice = pickVoice()
      try {
        for (const t of await genBatch(voice, feeling)) {
          cands.push({ text: t, target: feeling, voice })
        }
      } catch (err) {
        console.warn(`\n  gen batch (${feeling}) failed: ${err}`)
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
  const annotated = await annotate(
    cands.map((c) => c.text),
    () => annBar.increment(),
  )
  annBar.stop()

  const lines: string[] = []
  for (let i = 0; i < cands.length; i++) {
    const label = annotated.get(i)
    if (!label) continue
    lines.push(
      JSON.stringify({
        text: cands[i].text,
        feeling: label.feeling,
        emoji: label.emoji,
        bg: label.bg,
        fg: label.fg,
        meta: rowMeta({
          src: "upsample-feelings",
          v: 1,
          target_feeling: cands[i].target,
          params: {
            voice: cands[i].voice,
            batchesPerFeeling: BATCHES_PER_FEELING,
          },
        }),
      }),
    )
  }
  await appendJsonl(TRAIN, lines)

  console.log("\n--- summary ---")
  console.log(`targets (rarest)     : ${targets.join(", ")}`)
  console.log(`generated            : ${cands.length}`)
  console.log(`annotated -> train   : ${lines.length}`)
  console.log(`dropped (no label)   : ${cands.length - lines.length}`)
  process.exit(0)
}
