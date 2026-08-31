import { generateText } from "ai"
import cliProgress from "cli-progress"
import PQueue from "p-queue"

import { MODEL, annotate, annotateBatchCount } from "./annotate.ts"
import { appendJsonl } from "./io.ts"

const TRAIN = "./train.jsonl"

const BATCH_SIZE = 50
const BATCH_COUNT = 50
const CONCURRENCY = 25

const MIN_LEN = 4
const MAX_LEN = 48

const GEN_PROMPT = [
  `Write ${BATCH_SIZE} short text messages, one per line.`,
  `Each message between ${MIN_LEN} and ${MAX_LEN} characters.`,
  "Across the whole set, cover a wide range: many different senders and",
  "personalities, every mood (positive, negative, flat), every intent",
  "(statements, questions, requests, reactions, reminders, small talk), and",
  "every length from very short to near the maximum.",
  "Do not lean on any single persona, topic, tone, or sentence shape.",
  "No numbering, no bullets, no quotes, no emoji, no commentary.",
].join("\n")

async function genBatch(): Promise<string[]> {
  const { text } = await generateText({ model: MODEL, prompt: GEN_PROMPT })
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
  const genBar = new cliProgress.SingleBar(
    {
      format:
        "generating |{bar}| {percentage}% | {value}/{total} batches | ETA: {eta}s",
    },
    cliProgress.Presets.shades_classic,
  )
  genBar.start(BATCH_COUNT, 0)

  const texts: string[] = []
  const genQ = new PQueue({ concurrency: CONCURRENCY })
  genQ.addAll(
    Array.from({ length: BATCH_COUNT }, () => async () => {
      try {
        texts.push(...(await genBatch()))
      } catch (err) {
        console.warn(`\n  gen batch failed: ${err}`)
      }
      genBar.increment()
    }),
  )
  await genQ.onIdle()
  genBar.stop()

  console.log(`\n${texts.length} texts generated, annotating`)

  const annBar = new cliProgress.SingleBar(
    {
      format:
        "annotating |{bar}| {percentage}% | {value}/{total} batches | ETA: {eta}s",
    },
    cliProgress.Presets.shades_classic,
  )
  annBar.start(annotateBatchCount(texts.length), 0)
  const labels = await annotate(texts, () => annBar.increment())
  annBar.stop()

  const lines: string[] = []
  for (let i = 0; i < texts.length; i++) {
    const label = labels.get(i)
    if (!label) continue
    lines.push(
      JSON.stringify({
        text: texts[i],
        feeling: label.feeling,
        emoji: label.emoji,
      }),
    )
  }
  await appendJsonl(TRAIN, lines)

  console.log("\n--- summary ---")
  console.log(`generated           : ${texts.length}`)
  console.log(`annotated -> train  : ${lines.length}`)
  console.log(`dropped (no label)  : ${texts.length - lines.length}`)
  process.exit(0)
}
