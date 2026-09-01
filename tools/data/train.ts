import { generateText } from "ai"
import cliProgress from "cli-progress"
import PQueue from "p-queue"

import { MODEL, annotate, annotateBatchCount } from "./annotate.ts"
import { appendJsonl } from "./io.ts"
import { rowMeta } from "./meta.ts"

const TRAIN = "./train.jsonl"

const BATCH_SIZE = 50
const BATCH_COUNT = 50
const CONCURRENCY = 25

const MIN_LEN = 4
const MAX_LEN = 42

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

function genPrompt(topic: string): string {
  return [
    `Write ${BATCH_SIZE} short text messages, one per line.`,
    `Each message between ${MIN_LEN} and ${MAX_LEN} characters.`,
    `Every message is about ${topic}. Do not announce the topic or use it as a`,
    "label; let it come through naturally in what is said.",
    "Across the whole set, still cover a wide range: many different senders and",
    "personalities, every mood (positive, negative, flat), every intent",
    "(statements, questions, requests, reactions, reminders, small talk), and",
    "every length from very short to near the maximum.",
    "Do not lean on any single persona, tone, or sentence shape.",
    "No numbering, no bullets, no quotes, no emoji, no commentary.",
  ].join("\n")
}

async function genBatch(topic: string): Promise<string[]> {
  const { text } = await generateText({ model: MODEL, prompt: genPrompt(topic) })
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

  const texts: { text: string; topic: string }[] = []
  const genQ = new PQueue({ concurrency: CONCURRENCY })
  genQ.addAll(
    Array.from({ length: BATCH_COUNT }, (_, i) => async () => {
      const topic = topicForBatch(i)
      try {
        for (const t of await genBatch(topic)) texts.push({ text: t, topic })
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
  const labels = await annotate(
    texts.map((t) => t.text),
    () => annBar.increment(),
  )
  annBar.stop()

  const lines: string[] = []
  for (let i = 0; i < texts.length; i++) {
    const label = labels.get(i)
    if (!label) continue
    lines.push(
      JSON.stringify({
        text: texts[i].text,
        feeling: label.feeling,
        emoji: label.emoji,
        bg: label.bg,
        fg: label.fg,
        meta: rowMeta({
          src: "train",
          v: 1,
          topic: texts[i].topic,
          params: { batchSize: BATCH_SIZE, minLen: MIN_LEN, maxLen: MAX_LEN },
        }),
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
