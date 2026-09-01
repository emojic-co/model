import { generateText } from "ai"
import cliProgress from "cli-progress"
import PQueue from "p-queue"

import { MODEL, annotate, annotateBatchCount } from "./annotate.ts"
import { appendJsonl, readJsonl } from "./io.ts"
import { rowMeta } from "./meta.ts"
import { rarest } from "./rarest.ts"

const TRAIN = "./train.jsonl"
const LABELS = "./labels.json"

const RARE_EMOJI_COUNT = 60
const TEXTS_PER_EMOJI = 40
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

export function countPalette(
  rows: { emoji: string }[],
  palette: string[],
): Map<string, number> {
  const counts = new Map<string, number>(palette.map((e) => [e, 0]))
  for (const r of rows) {
    const c = counts.get(r.emoji)
    if (c !== undefined) counts.set(r.emoji, c + 1)
  }
  return counts
}

async function genBatch(voice: string, emoji: string): Promise<string[]> {
  const { text } = await generateText({
    model: MODEL,
    prompt: [
      `Write ${TEXTS_PER_EMOJI} short WhatsApp-style messages as if sent by ${voice}.`,
      `Every message must read naturally as one that would be sent together with the emoji ${emoji} -- its subject, mood, or occasion should fit that emoji.`,
      `Do not put any emoji in the output, and never name or describe the emoji.`,
      `One message per line. No numbering, no bullets, no quotes, no commentary.`,
      `Each message at most ${MAX_RAW_LEN} characters.`,
      `Vary tone and intent: quick updates, dry humor, complaints, questions, sudden news, invitations, low-effort replies.`,
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
  const palette = labels.emojis

  const rows = await readJsonl<{ emoji: string }>(TRAIN)
  const counts = countPalette(rows, palette)
  const targets = rarest(palette, counts, RARE_EMOJI_COUNT)
  console.log(
    `${palette.length} palette emojis -> upsampling ${targets.length} rarest ` +
      `(${counts.get(targets[0])}..${counts.get(targets[targets.length - 1])} rows each)`,
  )

  const genBar = new cliProgress.SingleBar(
    {
      format:
        "generating |{bar}| {percentage}% | {value}/{total} emojis | ETA: {eta}s",
    },
    cliProgress.Presets.shades_classic,
  )
  genBar.start(targets.length, 0)

  const cands: { text: string; target: string; voice: string }[] = []
  const genQ = new PQueue({ concurrency: GEN_CONCURRENCY })
  genQ.addAll(
    targets.map((emoji) => async () => {
      const voice = pickVoice()
      try {
        for (const t of await genBatch(voice, emoji)) {
          cands.push({ text: t, target: emoji, voice })
        }
      } catch (err) {
        console.warn(`\n  gen batch (${emoji}) failed: ${err}`)
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
          src: "upsample-emojis",
          v: 1,
          target_emoji: cands[i].target,
          params: { voice: cands[i].voice, textsPerEmoji: TEXTS_PER_EMOJI },
        }),
      }),
    )
  }
  await appendJsonl(TRAIN, lines)

  console.log("\n--- summary ---")
  console.log(`targets (rarest)     : ${targets.length}`)
  console.log(`generated            : ${cands.length}`)
  console.log(`annotated -> train   : ${lines.length}`)
  console.log(`dropped (no label)   : ${cands.length - lines.length}`)
  process.exit(0)
}
