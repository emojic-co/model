import { generateText, Output } from "ai"
import PQueue from "p-queue"
import { z } from "zod"

export const MODEL = "openai/gpt-5.6-luna"

export const ANNOTATE_BATCH_SIZE = 10
export const ANNOTATE_CONCURRENCY = 20

export type Label = { emoji: string; feeling: string }

const Annotation = z.object({
  id: z.number(),
  emoji: z.string(),
  feeling: z.string(),
})

const INSTRUCTIONS = [
  "You are an annotator. For each message below, choose two things:",
  "1. emoji - the single emoji that best fits the message. Any emoji is",
  "   allowed. Pick the one that fits best, not a safe default. Do not favor",
  "   rare emojis and do not favor common ones - just the best fit.",
  "2. feeling - one word for the emotion the message best conveys, e.g.",
  "   Happy, Sad, Angry, Calm, Anxious, Love, Excited, Grateful, Tired,",
  "   Annoyed, Hopeful, Proud. This list is not exhaustive; use whatever",
  "   single word fits best, capitalized. Prefer common, everyday feeling",
  "   words over unusual ones when both fit. \"Neutral\" is allowed for a",
  "   message that carries no real emotion, but do not reach for it.",
  "",
  "Return exactly one annotation object per input message, echoing its id.",
  "Do not add, drop, reorder, or merge items.",
  'Format: {"annotations": [{"id": 0, "emoji": "\u{1F600}", "feeling": "Happy"}]}',
].join("\n")

function chunk<T>(arr: T[], n: number): T[][] {
  const out: T[][] = []
  for (let i = 0; i < arr.length; i += n) out.push(arr.slice(i, i + n))
  return out
}

function cleanFeeling(raw: string): string {
  const f = raw.trim().replace(/\s+/g, " ")
  return f ? f[0].toUpperCase() + f.slice(1) : f
}

function cleanEmoji(raw: string): string {
  return raw.trim().split(/\s+/)[0] ?? ""
}

async function annotateBatch(
  batch: { id: number; text: string }[],
): Promise<Map<number, Label>> {
  const ids = new Set(batch.map((b) => b.id))
  const byId = new Map<number, Label>()

  for (let attempt = 0; attempt < 2 && byId.size < ids.size; attempt++) {
    try {
      const { output } = await generateText({
        model: MODEL,
        output: Output.object({
          schema: z.object({ annotations: z.array(Annotation) }),
        }),
        prompt: [INSTRUCTIONS, "", "Messages:", JSON.stringify(batch)].join("\n"),
      })
      for (const a of output.annotations) {
        if (!ids.has(a.id) || byId.has(a.id)) continue
        const emoji = cleanEmoji(a.emoji)
        const feeling = cleanFeeling(a.feeling)
        if (emoji && feeling) byId.set(a.id, { emoji, feeling })
      }
    } catch (err) {
      if (attempt === 1) {
        console.warn(`\n  annotate batch of ${batch.length} failed: ${err}`)
      }
    }
  }

  for (const b of batch) {
    if (!byId.has(b.id)) console.warn(`\n  dropped id ${b.id}: no annotation`)
  }
  return byId
}

export async function annotate(
  texts: string[],
  onBatchDone?: () => void,
): Promise<Map<number, Label>> {
  const items = texts.map((text, id) => ({ id, text }))
  const result = new Map<number, Label>()
  const queue = new PQueue({ concurrency: ANNOTATE_CONCURRENCY })

  queue.addAll(
    chunk(items, ANNOTATE_BATCH_SIZE).map((batch) => async () => {
      for (const [id, label] of await annotateBatch(batch)) result.set(id, label)
      onBatchDone?.()
    }),
  )
  await queue.onIdle()
  return result
}

export function annotateBatchCount(n: number): number {
  return Math.ceil(n / ANNOTATE_BATCH_SIZE)
}
