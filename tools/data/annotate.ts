import { generateText, Output } from "ai"
import PQueue from "p-queue"
import { z } from "zod"

export const MODEL = "openai/gpt-5.6-luna"

export const ANNOTATE_BATCH_SIZE = 10
export const ANNOTATE_CONCURRENCY = 40

export const MIN_CONTRAST = 3

export type Label = {
  emoji: string
  feeling: string
  bg: [string, string]
  fg: string
}

const Annotation = z.object({
  id: z.number(),
  emoji: z.string(),
  feeling: z.string(),
  bg: z.array(z.string()),
  fg: z.string(),
})

const INSTRUCTIONS = [
  "You are an annotator. For each message below, choose three things:",
  "1. emoji - the single emoji that best fits the message. Any emoji is",
  "   allowed. Pick the one that fits best, not a safe default. Do not favor",
  "   rare emojis and do not favor common ones - just the best fit.",
  "2. feeling - one word for the emotion the message best conveys, e.g.",
  "   Happy, Sad, Angry, Calm, Anxious, Love, Excited, Grateful, Tired,",
  "   Annoyed, Hopeful, Proud. This list is not exhaustive; use whatever",
  "   single word fits best, capitalized. Prefer common, everyday feeling",
  "   words over unusual ones when both fit. \"Neutral\" is allowed for a",
  "   message that carries no real emotion, but do not reach for it.",
  "3. colors - a 3-color palette that captures the mood and imagery of the",
  "   message. \"bg\" is two colors [top, bottom] for a background gradient;",
  "   they must sit close enough to read as one gradient, not a clash. \"fg\"",
  "   is one color for text laid over that gradient and must stay clearly",
  "   readable against both \"bg\" stops (strong contrast). All three are",
  "   lowercase hex in #rrggbb form.",
  "",
  "Return exactly one annotation object per input message, echoing its id.",
  "Do not add, drop, reorder, or merge items.",
  'Format: {"annotations": [{"id": 0, "emoji": "\u{1F600}", "feeling": "Happy", "bg": ["#ffd76a", "#ff9a5a"], "fg": "#3a1d00"}]}',
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

const HEX6 = /^#?([0-9a-f]{6})$/i
const HEX3 = /^#?([0-9a-f])([0-9a-f])([0-9a-f])$/i

function cleanHex(raw: string): string {
  const s = (raw ?? "").trim().toLowerCase()
  const m6 = HEX6.exec(s)
  if (m6) return `#${m6[1]}`
  const m3 = HEX3.exec(s)
  if (m3) return `#${m3[1]}${m3[1]}${m3[2]}${m3[2]}${m3[3]}${m3[3]}`
  return ""
}

function luminance(hex: string): number {
  const n = parseInt(hex.slice(1), 16)
  const lin = [(n >> 16) & 255, (n >> 8) & 255, n & 255].map((v) => {
    const c = v / 255
    return c <= 0.03928 ? c / 12.92 : ((c + 0.055) / 1.055) ** 2.4
  })
  return 0.2126 * lin[0] + 0.7152 * lin[1] + 0.0722 * lin[2]
}

function contrast(a: string, b: string): number {
  const la = luminance(a)
  const lb = luminance(b)
  return (Math.max(la, lb) + 0.05) / (Math.min(la, lb) + 0.05)
}

function cleanPalette(
  bg: unknown,
  fg: unknown,
): Pick<Label, "bg" | "fg"> | null {
  if (!Array.isArray(bg) || bg.length < 2) return null
  const bg0 = cleanHex(typeof bg[0] === "string" ? bg[0] : "")
  const bg1 = cleanHex(typeof bg[1] === "string" ? bg[1] : "")
  const fgHex = cleanHex(typeof fg === "string" ? fg : "")
  if (!bg0 || !bg1 || !fgHex) return null
  if (contrast(fgHex, bg0) < MIN_CONTRAST) return null
  if (contrast(fgHex, bg1) < MIN_CONTRAST) return null
  return { bg: [bg0, bg1], fg: fgHex }
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
        const palette = cleanPalette(a.bg, a.fg)
        if (emoji && feeling && palette) {
          byId.set(a.id, { emoji, feeling, ...palette })
        }
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
