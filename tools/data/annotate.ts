import { generateText, Output } from "ai"
import PQueue from "p-queue"
import { z } from "zod"

import { splitEmojis } from "./emoji.ts"
import { STYLE_LINES, STYLE_SET, STYLES } from "./styles.ts"

export const MODEL = "openai/gpt-5.6-luna"

export const ANNOTATE_BATCH_SIZE = Number(process.env.ANNOTATE_BATCH_SIZE) || 10
export const ANNOTATE_CONCURRENCY = Number(process.env.ANNOTATE_CONCURRENCY) || 25
export const ANNOTATE_ATTEMPTS = Number(process.env.ANNOTATE_ATTEMPTS) || 1

export const MIN_CONTRAST = 3
export const MAX_EMOJIS = 6
export const MAX_STYLES = 3

export type Label = {
  emojis: string[]
  styles: string[]
  bg?: [string, string]
  fg?: string
}

export type AnnotateOpts = {
  colors?: boolean
  onBatchDone?: () => void
}

export type Usage = {
  calls: number
  input: number
  output: number
  total: number
}

export const lastUsage: Usage = { calls: 0, input: 0, output: 0, total: 0 }

function addUsage(u: Usage, raw: unknown): void {
  const r = (raw ?? {}) as Record<string, number | undefined>
  const input = r.inputTokens ?? r.promptTokens ?? 0
  const output = r.outputTokens ?? r.completionTokens ?? 0
  u.calls += 1
  u.input += input
  u.output += output
  u.total += r.totalTokens ?? input + output
}

export function formatUsage(u: Usage): string {
  return (
    `tokens: input ${u.input.toLocaleString()} · `
    + `output ${u.output.toLocaleString()} · `
    + `total ${u.total.toLocaleString()} over ${u.calls} calls`
  )
}

export type DropReason = "noStyle" | "noPalette"
export type Drops = Record<DropReason | "batch" | "missingId", number>

export const lastDrops: Drops = {
  batch: 0,
  missingId: 0,
  noStyle: 0,
  noPalette: 0,
}

function resetDrops(d: Drops): void {
  d.batch = 0
  d.missingId = 0
  d.noStyle = 0
  d.noPalette = 0
}

export function formatDrops(d: Drops): string {
  const total = d.batch + d.missingId + d.noStyle + d.noPalette
  return (
    `drops: ${total} `
    + `(batch ${d.batch} · missingId ${d.missingId} · `
    + `noStyle ${d.noStyle} · noPalette ${d.noPalette})`
  )
}

const Annotation = z.object({
  id: z.number(),
  emojis: z.string(),
  styles: z.array(z.string()),
  bg: z.array(z.string()).optional(),
  fg: z.string().optional(),
})

const STYLE_BLOCK = STYLES.map((s) => `   ${s} - ${STYLE_LINES[s]}`).join("\n")

const EMOJI_RULES = [
  `1. emojis - a single space-separated string of 0 to ${MAX_EMOJIS} emojis,`,
  "   most associated first. Include:",
  "   - the concrete things, places, activities, animals, food, or objects the",
  "     message names or is plainly about;",
  "   - a mood emoji ONLY when the mood is strong - skip it otherwise.",
  "   Prefer a specific emoji over a generic one, no decorative picks, do not",
  "   lean on smiley faces. Return an empty string when no emoji is even loosely",
  "   relevant - do not force a pick.",
]

const STYLE_RULES = [
  `2. styles - 1 to ${MAX_STYLES} labels from this fixed set, describing how the`,
  "   message feels to read (its voice and tone), not only its emotion. These",
  "   are the ONLY allowed values:",
  STYLE_BLOCK,
]

const COLOR_RULES = [
  "3. bg, fg - a 3-color palette that captures the mood and imagery of the",
  '   message. "bg" is two colors [top, bottom] for a background gradient; they',
  '   must sit close enough to read as one gradient, not a clash. "fg" is one',
  '   color for text over that gradient and must stay clearly readable against',
  "   both bg stops (strong contrast). All three are lowercase #rrggbb hex.",
]

function instructions(colors: boolean): string {
  const parts = [
    "You are an annotator. For each message below choose"
    + (colors ? " three things:" : " two things:"),
    ...EMOJI_RULES,
    ...STYLE_RULES,
    ...(colors ? COLOR_RULES : []),
    "",
    "Return exactly one object per input message, echoing its id.",
    "Do not add, drop, reorder, or merge items.",
    colors
      ? 'Format: {"annotations": [{"id": 0, "emojis": "\u{1F68C} \u{1F624}",'
      + ' "styles": ["Irritated"], "bg": ["#c9d8e5", "#9fb4c8"],'
      + ' "fg": "#172b3a"}]}'
      : 'Format: {"annotations": [{"id": 0, "emojis": "\u{1F68C} \u{1F624}",'
      + ' "styles": ["Irritated"]}, {"id": 1, "emojis": "",'
      + ' "styles": ["Deadpan"]}]}',
  ]
  return parts.join("\n")
}

function chunk<T>(arr: T[], n: number): T[][] {
  const out: T[][] = []
  for (let i = 0; i < arr.length; i += n) out.push(arr.slice(i, i + n))
  return out
}

function cleanStyle(raw: string): string {
  const s = raw.trim()
  return s ? s[0].toUpperCase() + s.slice(1).toLowerCase() : s
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

function cleanLabel(
  a: z.infer<typeof Annotation>,
  colors: boolean,
): Label | DropReason {
  const emojis = splitEmojis(a.emojis).slice(0, MAX_EMOJIS)
  const styles = [
    ...new Set(a.styles.map(cleanStyle).filter((s) => STYLE_SET.has(s))),
  ].slice(0, MAX_STYLES)
  if (!styles.length) return "noStyle"

  if (!colors) return { emojis, styles }

  const palette = cleanPalette(a.bg, a.fg)
  if (!palette) return "noPalette"
  return { emojis, styles, ...palette }
}

async function annotateBatch(
  batch: { id: number; text: string }[],
  colors: boolean,
  usage: Usage,
  drops: Drops,
): Promise<Map<number, Label>> {
  const ids = new Set(batch.map((b) => b.id))
  const byId = new Map<number, Label>()
  const reason = new Map<number, DropReason>()
  let threw = false

  for (
    let attempt = 0;
    attempt < ANNOTATE_ATTEMPTS && byId.size < ids.size;
    attempt++
  ) {
    try {
      const res = await generateText({
        model: MODEL,
        output: Output.object({
          schema: z.object({ annotations: z.array(Annotation) }),
        }),
        prompt: [
          instructions(colors),
          "",
          "Messages:",
          JSON.stringify(batch),
        ].join("\n"),
      })
      addUsage(usage, res.usage)
      threw = false
      for (const a of res.output.annotations) {
        if (!ids.has(a.id) || byId.has(a.id)) continue
        const label = cleanLabel(a, colors)
        if (typeof label === "string") {
          reason.set(a.id, label)
          if (process.env.ANNOTATE_DEBUG)
            console.warn(
              `\n  #${a.id} ${label}: emojis=${JSON.stringify(a.emojis)}`
              + ` styles=${JSON.stringify(a.styles)}`,
            )
        } else {
          byId.set(a.id, label)
          reason.delete(a.id)
        }
      }
    } catch (err) {
      threw = true
      if (attempt === ANNOTATE_ATTEMPTS - 1) {
        console.warn(`\n  annotate batch of ${batch.length} failed: ${err}`)
      }
    }
  }

  for (const b of batch) {
    if (byId.has(b.id)) continue
    const r = reason.get(b.id)
    if (r) drops[r]++
    else if (threw) drops.batch++
    else drops.missingId++
    console.warn(`\n  dropped id ${b.id}: ${r ?? (threw ? "batch" : "missingId")}`)
  }
  return byId
}

export async function annotate(
  texts: string[],
  opts: AnnotateOpts = {},
): Promise<Map<number, Label>> {
  const colors = opts.colors ?? false
  const items = texts.map((text, id) => ({ id, text }))
  const result = new Map<number, Label>()
  const queue = new PQueue({ concurrency: ANNOTATE_CONCURRENCY })

  lastUsage.calls = 0
  lastUsage.input = 0
  lastUsage.output = 0
  lastUsage.total = 0
  resetDrops(lastDrops)

  queue.addAll(
    chunk(items, ANNOTATE_BATCH_SIZE).map((batch) => async () => {
      const got = await annotateBatch(batch, colors, lastUsage, lastDrops)
      for (const [id, label] of got) result.set(id, label)
      opts.onBatchDone?.()
    }),
  )
  await queue.onIdle()

  console.log(`\n${formatUsage(lastUsage)}`)
  console.log(formatDrops(lastDrops))
  return result
}

export function annotateBatchCount(n: number): number {
  return Math.ceil(n / ANNOTATE_BATCH_SIZE)
}
