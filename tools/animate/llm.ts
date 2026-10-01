// Asks an LLM (via the Vercel AI Gateway, key AI_GATEWAY_API_KEY) to write a Lottie animation for one emoji.
import { existsSync, readFileSync, writeFileSync } from "node:fs"

import { generateText } from "ai"

import { ANIMATE_USAGE_JSON } from "../../files.ts"
import type { Scene } from "./svg.ts"

// Best fit for this task: strong spatial reasoning over coordinates + vision (to check the contact
// sheet) + structured output. Override with ANIMATE_MODEL.
export const MODEL = process.env.ANIMATE_MODEL ?? "anthropic/claude-opus-5.5"

// Guardrails against runaway spend. Env overrides: ANIMATE_MAX_RUN_CALLS, ANIMATE_MAX_DAY_CALLS,
// ANIMATE_MAX_DAY_TOKENS, ANIMATE_MAX_OUTPUT_TOKENS, ANIMATE_TIMEOUT_S.
const num = (k: string, d: number) => (Number(process.env[k]) > 0 ? Number(process.env[k]) : d)
export const MAX_REFINE = 3
const MAX_RUN_CALLS = num("ANIMATE_MAX_RUN_CALLS", 8)
const MAX_DAY_CALLS = num("ANIMATE_MAX_DAY_CALLS", 60)
const MAX_DAY_TOKENS = num("ANIMATE_MAX_DAY_TOKENS", 3_000_000)
const MAX_OUTPUT_TOKENS = num("ANIMATE_MAX_OUTPUT_TOKENS", 32_000)
const TIMEOUT_MS = num("ANIMATE_TIMEOUT_S", 300) * 1000

type Ledger = { day: string; calls: number; tokens: number }
const today = () => new Date().toISOString().slice(0, 10)
function readLedger(): Ledger {
  try {
    if (existsSync(ANIMATE_USAGE_JSON)) {
      const l: Ledger = JSON.parse(readFileSync(ANIMATE_USAGE_JSON, "utf8"))
      if (l.day === today()) return l
    }
  } catch {}
  return { day: today(), calls: 0, tokens: 0 }
}
let runCalls = 0
const run = { start: Date.now(), modelMs: 0, inTok: 0, outTok: 0 }
const secs = (ms: number) => `${(ms / 1000).toFixed(1)}s`
const k = (n: number) => (n >= 1000 ? `${(n / 1000).toFixed(1)}k` : `${n}`)

/** One-line recap of the whole run: wall time, time spent waiting on the model, calls and tokens. */
export const runSummary = () =>
  `run: ${secs(Date.now() - run.start)} total, ${secs(run.modelMs)} in the model; ${runCalls} call${runCalls === 1 ? "" : "s"}, ${k(run.inTok)} in / ${k(run.outTok)} out tokens`

type GenArgs = Parameters<typeof generateText>[0]

/** Every model call goes through here: per-run and per-day limits, output cap, timeout, no hidden SDK retries.
 *  The ledger is bumped before the call so a crash or timeout still counts. */
export async function guardedGenerate<T extends GenArgs>(args: T, tag: string): Promise<Awaited<ReturnType<typeof generateText<any, any>>>> {
  const l = readLedger()
  if (runCalls >= MAX_RUN_CALLS) throw new Error(`run limit reached: ${MAX_RUN_CALLS} model calls (ANIMATE_MAX_RUN_CALLS)`)
  if (l.calls >= MAX_DAY_CALLS) throw new Error(`daily limit reached: ${MAX_DAY_CALLS} model calls today (ANIMATE_MAX_DAY_CALLS)`)
  if (l.tokens >= MAX_DAY_TOKENS) throw new Error(`daily limit reached: ${l.tokens} tokens today (ANIMATE_MAX_DAY_TOKENS)`)
  runCalls++
  l.calls++
  writeFileSync(ANIMATE_USAGE_JSON, JSON.stringify(l))
  console.log(`-> ${tag} (call ${runCalls}/${MAX_RUN_CALLS}, today ${l.calls}/${MAX_DAY_CALLS}) ...`)
  const t0 = Date.now()
  try {
    const res = await generateText({ ...args, maxRetries: 0, maxOutputTokens: MAX_OUTPUT_TOKENS, abortSignal: AbortSignal.timeout(TIMEOUT_MS) })
    const inT = res.usage.inputTokens ?? 0, outT = res.usage.outputTokens ?? 0
    l.tokens += inT + outT
    run.inTok += inT
    run.outTok += outT
    writeFileSync(ANIMATE_USAGE_JSON, JSON.stringify(l))
    console.log(`<- ${tag}: ${secs(Date.now() - t0)}, ${k(inT)} in / ${k(outT)} out tokens (today ${k(l.tokens)}/${k(MAX_DAY_TOKENS)}), finish: ${res.finishReason}`)
    return res
  } catch (e) {
    console.log(`<- ${tag}: failed after ${secs(Date.now() - t0)}`)
    throw e
  } finally {
    run.modelMs += Date.now() - t0
  }
}

export const SYSTEM = `Your task is to generate a Lottie animation for a static emoji.
The emoji is given to you as a static SVG file (viewBox 0 0 128 128) and you should return a Lottie JSON animation that animates the emoji.

- Make an animation that fits the emoji and what it depicts: lively, readable at small size. You decide what moves; you
  may change, add or remove shapes as long as everything relates to the original emoji.
- The first frame or the last frame must show the static emoji exactly as given.
- Use a 128x128 canvas (w = h = 128) so SVG coordinates can be used as they are. 60 fps, 60-240 frames.
- Shape layers only, no expressions, images or precomps.

Return only the Lottie JSON.`

export function describe(scene: Scene): string {
  const col = (p: Scene["parts"][number]["fill"]) =>
    !p ? "none" : p.kind === "solid" ? p.color : `${p.grad.type}-gradient ${p.grad.stops.map((s) => s.color).join(">")}`
  return scene.parts
    .map((p) => `${p.id}: fill ${col(p.fill)}${p.stroke ? `, stroke ${col(p.stroke.paint)}` : ""}, bbox [${p.bbox.map((v) => Math.round(v)).join(", ")}], ${p.nodes} nodes`)
    .join("\n")
}

type Img = { data: Buffer; label: string }

function parseDoc(text: string): unknown {
  const i = text.indexOf("{"), j = text.lastIndexOf("}")
  if (i < 0 || j < i) throw new Error("the answer contained no JSON object")
  return JSON.parse(text.slice(i, j + 1))
}

export const userText = (emoji: string, svg: string, text: string, context?: string) =>
  `Emoji: ${emoji}\n\n${context ? `${context}\n\n` : ""}Static SVG:\n${svg}\n\n${text}`

async function ask(tag: string, emoji: string, svg: string, images: Img[], text: string, context?: string): Promise<unknown> {
  const res = await guardedGenerate({
    model: MODEL,
    system: SYSTEM,
    messages: [{
      role: "user",
      content: [
        { type: "text", text: userText(emoji, svg, text, context) },
        ...images.flatMap((i) => [{ type: "text" as const, text: i.label }, { type: "file" as const, data: i.data, mediaType: "image/png" }]),
      ],
    }],
  }, tag)
  return parseDoc(res.text)
}

type Fix = { doc: unknown; errors: string[] }
const fixNote = (fix?: Fix) =>
  fix ? `\n\nYour previous answer was rejected:\n${JSON.stringify(fix.doc)}\n\nProblems:\n- ${fix.errors.join("\n- ")}\nReturn the full corrected JSON.` : ""

export const DESIGN_TASK = "Animate this emoji."

export const design = (emoji: string, svg: string, staticImg: Buffer, context?: string, fix?: Fix) =>
  ask(`design${fix ? " (fix)" : ""}`, emoji, svg, [{ data: staticImg, label: "The static emoji:" }], `${DESIGN_TASK}${fixNote(fix)}`, context)

export const refine = (emoji: string, svg: string, doc: unknown, sheet: Buffer, context?: string, fix?: Fix) =>
  ask(`refine${fix ? " (fix)" : ""}`, emoji, svg, [{ data: sheet, label: "Original (top-left) and frames of your animation, evenly spaced over its length:" }],
    `Here is the animation you produced:\n${JSON.stringify(doc)}\n\nReview the frames and improve the animation where it is broken, weak or overdone. Return the full JSON.${fixNote(fix)}`, context)
