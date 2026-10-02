// Asks an LLM (via the Vercel AI Gateway, key AI_GATEWAY_API_KEY) to design the motion for one emoji.
import { existsSync, readFileSync, writeFileSync } from "node:fs"

import { generateText, Output } from "ai"

import { ANIMATE_USAGE_JSON } from "../../files.ts"
import type { Scene } from "./svg.ts"
import { Spec } from "./spec.ts"
import type { z } from "zod"

// Best fit for this task: strong spatial reasoning over coordinates + vision (to check the contact
// sheet) + structured output. Override with ANIMATE_MODEL.
export const MODEL = process.env.ANIMATE_MODEL ?? "anthropic/claude-opus-5.5"

// Guardrails against runaway spend. Env overrides: ANIMATE_MAX_RUN_CALLS, ANIMATE_MAX_DAY_CALLS,
// ANIMATE_MAX_DAY_TOKENS, ANIMATE_MAX_OUTPUT_TOKENS, ANIMATE_TIMEOUT_S, ANIMATE_REASONING.
const num = (k: string, d: number) => (Number(process.env[k]) > 0 ? Number(process.env[k]) : d)
export const MAX_REFINE = 3
const MAX_RUN_CALLS = num("ANIMATE_MAX_RUN_CALLS", 8)
const MAX_DAY_CALLS = num("ANIMATE_MAX_DAY_CALLS", 60)
const MAX_DAY_TOKENS = num("ANIMATE_MAX_DAY_TOKENS", 3_000_000)
const MAX_OUTPUT_TOKENS = num("ANIMATE_MAX_OUTPUT_TOKENS", 128_000)
// Thinking effort (none|minimal|low|medium|high|xhigh|provider-default); less thinking = faster and cheaper.
const REASONING = (process.env.ANIMATE_REASONING ?? "provider-default") as "none" | "minimal" | "low" | "medium" | "high" | "xhigh" | "provider-default"
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
const run = { start: Date.now(), modelMs: 0, inTok: 0, outTok: 0, thinkTok: 0, cacheReadTok: 0 }
const secs = (ms: number) => `${(ms / 1000).toFixed(1)}s`
const k = (n: number) => (n >= 1000 ? `${(n / 1000).toFixed(1)}k` : `${n}`)

/** One-line recap of the whole run: wall time, time spent waiting on the model, calls and tokens. */
export const runSummary = () =>
  `run: ${secs(Date.now() - run.start)} total, ${secs(run.modelMs)} in the model; ${runCalls} call${runCalls === 1 ? "" : "s"}, ${k(run.inTok)} in${run.cacheReadTok ? ` (${k(run.cacheReadTok)} cached)` : ""} / ${k(run.outTok)} out${run.thinkTok ? ` (${k(run.thinkTok)} thinking)` : ""} tokens`

type GenArgs = Parameters<typeof generateText>[0]

/** Text size and attached images of a request, e.g. "system 1.1k chars, user 119.8k chars (~40.0k tokens), 2 images". */
function describeRequest(args: GenArgs): string {
  const parts = ((args as { messages?: { content: unknown }[] }).messages ?? []).flatMap((m) => (Array.isArray(m.content) ? m.content : [m.content])) as { type?: string; text?: string; data?: Buffer }[]
  const text = parts.reduce((n, p) => n + (p.text?.length ?? 0), 0)
  const imgs = parts.filter((p) => p.type === "file")
  const sys = typeof (args as { system?: unknown }).system === "string" ? (args as { system: string }).system.length : 0
  return `system ${k(sys)} chars, user ${k(text)} chars (~${k(Math.round((sys + text) / 3))} tokens), ${imgs.length} image${imgs.length === 1 ? "" : "s"}${imgs.length ? ` (${imgs.map((i) => `${Math.round((i.data?.length ?? 0) / 1024)} KB`).join(", ")})` : ""}`
}

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
  console.log(`   request: ${String(args.model ?? MODEL)}, reasoning ${REASONING}, output cap ${k(MAX_OUTPUT_TOKENS)}, timeout ${secs(TIMEOUT_MS)}; ${describeRequest(args)}`)
  const t0 = Date.now()
  // Running clock: rewritten in place every second on a terminal, a line every 15 s otherwise (logs, pipes).
  const tty = process.stdout.isTTY
  const clock = setInterval(() => {
    const t = secs(Date.now() - t0)
    if (tty) process.stdout.write(`\r\x1b[2K   waiting for the model ${t}`)
    else if (Math.round((Date.now() - t0) / 1000) % 15 === 0) console.log(`   waiting for the model ${t}`)
  }, 1000)
  const stopClock = () => {
    clearInterval(clock)
    if (tty) process.stdout.write("\r\x1b[2K")
  }
  try {
    const res = await generateText({ ...args, maxRetries: 0, reasoning: REASONING, maxOutputTokens: MAX_OUTPUT_TOKENS, abortSignal: AbortSignal.timeout(TIMEOUT_MS) })
    stopClock()
    const u = res.usage
    const inT = u.inputTokens ?? 0, outT = u.outputTokens ?? 0
    const think = u.outputTokenDetails?.reasoningTokens ?? 0, text = u.outputTokenDetails?.textTokens
    const { noCacheTokens: fresh, cacheReadTokens: cacheRead = 0, cacheWriteTokens: cacheWrite = 0 } = u.inputTokenDetails ?? {}
    l.tokens += inT + outT
    run.inTok += inT
    run.outTok += outT
    run.thinkTok += think
    run.cacheReadTok += cacheRead ?? 0
    writeFileSync(ANIMATE_USAGE_JSON, JSON.stringify(l))
    const ms = Date.now() - t0
    console.log(`<- ${tag}: ${secs(ms)}, finish: ${res.finishReason}${res.warnings?.length ? `, warnings: ${JSON.stringify(res.warnings)}` : ""}`)
    console.log(`   input  ${k(inT)}: ${fresh === undefined ? "n/a" : `${k(fresh)} uncached`}, ${k(cacheRead ?? 0)} cache read, ${k(cacheWrite ?? 0)} cache write`)
    console.log(`   output ${k(outT)}: ${text === undefined ? "n/a" : `${k(text)} answer`}, ${k(think)} thinking; ${outT ? Math.round(outT / (ms / 1000)) : 0} tokens/s; answer ${k(res.text.length)} chars`)
    console.log(`   today ${l.calls}/${MAX_DAY_CALLS} calls, ${k(l.tokens)}/${k(MAX_DAY_TOKENS)} tokens`)
    if (process.env.ANIMATE_USAGE) console.log(`   raw usage: ${JSON.stringify(u.raw)}`)
    return res
  } catch (e) {
    stopClock()
    console.log(`<- ${tag}: failed after ${secs(Date.now() - t0)}`)
    throw e
  } finally {
    clearInterval(clock)
    run.modelMs += Date.now() - t0
  }
}

const SYSTEM = `You are a motion designer for tiny looping emoji animations (like Google's animated Noto emoji).
You do NOT draw: the emoji art is fixed and given as numbered parts (p0, p1, ... in back-to-front order,
128x128 grid, y down). You only decide how parts move.

Rules
- Loop length 60-240 frames at 60 fps; typical 90-150. The loop must be seamless: every track ends where it starts.
- Frame 0 is the poster frame: every group must be at rest there (rotation 0, move [0,0], scale [1,1], opacity 1).
- Motion must suit the thing the emoji depicts (a flame flickers, a pendulum swings, a bell rings, a drop falls, a
  heart pulses). Keep it charming and subtle: rotations under ~15deg, moves under ~6 grid units, scales 0.9-1.12.
- Everything must stay inside the 128x128 grid; parts must not tear apart (parts that touch should share a group or
  move by matching amounts). Put the pivot where the real object would hinge (a hanging thing: its attachment point;
  a bouncing thing: its bottom centre).
- Group parts that move together (list their ids); parts you leave out stay still. Prefer 1-4 groups. Do not
  animate every part separately unless it clearly helps.
- Optional extras (max 6): small circles/4-point stars (sparkles, bubbles, dust) with their own tracks, for an
  accent. Use colours from the emoji's palette or a soft white/yellow. Skip extras if they add nothing.
- Keyframes: each track needs at least 2 keys {t, v}; t in frames. v shapes: rotation [deg], move [dx,dy],
  scale [sx,sy], opacity [0..1].`

export function describe(scene: Scene): string {
  const col = (p: Scene["parts"][number]["fill"]) =>
    !p ? "none" : p.kind === "solid" ? p.color : `${p.grad.type}-gradient ${p.grad.stops.map((s) => s.color).join(">")}`
  return scene.parts
    .map((p) => `${p.id}: fill ${col(p.fill)}${p.stroke ? `, stroke ${col(p.stroke.paint)}` : ""}, bbox [${p.bbox.map((v) => Math.round(v)).join(", ")}], ${p.nodes} nodes`)
    .join("\n")
}

type Img = { data: Buffer; label: string }

async function ask(tag: string, emoji: string, scene: Scene, images: Img[], text: string, context?: string): Promise<z.infer<typeof Spec>> {
  const res = await guardedGenerate({
    model: MODEL,
    system: SYSTEM,
    output: Output.object({ schema: Spec }),
    messages: [{
      role: "user",
      content: [
        { type: "text", text: `Emoji: ${emoji}\n\n${context ? `${context}\n\n` : ""}Parts:\n${describe(scene)}\n\n${text}` },
        ...images.flatMap((i) => [{ type: "text" as const, text: i.label }, { type: "file" as const, data: i.data, mediaType: "image/png" }]),
      ],
    }],
  }, tag)
  return res.output
}

export const design = (emoji: string, scene: Scene, staticImg: Buffer, context?: string) =>
  ask("design", emoji, scene, [{ data: staticImg, label: "The static emoji:" }], "Design the motion spec for this emoji.", context)

export const refine = (emoji: string, scene: Scene, spec: z.infer<typeof Spec>, sheet: Buffer, context?: string) =>
  ask("refine", emoji, scene, [{ data: sheet, label: "Original (top-left) and frames of your animation, evenly spaced over the loop:" }],
    `Here is the spec you produced:\n${JSON.stringify(spec)}\n\nReview the frames critically: parts tearing apart or leaving gaps, motion leaving the 128 grid,
wrong pivots, motion too weak/violent, extras that look odd or hide the emoji. Return the improved full spec
(identical if it is already good).`, context)
