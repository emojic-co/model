// Asks an LLM (via the Vercel AI Gateway, key AI_GATEWAY_API_KEY) to write a Lottie animation for one emoji.
import { generateText, Output } from "ai"
import type { Scene } from "./svg.ts"

// Best fit for this task: strong spatial reasoning over coordinates + vision (to check the contact
// sheet) + structured output. Override with ANIMATE_MODEL.
export const MODEL = process.env.ANIMATE_MODEL ?? "anthropic/claude-opus-5.5"

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

async function ask(emoji: string, svg: string, images: Img[], text: string, context?: string): Promise<unknown> {
  const res = await generateText({
    model: MODEL,
    system: SYSTEM,
    messages: [{
      role: "user",
      content: [
        { type: "text", text: userText(emoji, svg, text, context) },
        ...images.flatMap((i) => [{ type: "text" as const, text: i.label }, { type: "file" as const, data: i.data, mediaType: "image/png" }]),
      ],
    }],
  })
  if (process.env.ANIMATE_USAGE) console.log(`usage: ${res.usage.inputTokens} in / ${res.usage.outputTokens} out`)
  return parseDoc(res.text)
}

type Fix = { doc: unknown; errors: string[] }
const fixNote = (fix?: Fix) =>
  fix ? `\n\nYour previous answer was rejected:\n${JSON.stringify(fix.doc)}\n\nProblems:\n- ${fix.errors.join("\n- ")}\nReturn the full corrected JSON.` : ""

export const DESIGN_TASK = "Animate this emoji."

export const design = (emoji: string, svg: string, staticImg: Buffer, context?: string, fix?: Fix) =>
  ask(emoji, svg, [{ data: staticImg, label: "The static emoji:" }], `${DESIGN_TASK}${fixNote(fix)}`, context)

export const refine = (emoji: string, svg: string, doc: unknown, sheet: Buffer, context?: string, fix?: Fix) =>
  ask(emoji, svg, [{ data: sheet, label: "Original (top-left) and frames of your animation, evenly spaced over its length:" }],
    `Here is the animation you produced:\n${JSON.stringify(doc)}\n\nReview the frames and improve the animation where it is broken, weak or overdone. Return the full JSON.${fixNote(fix)}`, context)
