// Asks an LLM (via the Vercel AI Gateway, key AI_GATEWAY_API_KEY) to design the motion for one emoji.
import { generateText, Output } from "ai"
import type { Scene } from "./svg.ts"

// Best fit for this task: strong spatial reasoning over coordinates + vision (to check the contact
// sheet) + structured output. Override with ANIMATE_MODEL.
export const MODEL = process.env.ANIMATE_MODEL ?? "anthropic/claude-opus-5.5"

const SYSTEM = `You are a motion designer for tiny emoji animations (like Google's animated Noto emoji).
You write the animation as Lottie JSON. The emoji art is given as numbered parts (p0, p1, ... in back-to-front
order) on a 128x128 grid, y down. You are free to animate the parts, hide them, rebuild them, or draw anything
new, as long as it relates to the original emoji and its meaning. Decide for yourself what motion tells the story
best; make it lively and readable at small size.

Guidelines
- Either the first frame or the last frame must show the original emoji exactly (every part, unmoved). Everything
  in between is up to you. 60-240 frames at 60 fps.
- Stay inside the 128x128 grid.
- Use only shape layers (ty 4) and null layers (ty 3), no expressions, no images or precomps.

Answer with a single JSON object and nothing else: {"frames": N, "layers": [...]}. Layers are listed back to front
(like the parts, so the first layer is drawn first and the last one on top) and use grid coordinates (the file is scaled up for you).
- A layer with "ref": "pK" shows the original part pK; do not write its shapes. Reference every part.
- Any other shape layer carries your own "shapes" (Lottie groups "gr" with "el"/"rc"/"sh" paths, "fl"/"st"/"gf"
  paints, "tr" transform, optional "tm" trim paths). Colours are [r,g,b,a] in 0..1.
- Each layer may have "ind" (a number) and "parent" (another layer's ind) to build rigs, and Lottie transform "ks"
  with o (opacity 0-100), r (rotation deg), p (position), a (anchor), s (scale %). Defaults are identity.
  Rotation and scale act around the anchor: to rotate about a point P, set a = P and p = P (animate p by adding
  offsets to P).
- Animated properties are {"a":1,"k":[{"t":frame,"s":[value],"i":{"x":[..],"y":[..]},"o":{"x":[..],"y":[..]}}, ...]}
  (the last key has only t and s; i/o are the usual Lottie easing handles); static ones are {"a":0,"k":value}.`

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

async function ask(emoji: string, scene: Scene, images: Img[], text: string, context?: string): Promise<unknown> {
  const res = await generateText({
    model: MODEL,
    system: SYSTEM,
    messages: [{
      role: "user",
      content: [
        { type: "text", text: `Emoji: ${emoji}\n\n${context ? `${context}\n\n` : ""}Parts:\n${describe(scene)}\n\n${text}` },
        ...images.flatMap((i) => [{ type: "text" as const, text: i.label }, { type: "file" as const, data: i.data, mediaType: "image/png" }]),
      ],
    }],
  })
  if (process.env.ANIMATE_USAGE) console.log(`usage: ${res.usage.inputTokens} in / ${res.usage.outputTokens} out`)
  return parseDoc(res.text)
}

const fixNote = (fix?: { doc: unknown; errors: string[] }) =>
  fix ? `\n\nYour previous answer was rejected:\n${JSON.stringify(fix.doc)}\n\nProblems:\n- ${fix.errors.join("\n- ")}\nReturn the full corrected JSON.` : ""

export const design = (emoji: string, scene: Scene, staticImg: Buffer, context?: string, fix?: { doc: unknown; errors: string[] }) =>
  ask(emoji, scene, [{ data: staticImg, label: "The static emoji:" }], `Design the animation for this emoji.${fixNote(fix)}`, context)

export const refine = (emoji: string, scene: Scene, doc: unknown, sheet: Buffer, context?: string, fix?: { doc: unknown; errors: string[] }) =>
  ask(emoji, scene, [{ data: sheet, label: "Original (top-left) and frames of your animation, evenly spaced over its length:" }],
    `Here is the animation you produced:\n${JSON.stringify(doc)}\n\nReview the frames critically as a motion designer: does it look alive and fit the emoji? Fix anything broken
(parts tearing apart, gaps, leaving the 128 grid, wrong pivots, odd or distracting drawing) and anything weak or
overdone. Return the improved full JSON.${fixNote(fix)}`, context)
