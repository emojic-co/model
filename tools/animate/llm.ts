// Asks an LLM (via the Vercel AI Gateway, key AI_GATEWAY_API_KEY) to design the motion for one emoji.
import { generateText, Output } from "ai"
import type { Scene } from "./svg.ts"
import { Spec } from "./spec.ts"
import type { z } from "zod"

// Best fit for this task: strong spatial reasoning over coordinates + vision (to check the contact
// sheet) + structured output. Override with ANIMATE_MODEL.
export const MODEL = process.env.ANIMATE_MODEL ?? "anthropic/claude-opus-5.5"

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

async function ask(emoji: string, scene: Scene, images: Img[], text: string): Promise<z.infer<typeof Spec>> {
  const res = await generateText({
    model: MODEL,
    system: SYSTEM,
    output: Output.object({ schema: Spec }),
    messages: [{
      role: "user",
      content: [
        { type: "text", text: `Emoji: ${emoji}\n\nParts:\n${describe(scene)}\n\n${text}` },
        ...images.flatMap((i) => [{ type: "text" as const, text: i.label }, { type: "file" as const, data: i.data, mediaType: "image/png" }]),
      ],
    }],
  })
  return res.output
}

export const design = (emoji: string, scene: Scene, staticImg: Buffer) =>
  ask(emoji, scene, [{ data: staticImg, label: "The static emoji:" }], "Design the motion spec for this emoji.")

export const refine = (emoji: string, scene: Scene, spec: z.infer<typeof Spec>, sheet: Buffer) =>
  ask(emoji, scene, [{ data: sheet, label: "Original (top-left) and frames of your animation, evenly spaced over the loop:" }],
    `Here is the spec you produced:\n${JSON.stringify(spec)}\n\nReview the frames critically: parts tearing apart or leaving gaps, motion leaving the 128 grid,
wrong pivots, motion too weak/violent, extras that look odd or hide the emoji. Return the improved full spec
(identical if it is already good).`)
