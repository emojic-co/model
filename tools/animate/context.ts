// Extra context for the motion designer: what the emoji is (name, keywords, category) and what each
// numbered SVG part is (a vision pass over every part rendered on its own).
import { readFileSync } from "node:fs"

import { generateText, Output } from "ai"
import { z } from "zod"

import { describe, MODEL } from "./llm.ts"
import { partsPng, staticPng } from "./render.ts"
import type { Part, Scene } from "./svg.ts"

type Emojibase = { emoji: string; label: string; tags?: string[]; group?: number; subgroup?: number }
type Messages = { groups: { order: number; message: string }[]; subgroups: { order: number; message: string }[] }
const dir = "node_modules/emojibase-data/en"

export function emojiMeta(emoji: string): string {
  const bare = (e: string) => e.replace(/️/g, "")
  const data: Emojibase[] = JSON.parse(readFileSync(`${dir}/data.json`, "utf8"))
  const msg: Messages = JSON.parse(readFileSync(`${dir}/messages.json`, "utf8"))
  const e = data.find((d) => bare(d.emoji) === bare(emoji))
  if (!e) return ""
  const group = msg.groups.find((g) => g.order === e.group)?.message
  const sub = msg.subgroups.find((g) => g.order === e.subgroup)?.message
  return `Name: ${e.label}\nCategory: ${[group, sub].filter(Boolean).join(" / ")}\nKeywords: ${(e.tags ?? []).join(", ")}`
}

const r1 = (n: number) => Math.round(n * 10) / 10
/** SVG path elements for one part (solid colour, or a gradient's first stop): enough to recognise it. */
function partPaths(p: Part, opacity = 1): string {
  const fill = !p.fill ? "none" : p.fill.kind === "solid" ? p.fill.color : p.fill.grad.stops[0].color
  return p.shapes
    .map((s) => {
      if (s.t === "ellipse") return `<ellipse cx="${s.cx}" cy="${s.cy}" rx="${s.rx}" ry="${s.ry}" fill="${fill}" opacity="${opacity}"/>`
      const d = s.subs
        .map((u) => {
          const n = u.v.length
          let out = `M${r1(u.v[0][0])} ${r1(u.v[0][1])}`
          for (let k = 1; k < (u.c ? n + 1 : n); k++) {
            const a = k - 1, b = k % n
            out += `C${r1(u.v[a][0] + u.o[a][0])} ${r1(u.v[a][1] + u.o[a][1])} ${r1(u.v[b][0] + u.i[b][0])} ${r1(u.v[b][1] + u.i[b][1])} ${r1(u.v[b][0])} ${r1(u.v[b][1])}`
          }
          return out + (u.c ? "Z" : "")
        })
        .join("")
      return `<path d="${d}" fill="${fill}" opacity="${opacity}"/>`
    })
    .join("")
}

const Labels = z.object({
  parts: z.array(z.object({
    id: z.string(),
    role: z.string().describe("what this part is, e.g. 'head', 'left arm', 'tongue', 'shadow under the object'"),
    note: z.string().optional().describe("only if useful for animating: fused objects (e.g. 'legs and torso in one shape'), what it attaches to, where it hinges"),
  })),
})

/** One vision call: every part on its own (over a faint ghost of the whole emoji) -> role per part. */
export async function labelParts(emoji: string, scene: Scene, staticBody: string, meta: string): Promise<{ id: string; role: string; note?: string }[]> {
  const ghost = scene.parts.map((p) => partPaths(p, 0.12)).join("")
  const sheet = await partsPng(scene.parts.map((p) => ({ body: ghost + partPaths(p), label: p.id })))
  const res = await generateText({
    model: MODEL,
    system: "You identify the parts of a flat emoji illustration so a motion designer can animate them. Be concrete and brief.",
    output: Output.object({ schema: Labels }),
    messages: [{
      role: "user",
      content: [
        { type: "text", text: `Emoji ${emoji}\n${meta}\n\nParts (128 grid):\n${describe(scene)}\n\nBelow: the whole emoji, then each part alone (full colour) over a faint ghost of the emoji. Label every part.` },
        { type: "file", data: await staticPng(staticBody), mediaType: "image/png" },
        { type: "file", data: sheet, mediaType: "image/png" },
      ],
    }],
  })
  if (process.env.ANIMATE_USAGE) console.log(`usage: ${res.usage.inputTokens} in / ${res.usage.outputTokens} out (labels)`)
  return res.output.parts
}

export const contextText = (meta: string, labels: Awaited<ReturnType<typeof labelParts>>) =>
  `About this emoji:\n${meta}\n\nWhat each part is:\n${labels.map((l) => `${l.id}: ${l.role}${l.note ? ` (${l.note})` : ""}`).join("\n")}`
