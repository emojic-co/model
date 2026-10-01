// The motion spec an LLM returns: it describes *movement only*; the art always comes from the
// emoji's own SVG parts, so the animated emoji keeps Noto's look.
import { z } from "zod"

const Key = z.object({ t: z.number().describe("frame, 0..frames"), v: z.array(z.number()) })
const Track = z.object({
  ease: z.enum(["inOut", "linear", "in", "out"]).optional().describe("easing for every segment; default inOut"),
  keys: z.array(Key).min(2),
})
const Anim = {
  rotation: Track.optional().describe("v=[degrees], clockwise positive, about the pivot"),
  move: Track.optional().describe("v=[dx,dy] in 128-grid units, offset from rest; +y is down"),
  scale: Track.optional().describe("v=[sx,sy] multipliers about the pivot, 1 = rest"),
  opacity: Track.optional().describe("v=[0..1]"),
}

export const Group = z.object({
  name: z.string(),
  parts: z.array(z.string()).min(1).describe("part ids (e.g. p0) that move together"),
  pivot: z.array(z.number()).length(2).describe("[x,y] in the 128 grid that rotation/scale act around"),
  ...Anim,
})

export const Extra = z.object({
  name: z.string(),
  shape: z.enum(["circle", "star4"]),
  color: z.string().describe("#rrggbb"),
  center: z.array(z.number()).length(2),
  size: z.number().describe("radius in the 128 grid"),
  behind: z.boolean().optional().describe("draw behind the emoji instead of in front"),
  ...Anim,
})

export const Spec = z.object({
  frames: z.number().int().min(60).max(240).describe("loop length at 60 fps"),
  groups: z.array(Group),
  extras: z.array(Extra).max(6),
})
export type Spec = z.infer<typeof Spec>
export type TrackT = z.infer<typeof Track>
