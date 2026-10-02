// Step 1: writes animation/json/<emoji_code>.json, a JSON array of 20 short descriptions, one per animation frame.
//   bun run animation-describe 🔴            (or the code: 1f534)
//   bun run animation-describe 🔴 --force    overwrite an existing file
import { existsSync, mkdirSync, writeFileSync } from "node:fs"
import { dirname } from "node:path"

import { Output } from "ai"
import { cac } from "cac"
import { z } from "zod"

import { ask, emojiMeta, emojiSvg, FRAMES, jsonPath, parseEmoji, stemOf } from "./common.ts"

const MODEL = process.env.ANIMATION_DESCRIBE_MODEL ?? "openai/gpt-5.6-luna"

const SYSTEM = `You are a motion designer planning a tiny looping emoji animation of exactly ${FRAMES} frames, played at ~12 fps.
Write one short description (one sentence) per frame, in order, saying exactly what the emoji looks like in that frame.
Rules:
- Frame 1 is the emoji at rest, as drawn. Frame ${FRAMES} is one small step before frame 1, so the loop is seamless.
- The motion must suit what the emoji depicts (a flame flickers, a bell swings, a heart pulses, a drop falls...). Charming, clear, not violent.
- Name concrete parts of the drawing and concrete amounts ("tilts 8 degrees left", "moves 3 units up", "eyes half closed"). Use the SVG to see what parts exist.
- Describe the change relative to the static art, not the art itself. Keep everything inside the canvas. Small accents (sparkles, drops, steam) are fine.
- Use the full range of the motion: a swing goes both left and right, a pulse grows and shrinks, a bounce goes up and comes down.
- Smooth progression: each frame differs a little from the previous one; the motion peaks around the middle of the loop.`

const Out = z.object({ frames: z.array(z.string()).length(FRAMES).describe(`exactly ${FRAMES} frame descriptions, frame 1 first`) })

const cli = cac("animation-describe")
cli
  .command("<emoji>", "write the 20 frame descriptions for one emoji")
  .option("--force", "overwrite an existing file")
  .action(async (arg: string, o: { force?: boolean }) => {
    const emoji = parseEmoji(arg)
    const out = jsonPath(stemOf(emoji))
    if (existsSync(out) && !o.force) {
      console.warn(`warn: ${out} exists; use --force to overwrite`)
      return
    }
    const user = `${emojiMeta(emoji)}\n\nIts SVG:\n${emojiSvg(emoji)}\n\nWrite the ${FRAMES} frame descriptions.`
    const call = () => ask(MODEL, SYSTEM, user, "describe frames", { output: Output.object({ schema: Out }) })
    const { output } = await call().catch(call) // structured output occasionally misses the schema
    const frames = (output as z.infer<typeof Out>).frames.map((f) => f.replace(/\s+/g, " ").trim())
    mkdirSync(dirname(out), { recursive: true })
    writeFileSync(out, JSON.stringify(frames, null, 2) + "\n")
    console.log(out)
  })
cli.help()
cli.parse()
