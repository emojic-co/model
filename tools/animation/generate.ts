// Step 2: for one emoji, writes animation/svg/<emoji_code>/frame_01.svg ... frame_20.svg from the emoji's SVG,
// its metadata and animation/json/<emoji_code>.json. One model call per frame, a few in parallel.
// Frames that already exist are skipped (so an interrupted run resumes) unless --force.
//   bun run animation-svg 🔴
//   bun run animation-svg 🔴 --frame 7 --force     redo one frame
import { existsSync, mkdirSync, readFileSync, writeFileSync } from "node:fs"

import { cac } from "cac"

import { ask, emojiMeta, emojiSvg, FRAMES, framePath, jsonPath, parseEmoji, stemOf, svgDir } from "./common.ts"

const MODEL = process.env.ANIMATION_SVG_MODEL ?? "anthropic/claude-opus-5.5" // or anthropic/claude-sonnet-5.5
const CONCURRENCY = Number(process.env.ANIMATION_CONCURRENCY) || 4

const SYSTEM = `You are an SVG animator. You get a static emoji as an SVG, its metadata, the description of all frames of a ${FRAMES}-frame looping animation, and the number of ONE frame to draw.
Output the complete SVG of that single frame and nothing else (no markdown fence, no commentary).
Rules:
- Start from the given SVG: keep its viewBox, xmlns, colours, gradients and drawing style. Edit it to match the frame description: wrap parts in <g transform="..."> to rotate/translate/scale them (set transform-origin sensibly via rotate(angle cx cy)), change shapes, add small simple shapes for accents.
- Keep the same canvas, size and centering; the result must stay inside the viewBox.
- Parts that are attached must stay attached; do not tear the drawing apart. Parts a frame does not mention stay as drawn.
- Make the change visible but proportionate to the description. Neighbouring frames differ a little, so apply exactly this frame's pose, not an earlier or later one.
- Valid standalone SVG 1.1: no scripts, no external references, no CSS animation, no <animate>. Unique ids if you add any.`

export function extractSvg(text: string): string {
  const m = text.match(/<svg[\s\S]*<\/svg>/i)
  if (!m) throw new Error("no <svg> in the answer")
  const svg = m[0]
  if ((svg.match(/<svg[\s>]/gi) ?? []).length !== 1) throw new Error("nested or multiple <svg> roots")
  if (/<script|<animate|<foreignObject|\b(?:xlink:)?href\s*=\s*["'](?!#)/i.test(svg)) throw new Error("svg has scripts, animation or external references")
  if (!/viewBox\s*=/.test(svg)) throw new Error("svg has no viewBox")
  return svg + "\n"
}

const cli = cac("animation-svg")
cli
  .command("<emoji>", "generate the 20 frame SVGs for one emoji")
  .option("--force", "regenerate frames that already exist")
  .option("--frame <n>", "only this frame (1-20)")
  .action(async (arg: string, o: { force?: boolean; frame?: string }) => {
    const emoji = parseEmoji(arg)
    const stem = stemOf(emoji)
    if (!existsSync(jsonPath(stem))) {
      console.warn(`warn: no descriptions at ${jsonPath(stem)}; run: bun run animation-describe ${emoji}`)
      return
    }
    const descriptions: string[] = JSON.parse(readFileSync(jsonPath(stem), "utf8"))
    if (descriptions.length !== FRAMES) throw new Error(`${jsonPath(stem)} has ${descriptions.length} descriptions, expected ${FRAMES}`)
    const svg = emojiSvg(emoji)
    const meta = emojiMeta(emoji)
    const all = descriptions.map((d, i) => `${i + 1}. ${d}`).join("\n")
    mkdirSync(svgDir(stem), { recursive: true })

    let todo = Array.from({ length: FRAMES }, (_, i) => i + 1)
    if (o.frame) {
      const n = Number(o.frame)
      if (!Number.isInteger(n) || n < 1 || n > FRAMES) throw new Error(`--frame must be 1-${FRAMES}`)
      todo = [n]
    }
    todo = todo.filter((n) => o.force || !existsSync(framePath(stem, n)))
    console.error(`${todo.length} frame(s) to generate for ${emoji} with ${MODEL}`)

    const failed: number[] = []
    const work = async () => {
      for (let n = todo.shift(); n !== undefined; n = todo.shift()) {
        const user = `${meta}\n\nStatic SVG:\n${svg}\n\nAll frame descriptions:\n${all}\n\nDraw frame ${n} of ${FRAMES}: ${descriptions[n - 1]}`
        try {
          const call = async () => extractSvg((await ask(MODEL, SYSTEM, user, `frame ${n}`, { maxOutputTokens: 64_000 })).text)
          writeFileSync(framePath(stem, n), await call().catch(call)) // one retry on a malformed answer
        } catch (e) {
          console.error(`frame ${n} failed: ${(e as Error).message}`)
          failed.push(n)
        }
      }
    }
    await Promise.all(Array.from({ length: CONCURRENCY }, work))
    if (failed.length) {
      console.error(`failed frames: ${failed.sort((a, b) => a - b).join(", ")} (re-run to retry them)`)
      process.exitCode = 1
    }
    console.log(svgDir(stem))
  })
cli.help()
cli.parse()
