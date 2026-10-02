// Generates a 5x5 sprite animation sheet (25 frames, row-major, one seamless loop) for one emoji that has no
// Noto animation: the Noto SVG rasterised to PNG plus its animation_text brief go to the Vercel AI Gateway
// (openai/gpt-image-2, key AI_GATEWAY_API_KEY). The sheet is saved to sprites/<stem>.png and `git add`ed.
// stdout is only the saved path (so it pipes into xargs); warnings and progress go to stderr, and a warned run prints nothing.
//   bun run animate-sprite 🔴            refuses if Noto animates it, we already have a sprite, or there is no brief
//   bun run animate-sprite 🔴 --force    regenerate over an existing sprite
import { existsSync, mkdirSync, mkdtempSync, readFileSync, writeFileSync } from "node:fs"
import { tmpdir } from "node:os"
import { join, resolve } from "node:path"
import { generateImage } from "ai"
import { cac } from "cac"

import { SPRITE_DIR } from "../../files.ts"
import { resolveEmojiSvg } from "../cli/emoji-svg.ts"
import { readRows, stemOf } from "./csv.ts"

const MODEL = process.env.SPRITE_MODEL ?? "openai/gpt-image-2"
const SIZE = (process.env.SPRITE_SIZE ?? "1280x1280") as `${number}x${number}` // 5 cells of 256px
const SRC = 512 // px of the reference PNG
const CHROME = process.env.CHROME ?? ["google-chrome", "chromium", "chromium-browser"].find((b) => Bun.which(b))

const warn = (msg: string): never => {
  console.warn(`warn: ${msg}`)
  process.exit(0)
}

async function svgToPng(body: string, w: number, h: number): Promise<Buffer> {
  if (!CHROME) throw new Error("no Chrome/Chromium found (set CHROME=/path/to/chrome)")
  const dir = mkdtempSync(join(tmpdir(), "emoji-sprite-"))
  const page = join(dir, "emoji.html")
  const png = join(dir, "emoji.png")
  writeFileSync(page, `<!doctype html><body style="margin:0;background:#fff"><svg viewBox="0 0 ${w} ${h}" width="${SRC}" height="${SRC}">${body}</svg>`)
  const p = Bun.spawn(
    [CHROME, "--headless=new", "--no-sandbox", "--disable-gpu", "--hide-scrollbars", `--window-size=${SRC},${SRC}`, `--screenshot=${png}`, `file://${page}`],
    { stdout: "ignore", stderr: "ignore" },
  )
  await p.exited
  return readFileSync(png)
}

const prompt = (name: string, text: string) => `Using the attached emoji image ("${name}") as the exact character and art style, draw a sprite sheet of its animation.
The sheet is a 5x5 grid of 25 equally sized square frames, read left-to-right, top-to-bottom, forming one seamless loop (frame 25 leads back into frame 1).
Every frame shows the same emoji, same size, centred in its cell, on a plain solid white background. No text, no grid lines, no borders, no labels.
Animation: ${text}`

cli()
function cli() {
  const c = cac("animate-sprite")
  c.command("<emoji>", "generate a sprite sheet for one emoji")
    .option("--force", "regenerate even if a sprite already exists")
    .action(async (emoji: string, o: { force?: boolean }) => {
      const row = readRows().find((r) => r.emoji === emoji)
      if (!row) throw new Error(`${emoji} is not in animation.csv`)
      if (row.noto) warn(`${emoji} ${row.name} already has a Noto animation`)
      const out = `${SPRITE_DIR}/${stemOf(emoji)}.png`
      if (existsSync(out) && !o.force) warn(`${emoji} already has a sprite (${out}); use --force to regenerate`)
      if (!row.animation_text) warn(`${emoji} ${row.name} has no animation_text; run: bun run animate-describe --emoji ${emoji}`)
      const svg = resolveEmojiSvg(emoji)
      if (!svg) throw new Error(`no SVG for ${emoji}`)

      const png = await svgToPng(svg.body, svg.width, svg.height)
      console.error(`generating ${MODEL} ${SIZE} sprite for ${emoji} ${row.name}...`)
      const { image } = await generateImage({
        model: MODEL,
        prompt: { text: prompt(row.name, row.animation_text), images: [png] },
        size: SIZE,
      })

      mkdirSync(resolve(SPRITE_DIR), { recursive: true })
      writeFileSync(out, image.uint8Array)
      const git = Bun.spawnSync(["git", "add", "-f", out])
      if (git.exitCode) throw new Error(`git add failed: ${git.stderr.toString()}`)
      console.log(out)
    })
  c.help()
  c.parse()
}
