import { mkdir } from "node:fs/promises"

import { cac } from "cac"
import cliProgress from "cli-progress"
import PQueue from "p-queue"

import { parseJsonlText, readJsonl } from "../data/io.ts"
import { ensureFonts, FONTS_XDG_DATA_HOME, type FontCache } from "./fonts.ts"
import { cardSvg, type Row } from "./svg-card.ts"

const DEFAULT_RESOLUTION = 512
const DEFAULT_CONCURRENCY = 8

async function renderCard(row: Row, resolution: number, fonts: FontCache, dest: string): Promise<void> {
  const svg = cardSvg(row, resolution, fonts)
  const proc = Bun.spawn(["rsvg-convert", "-f", "png", "-"], {
    stdin: "pipe",
    stdout: "pipe",
    stderr: "pipe",
    env: { ...process.env, XDG_DATA_HOME: FONTS_XDG_DATA_HOME },
  })
  proc.stdin.write(svg)
  proc.stdin.end()
  const [png, code] = await Promise.all([new Response(proc.stdout).arrayBuffer(), proc.exited])
  if (code !== 0) {
    const err = await new Response(proc.stderr).text()
    throw new Error(`rsvg-convert exited ${code}: ${err}`)
  }
  const jpeg = await new Bun.Image(png).jpeg({ quality: 90 }).bytes()
  await Bun.write(dest, jpeg)
}

const cli = cac("imgs")
cli.usage("<data.jsonl> <output_dir>")
cli.option("-p, --concurrency <n>", "generation concurrency", { default: DEFAULT_CONCURRENCY })
cli.option("--resolution <n>", "output image resolution in pixels", { default: DEFAULT_RESOLUTION })
cli.help()

if (import.meta.main) {
  const parsed = cli.parse(process.argv, { run: false })
  if (parsed.options.help) process.exit(0)

  const [src, outDir] = parsed.args
  if (!src || !outDir) {
    console.error("usage: imgs [-p <concurrency>] [--resolution <n>] <data.jsonl> <output_dir>")
    process.exit(1)
  }

  const concurrency = Number(parsed.options.concurrency) || DEFAULT_CONCURRENCY
  const resolution = Number(parsed.options.resolution) || DEFAULT_RESOLUTION
  const rows = src === "-"
    ? parseJsonlText<Row>(await Bun.stdin.text(), "stdin")
    : await readJsonl<Row>(src)
  if (!rows.length) {
    console.error(`imgs: no rows in ${src}`)
    process.exit(1)
  }

  await mkdir(outDir, { recursive: true })
  const width = Math.max(4, String(rows.length - 1).length)

  process.stdout.write("imgs: fetching fonts...\n")
  const fonts = await ensureFonts()

  const bar = new cliProgress.SingleBar(
    { format: "imgs |{bar}| {percentage}% | {value}/{total} | ETA: {eta}s" },
    cliProgress.Presets.shades_classic,
  )
  bar.start(rows.length, 0)

  let failed = 0
  const queue = new PQueue({ concurrency })
  await queue.addAll(
    rows.map((r, i) => async () => {
      try {
        const dest = `${outDir}/${String(i).padStart(width, "0")}.jpg`
        await renderCard(r, resolution, fonts, dest)
      } catch (err) {
        failed++
        console.error(`\nimgs: row ${i} failed: ${err}`)
      } finally {
        bar.increment()
      }
    }),
  )
  bar.stop()

  console.log(`wrote ${rows.length - failed}/${rows.length} images to ${outDir}`)
  process.exit(failed ? 1 : 0)
}
