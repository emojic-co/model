import { mkdir } from "node:fs/promises"

import { cac } from "cac"
import cliProgress from "cli-progress"
import PQueue from "p-queue"

import { cardHtml, firstEmoji, page } from "../data/preview-card.ts"
import { parseJsonlText, readJsonl } from "../data/io.ts"

const DEFAULT_RESOLUTION = 512
const DEFAULT_CONCURRENCY = 8

type Row = {
  text: string
  emojis: string
  styles: string[]
  bg: [string, string]
  fg: string
}

function cardCss(resolution: number): string {
  return `
html, body { margin: 0; padding: 0; width: ${resolution}px; height: ${resolution}px; }
.card { width: ${resolution}px; height: ${resolution}px; max-width: none; border-radius: 0; }
`
}

function rowHtml(r: Row, resolution: number): Promise<string> {
  return page({
    title: "card",
    extraCss: cardCss(resolution),
    body: cardHtml({
      text: r.text,
      emoji: firstEmoji(r.emojis),
      feeling: r.styles[0] ?? "Neutral",
      colors: { bg1: r.bg[0], bg2: r.bg[1], text_color: r.fg },
    }),
  })
}

async function browserChromeOffset(): Promise<number> {
  const probeHeight = 600
  const view = new Bun.WebView({ width: 512, height: probeHeight, backend: "chrome" })
  try {
    await view.navigate("data:text/html,<html><body></body></html>")
    const innerHeight = await view.evaluate<number>("window.innerHeight")
    return probeHeight - innerHeight
  } finally {
    view.close()
  }
}

async function renderCard(html: string, offset: number, resolution: number, dest: string): Promise<void> {
  const view = new Bun.WebView({ width: resolution, height: resolution + offset, backend: "chrome" })
  try {
    await view.navigate(`data:text/html,${encodeURIComponent(html)}`)
    await view.evaluate("document.fonts.ready.then(() => true)")
    const blob = await view.screenshot({ format: "jpeg", quality: 90 })
    await Bun.write(dest, blob)
  } finally {
    view.close()
  }
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
  const offset = await browserChromeOffset()

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
        const html = await rowHtml(r, resolution)
        const dest = `${outDir}/${String(i).padStart(width, "0")}.jpg`
        await renderCard(html, offset, resolution, dest)
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
