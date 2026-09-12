import { existsSync, readFileSync } from "node:fs"
import { readFile } from "node:fs/promises"

import { cac } from "cac"
import cliProgress from "cli-progress"
import emojilibData from "emojilib"

import { EMOJILIB_JSONL, II_JSON } from "../../files.ts"
import { annotate, annotateBatchCount } from "./annotate.ts"
import { type CldrRecord, invertedIndex, missingRecords } from "./cldr.ts"
import { appendJsonl, readJsonl, writeFileAtomic } from "./io.ts"

const EMOJIBASE_DATA = "node_modules/emojibase-data/en/data.json"
const MAX_EMOJIS_PER_KEYWORD = 10

type EmojibaseEntry = { emoji: string }

const VARIATION_SELECTOR = /️/g
const stripVariationSelector = (emoji: string) => emoji.replace(VARIATION_SELECTOR, "")

function byCodePoint(a: string, b: string): number {
  return (a.codePointAt(0) ?? 0) - (b.codePointAt(0) ?? 0) || (a < b ? -1 : a > b ? 1 : 0)
}

export function keywordText(raw: string): string {
  return raw.replace(/_/g, " ").trim()
}

export async function loadEmojilibAnnotations(): Promise<Map<string, string[]>> {
  const emojibase = JSON.parse(
    await readFile(EMOJIBASE_DATA, "utf8"),
  ) as EmojibaseEntry[]
  const canonicalById = new Map(
    emojibase.map((e) => [stripVariationSelector(e.emoji), e.emoji]),
  )
  const out = new Map<string, string[]>()
  for (const [glyph, keywords] of Object.entries(
    emojilibData as Record<string, string[]>,
  )) {
    const canon = canonicalById.get(stripVariationSelector(glyph))
    if (!canon) continue
    out.set(canon, keywords.map(keywordText))
  }
  return out
}

async function mergeIntoIndex(
  records: CldrRecord[],
): Promise<{ added: number; extended: number; addedEmojis: number }> {
  const ii = JSON.parse(readFileSync(II_JSON, "utf8")) as Record<string, string[]>
  let added = 0
  let extended = 0
  let addedEmojis = 0
  for (const r of records) {
    const existing = ii[r.text]
    if (!existing) {
      ii[r.text] = [...r.emojis].sort(byCodePoint)
      added++
      addedEmojis += r.emojis.length
      continue
    }
    const set = new Set(existing)
    let changed = false
    for (const e of r.emojis) {
      if (!set.has(e)) {
        set.add(e)
        addedEmojis++
        changed = true
      }
    }
    if (changed) {
      ii[r.text] = [...set].sort(byCodePoint)
      extended++
    }
  }
  const sorted: Record<string, string[]> = {}
  for (const k of Object.keys(ii).sort((a, b) => (a < b ? -1 : a > b ? 1 : 0))) {
    sorted[k] = ii[k]
  }
  await writeFileAtomic(II_JSON, JSON.stringify(sorted, null, 2) + "\n")
  return { added, extended, addedEmojis }
}

const cli = cac("emojilib")
cli.usage("[options]")
cli
  .option("-f, --force", `overwrite ${EMOJILIB_JSONL} if it already exists`)
  .option(
    "--missing",
    `annotate only keywords absent from ${EMOJILIB_JSONL} and append them`,
  )
  .option(
    "--max-emojis <n>",
    `drop keywords mapping to more than n emoji (default ${MAX_EMOJIS_PER_KEYWORD})`,
  )
  .option("--dry-run", "build and summarize the index only; no annotator, no write")
  .option("--merge-ii", `merge the keyword -> emoji index into ${II_JSON} and exit`)
cli.help()

if (import.meta.main) {
  const { options } = cli.parse(process.argv, { run: false })
  if (options.help) process.exit(0)

  const force = Boolean(options.force)
  const missing = Boolean(options.missing)
  const dryRun = Boolean(options.dryRun)
  const mergeIi = Boolean(options.mergeIi)
  const maxEmojis = Number(options.maxEmojis ?? MAX_EMOJIS_PER_KEYWORD)
  if (!(maxEmojis >= 1)) {
    console.error(`--max-emojis must be >= 1, got ${JSON.stringify(options.maxEmojis)}`)
    process.exit(1)
  }

  if (!dryRun && !mergeIi && !missing && !force && existsSync(EMOJILIB_JSONL)) {
    console.error(`${EMOJILIB_JSONL} already exists; pass -f / --force to overwrite`)
    process.exit(1)
  }
  if (missing && !existsSync(EMOJILIB_JSONL)) {
    console.error(`--missing needs an existing ${EMOJILIB_JSONL} to append to`)
    process.exit(1)
  }

  const annotations = await loadEmojilibAnnotations()
  const records = invertedIndex(annotations, maxEmojis)
  const emojiTotal = records.reduce((n, r) => n + r.emojis.length, 0)
  console.log(
    `${annotations.size} emojilib emoji -> ${records.length} keyword records `
    + `(<= ${maxEmojis} emoji each, ${(emojiTotal / (records.length || 1)).toFixed(1)} avg)`,
  )

  if (mergeIi) {
    const { added, extended, addedEmojis } = await mergeIntoIndex(records)
    console.log(
      `-> ${II_JSON} : +${added} new keywords, ${extended} extended, `
      + `+${addedEmojis} emoji link(s)`,
    )
    process.exit(0)
  }

  let targets = records
  if (missing) {
    const existing = await readJsonl<{ text?: unknown }>(EMOJILIB_JSONL)
    const existingTexts = new Set(
      existing.map((r) => r.text).filter((t): t is string => typeof t === "string"),
    )
    targets = missingRecords(records, existingTexts)
    console.log(
      `${existingTexts.size} keywords already in ${EMOJILIB_JSONL} -> ${targets.length} missing`,
    )
  }

  if (dryRun) {
    for (const r of targets.slice(0, 20)) console.log(`  ${r.text} -> ${r.emojis.join(" ")}`)
    if (targets.length > 20) console.log(`  ... ${targets.length - 20} more`)
    process.exit(0)
  }

  if (missing && !targets.length) {
    console.log(`nothing missing; ${EMOJILIB_JSONL} is up to date`)
    process.exit(0)
  }

  const annBar = new cliProgress.SingleBar(
    {
      format:
        "annotating |{bar}| {percentage}% | {value}/{total} batches | ETA: {eta}s",
    },
    cliProgress.Presets.shades_classic,
  )
  annBar.start(annotateBatchCount(targets.length), 0)
  const labels = await annotate(
    targets.map((r) => r.text),
    { colors: true, onBatchDone: () => annBar.increment() },
  )
  annBar.stop()

  const today = new Date().toISOString().slice(0, 10)
  const lines: string[] = []
  let noLabel = 0
  let noPalette = 0
  for (let i = 0; i < targets.length; i++) {
    const label = labels.get(i)
    if (!label) {
      noLabel++
      continue
    }
    if (!label.bg || !label.fg) {
      noPalette++
      continue
    }
    lines.push(
      JSON.stringify({
        text: targets[i].text,
        emojis: targets[i].emojis.join(" "),
        styles: label.styles,
        bg: label.bg,
        fg: label.fg,
        meta: { date: today, emojilib: true },
      }),
    )
  }
  if (missing) await appendJsonl(EMOJILIB_JSONL, lines)
  else await writeFileAtomic(EMOJILIB_JSONL, lines.join("\n") + "\n")

  console.log("\n--- summary ---")
  console.log(`keyword records      : ${targets.length}`)
  console.log(`${missing ? "appended" : "written "} -> emojilib : ${lines.length}`)
  console.log(`dropped no label     : ${noLabel}`)
  console.log(`dropped no palette   : ${noPalette}`)
  process.exit(0)
}
