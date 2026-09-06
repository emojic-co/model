import { existsSync } from "node:fs"
import { readFile } from "node:fs/promises"

import { cac } from "cac"
import cliProgress from "cli-progress"

import { CLDR_JSONL } from "../../files.ts"
import { annotate, annotateBatchCount } from "./annotate.ts"
import { appendJsonl, readJsonl, writeFileAtomic } from "./io.ts"

const CLDR_ANNOTATIONS =
  "node_modules/cldr-annotations-full/annotations/en/annotations.json"
const CLDR_ANNOTATIONS_DERIVED =
  "node_modules/cldr-annotations-derived-full/annotationsDerived/en/annotations.json"
const EMOJIBASE_DATA = "node_modules/emojibase-data/en/data.json"

const MAX_EMOJIS_PER_KEYWORD = 10
const MIN_KEYWORD_LEN = 2

type Annotation = { default?: string[]; tts?: string[] }
type CldrAnnotations = { annotations: { annotations: Record<string, Annotation> } }
type CldrAnnotationsDerived = {
  annotationsDerived: { annotations: Record<string, Annotation> }
}
type EmojibaseEntry = { emoji: string }

export type CldrRecord = { text: string; emojis: string[] }

const VARIATION_SELECTOR = /️/g
const stripVariationSelector = (emoji: string) =>
  emoji.replace(VARIATION_SELECTOR, "")

function byCodePoint(a: string, b: string): number {
  return (a.codePointAt(0) ?? 0) - (b.codePointAt(0) ?? 0) || (a < b ? -1 : a > b ? 1 : 0)
}

export async function loadCldrAnnotations(): Promise<Map<string, string[]>> {
  const [cldr, cldrDerived, emojibase] = await Promise.all([
    readFile(CLDR_ANNOTATIONS, "utf8").then(
      (s) => JSON.parse(s) as CldrAnnotations,
    ),
    readFile(CLDR_ANNOTATIONS_DERIVED, "utf8").then(
      (s) => JSON.parse(s) as CldrAnnotationsDerived,
    ),
    readFile(EMOJIBASE_DATA, "utf8").then((s) => JSON.parse(s) as EmojibaseEntry[]),
  ])
  const canonicalById = new Map(
    emojibase.map((e) => [stripVariationSelector(e.emoji), e.emoji]),
  )
  const merged = {
    ...cldr.annotations.annotations,
    ...cldrDerived.annotationsDerived.annotations,
  }
  const out = new Map<string, string[]>()
  for (const [key, { default: keywords = [] }] of Object.entries(merged)) {
    const glyph = canonicalById.get(stripVariationSelector(key))
    if (!glyph) continue
    out.set(glyph, keywords)
  }
  return out
}

export function invertedIndex(
  annotations: Map<string, string[]>,
  maxEmojis = MAX_EMOJIS_PER_KEYWORD,
): CldrRecord[] {
  const index = new Map<string, Set<string>>()
  for (const [glyph, keywords] of annotations) {
    for (const raw of keywords) {
      const keyword = raw.trim()
      if (keyword.length < MIN_KEYWORD_LEN) continue
      let set = index.get(keyword)
      if (!set) index.set(keyword, (set = new Set()))
      set.add(glyph)
    }
  }
  return [...index.entries()]
    .filter(([, set]) => set.size <= maxEmojis)
    .map(([text, set]) => ({ text, emojis: [...set].sort(byCodePoint) }))
    .sort((a, b) => (a.text < b.text ? -1 : a.text > b.text ? 1 : 0))
}

export function missingRecords(
  records: CldrRecord[],
  existingTexts: Set<string>,
): CldrRecord[] {
  return records.filter((r) => !existingTexts.has(r.text))
}

const cli = cac("cldr")
cli.usage("[options]")
cli
  .option("-f, --force", `overwrite ${CLDR_JSONL} if it already exists`)
  .option(
    "--missing",
    `annotate only keywords absent from ${CLDR_JSONL} and append them`,
  )
  .option(
    "--max-emojis <n>",
    `drop keywords mapping to more than n emoji (default ${MAX_EMOJIS_PER_KEYWORD})`,
  )
  .option("--dry-run", "build and summarize the index only; no annotator, no write")
cli.help()

if (import.meta.main) {
  const { options } = cli.parse(process.argv, { run: false })
  if (options.help) process.exit(0)

  const force = Boolean(options.force)
  const missing = Boolean(options.missing)
  const dryRun = Boolean(options.dryRun)
  const maxEmojis = Number(options.maxEmojis ?? MAX_EMOJIS_PER_KEYWORD)
  if (!(maxEmojis >= 1)) {
    console.error(`--max-emojis must be >= 1, got ${JSON.stringify(options.maxEmojis)}`)
    process.exit(1)
  }

  if (!dryRun && !missing && !force && existsSync(CLDR_JSONL)) {
    console.error(`${CLDR_JSONL} already exists; pass -f / --force to overwrite`)
    process.exit(1)
  }
  if (missing && !existsSync(CLDR_JSONL)) {
    console.error(`--missing needs an existing ${CLDR_JSONL} to append to`)
    process.exit(1)
  }

  const annotations = await loadCldrAnnotations()
  const records = invertedIndex(annotations, maxEmojis)
  const emojiTotal = records.reduce((n, r) => n + r.emojis.length, 0)
  console.log(
    `${annotations.size} CLDR emoji -> ${records.length} keyword records `
    + `(<= ${maxEmojis} emoji each, ${(emojiTotal / (records.length || 1)).toFixed(1)} avg)`,
  )

  let targets = records
  if (missing) {
    const existing = await readJsonl<{ text?: unknown }>(CLDR_JSONL)
    const existingTexts = new Set(
      existing.map((r) => r.text).filter((t): t is string => typeof t === "string"),
    )
    targets = missingRecords(records, existingTexts)
    console.log(
      `${existingTexts.size} keywords already in ${CLDR_JSONL} -> ${targets.length} missing`,
    )
  }

  if (dryRun) {
    for (const r of targets.slice(0, 20)) console.log(`  ${r.text} -> ${r.emojis.join(" ")}`)
    if (targets.length > 20) console.log(`  ... ${targets.length - 20} more`)
    process.exit(0)
  }

  if (missing && !targets.length) {
    console.log(`nothing missing; ${CLDR_JSONL} is up to date`)
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
        meta: { date: today, cldr: true },
      }),
    )
  }
  if (missing) await appendJsonl(CLDR_JSONL, lines)
  else await writeFileAtomic(CLDR_JSONL, lines.join("\n") + "\n")

  console.log("\n--- summary ---")
  console.log(`keyword records      : ${targets.length}`)
  console.log(`${missing ? "appended" : "written "} -> cldr     : ${lines.length}`)
  console.log(`dropped no label     : ${noLabel}`)
  console.log(`dropped no palette   : ${noPalette}`)
  process.exit(0)
}
