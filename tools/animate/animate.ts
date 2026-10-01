// Animates one emoji that has no Noto animation: pick the least complex -> LLM designs the motion ->
// Lottie is built from the emoji's own SVG parts -> added to the animated-emoji collection (web +
// Android assets) -> emoji_animation.csv and the review page are updated.
//   bun run animate-list            rebuild emoji_animation.csv
//   bun run animate                 animate the next emoji     (--emoji 🔴 to force one, --refine N)
//   bun run animate-page            rebuild the review page only
import { readFileSync, writeFileSync } from "node:fs"

import { cac } from "cac"

import { ANDROID_ASSETS_DIR, NOTO_LOTTIE_DIR } from "../../files.ts"
import { resolveEmojiSvg } from "../cli/emoji-svg.ts"
import { indexKeys, refreshRows, stemOf, writeRows } from "./csv.ts"
import { buildLottie } from "./lottie.ts"
import { contextText, emojiMeta, labelParts } from "./context.ts"
import { design, MAX_REFINE, MODEL, refine, runSummary } from "./llm.ts"
import { writePage } from "./page.ts"
import { sheetPng, staticPng } from "./render.ts"
import { parseScene } from "./svg.ts"

const cli = cac("animate")

cli.command("list", "rebuild emoji_animation.csv").action(() => {
  const rows = refreshRows()
  writeRows(rows)
  const ok = rows.filter((r) => !r.unsupported).length
  console.log(`${rows.length} emojis without animation; ${ok} animatable, ${rows.length - ok} unsupported`)
})

cli.command("page", "rebuild the review page").action(() => console.log(`${writePage()} animations on page`))

cli
  .command("run", "animate one emoji")
  .option("--emoji <emoji>", "animate this emoji instead of the least complex")
  .option("--refine <n>", "vision review rounds, 0-3", { default: 1 })
  .option("--no-context", "skip the emoji metadata and the LLM label for every part (default: both go into the prompt)")
  .option("--out <json>", "write the Lottie here and stop (nothing is added to the collection)")
  .option("--dry", "do not write anything")
  .option("--missing", "print every emoji that still has no animation (in the order they would be picked, skipped ones marked) and stop")
  .option("--force", "overwrite an emoji that already has a generated animation (without it, the run refuses unless --out or --dry)")
  .action(async (o: { emoji?: string; refine: number; context?: boolean; out?: string; dry?: boolean; force?: boolean; missing?: boolean }) => {
    const refineRounds = Number(o.refine)
    if (!Number.isInteger(refineRounds) || refineRounds < 0 || refineRounds > MAX_REFINE) throw new Error(`--refine must be an integer 0-${MAX_REFINE}`)
    const rows = refreshRows()
    if (o.missing) {
      const todo = rows.filter((r) => !r.lottie)
      for (const r of todo) console.log(`${r.emoji}\t${r.codepoints}\tcomplexity ${r.complexity}${r.unsupported ? `\tskipped: unsupported ${r.unsupported}` : r.note ? `\tskipped: ${r.note}` : ""}`)
      const skipped = todo.filter((r) => r.unsupported || r.note).length
      console.log(`${todo.length} without animation; ${todo.length - skipped} can be picked, ${skipped} skipped`)
      return
    }
    const row = o.emoji
      ? rows.find((r) => r.emoji === o.emoji)
      : rows.find((r) => !r.lottie && !r.unsupported && !r.note)
    if (!row) throw new Error(o.emoji ? `${o.emoji} is not in emoji_animation.csv` : "nothing left to animate")
    if (row.lottie && !o.force && !o.out && !o.dry) throw new Error(`${row.emoji} already has an animation (noto/${row.lottie}.json); pass --force to overwrite it`)
    if (row.unsupported) throw new Error(`${row.emoji} uses unsupported SVG features: ${row.unsupported}`)
    const svg = resolveEmojiSvg(row.emoji)!
    const scene = parseScene(svg.body, svg.width)
    console.log(`${row.emoji} complexity ${row.complexity} (${scene.parts.length} parts) with ${MODEL}`)

    try {
      const png = await staticPng(svg.body)
      let ctx: string | undefined
      if (o.context !== false) {
        const labels = await labelParts(row.emoji, scene, svg.body, emojiMeta(row.emoji))
        ctx = contextText(emojiMeta(row.emoji), labels)
        console.log(ctx)
      }
      // Structured output occasionally misses the schema; one retry is enough in practice.
      let spec = await design(row.emoji, scene, png, ctx).catch(() => design(row.emoji, scene, png, ctx))
      let lottie = buildLottie(scene, spec, row.emoji)
      for (let n = 0; n < refineRounds; n++) {
        console.log(`refine round ${n + 1}`)
        spec = await refine(row.emoji, scene, spec, await sheetPng(svg.body, lottie, spec.frames), ctx)
        lottie = buildLottie(scene, spec, row.emoji)
      }
      const stem = stemOf(row.emoji)
      console.log(`${spec.frames} frames, ${spec.groups.length} groups, ${spec.extras.length} extras, ${JSON.stringify(lottie).length} bytes`)
      if (o.out) writeFileSync(o.out, JSON.stringify(lottie))
      if (o.dry || o.out) return void console.log(runSummary())
      const json = JSON.stringify(lottie)
      const idx: Record<string, string> = JSON.parse(readFileSync(`${NOTO_LOTTIE_DIR}/index.json`, "utf8"))
      for (const k of indexKeys(row.emoji)) idx[k] ??= stem
      const sorted = JSON.stringify(Object.fromEntries(Object.entries(idx).sort(([a], [b]) => a.localeCompare(b))))
      for (const dir of [NOTO_LOTTIE_DIR, `${ANDROID_ASSETS_DIR}/noto`]) {
        writeFileSync(`${dir}/${stem}.json`, json)
        writeFileSync(`${dir}/index.json`, sorted)
      }
      Object.assign(row, { lottie: stem, model: MODEL, animated_at: new Date().toISOString().slice(0, 10), note: "" })
    } catch (e) {
      console.log(runSummary())
      row.note = `failed: ${(e as Error).message.slice(0, 120).replace(/\s+/g, " ")}`
      writeRows(rows)
      throw e
    }
    writeRows(rows)
    console.log(runSummary())
    console.log(`added ${row.emoji} -> noto/${row.lottie}.json; ${writePage()} animations on the review page`)
  })

cli.help()
cli.parse()
