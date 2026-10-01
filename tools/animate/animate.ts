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
import { compose, type Doc } from "./lottie.ts"
import { contextText, emojiMeta, labelParts } from "./context.ts"
import { design, MODEL, refine } from "./llm.ts"
import { writePage } from "./page.ts"
import { sheetPng, staticPng } from "./render.ts"
import { parseScene } from "./svg.ts"
import { validate } from "./validate.ts"

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
  .option("--refine <n>", "vision review rounds", { default: 1 })
  .option("--no-context", "skip the emoji metadata and the LLM label for every part (default: both go into the prompt)")
  .option("--out <json>", "write the Lottie here and stop (nothing is added to the collection)")
  .option("--dry", "do not write anything")
  .action(async (o: { emoji?: string; refine: number; context?: boolean; out?: string; dry?: boolean }) => {
    const rows = refreshRows()
    const row = o.emoji
      ? rows.find((r) => r.emoji === o.emoji)
      : rows.find((r) => !r.lottie && !r.unsupported && !r.note)
    if (!row) throw new Error(o.emoji ? `${o.emoji} is not in emoji_animation.csv` : "nothing left to animate")
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
      // The model's Lottie is composed with the original parts and checked in Chrome; problems go back to it.
      const build = async (ask: (fix?: { doc: unknown; errors: string[] }) => Promise<unknown>) => {
        let fix: { doc: unknown; errors: string[] } | undefined
        for (let n = 0; n < 3; n++) {
          const doc = await ask(fix)
          let errors: string[]
          try {
            const lottie = compose(scene, doc, row.emoji)
            errors = await validate(svg.body, lottie)
            if (!errors.length) return { doc: doc as Doc, lottie }
          } catch (e) {
            errors = [(e as Error).message]
          }
          console.log(`rejected: ${errors.join("; ")}`)
          fix = { doc, errors }
        }
        return undefined
      }
      let cur = await build((fix) => design(row.emoji, scene, png, ctx, fix))
      if (!cur) throw new Error("the model did not produce a valid animation in 3 tries")
      let spec = cur.doc
      let lottie = cur.lottie
      for (let n = 0; n < Number(o.refine); n++) {
        console.log(`refine round ${n + 1}`)
        const prev = cur
        const next = await build((fix) => sheetPng(svg.body, prev.lottie, prev.doc.frames).then((sheet) => refine(row.emoji, scene, prev.doc, sheet, ctx, fix)))
        if (next) { cur = next; spec = next.doc; lottie = next.lottie } else console.log("refinement rejected; keeping the previous version")
      }
      const stem = stemOf(row.emoji)
      console.log(`${spec.frames} frames, ${spec.layers.length} layers, ${JSON.stringify(lottie).length} bytes`)
      if (o.out) writeFileSync(o.out, JSON.stringify(lottie))
      if (o.dry || o.out) return
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
      row.note = `failed: ${(e as Error).message.slice(0, 120).replace(/\s+/g, " ")}`
      writeRows(rows)
      throw e
    }
    writeRows(rows)
    console.log(`added ${row.emoji} -> noto/${row.lottie}.json; ${writePage()} animations on the review page`)
  })

cli.help()
cli.parse()
