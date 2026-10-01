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
import { exampleText } from "./example.ts"
import { prepare } from "./lottie.ts"
import { contextText, emojiMeta, labelParts } from "./context.ts"
import { describe, design, DESIGN_TASK, MAX_REFINE, MODEL, refine, runSummary, SYSTEM, userText } from "./llm.ts"
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
  .command("run", "animate one emoji with an LLM and add it to the animated-emoji collection")
  .usage("[options]   (via package.json: bun run animate [options])")
  .option("--emoji <emoji>", "emoji to animate (default: the least complex one with no animation yet)")
  .option("--refine <n>", "review rounds after the first design: the model sees a contact sheet of its own animation and improves it; 0 skips, max 3 (default: 1)", { default: 1 })
  .option("--labels", "add more context to the prompt: the emoji's category and keywords plus an LLM-written label for every SVG part; costs one extra model call (default: only the emoji name)")
  .option("--no-example", "leave out the example: the full SVG + Noto Lottie of the nearest originally animated emoji, from data/animation_neighbors.json")
  .option("--print-prompt", "print the full design prompt (system + user message) and stop; no animation call (add --labels and it still makes the labelling call)")
  .option("--out <json>", "write the Lottie to this file and stop; the collection, CSV and assets are not touched")
  .option("--dry", "run everything but write nothing")
  .option("--missing", "print every emoji that still has no animation (in the order they would be picked, skipped ones marked) and stop")
  .example("bun run animate --missing                   # list the emojis still without an animation")
  .example("bun run animate --emoji 💦 --print-prompt     # inspect the prompt, no model call")
  .example("bun run animate --emoji 💦 --out /tmp/x.json  # try it, keep the result out of the collection")
  .example("bun run animate --emoji 💦                    # animate and add to the collection")
  .example("bun run animate --refine 0 --labels           # next emoji, no review round, with the extra context")
  .example("MODEL: set ANIMATE_MODEL=<gateway model id> to change the model (default anthropic/claude-opus-5.5); limits: ANIMATE_MAX_RUN_CALLS/MAX_DAY_CALLS/MAX_DAY_TOKENS/MAX_OUTPUT_TOKENS/TIMEOUT_S")
  .action(async (o: { emoji?: string; refine: number; labels?: boolean; missing?: boolean; example?: boolean; out?: string; dry?: boolean; printPrompt?: boolean }) => {
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
    if (row.unsupported) throw new Error(`${row.emoji} uses unsupported SVG features: ${row.unsupported}`)
    const svg = resolveEmojiSvg(row.emoji)!
    const scene = parseScene(svg.body, svg.width)
    const svgText = `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 ${svg.width} ${svg.height}">${svg.body}</svg>`
    console.log(`${row.emoji} complexity ${row.complexity} (${scene.parts.length} parts) with ${MODEL}`)

    try {
      const png = await staticPng(svg.body)
      let ctx: string | undefined = emojiMeta(row.emoji).split("\n")[0] || undefined // always: "Name: ..."
      if (o.labels) {
        const labels = await labelParts(row.emoji, scene, svg.body, emojiMeta(row.emoji))
        ctx = `${contextText(emojiMeta(row.emoji), labels)}\n\nParts (in drawing order, p0 first):\n${describe(scene)}`
        console.log(ctx)
      }
      if (o.example !== false) {
        const ex = exampleText(row.emoji)
        if (ex) ctx = [ctx, ex].filter(Boolean).join("\n\n")
      }
      if (o.printPrompt) {
        console.log(`=== SYSTEM ===\n${SYSTEM}\n\n=== USER (plus the static emoji image) ===\n${userText(row.emoji, svgText, DESIGN_TASK, ctx)}`)
        return
      }
      // The model's Lottie is checked (syntax, then a render in Chrome); problems go back to it.
      const build = async (ask: (fix?: { doc: unknown; errors: string[] }) => Promise<unknown>) => {
        let fix: { doc: unknown; errors: string[] } | undefined
        for (let n = 0; n < 3; n++) {
          let doc: unknown
          let errors: string[]
          try {
            doc = await ask(fix)
            const lottie = prepare(doc, row.emoji)
            const t0 = Date.now()
            errors = await validate(lottie)
            console.log(`   checked in Chrome: ${errors.length ? `${errors.length} problem(s)` : "ok"} (${((Date.now() - t0) / 1000).toFixed(1)}s)`)
            if (!errors.length) return { doc, lottie }
          } catch (e) {
            errors = [(e as Error).message]
          }
          console.log(`rejected: ${errors.join("; ")}`)
          fix = { doc, errors }
        }
        return undefined
      }
      let cur = await build((fix) => design(row.emoji, svgText, png, ctx, fix))
      if (!cur) throw new Error("the model did not produce a valid animation in 3 tries")
      for (let n = 0; n < refineRounds; n++) {
        console.log(`refine round ${n + 1}`)
        const prev = cur
        const next = await build((fix) => sheetPng(svg.body, prev.lottie, prev.lottie.op).then((sheet) => refine(row.emoji, svgText, prev.doc, sheet, ctx, fix)))
        if (next) cur = next
        else console.log("refinement rejected; keeping the previous version")
      }
      const lottie = cur.lottie
      const stem = stemOf(row.emoji)
      console.log(`${lottie.op} frames, ${lottie.layers.length - 1} layers, ${JSON.stringify(lottie).length} bytes`)
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
