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
import { refreshRows, stemOf, writeRows } from "./csv.ts"
import { buildLottie } from "./lottie.ts"
import { design, MODEL, refine } from "./llm.ts"
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
  .option("--refine <n>", "vision review rounds", { default: 1 })
  .option("--dry", "do not write anything")
  .action(async (o: { emoji?: string; refine: number; dry?: boolean }) => {
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
      let spec = await design(row.emoji, scene, await staticPng(svg.body))
      let lottie = buildLottie(scene, spec, row.emoji)
      for (let n = 0; n < Number(o.refine); n++) {
        console.log(`refine round ${n + 1}`)
        spec = await refine(row.emoji, scene, spec, await sheetPng(svg.body, lottie, spec.frames))
        lottie = buildLottie(scene, spec, row.emoji)
      }
      const stem = stemOf(row.emoji)
      console.log(`${spec.frames} frames, ${spec.groups.length} groups, ${spec.extras.length} extras, ${JSON.stringify(lottie).length} bytes`)
      if (o.dry) return
      const json = JSON.stringify(lottie)
      const idx: Record<string, string> = JSON.parse(readFileSync(`${NOTO_LOTTIE_DIR}/index.json`, "utf8"))
      idx[row.emoji] = stem
      idx[row.emoji.replace(/️/g, "")] = stem
      idx[`${row.emoji.replace(/️/g, "")}️`] ??= stem
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
