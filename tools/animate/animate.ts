// Animates one emoji: the LLM designs the motion, a Lottie is built from the emoji's own SVG parts and
// added to the animated-emoji collection (web + Android assets), then the preview page is rebuilt.
//   bun run animate <emoji> [--force]   animate an emoji (--force: replace an existing animation)
//   bun run animate list                emojis that have no animation yet (one line)
//   bun run animate preview             rebuild the preview page of all animated emojis
import { cac } from "cac"

import { resolveEmojiSvg } from "../cli/emoji-svg.ts"
import { contextText, emojiMeta, labelParts } from "./context.ts"
import { addAnimation, isAnimated, missing } from "./index.ts"
import { design, MODEL, refine } from "./llm.ts"
import { buildLottie } from "./lottie.ts"
import { writePage } from "./page.ts"
import { sheetPng, staticPng } from "./render.ts"
import { parseScene } from "./svg.ts"

const cli = cac("animate")

cli.command("list", "list the emojis that have no animation yet").action(() => console.log(missing().join(" ")))

cli.command("preview", "rebuild the preview page").action(() => console.log(`${writePage()} animations on page`))

cli
  .command("[emoji]", "animate this emoji")
  .option("--force", "replace an existing animation")
  .action(async (emoji: string | undefined, o: { force?: boolean }) => {
    if (!emoji) return cli.outputHelp()
    if (isAnimated(emoji) && !o.force) throw new Error(`${emoji} already has an animation; pass --force to replace it`)
    const svg = resolveEmojiSvg(emoji)
    if (!svg) throw new Error(`no SVG for ${emoji}`)
    const scene = parseScene(svg.body, svg.width)
    if (scene.unsupported.length) throw new Error(`${emoji} uses unsupported SVG features: ${scene.unsupported.join(", ")}`)
    console.log(`${emoji} (${scene.parts.length} parts) with ${MODEL}`)

    const png = await staticPng(svg.body)
    const labels = await labelParts(emoji, scene, svg.body, emojiMeta(emoji))
    const ctx = contextText(emojiMeta(emoji), labels)
    console.log(ctx)
    // Structured output occasionally misses the schema; one retry is enough in practice.
    let spec = await design(emoji, scene, png, ctx).catch(() => design(emoji, scene, png, ctx))
    let lottie = buildLottie(scene, spec, emoji)
    spec = await refine(emoji, scene, spec, await sheetPng(svg.body, lottie, spec.frames), ctx)
    lottie = buildLottie(scene, spec, emoji)
    const stem = addAnimation(emoji, JSON.stringify(lottie))
    console.log(`added ${emoji} -> noto/${stem}.json; ${writePage()} animations on the preview page`)
  })

cli.help()
cli.parse()
