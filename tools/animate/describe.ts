// Fills animation.csv `animation_text`: for every emoji without a Noto animation, an LLM describes what a
// short looping animation of it should look like. Batched (one call per --batch emojis); the CSV is saved after
// every batch, so a stopped run resumes where it left off. Rows that already have text are skipped.
//   bun run animate-describe                  all emojis still missing text
//   bun run animate-describe --emoji 🔴       just this one (overwrites)
//   bun run animate-describe --limit 20 --dry preview the first 20 without writing
// Default batch size keeps a full run within the default per-run call cap (ANIMATE_MAX_RUN_CALLS).
import { cac } from "cac"
import { Output } from "ai"
import { z } from "zod"

import { hasLottie, readRows, writeRows, type Row } from "./csv.ts"
import { emojiMeta } from "./context.ts"
import { guardedGenerate, runSummary } from "./llm.ts"

// The text is cheap prose, so it does not need the Lottie designer model. Override with ANIMATE_DESCRIBE_MODEL.
const MODEL = process.env.ANIMATE_DESCRIBE_MODEL ?? "openai/gpt-5.6-luna"

const SYSTEM = `You write briefs for a motion designer who animates emojis as tiny seamless loops (2-4 seconds, like Google's animated Noto emoji).
For each emoji, describe in 1-2 plain sentences what its animation looks like: which part of the depicted thing moves, how
(sway, bounce, blink, flicker, spin, pulse, shake...), and any small accent (sparkles, bubbles, steam). The motion must suit
the real object, be charming and subtle, and loop. Describe only the motion, not the static art. No preamble, no emoji in the text.`

const Out = z.object({
  items: z.array(z.object({ emoji: z.string(), text: z.string() })).describe("one entry per requested emoji, same emoji string as given"),
})

const cli = cac("animate-describe")
cli
  .command("")
  .option("--emoji <emoji>", "describe only this emoji (replaces its text)")
  .option("--batch <n>", "emojis per model call", { default: 60 })
  .option("--limit <n>", "stop after this many emojis")
  .option("--dry", "print the descriptions, do not write the CSV")
  .action(async (o: { emoji?: string; batch: number; limit?: string; dry?: boolean }) => {
    const rows = readRows()
    const batch = Number(o.batch)
    if (!Number.isInteger(batch) || batch < 1) throw new Error("--batch must be a positive integer")
    // Emojis that already got a Lottie have their motion; they are only described when asked for by name.
    let todo = o.emoji ? rows.filter((r) => r.emoji === o.emoji) : rows.filter((r) => !r.noto && !r.animation_text && !hasLottie(r))
    if (o.emoji && !todo.length) throw new Error(`${o.emoji} is not in animation.csv`)
    if (todo.some((r) => r.noto)) throw new Error(`${o.emoji} has a Noto animation; nothing to describe`)
    if (o.limit) todo = todo.slice(0, Number(o.limit))
    console.log(`${todo.length} emojis to describe with ${MODEL}, ${Math.ceil(todo.length / batch)} call(s)`)

    try {
      for (let i = 0; i < todo.length; i += batch) {
        const chunk: Row[] = todo.slice(i, i + batch)
        const list = chunk.map((r) => `${r.emoji}\n${emojiMeta(r.emoji)}`).join("\n\n")
        // Structured output occasionally misses the schema; one retry is enough in practice.
        const ask = () =>
          guardedGenerate({
            model: MODEL,
            system: SYSTEM,
            output: Output.object({ schema: Out }),
            messages: [{ role: "user", content: `Describe the animation of each of these ${chunk.length} emojis:\n\n${list}` }],
          }, `describe ${i + 1}-${i + chunk.length}`)
        const { output } = await ask().catch(ask)
        const got = new Map((output as z.infer<typeof Out>).items.map((x) => [x.emoji, x.text.replace(/\s+/g, " ").trim()]))
        for (const r of chunk) {
          const text = got.get(r.emoji)
          if (!text) { console.log(`missing: ${r.emoji} ${r.name}`); continue }
          r.animation_text = text
          if (o.dry) console.log(`${r.emoji} ${r.name}: ${text}`)
        }
        if (!o.dry) writeRows(rows)
      }
    } finally {
      console.log(runSummary())
    }
  })
cli.help()
cli.parse()
