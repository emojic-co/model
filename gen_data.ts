/**
 * Synthetic dataset generator for `emojic`.
 *
 * Uses the Vercel AI SDK (via the AI Gateway) with GPT-5.6 Luna to generate
 * short, WhatsApp-style texts for a randomly chosen (emoji, feeling) pair and
 * appends them to `data.jsonl` as `{text, emoji, feeling}` rows.
 *
 * The model is only asked for the `text`; the emoji and feeling are fixed by
 * this script and written alongside each generated line.
 *
 * Run:
 *   bun run gen_data.ts        # 10 batches in parallel: ~100 samples,
 *                              # a fresh random (emoji, feeling) pair per batch
 *
 * Requires AI_GATEWAY_API_KEY (Bun auto-loads it from .env).
 */
import { appendFile, readFile } from "node:fs/promises"

import { generateText, Output } from "ai"
import cliProgress from "cli-progress"
import { z } from "zod"

// Label sets live in labels.json, shared with main.py (see `EMOJIS` / `feeling`).




const MODEL = "openai/gpt-5.6-luna"
const OUT_PATH = new URL("./data.jsonl", import.meta.url)
const BATCH_COUNT = 10
const SAMPLES_PER_BATCH = 50

const labels = z.object({
  feelings: z.array(z.string()),
  emojis: z.array(z.string()),
})

type Record = z.infer<typeof Record>
const Record = z.object({
  text: z.string().max(64),
  emoji: z.string(),
  feeling: z.string(),
})
const EXAMPLE_RECORD: Record = {
  text: "I can't believe this happened!",
  emoji: "😭",
  feeling: "Sad",
}

if (import.meta.main) {
  // Each batch picks its own random (emoji, feeling) pair; run them together.
  const { feelings, emojis } = labels.parse(JSON.parse(
    await readFile('./labels.json', "utf8"),
  ))

  async function batch() {
    const { output } = await generateText({
      model: MODEL,
      output: Output.object({
        schema: z.object({
          records: z
            .array(Record)
            .describe("Generated messages, each with an emoji and feeling."),
        }),
      }),
      prompt: [
        'Generate a list of short (1-6 words, no more than 64 characters), WhatsApp-style diverse messages.',
        'Ensure high variance in tone, intent, and style (e.g., quick updates, dry humor, complaints, sudden news, invitations, reactions, low-effort replies). Avoid repetitive or cliché text.',
        `For each message associate one of the feelings: ${feelings.join(", ")}`,
        `For each message associate one of the emojis: ${emojis.join(", ")}`,
        `Note that emojis and feelings are independent; the same emoji can be used for different feelings, and vice versa.`,
        `Return at least ${SAMPLES_PER_BATCH} records like this example:`,
        JSON.stringify([EXAMPLE_RECORD]),
      ].join("\n")
    })

    const parsed = z.array(Record).parse(output.records)

    // Drop anything the model invented outside the labels.json label sets.
    const feelingSet = new Set(feelings)
    const emojiSet = new Set(emojis)
    const records = parsed.filter(
      (r) => emojiSet.has(r.emoji) && feelingSet.has(r.feeling),
    )
    const dropped = parsed.length - records.length
    if (dropped > 0) {
      console.warn(`Dropped ${dropped} record(s) with unknown emoji/feeling`)
    }
    return records
  }

  const bar = new cliProgress.SingleBar(
    { format: "generating |{bar}| {percentage}% | {value}/{total} batches | ETA: {eta}s" },
    cliProgress.Presets.shades_classic,
  )
  bar.start(BATCH_COUNT, 0)
  await Promise.all(Array.from({ length: BATCH_COUNT }).map(async () => {
    const records = await batch()
    bar.increment()
    if (records.length > 0) {
      const lines = records
        .map((r) => JSON.stringify({ emoji: r.emoji, feeling: r.feeling, text: r.text }))
        .join("\n")
      await appendFile(OUT_PATH, lines + "\n")
    }
  }))
  bar.stop()

}
