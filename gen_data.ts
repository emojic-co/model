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
import { z } from "zod"

// Label sets live in labels.json, shared with main.py (see `EMOJIS` / `feeling`).
const LABELS_PATH = new URL("./labels.json", import.meta.url)
const { emojis: EMOJIS, feelings: FEELINGS } = JSON.parse(
  await readFile(LABELS_PATH, "utf8"),
) as { emojis: string[]; feelings: string[] }



const pick = <T>(xs: readonly T[]): T =>
  xs[Math.floor(Math.random() * xs.length)]

async function generateBatch(emoji: string, feeling: string): Promise<string[]> {
  const { output } = await generateText({
    model: MODEL,
    output: Output.object({
      schema: z.object({
        texts: z
          .array(z.string())
          .describe("The generated messages, text only."),
      }),
    }),
    prompt:
      `Write ${SAMPLES_PER_BATCH} different short messages a person might send ` +
      `in a WhatsApp chat.\n\n` +
      `Every message must:\n` +
      `- clearly convey the feeling "${feeling}"\n` +
      `- fit the emoji ${emoji} in meaning (do NOT put the emoji or any ` +
      `emoji in the text)\n` +
      `- be 1 to 6 words, casual, like a real texter\n` +
      `- contain no digits and no emoji\n\n` +
      '- you can use special chars like !?:()@$%&* if appropriate\n\n' +
      `Return only the message text for each item.`,
  })

  return output.texts
}

const MODEL = "openai/gpt-5.6-luna"
const OUT_PATH = new URL("./data.jsonl", import.meta.url)
const SAMPLES_PER_BATCH = 20
const BATCH_COUNT = 50

if (import.meta.main) {
  // Each batch picks its own random (emoji, feeling) pair; run them together.
  const pairs = Array.from({ length: BATCH_COUNT }, () => ({
    emoji: pick(EMOJIS),
    feeling: pick(FEELINGS),
  }))

  const results = await Promise.allSettled(
    pairs.map(({ emoji, feeling }) => generateBatch(emoji, feeling)),
  )

  const lines = results
    .flatMap((result, i) => {
      const { emoji, feeling } = pairs[i]
      if (result.status === "rejected") {
        console.error(`batch ${i} (${emoji} ${feeling}) failed:`, result.reason)
        return []
      }
      return result.value
        .map((t) => t.trim())
        .filter(Boolean)
        .map((text) => JSON.stringify({ emoji, feeling, text }))
    })
    .join("\n")

  if (lines) await appendFile(OUT_PATH, lines + "\n", "utf8")
}
