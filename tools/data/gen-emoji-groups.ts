import { writeFile } from "node:fs/promises"

import data from "emojibase-data/en/data.json" with { type: "json" }

const GROUP_NAMES = [
  "Smileys & Emotion",
  "People & Body",
  "Component",
  "Animals & Nature",
  "Food & Drink",
  "Travel & Places",
  "Activities",
  "Objects",
  "Symbols",
  "Flags",
]

const strip = (s: string) =>
  [...s].filter((c) => {
    const cp = c.codePointAt(0) ?? 0
    return cp !== 0xfe0f && !(cp >= 0x1f3fb && cp <= 0x1f3ff)
  }).join("")

const out: Record<string, string> = {}
for (const e of data as { emoji: string; group?: number }[]) {
  if (e.group === undefined) continue
  const name = GROUP_NAMES[e.group]
  if (!name) continue
  out[e.emoji] = name
  out[strip(e.emoji)] = name
}

await writeFile(
  "tools/data/emoji-groups.json",
  JSON.stringify(out, null, 0) + "\n",
)
console.log(`wrote tools/data/emoji-groups.json: ${Object.keys(out).length} keys`)
