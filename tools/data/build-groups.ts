import { readFile, writeFile } from "node:fs/promises"

import { GROUP_JSON } from "../../files.ts"

const EMOJIBASE_DATA = "node_modules/emojibase-data/en/data.json"
const EMOJIBASE_MESSAGES = "node_modules/emojibase-data/en/messages.json"

type EmojiRow = { emoji?: string; group?: number; subgroup?: number }
type Messages = { subgroups: { key: string; order: number }[] }

async function main(): Promise<void> {
  const data = JSON.parse(await readFile(EMOJIBASE_DATA, "utf8")) as EmojiRow[]
  const messages = JSON.parse(
    await readFile(EMOJIBASE_MESSAGES, "utf8"),
  ) as Messages
  const keyOf = new Map(messages.subgroups.map((s) => [s.order, s.key]))

  const out: Record<string, string[]> = {}
  for (const s of messages.subgroups) out[s.key] = []
  for (const row of data) {
    if (row.subgroup === undefined || typeof row.emoji !== "string") continue
    const key = keyOf.get(row.subgroup)
    if (!key) continue
    out[key].push(row.emoji)
  }

  const body = messages.subgroups
    .map((s) => `  ${JSON.stringify(s.key)}: ${JSON.stringify(out[s.key])}`)
    .join(",\n")
  await writeFile(GROUP_JSON, `{\n${body}\n}\n`, "utf8")

  const empty = messages.subgroups.filter((s) => out[s.key].length === 0)
  console.log(
    `${GROUP_JSON}: ${messages.subgroups.length} groups, ` +
      `${data.filter((r) => r.subgroup !== undefined && r.emoji).length} emoji` +
      (empty.length ? ` (${empty.map((s) => s.key).join(", ")} empty)` : ""),
  )
}

main()
