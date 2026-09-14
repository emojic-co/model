import { readFile } from "node:fs/promises"

import { WA_KEYWORDS_JSON } from "../../files.ts"
import type { SourceRow } from "./keywords.ts"

export function toSourceRows(data: Record<string, string[]>): SourceRow[] {
  return Object.entries(data).map(([text, emojis]) => ({
    text,
    emojis: emojis.join(" "),
    styles: [],
  }))
}

export async function loadWaKeywords(): Promise<SourceRow[]> {
  const data = JSON.parse(await readFile(WA_KEYWORDS_JSON, "utf8")) as Record<string, string[]>
  return toSourceRows(data)
}
