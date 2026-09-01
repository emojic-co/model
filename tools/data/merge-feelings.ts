import { readFile } from "node:fs/promises"
import { writeFileAtomic } from "./io.ts"

const FILES = ["./train.jsonl", "./eval.jsonl"]

const MERGES: Record<string, string> = {
  Loving: "Love",
  Worried: "Concerned",
}

const AT = new Date().toISOString().slice(0, 10)

if (import.meta.main) {
  for (const path of FILES) {
    const src = await readFile(path, "utf8")
    const out: string[] = []
    let changed = 0
    let bad = 0
    for (const line of src.split("\n")) {
      if (!line.trim()) continue
      let row: Record<string, unknown>
      try {
        row = JSON.parse(line)
      } catch {
        bad++
        out.push(line)
        continue
      }
      const from = row.feeling as string
      const to = MERGES[from]
      if (to) {
        row.feeling = to
        const meta =
          row.meta && typeof row.meta === "object"
            ? (row.meta as Record<string, unknown>)
            : {}
        meta.feeling_merge = { from, to, at: AT }
        row.meta = meta
        changed++
      }
      out.push(JSON.stringify(row))
    }
    await writeFileAtomic(path, out.join("\n") + "\n")
    console.log(
      `${path}: ${changed} rows remapped${
        bad ? `, ${bad} malformed line(s) left as-is` : ""
      }`,
    )
  }
  process.exit(0)
}
