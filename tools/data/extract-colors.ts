import { cac } from "cac"

import { DATA_JSONL } from "../../files.ts"
import { readJsonl } from "./io.ts"

type Row = Record<string, unknown> & { color?: unknown }

export function pickColorRows(rows: Row[]): Row[] {
  return rows.filter((r) => typeof r.color === "string" && r.color.length > 0)
}

const cli = cac("extract-colors")
cli.usage("[file]  # defaults to data/data.jsonl; writes matching rows to stdout")
cli.help()

if (import.meta.main) {
  const parsed = cli.parse(process.argv, { run: false })
  if (cli.options.help) process.exit(0)

  const src = parsed.args[0] ?? DATA_JSONL
  const rows = await readJsonl<Row>(src)
  const kept = pickColorRows(rows)
  process.stdout.write(kept.map((r) => JSON.stringify(r)).join("\n") + "\n")
  process.stderr.write(`${kept.length}/${rows.length} rows with a color field\n`)
  process.exit(0)
}
