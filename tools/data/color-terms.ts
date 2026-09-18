import { cac } from "cac"

import { COLOR_NAMES_SOURCE_JSON, COLOR_TERMS_JSONL } from "../../files.ts"
import { buildColorTerms, type ColorTerm, type NamedColor } from "./color-names.ts"
import { writeFileAtomic } from "./io.ts"

function toLine(t: ColorTerm): string {
  return JSON.stringify({
    text: t.text,
    emojis: "",
    styles: [],
    bg: t.bg,
    fg: t.fg,
    src: "colors",
  })
}

if (import.meta.main) {
  const cli = cac("color-terms")
  cli.usage("[options]  # writes data/color_terms.jsonl")
  cli.help()
  cli.parse(process.argv, { run: false })
  if (cli.options.help) process.exit(0)

  const names = (await Bun.file(COLOR_NAMES_SOURCE_JSON).json()) as NamedColor[]
  const terms = buildColorTerms(names)
  await writeFileAtomic(COLOR_TERMS_JSONL, terms.map(toLine).join("\n") + "\n")
  console.log(`-> ${COLOR_TERMS_JSONL} : ${terms.length}`)
  process.exit(0)
}
