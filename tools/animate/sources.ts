// emoji_animation_sources.csv: every labelled emoji x every animated-emoji source we know of (1 = has one).
// Remote sources are matched by emoji name (emojibase labels); their file paths go in `<source>_file`.
//   bun run animate-sources
import { readFileSync, writeFileSync } from "node:fs"

import { ANIMATION_SOURCES_CSV, LABELS_JSON, NOTO_LOTTIE_DIR } from "../../files.ts"
import { indexKeys, readRows } from "./csv.ts"

const TELEGRAM = "Tarikul-Islam-Anik/Telegram-Animated-Emojis"
const FLUENT = "microsoft/fluentui-emoji-animated"

const norm = (s: string) => s.toLowerCase().normalize("NFKD").replace(/[^a-z0-9]+/g, "")
const bare = (e: string) => e.replace(/️/g, "")

async function tree(repo: string): Promise<string[]> {
  const res = await fetch(`https://api.github.com/repos/${repo}/git/trees/main?recursive=1`)
  return ((await res.json()) as { tree: { path: string }[] }).tree.map((t) => t.path)
}

const labels: { emojis: string[] } = JSON.parse(readFileSync(LABELS_JSON, "utf8"))
const noto: Record<string, string> = JSON.parse(readFileSync(`${NOTO_LOTTIE_DIR}/index.json`, "utf8"))
// index.json also holds our own LLM-made animations (emoji_animation.csv `lottie`); those are not Noto.
const generated = new Set(readRows().filter((r) => r.lottie).map((r) => r.emoji))
const names: { emoji: string; label: string }[] = JSON.parse(readFileSync("node_modules/emojibase-data/en/data.json", "utf8"))
const nameOf = new Map(names.map((n) => [bare(n.emoji), n.label]))

const telegram = new Map<string, string>() // normalized name -> path
for (const p of await tree(TELEGRAM)) if (p.endsWith(".webp") && !p.startsWith("web/")) telegram.set(norm(p.split("/").pop()!.replace(/\.webp$/, "")), p)
const fluent = new Map<string, string>()
for (const p of await tree(FLUENT)) {
  const m = p.match(/^assets\/([^/]+)\/animated\/.*\.png$/)
  if (m) fluent.set(norm(m[1]), p)
}

const q = (v: string) => (/[",\n]/.test(v) ? `"${v.replace(/"/g, '""')}"` : v)
const rows = labels.emojis.map((emoji) => {
  const name = nameOf.get(bare(emoji)) ?? ""
  const tg = telegram.get(norm(name)) ?? ""
  const fl = fluent.get(norm(name)) ?? ""
  return [emoji, name, !generated.has(emoji) && indexKeys(emoji).some((k) => k in noto) ? "1" : "", tg ? "1" : "", fl ? "1" : "", tg, fl]
})
const head = ["emoji", "name", "noto", "telegram", "fluent", "telegram_file", "fluent_file"]
writeFileSync(ANIMATION_SOURCES_CSV, [head, ...rows].map((r) => r.map(q).join(",")).join("\n") + "\n")
const n = (i: number) => rows.filter((r) => r[i]).length
console.log(`${rows.length} emojis; noto ${n(2)}, telegram ${n(3)}, fluent ${n(4)}; no noto but another: ${rows.filter((r) => !r[2] && (r[3] || r[4])).length}`)
