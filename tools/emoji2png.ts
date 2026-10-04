// Usage: bun run tools/emoji2png.ts <emoji> [--res <resolution>] [--output <output_file>] [--transparent]
// Prints the output path so it can be piped: `emoji2png 🦊 | xargs google-chrome`.
import { mkdirSync, writeFileSync } from "node:fs"
import { dirname } from "node:path"
import { resolveEmojiSvg } from "./cli/emoji-svg.ts"

const { values, positionals } = (await import("node:util")).parseArgs({
  args: Bun.argv.slice(2),
  allowPositionals: true,
  options: { res: { type: "string", default: "512" }, output: { type: "string" }, transparent: { type: "boolean", default: false } },
})

const emoji = positionals[0]
if (!emoji) {
  console.error("usage: emoji2png <emoji> [--res <resolution>] [--output <output_file>] [--transparent]")
  process.exit(1)
}
const res = Number.parseInt(values.res as string, 10)
if (!Number.isFinite(res) || res <= 0) {
  console.error(`invalid --res: ${values.res}`)
  process.exit(1)
}
const svg = resolveEmojiSvg(emoji)
if (!svg) {
  console.error(`no artwork for ${emoji}`)
  process.exit(1)
}

const code = [...emoji].map((c) => (c.codePointAt(0) as number).toString(16)).filter((c) => c !== "fe0f").join("-")
const output = values.output ?? `pngs/${code}.png`
const doc = `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 ${svg.width} ${svg.height}" width="${res}" height="${res}">${svg.body}</svg>`

const proc = Bun.spawn(["rsvg-convert", "-f", "png", ...(values.transparent ? [] : ["-b", "white"]), "-w", String(res), "-h", String(res), "-"], {
  stdin: new TextEncoder().encode(doc),
  stdout: "pipe",
  stderr: "inherit",
})
const [png, status] = await Promise.all([new Response(proc.stdout).arrayBuffer(), proc.exited])
if (status !== 0) process.exit(status)

mkdirSync(dirname(output), { recursive: true })
writeFileSync(output, new Uint8Array(png))
console.log(output)
