import { execFileSync } from "node:child_process"
import { existsSync } from "node:fs"
import { mkdir, rename, stat } from "node:fs/promises"

import { cac } from "cac"

import { ARCHIVE_DIR, DATA_JSONL as DATA } from "../../files.ts"
import { readJsonl, writeFileAtomic } from "./io.ts"
import { collapse, type Row } from "./regen.ts"

export function minimalLine(r: Row, sha: string): string {
  const rec: Record<string, unknown> =
    r.bg && r.fg
      ? { text: r.text, emojis: r.emojis, styles: r.styles, bg: r.bg, fg: r.fg }
      : { text: r.text, emojis: r.emojis, styles: r.styles }
  rec.min = sha
  return JSON.stringify(rec)
}

export function minimize(rows: unknown[], sha: string): string {
  return collapse(rows).map((r) => minimalLine(r, sha)).join("\n") + "\n"
}

function shortSha(): string {
  try {
    return execFileSync("git", ["rev-parse", "--short", "HEAD"], { encoding: "utf8" }).trim()
  } catch {
    console.error("minimize: not a git repo, or HEAD has no commit")
    process.exit(1)
  }
}

function dataDirty(): boolean {
  try {
    return (
      execFileSync("git", ["status", "--porcelain", "--", DATA], { encoding: "utf8" }).trim() !== ""
    )
  } catch {
    return false
  }
}

function fmtBytes(n: number): string {
  if (n < 1024) return `${n} B`
  if (n < 1024 * 1024) return `${(n / 1024).toFixed(1)} KB`
  return `${(n / (1024 * 1024)).toFixed(1)} MB`
}

const cli = cac("minimize")
cli.usage("[options]")
cli.option("--dry", "report row counts and sizes; move and write nothing")
cli.help()

if (import.meta.main) {
  const { options } = cli.parse(process.argv, { run: false })
  if (options.help) process.exit(0)

  if (!existsSync(DATA)) {
    console.error(`minimize: ${DATA} not found`)
    process.exit(1)
  }

  const sha = shortSha()
  const archivePath = `${ARCHIVE_DIR}/data.${sha}.jsonl`
  const beforeBytes = (await stat(DATA)).size

  if (dataDirty()) {
    console.warn(
      `minimize: ${DATA} has uncommitted changes; the archive is named for HEAD (${sha}) `
      + "but will hold the current working-tree contents",
    )
  }

  if (options.dry) {
    const rows = await readJsonl<unknown>(DATA)
    const out = minimize(rows, sha)
    const kept = out.trimEnd().split("\n").length
    console.log("--- minimize --dry ---")
    console.log(`would archive : ${DATA} -> ${archivePath}`)
    console.log(`lines read    : ${rows.length}`)
    console.log(`rows kept     : ${kept} (would collapse ${rows.length - kept})`)
    console.log(`marker        : min=${sha}`)
    console.log(`size          : ${fmtBytes(beforeBytes)} -> ${fmtBytes(Buffer.byteLength(out))}`)
    process.exit(0)
  }

  if (existsSync(archivePath)) {
    console.error(
      `minimize: ${archivePath} already exists; commit the current ${DATA} `
      + "or remove the stale archive first",
    )
    process.exit(1)
  }

  await mkdir(ARCHIVE_DIR, { recursive: true })
  await rename(DATA, archivePath)

  const rows = await readJsonl<unknown>(archivePath)
  const out = minimize(rows, sha)
  await writeFileAtomic(DATA, out)

  const kept = out.trimEnd().split("\n").length
  const afterBytes = Buffer.byteLength(out)

  console.log("--- minimize ---")
  console.log(`archived   : ${DATA} -> ${archivePath}`)
  console.log(`lines read : ${rows.length}`)
  console.log(`rows kept  : ${kept} (collapsed ${rows.length - kept})`)
  console.log(`marker     : min=${sha}`)
  console.log(`size       : ${fmtBytes(beforeBytes)} -> ${fmtBytes(afterBytes)}`)
  console.log("")
  console.log("next steps (not run by this tool):")
  console.log(`  git lfs install && git lfs track "${DATA}" "${ARCHIVE_DIR}/*.jsonl"`)
  console.log(`  git add .gitattributes ${ARCHIVE_DIR}/ ${DATA}`)
  console.log("  # purge the pre-LFS blobs from history (deliberate, needs a force-push):")
  console.log(`  git lfs migrate import --include="${DATA}" --everything`)
  process.exit(0)
}
