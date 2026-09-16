import { existsSync } from "node:fs"
import { copyFile, open, readFile, rename, writeFile } from "node:fs/promises"

export async function appendJsonl(path: string, rows: string[]): Promise<void> {
  if (!rows.length) return
  const fh = await open(path, "a")
  try {
    await fh.write(`\n${rows.join("\n")}\n`)
    await fh.sync()
  } finally {
    await fh.close()
  }
}

export async function writeFileAtomic(
  path: string,
  data: string,
  backup = false,
): Promise<void> {
  const tmp = `${path}.tmp`
  await writeFile(tmp, data)
  if (backup && existsSync(path)) await copyFile(path, `${path}.bak`)
  await rename(tmp, path)
}

export function parseJsonlText<T = unknown>(text: string, label = "input"): T[] {
  const out: T[] = []
  let bad = 0
  for (const line of text.split("\n")) {
    if (!line.trim()) continue
    try {
      out.push(JSON.parse(line) as T)
    } catch {
      bad++
    }
  }
  if (bad) console.warn(`${label}: skipped ${bad} malformed line(s)`)
  return out
}

export async function readJsonl<T = unknown>(path: string): Promise<T[]> {
  return parseJsonlText<T>(await readFile(path, "utf8"), path)
}
