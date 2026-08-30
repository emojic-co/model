import { existsSync } from "node:fs"
import { copyFile, open, readFile, rename, writeFile } from "node:fs/promises"

async function ensureTrailingNewline(path: string): Promise<void> {
  if (!existsSync(path)) return
  const fh = await open(path, "r+")
  try {
    const { size } = await fh.stat()
    if (size === 0) return
    const buf = Buffer.alloc(1)
    await fh.read(buf, 0, 1, size - 1)
    if (buf[0] !== 0x0a) await fh.write("\n", size)
  } finally {
    await fh.close()
  }
}

export async function appendJsonl(path: string, rows: string[]): Promise<void> {
  if (!rows.length) return
  await ensureTrailingNewline(path)
  const fh = await open(path, "a")
  try {
    await fh.write(rows.join("\n") + "\n")
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

export async function readJsonl<T = unknown>(path: string): Promise<T[]> {
  const out: T[] = []
  let bad = 0
  for (const line of (await readFile(path, "utf8")).split("\n")) {
    if (!line.trim()) continue
    try {
      out.push(JSON.parse(line) as T)
    } catch {
      bad++
    }
  }
  if (bad) console.warn(`${path}: skipped ${bad} malformed line(s)`)
  return out
}
