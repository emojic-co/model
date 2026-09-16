import { afterEach, expect, test } from "bun:test"
import { mkdtemp, readFile, rm } from "node:fs/promises"
import { tmpdir } from "node:os"
import { join } from "node:path"

import { appendJsonl, parseJsonlText } from "./io.ts"

const dirs: string[] = []

afterEach(async () => {
  while (dirs.length) await rm(dirs.pop() as string, { recursive: true, force: true })
})

async function tmpPath(): Promise<string> {
  const dir = await mkdtemp(join(tmpdir(), "io-test-"))
  dirs.push(dir)
  return join(dir, "data.jsonl")
}

test("appendJsonl writes rows that parse back out, in order", async () => {
  const path = await tmpPath()
  await appendJsonl(path, [JSON.stringify({ n: 1 }), JSON.stringify({ n: 2 })])
  await appendJsonl(path, [JSON.stringify({ n: 3 })])
  const rows = parseJsonlText<{ n: number }>(await readFile(path, "utf8"))
  expect(rows.map((r) => r.n)).toEqual([1, 2, 3])
})

test("concurrent appendJsonl calls never tear a line across two writers", async () => {
  const path = await tmpPath()
  const batches = Array.from({ length: 40 }, (_, i) => [
    JSON.stringify({ n: i, pad: "x".repeat(200) }),
  ])
  await Promise.all(batches.map((rows) => appendJsonl(path, rows)))
  const text = await readFile(path, "utf8")
  const seen = new Set<number>()
  let bad = 0
  for (const line of text.split("\n")) {
    if (!line.trim()) continue
    try {
      seen.add((JSON.parse(line) as { n: number }).n)
    } catch {
      bad++
    }
  }
  expect(bad).toBe(0)
  expect(seen.size).toBe(40)
})
