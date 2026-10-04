// Copies the Noto Lottie clips used by configs/*.json into public/noto/ (deterministic, no model).
import { readdirSync, readFileSync, writeFileSync, mkdirSync, copyFileSync, rmSync } from 'node:fs'
import { dirname, join } from 'node:path'
import { fileURLToPath } from 'node:url'

const here = dirname(fileURLToPath(import.meta.url))
const src = join(here, '..', '..', 'web', 'public', 'noto')
const dst = join(here, 'public', 'noto')
const index = JSON.parse(readFileSync(join(src, 'index.json'), 'utf8'))
const configs = readdirSync(join(here, 'configs'))
  .filter((f) => f.endsWith('.json'))
  .map((f) => JSON.parse(readFileSync(join(here, 'configs', f), 'utf8')))

rmSync(dst, { recursive: true, force: true })
mkdirSync(dst, { recursive: true })
const used = {}
for (const c of configs) {
  for (const { emoji } of [...c.cards, c.outro]) {
    const stem = index[emoji]
    if (!stem) throw new Error(`${emoji} has no animation`)
    used[emoji] = stem
    copyFileSync(join(src, `${stem}.json`), join(dst, `${stem}.json`))
  }
}
writeFileSync(join(dst, 'index.json'), JSON.stringify(used))
console.log(`synced ${Object.keys(used).length} clips`)
