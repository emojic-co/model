// node render.mjs [id ...]   renders out/<id>.mp4 for the given configs (default: all of configs/*.json)
import path from 'node:path'
import { readdirSync } from 'node:fs'
import { bundle } from '@remotion/bundler'
import { renderMedia, selectComposition, ensureBrowser } from '@remotion/renderer'
import { webpackOverride } from './webpack-override.mjs'

const ids = process.argv.length > 2
  ? process.argv.slice(2)
  : readdirSync('configs').filter((f) => f.endsWith('.json')).map((f) => f.replace(/\.json$/, ''))

await ensureBrowser()
const serveUrl = await bundle({ entryPoint: path.resolve('src/index.ts'), webpackOverride })
for (const id of ids) {
  const composition = await selectComposition({ serveUrl, id, inputProps: {} })
  const outputLocation = path.resolve('out', `${id}.mp4`)
  await renderMedia({ composition, serveUrl, codec: 'h264', outputLocation, concurrency: 2 })
  console.log('✓', outputLocation)
}
