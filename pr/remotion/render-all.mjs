import path from 'node:path'
import { bundle } from '@remotion/bundler'
import { renderMedia, selectComposition, ensureBrowser } from '@remotion/renderer'
import { webpackOverride } from './webpack-override.mjs'

const RENDER = process.argv.slice(2)
const slugs = RENDER.length
  ? RENDER
  : ['pizza-s-finally-here', 'rent-is-due-and-i-m-broke', 'quiet-night-just-me-and-tea']

await ensureBrowser()

console.log('bundling…')
const serveUrl = await bundle({
  entryPoint: path.resolve('src/index.ts'),
  webpackOverride,
})

for (const id of slugs) {
  const composition = await selectComposition({ serveUrl, id, inputProps: {} })
  const outputLocation = path.resolve('out', `${id}.mp4`)
  await renderMedia({
    composition,
    serveUrl,
    codec: 'h264',
    outputLocation,
    inputProps: {},
    concurrency: 1,
  })
  console.log('✓', outputLocation)
}
