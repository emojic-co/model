import path from 'node:path'
import { bundle } from '@remotion/bundler'
import { renderMedia, selectComposition, ensureBrowser } from '@remotion/renderer'
import { webpackOverride } from './webpack-override.mjs'

await ensureBrowser()
const serveUrl = await bundle({ entryPoint: path.resolve('src/index.ts'), webpackOverride })
const composition = await selectComposition({ serveUrl, id: 'showcase', inputProps: {} })
const outputLocation = path.resolve('out', 'showcase.mp4')
await renderMedia({ composition, serveUrl, codec: 'h264', outputLocation, concurrency: 2 })
console.log('✓', outputLocation)
