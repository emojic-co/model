import { readFileSync } from "node:fs"
import { expect, test } from "bun:test"
import { parse } from "yaml"

import { ANDROID_ASSETS_DIR, STYLE_YML } from "../../files.ts"
import { STYLES } from "./styles.ts"
import { buildStyleFile } from "./export-style.ts"

test("style.yml matches what web/src/feelings.js and patterns.js currently produce", () => {
  const fresh = buildStyleFile()
  const onDisk = parse(readFileSync(STYLE_YML, "utf-8"))
  expect({ global: onDisk.global, patterns: onDisk.patterns, styles: onDisk.styles }).toEqual({
    global: fresh.global,
    patterns: fresh.patterns,
    styles: fresh.styles,
  })
})

test("the android asset copy is in sync with the web copy", () => {
  const web = parse(readFileSync(STYLE_YML, "utf-8"))
  const android = parse(readFileSync(`${ANDROID_ASSETS_DIR}/style.yml`, "utf-8"))
  expect(android.global).toEqual(web.global)
  expect(android.patterns).toEqual(web.patterns)
  expect(android.styles).toEqual(web.styles)
})

test("every style used by the model has an entry in style.yml", () => {
  const built = buildStyleFile()
  for (const s of STYLES) expect(built.styles[s]).toBeDefined()
  expect(built.styles.Neutral).toBeDefined()
})
