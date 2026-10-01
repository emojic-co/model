import { expect, test } from "bun:test"

import { resolveEmojiSvg } from "../cli/emoji-svg.ts"
import { compose } from "./lottie.ts"
import { parsePath } from "./path.ts"
import { complexity, parseScene } from "./svg.ts"

test("arcs and quadratics become cubics ending at the target", () => {
  const [a] = parsePath("M0 0A10 10 0 0 1 20 0")
  expect(a.v.length).toBeGreaterThan(1)
  expect(a.v.at(-1)).toEqual([20, 0])
  const [q] = parsePath("M0 0Q5 10 10 0")
  expect(q.v.at(-1)).toEqual([10, 0])
})

test("unsupported constructs make an emoji too complex to animate", () => {
  const masked = parseScene('<mask id="m"><path d="M0 0L1 1"/></mask><path fill="#fff" d="M0 0L1 1"/>')
  expect(masked.unsupported).toContain("<mask>")
  expect(complexity(masked)).toBeGreaterThan(1000)
})

test("composes the model's layers with the original parts", () => {
  const scene = parseScene(resolveEmojiSvg("🌙")!.body)
  const refs = scene.parts.map((p) => ({ ref: p.id }))
  const l = compose(scene, { frames: 60, layers: refs }, "moon")
  expect(l.layers).toHaveLength(scene.parts.length + 1) // + root scale layer
  expect(() => compose(scene, { frames: 60, layers: [{ ref: "nope" }] }, "x")).toThrow()
  expect(() => compose(scene, { frames: 60, layers: refs.slice(1) }, "x")).toThrow(/not used/)
  expect(() => compose(scene, { frames: 60, layers: [...refs, { ty: 0 }] }, "x")).toThrow()
})
