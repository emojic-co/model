import { expect, test } from "bun:test"

import { resolveEmojiSvg } from "../cli/emoji-svg.ts"
import { prepare } from "./lottie.ts"
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

test("prepare scales the model's canvas to 1024 and checks syntax only", () => {
  const doc = { w: 128, h: 128, op: 60, layers: [{ ind: 1, ty: 4, shapes: [] }] }
  const l = prepare(doc, "x")
  expect(l.w).toBe(1024)
  expect(l.layers).toHaveLength(2) // + root scale layer
  expect(l.layers[0].parent).toBe(9999)
  expect(() => prepare({ ...doc, layers: [] }, "x")).toThrow()
  expect(() => prepare({ ...doc, w: "a" }, "x")).toThrow()
  expect(() => prepare({ ...doc, layers: [{ ks: { x: "wiggle(2,3)" } }] }, "x")).toThrow(/expression/)
})
