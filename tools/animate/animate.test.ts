import { expect, test } from "bun:test"

import { resolveEmojiSvg } from "../cli/emoji-svg.ts"
import { buildLottie, normalizeTrack } from "./lottie.ts"
import { parsePath } from "./path.ts"
import { complexity, parseScene } from "./svg.ts"

test("arcs and quadratics become cubics ending at the target", () => {
  const [a] = parsePath("M0 0A10 10 0 0 1 20 0")
  expect(a.v.length).toBeGreaterThan(1)
  expect(a.v.at(-1)).toEqual([20, 0])
  const [q] = parsePath("M0 0Q5 10 10 0")
  expect(q.v.at(-1)).toEqual([10, 0])
})

test("tracks are made loop-safe with a rest pose at frame 0", () => {
  const t = normalizeTrack({ keys: [{ t: 30, v: [5] }, { t: 60, v: [-5] }] }, 120, [0])!
  expect(t.keys.map((k) => [k.t, k.v[0]])).toEqual([[0, 0], [30, 5], [60, -5], [120, 0]])
})

test("unsupported constructs make an emoji too complex to animate", () => {
  const masked = parseScene('<mask id="m"><path d="M0 0L1 1"/></mask><path fill="#fff" d="M0 0L1 1"/>')
  expect(masked.unsupported).toContain("<mask>")
  expect(complexity(masked)).toBeGreaterThan(1000)
})

test("builds a valid lottie for a real emoji", () => {
  const scene = parseScene(resolveEmojiSvg("🌙")!.body)
  const l = buildLottie(scene, {
    frames: 60,
    groups: [{ name: "all", parts: ["p0", "p1"], pivot: [64, 64], rotation: { keys: [{ t: 0, v: [0] }, { t: 30, v: [5] }, { t: 60, v: [0] }] } }],
    extras: [],
  }, "moon")
  expect(l.layers).toHaveLength(2)
  expect(() => buildLottie(scene, { frames: 60, groups: [{ name: "x", parts: ["nope"], pivot: [0, 0] }], extras: [] }, "x")).toThrow()
})
