import { test, expect } from "bun:test"

import { MODEL } from "./annotate.ts"
import { rowMeta } from "./meta.ts"

test("rowMeta fills at + model and passes params through", () => {
  const m = rowMeta({ src: "train", v: 1, params: { batchSize: 50 } })
  expect(m.src).toBe("train")
  expect(m.v).toBe(1)
  expect(m.model).toBe(MODEL)
  expect(m.params).toEqual({ batchSize: 50 })
  expect(m.at as string).toMatch(/^\d{4}-\d{2}-\d{2}$/)
})

test("rowMeta omits undefined optionals", () => {
  const m = rowMeta({ src: "train", v: 1, params: {} })
  expect("topic" in m).toBe(false)
  expect("target_emoji" in m).toBe(false)
  expect("target_feeling" in m).toBe(false)
})

test("rowMeta includes optionals when provided", () => {
  const m = rowMeta({
    src: "upsample-emojis",
    v: 1,
    target_emoji: "📺",
    topic: undefined,
    params: { voice: "a nurse" },
  })
  expect(m.target_emoji).toBe("📺")
  expect("topic" in m).toBe(false)
})
