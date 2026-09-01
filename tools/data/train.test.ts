import { test, expect } from "bun:test"

import { TOPICS, topicForBatch } from "./train.ts"

test("TOPICS is a non-trivial unique list", () => {
  expect(TOPICS.length).toBeGreaterThan(15)
  expect(new Set(TOPICS).size).toBe(TOPICS.length)
})

test("topicForBatch round-robins over TOPICS", () => {
  expect(topicForBatch(0)).toBe(TOPICS[0])
  expect(topicForBatch(TOPICS.length)).toBe(TOPICS[0])
  expect(topicForBatch(TOPICS.length + 1)).toBe(TOPICS[1])
})
