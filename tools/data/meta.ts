import { MODEL } from "./annotate.ts"

export function rowMeta(f: {
  src: string
  v: number
  topic?: string
  target_emoji?: string
  target_feeling?: string
  params: Record<string, unknown>
}): Record<string, unknown> {
  const m: Record<string, unknown> = {
    src: f.src,
    v: f.v,
    at: new Date().toISOString().slice(0, 10),
    model: MODEL,
    params: f.params,
  }
  if (f.topic !== undefined) m.topic = f.topic
  if (f.target_emoji !== undefined) m.target_emoji = f.target_emoji
  if (f.target_feeling !== undefined) m.target_feeling = f.target_feeling
  return m
}
