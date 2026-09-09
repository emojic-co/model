const KEEP = new Set(
  "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789!?:()@$%&* ",
)

export function normalize(text: string): string {
  let t = text.replace(/\s+/g, " ").trim()
  t = t.replace(/(.)\1{2,}/g, "$1$1")
  return [...t].filter((c) => KEEP.has(c)).join("")
}
