const KEEP = new Set("abcdefghijklmnopqrstuvwxyz!?:()@$%&* ")

export function normalize(text: string): string {
  let t = text.replace(/\s+/g, " ").trim().toLowerCase()
  t = t.replace(/(.)\1{2,}/g, "$1$1")
  return [...t].filter((c) => KEEP.has(c)).join("")
}
