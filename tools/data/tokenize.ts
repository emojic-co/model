export const STOPWORDS = new Set(
  ("a an the to of in on at is it its i you we they he she this that for and or but"
    + " not with my your me am are was were be been being do does did have has had"
    + " will would can could just so if").split(" "),
)

export function queryTokens(text: string): string[] {
  return text
    .toLowerCase()
    .replace(/[^a-z0-9\s]/g, " ")
    .split(/\s+/)
    .filter((w) => w.length >= 2 && !STOPWORDS.has(w))
}
