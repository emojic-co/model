const KEYBOARD: Record<string, string> = {
  a: "qwsz",
  b: "vghn",
  c: "xdfv",
  d: "serfcx",
  e: "wsdr",
  f: "drtgvc",
  g: "ftyhbv",
  h: "gyujnb",
  i: "ujko",
  j: "huikmn",
  k: "jiolm",
  l: "kop",
  m: "njk",
  n: "bhjm",
  o: "iklp",
  p: "ol",
  q: "wa",
  r: "edft",
  s: "awdxz",
  t: "rfgy",
  u: "yhji",
  v: "cfgb",
  w: "qase",
  x: "zsdc",
  y: "tghu",
  z: "asx",
}

const MISSPELL: Record<string, string[]> = {
  receive: ["recieve"],
  believe: ["beleive"],
  friend: ["freind"],
  weird: ["wierd"],
  piece: ["peice"],
  their: ["thier"],
  because: ["becuase", "becase"],
  definitely: ["definately"],
  separate: ["seperate"],
  necessary: ["neccessary"],
  tomorrow: ["tomorow"],
  restaurant: ["restaraunt"],
  beautiful: ["beatiful"],
  favourite: ["favorite"],
  colour: ["color"],
}

function idx(word: string, salt: number, span: number): number {
  let h = salt
  for (let i = 0; i < word.length; i++) h = (h * 31 + word.charCodeAt(i)) >>> 0
  return span > 0 ? h % span : 0
}

export function fuzzVariants(keyword: string): string[] {
  const w = keyword.toLowerCase()
  const out = new Set<string>()

  if (w.length >= 3) {
    const i = idx(w, 1, w.length)
    out.add(w.slice(0, i) + w[i] + w.slice(i))
  }
  if (w.length >= 4) {
    const i = idx(w, 2, w.length)
    out.add(w.slice(0, i) + w.slice(i + 1))
  }
  if (w.length >= 4) {
    const i = idx(w, 3, w.length - 1)
    out.add(w.slice(0, i) + w[i + 1] + w[i] + w.slice(i + 2))
  }
  if (w.length >= 3) {
    const i = idx(w, 4, w.length)
    const nb = KEYBOARD[w[i]]
    if (nb) out.add(w.slice(0, i) + nb[idx(w, 5, nb.length)] + w.slice(i + 1))
  }

  out.add(w.endsWith("s") ? w.slice(0, -1) : w + "s")
  if (w.endsWith("y") && w.length >= 3) out.add(w.slice(0, -1) + "ies")
  else if (!w.endsWith("es")) out.add(w + "es")
  if (!w.endsWith("ing")) out.add((w.endsWith("e") ? w.slice(0, -1) : w) + "ing")
  if (!w.endsWith("ed")) out.add(w + (w.endsWith("e") ? "d" : "ed"))

  for (const m of MISSPELL[w] ?? []) out.add(m)

  out.delete(w)
  return [...out]
}
