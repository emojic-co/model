export function normalize(text, char2idx) {
  const t = text
    .replace(/\s+/g, ' ')
    .trim()
    .toLowerCase()
    .replace(/(.)\1{2,}/g, '$1$1')
  let out = ''
  for (const c of t) if (char2idx.has(c)) out += c
  return out
}

export function encode(text, meta, char2idx) {
  const norm = normalize(text, char2idx).slice(0, meta.max_text_len)
  const ids = new Array(meta.max_text_len).fill(meta.pad_idx)
  for (let i = 0; i < norm.length; i++) ids[i] = char2idx.get(norm[i])
  return BigInt64Array.from(ids, BigInt)
}

export function argmax(arr) {
  let best = 0
  for (let i = 1; i < arr.length; i++) if (arr[i] > arr[best]) best = i
  return best
}

export function softmax(arr) {
  let m = -Infinity
  for (const x of arr) if (x > m) m = x
  const exps = Array.from(arr, (x) => Math.exp(x - m))
  const sum = exps.reduce((a, b) => a + b, 0)
  return exps.map((e) => e / sum)
}
