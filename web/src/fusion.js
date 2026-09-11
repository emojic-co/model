export function fuse(gate, emojiSigmoid, kw) {
  const out = new Float32Array(emojiSigmoid.length)
  for (let i = 0; i < emojiSigmoid.length; i++) {
    out[i] = gate * emojiSigmoid[i] + (1 - gate) * kw[i]
  }
  return out
}
