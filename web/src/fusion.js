export function makeFusion(mf) {
  const { w_dl, w_search, b } = mf
  return {
    fuse(logits, kw) {
      const out = new Float32Array(logits.length)
      for (let i = 0; i < logits.length; i++) {
        out[i] = w_dl[i] * logits[i] + w_search[i] * kw[i] + b[i]
      }
      return out
    },
  }
}
