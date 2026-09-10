function zscore(a) {
  let m = 0
  for (const x of a) m += x
  m /= a.length
  let v = 0
  for (const x of a) v += (x - m) ** 2
  v = Math.sqrt(v / a.length) + 1e-6
  const out = new Float32Array(a.length)
  for (let i = 0; i < a.length; i++) out[i] = (a[i] - m) / v
  return out
}

const softplus = (x) => Math.log1p(Math.exp(-Math.abs(x))) + Math.max(x, 0)

export function makeFusion(mf) {
  if (mf.variant === 'gain') {
    const beta = softplus(mf.beta)
    return {
      fuse(logits, kw) {
        const out = new Float32Array(logits.length)
        for (let i = 0; i < logits.length; i++) out[i] = logits[i] + beta * kw[i]
        return out
      },
    }
  }
  if (mf.variant === 'mix') {
    const g = mf.g
    return {
      fuse(logits, kw) {
        const z = zscore(logits)
        const out = new Float32Array(logits.length)
        for (let i = 0; i < logits.length; i++) out[i] = (1 - g) * z[i] + g * kw[i]
        return out
      },
    }
  }
  const { bn_mean, bn_var, w, b } = mf
  return {
    fuse(logits, kw) {
      let mx = -Infinity
      let t1 = -Infinity
      let t2 = -Infinity
      let sum = 0
      let kmax = 0
      let kcount = 0
      let ksum = 0
      let smax = -Infinity
      for (const x of logits) if (x > smax) smax = x
      for (const x of logits) sum += Math.exp(x - smax)
      let ent = 0
      for (const x of logits) {
        const p = Math.exp(x - smax) / sum
        ent -= p * Math.log(p + 1e-9)
        if (x > t1) {
          t2 = t1
          t1 = x
        } else if (x > t2) t2 = x
        if (x > mx) mx = x
      }
      for (const x of kw) {
        if (x > kmax) kmax = x
        if (x > 0) kcount++
        ksum += x
      }
      const feat = [mx, t1 - t2, ent, kmax, kcount, ksum]
      let lin = b
      for (let i = 0; i < 6; i++) {
        const n = (feat[i] - bn_mean[i]) / Math.sqrt(bn_var[i] + 1e-5)
        lin += w[i] * n
      }
      const a = 1 / (1 + Math.exp(-lin))
      const z = zscore(logits)
      const out = new Float32Array(logits.length)
      for (let i = 0; i < logits.length; i++) out[i] = a * z[i] + (1 - a) * kw[i]
      return out
    },
  }
}
