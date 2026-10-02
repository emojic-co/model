// SVG path `d` -> Lottie-style bezier subpaths (absolute coords, in/out tangents relative to vertex).
// Supports M L H V C S Q T A Z (absolute + relative); arcs and quadratics become cubics.
export type Sub = { v: number[][]; i: number[][]; o: number[][]; c: boolean }

export function parsePath(d: string): Sub[] {
  const toks = d.match(/[MmLlHhVvCcSsQqTtAaZz]|-?(?:\d+\.?\d*|\.\d+)(?:e[-+]?\d+)?/g) ?? []
  const subs: Sub[] = []
  let cur: Sub = { v: [], i: [], o: [], c: false }
  let k = 0
  let cmd = ""
  let x = 0, y = 0, sx = 0, sy = 0
  let lastC2: [number, number] | null = null
  let lastQ: [number, number] | null = null
  const num = () => Number(toks[k++])
  const flush = () => {
    if (cur.v.length) subs.push(cur)
    cur = { v: [], i: [], o: [], c: false }
  }
  const line = (nx: number, ny: number) => {
    cur.v.push([nx, ny]); cur.i.push([0, 0]); cur.o.push([0, 0])
    x = nx; y = ny; lastC2 = null
  }
  const cubic = (c1: number[], c2: number[], e: number[]) => {
    cur.o[cur.o.length - 1] = [c1[0] - x, c1[1] - y]
    cur.v.push([e[0], e[1]]); cur.i.push([c2[0] - e[0], c2[1] - e[1]]); cur.o.push([0, 0])
    x = e[0]; y = e[1]
  }
  while (k < toks.length) {
    if (/^[A-Za-z]$/.test(toks[k])) cmd = toks[k++]
    const rel = cmd === cmd.toLowerCase()
    const C = cmd.toUpperCase()
    if (!"MLHVCSQTAZ".includes(C)) throw new Error(`unsupported path command ${cmd}`)
    if (C === "Z") {
      cur.c = true; flush(); x = sx; y = sy; lastC2 = null
      continue
    }
    if (C === "M") {
      flush()
      const a = num(), b = num()
      x = rel ? x + a : a; y = rel ? y + b : b
      sx = x; sy = y
      cur.v.push([x, y]); cur.i.push([0, 0]); cur.o.push([0, 0])
      cmd = rel ? "l" : "L"; lastC2 = null
    } else if (C === "L") {
      const a = num(), b = num()
      line(rel ? x + a : a, rel ? y + b : b)
    } else if (C === "H") {
      const a = num(); line(rel ? x + a : a, y)
    } else if (C === "V") {
      const b = num(); line(x, rel ? y + b : b)
    } else if (C === "A") {
      const rx = num(), ry = num(), rot = num(), large = num(), sweep = num()
      const a = num(), b = num()
      const nx = rel ? x + a : a, ny = rel ? y + b : b
      for (const [c1, c2, e] of arcToCubics(x, y, rx, ry, rot, !!large, !!sweep, nx, ny)) cubic(c1, c2, e)
      lastC2 = null
    } else if (C === "Q" || C === "T") {
      const a = Array.from({ length: C === "Q" ? 4 : 2 }, num).map((v, j) => (rel ? v + (j % 2 ? y : x) : v))
      const q: [number, number] = C === "Q" ? [a[0], a[1]] : lastQ ? [2 * x - lastQ[0], 2 * y - lastQ[1]] : [x, y]
      const e: [number, number] = C === "Q" ? [a[2], a[3]] : [a[0], a[1]]
      cubic([x + (2 / 3) * (q[0] - x), y + (2 / 3) * (q[1] - y)], [e[0] + (2 / 3) * (q[0] - e[0]), e[1] + (2 / 3) * (q[1] - e[1])], e)
      lastQ = q; continue
    } else {
      const n = C === "C" ? 6 : 4
      const a = Array.from({ length: n }, num).map((v, j) => (rel ? v + (j % 2 ? y : x) : v))
      const c1: [number, number] = C === "C" ? [a[0], a[1]] : lastC2 ? [2 * x - lastC2[0], 2 * y - lastC2[1]] : [x, y]
      const c2: [number, number] = C === "C" ? [a[2], a[3]] : [a[0], a[1]]
      const e: [number, number] = C === "C" ? [a[4], a[5]] : [a[2], a[3]]
      cubic(c1, c2, e)
      lastC2 = c2
    }
    if (C !== "Q" && C !== "T") lastQ = null
  }
  flush()
  return subs
}

// SVG arc (endpoint form) -> cubic segments, per the SVG implementation notes (F.6.5).
function arcToCubics(
  x1: number, y1: number, rx: number, ry: number, rotDeg: number, large: boolean, sweep: boolean, x2: number, y2: number,
): [number[], number[], number[]][] {
  if (rx === 0 || ry === 0 || (x1 === x2 && y1 === y2)) return [[[x1, y1], [x2, y2], [x2, y2]]]
  rx = Math.abs(rx); ry = Math.abs(ry)
  const phi = (rotDeg * Math.PI) / 180, cp = Math.cos(phi), sp = Math.sin(phi)
  const dx = (x1 - x2) / 2, dy = (y1 - y2) / 2
  const x1p = cp * dx + sp * dy, y1p = -sp * dx + cp * dy
  const lam = (x1p * x1p) / (rx * rx) + (y1p * y1p) / (ry * ry)
  if (lam > 1) { rx *= Math.sqrt(lam); ry *= Math.sqrt(lam) }
  const num = rx * rx * ry * ry - rx * rx * y1p * y1p - ry * ry * x1p * x1p
  const den = rx * rx * y1p * y1p + ry * ry * x1p * x1p
  const co = (large === sweep ? -1 : 1) * Math.sqrt(Math.max(0, num / den))
  const cxp = (co * rx * y1p) / ry, cyp = (-co * ry * x1p) / rx
  const cx = cp * cxp - sp * cyp + (x1 + x2) / 2, cy = sp * cxp + cp * cyp + (y1 + y2) / 2
  const ang = (ux: number, uy: number, vx: number, vy: number) => Math.atan2(ux * vy - uy * vx, ux * vx + uy * vy)
  const th1 = ang(1, 0, (x1p - cxp) / rx, (y1p - cyp) / ry)
  let dth = ang((x1p - cxp) / rx, (y1p - cyp) / ry, (-x1p - cxp) / rx, (-y1p - cyp) / ry)
  if (!sweep && dth > 0) dth -= 2 * Math.PI
  if (sweep && dth < 0) dth += 2 * Math.PI
  const n = Math.max(1, Math.ceil(Math.abs(dth) / (Math.PI / 2) - 1e-9))
  const step = dth / n, t = (4 / 3) * Math.tan(step / 4)
  const pt = (a: number): [number, number] => [
    cx + rx * Math.cos(a) * cp - ry * Math.sin(a) * sp,
    cy + rx * Math.cos(a) * sp + ry * Math.sin(a) * cp,
  ]
  const dv = (a: number): [number, number] => [
    -rx * Math.sin(a) * cp - ry * Math.cos(a) * sp,
    -rx * Math.sin(a) * sp + ry * Math.cos(a) * cp,
  ]
  const out: [number[], number[], number[]][] = []
  for (let k = 0; k < n; k++) {
    const a0 = th1 + k * step, a1 = a0 + step
    const p0 = pt(a0), p1 = pt(a1), d0 = dv(a0), d1 = dv(a1)
    out.push([[p0[0] + t * d0[0], p0[1] + t * d0[1]], [p1[0] - t * d1[0], p1[1] - t * d1[1]], k === n - 1 ? [x2, y2] : p1])
  }
  return out
}
