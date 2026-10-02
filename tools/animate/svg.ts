// Parses a Noto emoji SVG body (iconify, 128x128) into drawable parts for the Lottie builder, and
// scores how hard the emoji is to animate. Anything the builder cannot reproduce faithfully
// (masks, clip paths, <use>, element transforms, dashes, arcs) lands in `unsupported`.
import { parsePath, type Sub } from "./path.ts"

export type Stop = { offset: number; color: string; opacity: number }
export type Gradient = {
  type: "linear" | "radial"
  p1: [number, number] // linear start | radial center
  p2: [number, number] // linear end | radial edge point
  stops: Stop[]
}
export type Paint = { kind: "solid"; color: string; opacity: number } | { kind: "grad"; grad: Gradient; opacity: number }
export type Shape = { t: "path"; subs: Sub[] } | { t: "ellipse"; cx: number; cy: number; rx: number; ry: number }
export type Part = {
  id: string
  shapes: Shape[]
  fill?: Paint
  stroke?: { paint: Paint; width: number; cap: number; join: number; miter: number }
  opacity: number
  bbox: [number, number, number, number]
  nodes: number
}
export type Scene = { size: number; parts: Part[]; unsupported: string[]; gradients: number; strokes: number }

type Tag = { name: string; attrs: Record<string, string>; open: boolean; close: boolean }
type Mat = [number, number, number, number, number, number]

const attrsOf = (s: string) => Object.fromEntries([...s.matchAll(/([\w:-]+)="([^"]*)"/g)].map((m) => [m[1], m[2]]))

function tags(body: string): Tag[] {
  return [...body.matchAll(/<(\/?)([\w]+)([^>]*?)(\/?)>/g)].map((m) => ({
    name: m[2],
    attrs: attrsOf(m[3]),
    open: !m[1],
    close: !!m[1] || !!m[4],
  }))
}

const NAMED: Record<string, string> = { silver: "#c0c0c0", gray: "#808080", grey: "#808080", white: "#ffffff", black: "#000000", red: "#ff0000" }

function hex(c: string | undefined): string | null {
  if (!c) return null
  if (NAMED[c.trim().toLowerCase()]) return NAMED[c.trim().toLowerCase()]
  const m = /^#([0-9a-f]{3}|[0-9a-f]{6})$/i.exec(c.trim())
  if (!m) return null
  const h = m[1].length === 3 ? [...m[1]].map((x) => x + x).join("") : m[1]
  return `#${h.toLowerCase()}`
}

const I: Mat = [1, 0, 0, 1, 0, 0]
const mul = (a: Mat, b: Mat): Mat => [
  a[0] * b[0] + a[2] * b[1], a[1] * b[0] + a[3] * b[1],
  a[0] * b[2] + a[2] * b[3], a[1] * b[2] + a[3] * b[3],
  a[0] * b[4] + a[2] * b[5] + a[4], a[1] * b[4] + a[3] * b[5] + a[5],
]

function matrix(t: string | undefined): Mat | null {
  if (!t) return I
  let m: Mat = I
  for (const f of t.matchAll(/(\w+)\(([^)]*)\)/g)) {
    const n = f[2].split(/[\s,]+/).filter(Boolean).map(Number)
    if (f[1] === "matrix" && n.length === 6) m = mul(m, n as Mat)
    else if (f[1] === "translate") m = mul(m, [1, 0, 0, 1, n[0], n[1] ?? 0])
    else if (f[1] === "scale") m = mul(m, [n[0], 0, 0, n[1] ?? n[0], 0, 0])
    else if (f[1] === "rotate") {
      const r = (n[0] * Math.PI) / 180, c = Math.cos(r), sn = Math.sin(r), [cx, cy] = [n[1] ?? 0, n[2] ?? 0]
      m = mul(m, [1, 0, 0, 1, cx, cy]); m = mul(m, [c, sn, -sn, c, 0, 0]); m = mul(m, [1, 0, 0, 1, -cx, -cy])
    }
    else return null
  }
  return m
}

const apply = (m: Mat, x: number, y: number): [number, number] => [m[0] * x + m[2] * y + m[4], m[1] * x + m[3] * y + m[5]]

function boundsOf(shapes: Shape[]): [number, number, number, number] {
  let x0 = Infinity, y0 = Infinity, x1 = -Infinity, y1 = -Infinity
  const add = (x: number, y: number) => { x0 = Math.min(x0, x); y0 = Math.min(y0, y); x1 = Math.max(x1, x); y1 = Math.max(y1, y) }
  for (const s of shapes) {
    if (s.t === "ellipse") { add(s.cx - s.rx, s.cy - s.ry); add(s.cx + s.rx, s.cy + s.ry); continue }
    for (const sub of s.subs) sub.v.forEach((p, k) => {
      add(p[0], p[1])
      add(p[0] + sub.o[k][0], p[1] + sub.o[k][1])
      add(p[0] + sub.i[k][0], p[1] + sub.i[k][1])
    })
  }
  return [x0, y0, x1, y1]
}

// Bakes an affine matrix into a shape; ellipses become 4-segment cubic paths.
function transformShape(sh: Shape, M: Mat): Shape {
  if (sh.t === "ellipse") {
    const k = 0.5522847498
    const { cx, cy, rx, ry } = sh
    const v = [[cx + rx, cy], [cx, cy + ry], [cx - rx, cy], [cx, cy - ry]]
    const o = [[0, k * ry], [-k * rx, 0], [0, -k * ry], [k * rx, 0]]
    const i = [[0, -k * ry], [k * rx, 0], [0, k * ry], [-k * rx, 0]]
    sh = { t: "path", subs: [{ v, i, o, c: true }] }
  }
  const lin = (p: number[]) => [M[0] * p[0] + M[2] * p[1], M[1] * p[0] + M[3] * p[1]]
  return { t: "path", subs: sh.subs.map((u) => ({ c: u.c, v: u.v.map((p) => apply(M, p[0], p[1])), i: u.i.map(lin), o: u.o.map(lin) })) }
}

export function parseScene(body: string, size = 128): Scene {
  const unsupported = new Set<string>()
  const grads = new Map<string, { tag: Tag; stops: Stop[] }>()
  const parts: Part[] = []
  const groupOpacity: number[] = [1]
  const groupMat: Mat[] = [I]
  let curGrad: { id: string; tag: Tag; stops: Stop[] } | null = null
  let inDefs = 0
  let gradients = 0, strokes = 0

  const paint = (value: string | undefined, opacity: number, bbox: Part["bbox"], M: Mat): Paint | undefined => {
    if (!value || value === "none") return undefined
    const ref = /^url\(#([^)]+)\)$/.exec(value)
    if (!ref) {
      const c = hex(value)
      if (!c) { unsupported.add(`color:${value}`); return undefined }
      return { kind: "solid", color: c, opacity }
    }
    const g = grads.get(ref[1])
    if (!g || !g.stops.length) { unsupported.add(`gradient:${ref[1]}`); return undefined }
    const a = g.tag.attrs
    const gm = matrix(a.gradientTransform)
    if (!gm) { unsupported.add("gradientTransform"); return undefined }
    const m = mul(M, gm)
    const [x0, y0, x1, y1] = bbox
    const abs = a.gradientUnits === "userSpaceOnUse"
    const ux = (v: number) => (abs ? v : x0 + v * (x1 - x0))
    const uy = (v: number) => (abs ? v : y0 + v * (y1 - y0))
    const n = (k: string, d: number) => (a[k] === undefined ? d : Number(a[k]))
    gradients++
    if (g.tag.name === "linearGradient") {
      const p1 = apply(m, ux(n("x1", 0)), uy(n("y1", 0)))
      const p2 = apply(m, ux(n("x2", abs ? 0 : 1)), uy(n("y2", 0)))
      return { kind: "grad", opacity, grad: { type: "linear", p1, p2, stops: g.stops } }
    }
    // Lottie radial gradients are circular: map the centre and a point at radius r through the matrix.
    const cx = n("cx", abs ? 0 : 0.5), cy = n("cy", abs ? 0 : 0.5), r = n("r", abs ? 0 : 0.5)
    const p1 = apply(m, ux(cx), uy(cy))
    const p2 = apply(m, ux(cx) + (abs ? r : r * (x1 - x0)), uy(cy))
    return { kind: "grad", opacity, grad: { type: "radial", p1, p2, stops: g.stops } }
  }

  for (const t of tags(body)) {
    const a = t.attrs
    if (t.name === "defs") { inDefs += t.open && !t.close ? 1 : t.close && !t.open ? -1 : 0; continue }
    if (t.name === "linearGradient" || t.name === "radialGradient") {
      if (t.open) {
        curGrad = { id: a.id, tag: t, stops: [] }
        grads.set(a.id, curGrad)
      }
      if (t.close && !t.open) curGrad = null
      continue
    }
    if (t.name === "stop") {
      const color = hex(a["stop-color"] ?? (/stop-color:([^;]+)/.exec(a.style ?? "") ?? [])[1])
      if (!color || !curGrad) { unsupported.add("stop-color"); continue }
      curGrad.stops.push({ offset: Number(a.offset ?? 0), color, opacity: Number(a["stop-opacity"] ?? 1) })
      continue
    }
    if (t.name === "g") {
      if (t.open && !t.close) {
        groupOpacity.push(groupOpacity[groupOpacity.length - 1] * Number(a.opacity ?? 1))
        const gt = matrix(a.transform)
        if (!gt) unsupported.add("g@transform")
        groupMat.push(mul(groupMat[groupMat.length - 1], gt ?? I))
      } else if (t.close && !t.open) { groupOpacity.pop(); groupMat.pop() }
      for (const k of ["mask", "clip-path"]) if (a[k]) unsupported.add(`g@${k}`)
      continue
    }
    if (!t.open) continue
    if (["use", "mask", "clipPath", "image", "text", "filter"].includes(t.name)) { unsupported.add(`<${t.name}>`); continue }
    if (!["path", "circle", "ellipse"].includes(t.name)) continue
    for (const k of ["mask", "clip-path", "stroke-dasharray"]) if (a[k]) unsupported.add(`${t.name}@${k}`)
    let shapes: Shape[]
    try {
      shapes = t.name === "path"
        ? [{ t: "path", subs: parsePath(a.d ?? "") }]
        : [{ t: "ellipse", cx: Number(a.cx), cy: Number(a.cy), rx: Number(a.rx ?? a.r), ry: Number(a.ry ?? a.r) }]
    } catch (e) {
      unsupported.add((e as Error).message)
      continue
    }
    const em = matrix(a.transform)
    if (!em) unsupported.add(`${t.name}@transform`)
    const M = mul(groupMat[groupMat.length - 1], em ?? I)
    const bbox = boundsOf(shapes) // local space, for objectBoundingBox gradients
    if (M !== I && M.some((v, k) => v !== I[k])) shapes = shapes.map((sh) => transformShape(sh, M))
    const op = Number(a.opacity ?? 1) * groupOpacity[groupOpacity.length - 1]
    const fill = paint(a.fill ?? "#000000", Number(a["fill-opacity"] ?? 1), bbox, M)
    let stroke: Part["stroke"]
    const sp = paint(a.stroke, Number(a["stroke-opacity"] ?? 1), bbox, M)
    if (sp) {
      strokes++
      stroke = {
        paint: sp,
        width: Number(a["stroke-width"] ?? 1) * Math.sqrt(Math.abs(M[0] * M[3] - M[1] * M[2])),
        cap: { butt: 1, round: 2, square: 3 }[a["stroke-linecap"] ?? "butt"] ?? 1,
        join: { miter: 1, round: 2, bevel: 3 }[a["stroke-linejoin"] ?? "miter"] ?? 1,
        miter: Number(a["stroke-miterlimit"] ?? 4),
      }
    }
    const nodes = shapes.reduce((n, s) => n + (s.t === "ellipse" ? 4 : s.subs.reduce((m, u) => m + u.v.length, 0)), 0)
    parts.push({ id: `p${parts.length}`, shapes, fill, stroke, opacity: op, bbox: boundsOf(shapes), nodes })
  }
  void inDefs
  return { size, parts, unsupported: [...unsupported].sort(), gradients, strokes }
}

// Animation difficulty: more independent parts / path nodes / gradients / strokes = harder to rig and
// to keep visually faithful. Unsupported constructs make the emoji un-animatable for now (+1000).
export function complexity(s: Scene): number {
  const nodes = s.parts.reduce((n, p) => n + p.nodes, 0)
  const score = s.parts.length + nodes / 20 + s.gradients * 1.5 + s.strokes * 1.5
  return Math.round((score + (s.unsupported.length ? 1000 : 0)) * 10) / 10
}
