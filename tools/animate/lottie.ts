// Scene (static SVG parts) + the model's Lottie layers -> one Lottie file, 1024x1024 at 60 fps like Noto's own
// files. The model works in the 128 grid; a root null layer scales everything up to the 1024 canvas.
import type { Gradient, Paint, Part, Scene, Shape } from "./svg.ts"
import { z } from "zod"

const FPS = 60
const ROOT = 9999
const r3 = (n: number) => Math.round(n * 1000) / 1000
const pt = (p: number[]) => [r3(p[0]), r3(p[1])]
const stat = (k: unknown) => ({ a: 0, k })
const rgb = (h: string) => [1, 3, 5].map((i) => parseInt(h.slice(i, i + 2), 16) / 255)

const paintItem = (p: Paint, kind: "fl" | "st", extra: object = {}) => {
  if (p.kind === "solid") return { ty: kind, c: stat([...rgb(p.color), 1]), o: stat(p.opacity * 100), ...extra }
  const g: Gradient = p.grad
  const colors = g.stops.flatMap((s) => [r3(s.offset), ...rgb(s.color).map((c) => r3(c))])
  const alphas = g.stops.flatMap((s) => [r3(s.offset), r3(s.opacity)])
  const hasAlpha = g.stops.some((s) => s.opacity < 1)
  return {
    ty: kind === "fl" ? "gf" : "gs",
    o: stat(p.opacity * 100),
    t: g.type === "linear" ? 1 : 2,
    s: stat(pt(g.p1)), e: stat(pt(g.p2)), h: stat(0), a: stat(0),
    g: { p: g.stops.length, k: stat(hasAlpha ? [...colors, ...alphas] : colors) },
    ...extra,
  }
}

function shapeItems(s: Shape) {
  if (s.t === "ellipse") return [{ ty: "el", p: stat(pt([s.cx, s.cy])), s: stat([r3(s.rx * 2), r3(s.ry * 2)]) }]
  return s.subs.map((u) => ({
    ty: "sh",
    ks: stat({ c: u.c, v: u.v.map(pt), i: u.i.map(pt), o: u.o.map(pt) }),
  }))
}

const ident = { ty: "tr", p: stat([0, 0]), a: stat([0, 0]), s: stat([100, 100]), r: stat(0), o: stat(100) }

function partGroup(p: Part) {
  const items: object[] = p.shapes.flatMap(shapeItems)
  if (p.stroke) items.push(paintItem(p.stroke.paint, "st", { w: stat(r3(p.stroke.width)), lc: p.stroke.cap, lj: p.stroke.join, ml: p.stroke.miter }))
  if (p.fill) items.push(paintItem(p.fill, "fl", { r: 1 }))
  return { ty: "gr", nm: p.id, it: [...items, { ...ident, o: stat(p.opacity * 100) }] }
}

/** What the model returns: Lottie shape/null layers in the 128 grid, listed back to front like the parts. A layer with
 * `"ref": "p3"` shows the original part p3 (its `shapes` are filled in here); any other shape layer carries the
 * model's own `shapes`. */
export const Doc = z.object({
  frames: z.number().int().min(60).max(240),
  layers: z.array(z.record(z.string(), z.unknown())).min(1),
})
export type Doc = z.infer<typeof Doc>

const hasExpression = (v: unknown): boolean =>
  Array.isArray(v) ? v.some(hasExpression)
    : v && typeof v === "object" ? Object.entries(v).some(([k, x]) => (k === "x" && typeof x === "string") || hasExpression(x))
    : false

export const MAX_BYTES = 250_000

export function compose(scene: Scene, docIn: unknown, name: string) {
  const doc = Doc.parse(docIn)
  const F = doc.frames
  const parts = new Map(scene.parts.map((p) => [p.id, p]))
  const used = new Set<string>()
  const seen = new Set<number>()
  const layers = doc.layers.map((l, n) => {
    const { ref, ...rest } = l as { ref?: string } & Record<string, unknown>
    const ty = (rest.ty as number | undefined) ?? 4
    if (ty !== 3 && ty !== 4) throw new Error(`layer ${n}: only shape layers (ty 4) and null layers (ty 3) are allowed`)
    const ind = typeof rest.ind === "number" ? rest.ind : n + 1
    if (ind === ROOT || seen.has(ind)) throw new Error(`layer ${n}: duplicate or reserved ind ${ind}`)
    seen.add(ind)
    const out: Record<string, unknown> = { ddd: 0, sr: 1, st: 0, bm: 0, nm: ref ?? `layer${n}`, ...rest, ty, ind, ip: 0, op: F }
    out.ks = { o: stat(100), r: stat(0), p: stat([0, 0]), a: stat([0, 0]), s: stat([100, 100]), ...(rest.ks as object) }
    if (ref !== undefined) {
      const p = parts.get(ref)
      if (!p) throw new Error(`layer ${n}: ref ${ref} is not a part of this emoji`)
      used.add(ref)
      out.shapes = [partGroup(p)]
    } else if (ty === 4 && !Array.isArray(rest.shapes)) throw new Error(`layer ${n}: shape layer without "shapes" or "ref"`)
    return out
  })
  for (const l of layers) {
    if (l.parent !== undefined && !seen.has(l.parent as number)) throw new Error(`layer ${l.ind}: parent ${l.parent} does not exist`)
    l.parent ??= ROOT
  }
  const unused = scene.parts.filter((p) => !used.has(p.id)).map((p) => p.id)
  if (unused.length) throw new Error(`parts not used: ${unused.join(", ")} (the first or last frame must show the original emoji, so reference every part)`)
  const root = {
    ddd: 0, ind: ROOT, ty: 3, nm: "root", sr: 1, ip: 0, op: F, st: 0, bm: 0,
    ks: { o: stat(100), r: stat(0), p: stat([0, 0]), a: stat([0, 0]), s: stat([800, 800]) },
  }
  const lottie = { v: "5.8.1", fr: FPS, ip: 0, op: F, w: 1024, h: 1024, nm: name, ddd: 0, assets: [], layers: [...layers.reverse(), root] }
  if (hasExpression(lottie)) throw new Error("expressions are not allowed")
  const bytes = JSON.stringify(lottie).length
  if (bytes > MAX_BYTES) throw new Error(`file too large (${bytes} bytes, max ${MAX_BYTES}); use fewer layers/keyframes`)
  return lottie
}
