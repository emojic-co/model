// Scene (static SVG parts) + Spec (motion) -> Lottie JSON, 1024x1024 at 60 fps like Noto's own files.
import type { Gradient, Paint, Part, Scene, Shape } from "./svg.ts"
import { Spec, type TrackT } from "./spec.ts"
import type { z } from "zod"

const SC = 8 // 128 grid -> 1024 canvas
const FPS = 60
const r2 = (n: number) => Math.round(n * 100) / 100
const pt = (p: number[]) => [r2(p[0] * SC), r2(p[1] * SC)]
const stat = (k: unknown) => ({ a: 0, k })
const rgb = (h: string) => [1, 3, 5].map((i) => parseInt(h.slice(i, i + 2), 16) / 255)

const EASE: Record<string, { o: [number, number]; i: [number, number] }> = {
  inOut: { o: [0.42, 0], i: [0.58, 1] },
  in: { o: [0.42, 0], i: [1, 1] },
  out: { o: [0, 0], i: [0.58, 1] },
  linear: { o: [0, 0], i: [1, 1] },
}

function kfs(track: TrackT, map: (v: number[]) => number[]) {
  const e = EASE[track.ease ?? "inOut"]
  return {
    a: 1,
    k: track.keys.map((k, n) => {
      const s = map(k.v)
      const kf: Record<string, unknown> = { t: k.t, s }
      if (n < track.keys.length - 1) {
        kf.i = { x: s.map(() => e.i[0]), y: s.map(() => e.i[1]) }
        kf.o = { x: s.map(() => e.o[0]), y: s.map(() => e.o[1]) }
      }
      return kf
    }),
  }
}

const paintItem = (p: Paint, kind: "fl" | "st", extra: object = {}) => {
  if (p.kind === "solid") return { ty: kind, c: stat([...rgb(p.color), 1]), o: stat(p.opacity * 100), ...extra }
  const g: Gradient = p.grad
  const colors = g.stops.flatMap((s) => [r2(s.offset), ...rgb(s.color).map((c) => r2(c))])
  const alphas = g.stops.flatMap((s) => [r2(s.offset), r2(s.opacity)])
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
  if (s.t === "ellipse") return [{ ty: "el", p: stat(pt([s.cx, s.cy])), s: stat([r2(s.rx * 2 * SC), r2(s.ry * 2 * SC)]) }]
  return s.subs.map((u) => ({
    ty: "sh",
    ks: stat({ c: u.c, v: u.v.map(pt), i: u.i.map(pt), o: u.o.map(pt) }),
  }))
}

const ident = { ty: "tr", p: stat([0, 0]), a: stat([0, 0]), s: stat([100, 100]), r: stat(0), o: stat(100) }

function partGroup(p: Part) {
  const items: object[] = p.shapes.flatMap(shapeItems)
  if (p.stroke) items.push(paintItem(p.stroke.paint, "st", { w: stat(r2(p.stroke.width * SC)), lc: p.stroke.cap, lj: p.stroke.join, ml: p.stroke.miter }))
  if (p.fill) items.push(paintItem(p.fill, "fl", { r: 1 }))
  return { ty: "gr", nm: p.id, it: [...items, { ...ident, o: stat(p.opacity * 100) }] }
}

function star4(cx: number, cy: number, r: number) {
  const v = Array.from({ length: 8 }, (_, k) => {
    const a = -Math.PI / 2 + (k * Math.PI) / 4
    const rr = k % 2 ? r * 0.28 : r
    return [cx + rr * Math.cos(a), cy + rr * Math.sin(a)]
  })
  return { ty: "sh", ks: stat({ c: true, v: v.map(pt), i: v.map(() => [0, 0]), o: v.map(() => [0, 0]) }) }
}

type Anim = { rotation?: TrackT; move?: TrackT; scale?: TrackT; opacity?: TrackT }

function layer(ind: number, nm: string, shapes: object[], pivot: number[], a: Anim, frames: number) {
  const pv = pt(pivot)
  return {
    ddd: 0, ind, ty: 4, nm, sr: 1, ip: 0, op: frames, st: 0, bm: 0,
    ks: {
      o: a.opacity ? kfs(a.opacity, (v) => [r2(v[0] * 100)]) : stat(100),
      r: a.rotation ? kfs(a.rotation, (v) => [v[0]]) : stat(0),
      p: a.move ? kfs(a.move, (v) => [r2(pv[0] + v[0] * SC), r2(pv[1] + v[1] * SC)]) : stat(pv),
      a: stat(pv),
      s: a.scale ? kfs(a.scale, (v) => [r2(v[0] * 100), r2((v[1] ?? v[0]) * 100)]) : stat([100, 100]),
    },
    shapes,
  }
}

/** Makes tracks loop-safe: sorted, spanning 0..frames, with identical first/last value. */
export function normalizeTrack(t: TrackT | undefined, frames: number, rest: number[] | null): TrackT | undefined {
  if (!t) return undefined
  const keys = t.keys.map((k) => ({ t: Math.max(0, Math.min(frames, Math.round(k.t))), v: k.v })).sort((a, b) => a.t - b.t)
  const first = rest ?? keys[0].v
  if (keys[0].t !== 0) keys.unshift({ t: 0, v: first })
  else keys[0] = { t: 0, v: first }
  if (keys[keys.length - 1].t !== frames) keys.push({ t: frames, v: first })
  else keys[keys.length - 1] = { t: frames, v: first }
  return { ease: t.ease, keys }
}

const REST = { rotation: [0], move: [0, 0], scale: [1, 1], opacity: [1] }

export function buildLottie(scene: Scene, specIn: z.infer<typeof Spec>, name: string) {
  const spec = Spec.parse(specIn)
  const F = spec.frames
  const known = new Set(scene.parts.map((p) => p.id))
  const byPart = new Map<string, { pivot: number[]; anim: Anim }>()
  for (const g of spec.groups) {
    const anim: Anim = {}
    for (const k of ["rotation", "move", "scale", "opacity"] as const) anim[k] = normalizeTrack(g[k], F, REST[k]) // rest pose at t=0 = the static emoji
    for (const id of g.parts) {
      if (!known.has(id)) throw new Error(`spec references unknown part ${id}`)
      byPart.set(id, { pivot: g.pivot, anim })
    }
  }
  const mk = (extra: (typeof spec.extras)[number]) => {
    const anim: Anim = {}
    for (const k of ["rotation", "move", "scale", "opacity"] as const) anim[k] = normalizeTrack(extra[k], F, null)
    const [cx, cy] = extra.center, r = extra.size
    const shape = extra.shape === "circle"
      ? { ty: "el", p: stat(pt([cx, cy])), s: stat([r * 2 * SC, r * 2 * SC]) }
      : star4(cx, cy, r)
    return (ind: number) =>
      layer(ind, extra.name, [{ ty: "gr", it: [shape, { ty: "fl", c: stat([...rgb(extra.color), 1]), o: stat(100), r: 1 }, ident] }], [cx, cy], anim, F)
  }
  const back = spec.extras.filter((e) => e.behind).map(mk)
  const front = spec.extras.filter((e) => !e.behind).map(mk)
  // Lottie layer order is top-first: front extras, parts in reverse SVG order, behind extras.
  const builders = [
    ...front,
    ...[...scene.parts].reverse().map((p) => (ind: number) => {
      const m = byPart.get(p.id)
      return layer(ind, p.id, [partGroup(p)], m?.pivot ?? [64, 64], m?.anim ?? {}, F)
    }),
    ...back,
  ]
  return {
    v: "5.8.1", fr: FPS, ip: 0, op: F, w: 1024, h: 1024, nm: name, ddd: 0, assets: [],
    layers: builders.map((b, n) => b(n + 1)),
  }
}
