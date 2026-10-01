// The model's Lottie -> a checked Lottie file at 1024x1024 like Noto's own files. The model works on a smaller canvas
// (128, the SVG's own units); a root null layer scales everything up.
const ROOT = 9999
const CANVAS = 1024
const stat = (k: unknown) => ({ a: 0, k })

const hasExpression = (v: unknown): boolean =>
  Array.isArray(v) ? v.some(hasExpression)
    : v && typeof v === "object" ? Object.entries(v).some(([k, x]) => (k === "x" && typeof x === "string") || hasExpression(x))
    : false

export const MAX_BYTES = 400_000

/** Syntax checks only. Scales the model's canvas up to 1024 with a root null layer that parents every top-level layer. */
export function prepare(docIn: unknown, name: string) {
  const d = docIn as Record<string, any>
  if (!d || typeof d !== "object" || !Array.isArray(d.layers) || !d.layers.length) throw new Error('expected a Lottie object with a non-empty "layers" array')
  for (const k of ["w", "h", "op"]) if (typeof d[k] !== "number" || !(d[k] > 0)) throw new Error(`"${k}" must be a positive number`)
  if (hasExpression(d)) throw new Error("expressions are not allowed")
  const scale = (CANVAS / d.w) * 100
  const inds = new Set<number>(d.layers.map((l: any) => l.ind).filter((n: unknown) => typeof n === "number"))
  if (inds.has(ROOT)) throw new Error(`layer ind ${ROOT} is reserved`)
  const layers = d.layers.map((l: any) => ({ ...l, parent: l.parent ?? ROOT }))
  const root = {
    ddd: 0, ind: ROOT, ty: 3, nm: "root", sr: 1, ip: d.ip ?? 0, op: d.op, st: 0, bm: 0,
    ks: { o: stat(100), r: stat(0), p: stat([0, 0]), a: stat([0, 0]), s: stat([scale, scale]) },
  }
  const lottie = { ...d, nm: name, w: CANVAS, h: CANVAS, layers: [...layers, root] }
  const bytes = JSON.stringify(lottie).length
  if (bytes > MAX_BYTES) throw new Error(`file too large (${bytes} bytes, max ${MAX_BYTES}); use fewer layers/keyframes`)
  return lottie
}
