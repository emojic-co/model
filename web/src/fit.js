export function wrapLines(measure, text, maxWidth, maxLines) {
  const words = text.split(/\s+/).filter(Boolean)
  const lines = []
  let line = ''
  for (const w of words) {
    const next = line ? line + ' ' + w : w
    if (line && measure(next) > maxWidth) {
      lines.push(line)
      line = w
    } else {
      line = next
    }
  }
  if (line) lines.push(line)
  return lines.slice(0, maxLines)
}

export function fitCanvasFont({ text, maxWidth, maxHeight, min, max, lineHeight, widthAt }) {
  let lo = min
  let hi = max
  let best = min
  while (lo <= hi) {
    const px = (lo + hi) >> 1
    const measure = (s) => widthAt(s, px)
    const lines = wrapLines(measure, text, maxWidth, 999)
    const widest = lines.reduce((w, l) => Math.max(w, measure(l)), 0)
    const tall = lines.length * px * lineHeight
    if (widest <= maxWidth && tall <= maxHeight) {
      best = px
      lo = px + 1
    } else {
      hi = px - 1
    }
  }
  return best
}
