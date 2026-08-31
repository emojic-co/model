export function cycle(list, current, dir) {
  if (!list || list.length === 0) return current
  const n = list.length
  const i = list.indexOf(current)
  if (i === -1) return dir > 0 ? list[0] : list[n - 1]
  return list[(i + dir + n) % n]
}
