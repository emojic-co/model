export function cycle(list, current, dir) {
  if (!list || list.length === 0) return current
  const n = list.length
  const i = list.indexOf(current)
  if (i === -1) return dir > 0 ? list[0] : list[n - 1]
  return list[(i + dir + n) % n]
}

export function textToPath(text) {
  const t = text.trim()
  return t ? `/${encodeURIComponent(t)}` : '/'
}

export function pathToText(pathname) {
  try {
    return decodeURIComponent(pathname.replace(/^\//, ''))
  } catch {
    return ''
  }
}

export function restoreRedirectedPath() {
  const params = new URLSearchParams(window.location.search)
  const p = params.get('p')
  if (p === null) return
  window.history.replaceState(null, '', '/' + p + window.location.hash)
}
