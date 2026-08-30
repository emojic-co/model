import { useLayoutEffect, useRef } from 'react'

export function useFitText(text, { min = 32, max = 104, key } = {}) {
  const ref = useRef(null)

  useLayoutEffect(() => {
    const el = ref.current
    if (!el) return
    const fit = () => {
      let lo = min
      let hi = max
      let best = min
      while (lo <= hi) {
        const mid = (lo + hi) >> 1
        el.style.fontSize = mid + 'px'
        if (el.scrollWidth <= el.clientWidth && el.scrollHeight <= el.clientHeight) {
          best = mid
          lo = mid + 1
        } else {
          hi = mid - 1
        }
      }
      el.style.fontSize = best + 'px'
    }
    fit()
    const ro = new ResizeObserver(fit)
    ro.observe(el)
    if (document.fonts && document.fonts.status !== 'loaded') {
      document.fonts.ready.then(() => {
        if (ref.current) fit()
      })
    }
    return () => ro.disconnect()
  }, [text, min, max, key])

  return ref
}
