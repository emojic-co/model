import { useLayoutEffect, useRef, useState } from 'react'

export function useFitText(text, { min = 32, max = 104 } = {}) {
  const ref = useRef(null)
  const [px, setPx] = useState(max)

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
      setPx(best)
    }
    fit()
    const ro = new ResizeObserver(fit)
    ro.observe(el)
    return () => ro.disconnect()
  }, [text, min, max])

  return [px, ref]
}
