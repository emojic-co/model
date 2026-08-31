import { useLayoutEffect, useRef } from 'react'

export function useFitText(text, { min = 5, max = 20, key } = {}) {
  const ref = useRef(null)

  useLayoutEffect(() => {
    const el = ref.current
    if (!el) return
    const fits = () =>
      el.scrollWidth <= el.clientWidth && el.scrollHeight <= el.clientHeight
    const fit = () => {
      let lo = min
      let hi = max
      el.style.fontSize = lo + 'cqw'
      if (!fits()) return
      for (let i = 0; i < 22; i++) {
        const mid = (lo + hi) / 2
        el.style.fontSize = mid + 'cqw'
        if (fits()) lo = mid
        else hi = mid
      }
      el.style.fontSize = lo + 'cqw'
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
