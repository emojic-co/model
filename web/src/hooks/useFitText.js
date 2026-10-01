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
    // Style fonts load lazily on first use, after this fit ran with the fallback font's
    // metrics; re-fit whenever any font finishes loading.
    const onFonts = () => {
      if (ref.current) fit()
    }
    const fonts = document.fonts
    fonts?.addEventListener?.('loadingdone', onFonts)
    fonts?.ready?.then(onFonts)
    return () => {
      ro.disconnect()
      fonts?.removeEventListener?.('loadingdone', onFonts)
    }
  }, [text, min, max, key])

  return ref
}
