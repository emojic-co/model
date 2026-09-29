import { useEffect, useRef, useState } from 'react'
import { loadNotoIndex, notoUrl, prefersReducedMotion } from '../notoLottie'

// Static glyph always renders (keeps layout and is the fallback); the Lottie clone is
// overlaid once loaded. Preview only: the shared/copied card image stays static.
export function AnimatedEmoji({ emoji }) {
  const box = useRef(null)
  const [live, setLive] = useState(false)

  useEffect(() => {
    if (prefersReducedMotion()) return
    let anim
    let cancelled = false
    ;(async () => {
      const stem = (await loadNotoIndex())[emoji]
      if (!stem || cancelled) return
      const [{ default: lottie }, data] = await Promise.all([
        import('lottie-web/build/player/lottie_light'),
        fetch(notoUrl(stem)).then((r) => r.json()),
      ])
      if (cancelled || !box.current) return
      anim = lottie.loadAnimation({
        container: box.current,
        renderer: 'svg',
        loop: true,
        autoplay: true,
        animationData: data,
      })
      setLive(true)
    })().catch(() => {})
    return () => {
      cancelled = true
      anim?.destroy()
      setLive(false)
    }
  }, [emoji])

  return (
    <span className="card-emoji" data-animated={live || undefined}>
      <span className="card-emoji-glyph">{emoji}</span>
      <span className="card-emoji-lottie" ref={box} aria-hidden="true" />
    </span>
  )
}
