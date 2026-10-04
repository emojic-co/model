import { useEffect, useRef, useState } from 'react'
import { useCardPlayer } from '../hooks/useCardPlayer'

// The card preview: the shared clip painted on a canvas (same painter as the jpg/gif/mp4 exports).
export function CardCanvas({ cardData, paused }) {
  const ref = useRef(null)
  const [scale, setScale] = useState(1)

  useEffect(() => {
    const el = ref.current
    const ro = new ResizeObserver(() => {
      const px = Math.round(el.clientWidth * Math.min(window.devicePixelRatio || 1, 2))
      if (px && el.width !== px) {
        el.width = px
        el.height = px
        setScale(px / 512)
      }
    })
    ro.observe(el)
    return () => ro.disconnect()
  }, [])

  useCardPlayer(cardData, ref, scale, paused)
  return <canvas ref={ref} className="card-canvas" aria-hidden="true" />
}
