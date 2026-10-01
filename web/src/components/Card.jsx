import { useEffect, useRef, useState } from 'react'
import { CARD_CSS_VARS, RATIOS } from '../hooks/useCardImage'
import { AnimatedEmoji } from './AnimatedEmoji'
import { CardShimmer } from './CardShimmer'
import { CharText } from './CharText'
import { useFitText } from '../hooks/useFitText'
import { resolveFeeling } from '../feelings'
import { contrastRatio, patternTint, toCssOklab, toHexColor, BLACK, WHITE } from '../model'
import { patternLayers, patternSizeCss } from '../patterns'

function watermarkInk(bg) {
  return contrastRatio(BLACK, bg) >= contrastRatio(WHITE, bg) ? '#000000' : '#ffffff'
}

const FADE_MS = 150

export function Card({ text, emoji, feeling, lang, colors, loading, onJpg, onGif, onMp4, ref }) {
  const [shown, setShown] = useState({ emoji, feeling, lang })
  const [phase, setPhase] = useState('in')
  const prev = useRef({ emoji, feeling, lang })
  const wasLoading = useRef(loading)

  useEffect(() => {
    if (loading) {
      wasLoading.current = true
      prev.current = { emoji, feeling, lang }
      setPhase('out')
      return
    }
    if (wasLoading.current) {
      wasLoading.current = false
      prev.current = { emoji, feeling, lang }
      setShown({ emoji, feeling, lang })
      setPhase('in')
      return
    }
    if (prev.current.emoji === emoji && prev.current.feeling === feeling && prev.current.lang === lang) return
    prev.current = { emoji, feeling, lang }
    setPhase('out')
    const t = setTimeout(() => {
      setShown({ emoji, feeling, lang })
      setPhase('in')
    }, FADE_MS)
    return () => clearTimeout(t)
  }, [emoji, feeling, lang, loading])

  const placeholder = !text.trim()
  const displayText = placeholder ? "What's on your mind?" : text
  const textRef = useFitText(displayText, { min: RATIOS.textMinRatio * 100, max: RATIOS.textMaxRatio * 100, key: shown.feeling })
  const r = shown.feeling ? resolveFeeling(shown.feeling, shown.lang) : null
  const style =
    !loading && colors && r
      ? (() => {
          const layers = patternLayers(shown.feeling, toHexColor(patternTint(colors.bg1, colors.bg2)))
          return {
            backgroundImage: [
              ...layers.map((l) => l.image),
              `linear-gradient(135deg, ${toCssOklab(colors.bg1)}, ${toCssOklab(colors.bg2)})`,
            ].join(', '),
            backgroundSize: [...patternSizeCss(layers), 'auto'].join(', '),
            color: toCssOklab(colors.text_color),
            fontFamily: r.font,
            ...r.vars,
          }
        })()
      : undefined

  return (
    <div
      ref={ref}
      className="card"
      data-feeling={shown.feeling || undefined}
      data-cluster={r?.cluster || undefined}
      data-emoji={r?.emoji || undefined}
      data-phase={phase}
      style={{ ...CARD_CSS_VARS, ...style }}
    >
      <CardShimmer
        cluster={r?.cluster}
        feeling={shown.feeling}
        motif={r?.entrance}
        text={displayText}
        replayKey={shown}
      />
      <AnimatedEmoji emoji={shown.emoji} />
      <div className="card-text-box" ref={textRef}>
        <CharText
          className={'card-text' + (placeholder ? ' card-text-placeholder' : '')}
          style={r?.style}
          dir="auto"
          text={displayText}
          motif={r?.entrance}
          feeling={shown.feeling}
          replayKey={shown}
        />
      </div>
      <div className="share-bar">
        <button type="button" aria-label="Copy card as jpg" onClick={onJpg}>
          jpg
        </button>
        <button type="button" aria-label="Save card as gif" onClick={onGif} disabled={!onGif}>
          gif
        </button>
        <button type="button" aria-label="Save card as mp4" onClick={onMp4} disabled={!onMp4}>
          mp4
        </button>
      </div>
      <span
        className="card-watermark"
        aria-hidden="true"
        style={colors ? { color: watermarkInk(colors.bg2) } : undefined}
      >
        emojify.ing
      </span>
    </div>
  )
}
