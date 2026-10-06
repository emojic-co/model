import { useEffect, useRef, useState } from 'react'
import { CARD_CSS_VARS } from '../hooks/useCardImage'
import { CardCanvas } from './CardCanvas'
import { resolveFeeling } from '../feelings'
import { toCssOklab } from '../model'
import { useI18n } from '../i18n'

const FADE_MS = 150

export function Card({ text, emoji, feeling, lang, colors, loading, onJpg, onMp4, exporting, progress = 0, onCancel, ref }) {
  const { t } = useI18n()
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
  const r = shown.feeling ? resolveFeeling(shown.feeling, shown.lang) : null
  const ready = !loading && colors && r
  // The canvas paints the full card; the CSS gradient only shows while it (re)builds.
  const style = ready
    ? { backgroundImage: `linear-gradient(135deg, ${toCssOklab(colors.bg1)}, ${toCssOklab(colors.bg2)})` }
    : undefined
  const cardData = ready
    ? { text: displayText, emoji: shown.emoji, feeling: shown.feeling, lang: shown.lang, colors, dim: placeholder }
    : null

  return (
    <div className="card-col">
      <div
        ref={ref}
        className="card"
        data-feeling={shown.feeling || undefined}
        data-cluster={r?.cluster || undefined}
        data-phase={phase}
        style={{ ...CARD_CSS_VARS, ...style }}
      >
        <CardCanvas cardData={cardData} paused={!!exporting} />
      </div>
      <div className="share-bar">
        {exporting ? (
          <>
            <progress className="share-progress" max="1" value={progress} aria-label={t('card.making', { format: exporting })} />
            <button type="button" aria-label={t('card.cancelAria')} onClick={onCancel}>
              {t('card.cancel')}
            </button>
          </>
        ) : (
          <>
            <button type="button" aria-label={t('card.copyJpg')} onClick={onJpg}>
              jpg
            </button>
            <button type="button" aria-label={t('card.saveMp4')} onClick={onMp4} disabled={!onMp4}>
              mp4
            </button>
          </>
        )}
      </div>
    </div>
  )
}
