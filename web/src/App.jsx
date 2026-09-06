import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { useOnnx } from './hooks/useOnnx'
import { argmax, normalize } from './model'
import { topFeelings, DEFAULT_COLORS } from './feelings'
import { cycle } from './nav'
import { cldrEmojis } from './cldrEmojis'
import GitHubButton from 'react-github-btn'
import { Card } from './components/Card'
import { FeelingBar } from './components/FeelingBar'
import { ColorBar } from './components/ColorBar'
import { EmojiList } from './components/EmojiList'
import { KeyHints } from './components/KeyHints'
import { useCardImage } from './hooks/useCardImage'
import { Toast } from './components/Toast'
import { useMediaQuery } from './hooks/useMediaQuery'

const MIN_CHARS = 3
const DEBOUNCE_MS = 250
const EMOJI_SOURCE_KEY = 'emojiSource'

function initialEmojiSource() {
  try {
    return localStorage.getItem(EMOJI_SOURCE_KEY) === 'cldr' ? 'cldr' : 'model'
  } catch {
    return 'model'
  }
}

function formatDate(iso) {
  if (!iso) return '—'
  const d = new Date(iso)
  return Number.isNaN(d.getTime())
    ? iso
    : d.toLocaleString(undefined, {
        year: 'numeric',
        month: 'short',
        day: 'numeric',
        hour: 'numeric',
        minute: '2-digit',
        timeZoneName: 'short',
      })
}

export function App() {
  const { meta, config, ready, predict } = useOnnx()
  const mobile = useMediaQuery('(max-width: 56.25em)')
  const emojiSlots = mobile ? 9 : 10
  const feelingCount = mobile ? 4 : 5
  const colorCount = mobile ? 4 : 5
  const [text, setText] = useState('')
  const [scores, setScores] = useState(null)
  const [override, setOverride] = useState({ emoji: null, feeling: null, color: 0 })
  const [emojiSource, setEmojiSource] = useState(initialEmojiSource)
  const useCldrEmojis = emojiSource === 'cldr'
  const [toast, setToast] = useState({ msg: '', n: 0 })
  const showToast = useCallback((msg) => setToast((s) => ({ msg, n: s.n + 1 })), [])
  const seq = useRef(0)
  const inputRef = useRef(null)

  const char2idx = useMemo(
    () => (meta ? new Map([...meta.chars].map((c, i) => [c, i])) : null),
    [meta],
  )

  useEffect(() => {
    try {
      localStorage.setItem(EMOJI_SOURCE_KEY, emojiSource)
    } catch {}
  }, [emojiSource])

  useEffect(() => {
    if (!ready || !char2idx) return
    if (normalize(text, char2idx).length < MIN_CHARS) {
      seq.current++
      setScores(null)
      setOverride({ emoji: null, feeling: null, color: 0 })
      return
    }
    const mine = ++seq.current
    const timer = setTimeout(async () => {
      const logits = await predict(text)
      if (mine !== seq.current) return
      setScores(logits)
      setOverride({ emoji: null, feeling: null, color: 0 })
    }, DEBOUNCE_MS)
    return () => clearTimeout(timer)
  }, [text, ready, char2idx, predict])

  const emojiScores = scores && scores.emoji
  const cldrTop = useMemo(
    () => (useCldrEmojis && scores ? cldrEmojis(text) : null),
    [useCldrEmojis, scores, text],
  )
  const predictedEmoji = useCldrEmojis
    ? (cldrTop?.[0] ?? null)
    : scores
      ? meta.emojis[argmax(scores.emoji)]
      : null
  const predictedFeeling = scores ? meta.styles[argmax(scores.feeling)] : null
  const shownEmoji = override.emoji ?? predictedEmoji
  const shownFeeling = override.feeling ?? predictedFeeling
  const feelingOptions = useMemo(
    () => topFeelings(scores?.feeling, meta?.styles ?? [], shownFeeling, feelingCount),
    [scores, meta, shownFeeling, feelingCount],
  )
  const emojiTop = useMemo(() => {
    if (useCldrEmojis) {
      return cldrTop
        ? cldrTop.slice(0, emojiSlots).map((emoji, i) => ({ emoji, p: 1 - i / emojiSlots }))
        : null
    }
    return emojiScores
      ? emojiScores
          .map((p, i) => ({ emoji: meta.emojis[i], p }))
          .sort((a, b) => b.p - a.p)
          .slice(0, emojiSlots)
      : null
  }, [useCldrEmojis, cldrTop, emojiScores, meta, emojiSlots])
  const emojiList = useMemo(() => emojiTop?.map((x) => x.emoji) ?? [], [emojiTop])

  const palettes = useMemo(() => scores?.palettes ?? [DEFAULT_COLORS], [scores])
  const colors = palettes[override.color] ?? palettes[0]

  const cardData =
    shownEmoji && shownFeeling
      ? { text, emoji: shownEmoji, feeling: shownFeeling, colors }
      : null
  const copyCard = useCardImage(cardData, showToast)

  useEffect(() => {
    const onKey = (e) => {
      if (e.key === 'Escape') {
        setText('')
        inputRef.current?.focus()
        return
      }
      if (e.key === 'Enter') {
        if (e.target instanceof HTMLButtonElement) return
        e.preventDefault()
        copyCard()
        return
      }
      if (e.key !== 'ArrowUp' && e.key !== 'ArrowDown') return
      const dir = e.key === 'ArrowDown' ? 1 : -1
      if (e.ctrlKey) {
        if (!feelingOptions.length) return
        e.preventDefault()
        setOverride((o) => ({
          ...o,
          feeling: cycle(feelingOptions, o.feeling ?? predictedFeeling, dir),
        }))
        return
      }
      if (e.altKey) {
        if (e.shiftKey || e.metaKey || palettes.length <= 1) return
        e.preventDefault()
        setOverride((o) => ({ ...o, color: (o.color + dir + colorCount) % colorCount }))
        return
      }
      if (e.shiftKey || e.metaKey || !emojiList.length) return
      e.preventDefault()
      setOverride((o) => ({ ...o, emoji: cycle(emojiList, o.emoji ?? predictedEmoji, dir) }))
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [
    emojiList,
    feelingOptions,
    predictedEmoji,
    predictedFeeling,
    copyCard,
    palettes,
    colorCount,
  ])

  const maxLen = config?.max_text_len ?? 0
  const tooShort =
    ready && char2idx ? normalize(text, char2idx).length < MIN_CHARS : false
  const displayEmoji = shownEmoji ?? '🙂'
  const displayFeeling = shownFeeling ?? 'Neutral'

  return (
    <main>
      <div className="stage">
        <div className="head">
          <header className="masthead">
            <h1>
              emojify<span className="tld">.ing</span>
            </h1>
            <span className="slug">emojify anything</span>
          </header>
          <input
            className="input"
            type="text"
            autoComplete="off"
            autoFocus
            ref={inputRef}
            maxLength={maxLen || undefined}
            placeholder="type at least 3 characters…"
            value={text}
            onChange={(e) => setText(e.target.value)}
          />
          <div className="input-meta">
            <p className={'warn' + (text.trim() && tooShort ? '' : ' is-hidden')}>
              text is too short — showing a default card
            </p>
            <div className={'counter' + (maxLen && text.length >= maxLen ? ' full' : '')}>
              {text.length}
              <span>/{maxLen}</span>
            </div>
          </div>
        </div>
        <EmojiList
          items={emojiTop}
          active={shownEmoji}
          slots={emojiSlots}
          onPick={(e) => setOverride((o) => ({ ...o, emoji: e }))}
        />
        <Card
          text={text}
          emoji={displayEmoji}
          feeling={displayFeeling}
          colors={colors}
          onCopy={copyCard}
        />
        <KeyHints />
        <div className="feelings-col">
          <ColorBar
            palettes={palettes}
            active={override.color}
            count={colorCount}
            ready={!tooShort && !!scores}
            onPick={(i) => setOverride((o) => ({ ...o, color: i }))}
          />
          <FeelingBar
            feelings={feelingOptions}
            active={shownFeeling}
            count={feelingCount}
            ready={!tooShort && !!shownFeeling}
            onPick={(f) => setOverride((o) => ({ ...o, feeling: f }))}
          />
          <div className="emoji-source" role="group" aria-label="emoji source">
            {['model', 'cldr'].map((src) => (
              <button
                key={src}
                type="button"
                className={emojiSource === src ? 'active' : undefined}
                aria-pressed={emojiSource === src}
                onClick={() => {
                  setEmojiSource(src)
                  setOverride((o) => ({ ...o, emoji: null }))
                }}
              >
                {src === 'model' ? 'model' : 'keywords'}
              </button>
            ))}
          </div>
          <footer className="footer">
            <span>
              model updated <span>{formatDate(meta?.exported_at)}</span>
            </span>
            <span>made with ❤️ by Gilad</span>
            <span className="gh">
              <GitHubButton
                href="https://github.com/emojic-co/model"
                data-icon="octicon-star"
                data-show-count="true"
                aria-label="Star emojic-co/model on GitHub"
              >
                Star
              </GitHubButton>
            </span>
          </footer>
        </div>
      </div>
      <Toast toast={toast} />
    </main>
  )
}
