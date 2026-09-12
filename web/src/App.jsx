import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { useOnnx } from './hooks/useOnnx'
import { argmax, normalize, fixContrast, sigmoid } from './model'
import { topFeelings, DEFAULT_COLORS } from './feelings'
import { cycle } from './nav'
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
const CONTRAST_FIX_KEY = 'contrastFix'

export function pickEmojiList(scores, meta, slots) {
  if (!scores || !meta) return []
  const arr = scores.emoji
  if (!arr) return []
  return [...arr.keys()]
    .sort((a, b) => arr[b] - arr[a])
    .slice(0, slots)
    .map((idx) => ({ emoji: meta.emojis[idx], p: sigmoid([arr[idx]])[0] }))
}

function initialContrastFix() {
  try {
    return localStorage.getItem(CONTRAST_FIX_KEY) !== 'off'
  } catch {
    return true
  }
}

function formatMs(ms) {
  if (ms == null) return '—'
  return ms < 10 ? `${ms.toFixed(1)} ms` : `${Math.round(ms)} ms`
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
  const [contrastFix, setContrastFix] = useState(initialContrastFix)
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
      localStorage.setItem(CONTRAST_FIX_KEY, contrastFix ? 'on' : 'off')
    } catch {}
  }, [contrastFix])

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

  const emojiTop = useMemo(
    () => pickEmojiList(scores, meta, emojiSlots),
    [scores, meta, emojiSlots],
  )
  const predictedEmoji = emojiTop[0]?.emoji ?? null
  const predictedFeeling = scores ? meta.styles[argmax(scores.feeling)] : null
  const shownEmoji = override.emoji ?? predictedEmoji
  const shownFeeling = override.feeling ?? predictedFeeling
  const feelingOptions = useMemo(
    () => topFeelings(scores?.feeling, meta?.styles ?? [], shownFeeling, feelingCount),
    [scores, meta, shownFeeling, feelingCount],
  )
  const emojiList = useMemo(() => emojiTop.map((x) => x.emoji), [emojiTop])

  const rawPalettes = useMemo(() => scores?.palettes ?? [DEFAULT_COLORS], [scores])
  const palettes = useMemo(
    () => (contrastFix ? rawPalettes.map((p) => fixContrast(p)) : rawPalettes),
    [rawPalettes, contrastFix],
  )
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
          <div className="head-top">
            <header className="masthead">
              <h1>
                emojify<span className="tld">.ing</span>
              </h1>
            </header>
          </div>
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
            <span className={'timing' + (scores ? '' : ' is-hidden')}>
              model ran in {formatMs(scores?.ms)}
            </span>
            <div className={'counter' + (maxLen && text.length >= maxLen ? ' full' : '')}>
              {text.length}
              <span>/{maxLen}</span>
            </div>
          </div>
        </div>
        <EmojiList
          items={scores ? emojiTop : null}
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
          <footer className="footer">
            <span>
              model updated <span>{formatDate(meta?.exported_at)}</span>
            </span>
            <span className="contrast-toggle">
              <label>
                <input
                  type="checkbox"
                  checked={contrastFix}
                  onChange={(e) => setContrastFix(e.target.checked)}
                />
                fix low-contrast palettes
              </label>
            </span>
            <span>made with ❤️ by Gilad</span>
            <span>
              <a
                href="https://github.com/emojic-co/model/blob/main/ABOUT.md"
                target="_blank"
                rel="noopener noreferrer"
              >
                about this model
              </a>
            </span>
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
