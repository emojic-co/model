import { useEffect, useState } from 'react'
import { parse } from 'yaml'
import { useFitText } from './hooks/useFitText'
import { contrastRatio, fixContrast, patternTint } from './model'

function watermarkInk(bg) {
  return contrastRatio('#000000', bg) >= contrastRatio('#ffffff', bg) ? '#000000' : '#ffffff'
}

function PreviewCard({ name, lang, entry, sample, globalSettings, patterns }) {
  const colors = fixContrast(sample.colors)
  const displayText = sample.text.trim() ? sample.text : "What's on your mind?"
  const textRef = useFitText(displayText, {
    min: globalSettings.textMinRatio * 100,
    max: globalSettings.textMaxRatio * 100,
    key: name,
  })
  const tint = patternTint(colors.bg1, colors.bg2)
  const maskUrl = `url("data:image/svg+xml,${encodeURIComponent(patterns[entry.pattern.name])}")`

  return (
    <figure className="style-preview-card">
      <div
        className="card"
        dir={lang === 'he' ? 'rtl' : 'ltr'}
        style={{
          background: `linear-gradient(135deg, ${colors.bg1}, ${colors.bg2})`,
          color: colors.text_color,
          fontFamily: `"${entry.font}", sans-serif`,
        }}
      >
        <div
          aria-hidden="true"
          style={{
            position: 'absolute',
            inset: 0,
            backgroundColor: tint,
            WebkitMaskImage: maskUrl,
            maskImage: maskUrl,
            WebkitMaskRepeat: 'repeat',
            maskRepeat: 'repeat',
            WebkitMaskSize: `${entry.pattern.widthRatio * 100}% ${entry.pattern.heightRatio * 100}%`,
            maskSize: `${entry.pattern.widthRatio * 100}% ${entry.pattern.heightRatio * 100}%`,
            opacity: globalSettings.maxPatternOpacity,
          }}
        />
        <span className="card-emoji" style={{ position: 'relative' }}>
          {sample.emoji}
        </span>
        <div className="card-text-box" ref={textRef} style={{ position: 'relative' }}>
          <p
            className="card-text"
            dir="auto"
            style={{
              fontWeight: entry.fontWeight,
              fontStyle: entry.italic ? 'italic' : 'normal',
              textTransform: entry.uppercase ? 'uppercase' : 'none',
              letterSpacing: entry.letterSpacingEm != null ? `${entry.letterSpacingEm}em` : undefined,
              opacity: entry.opacity,
            }}
          >
            {displayText}
          </p>
        </div>
        <span
          className="card-watermark"
          aria-hidden="true"
          style={{ position: 'relative', color: watermarkInk(colors.bg2) }}
        >
          emojify.ing
        </span>
      </div>
      <figcaption>
        {name} <span className="style-preview-lang">{lang}</span>
      </figcaption>
    </figure>
  )
}

const BASE = import.meta.env.BASE_URL

export function StylePreview() {
  const [state, setState] = useState({ status: 'loading' })

  useEffect(() => {
    Promise.all([
      fetch(BASE + 'style.yml').then((r) => r.text()),
      fetch(BASE + 'style-samples.json').then((r) => r.json()),
    ])
      .then(([yamlText, samples]) => {
        setState({ status: 'ready', style: parse(yamlText), samples })
      })
      .catch((err) => setState({ status: 'error', error: err }))
  }, [])

  if (state.status === 'loading') return <p className="style-preview-status">loading style.yml…</p>
  if (state.status === 'error') {
    return <p className="style-preview-status">failed to load style.yml: {String(state.error)}</p>
  }

  const { style, samples } = state
  const names = Object.keys(style.styles)

  return (
    <main className="style-preview">
      <header>
        <h1>Style preview</h1>
        <p>
          Every style rendered live from <code>style.yml</code> (exported{' '}
          {new Date(style.exportedAt).toLocaleString()}) — the same file the Android app syncs and renders from,
          so this page shows exactly what both apps ship.
        </p>
      </header>
      <div className="style-preview-grid">
        {names.flatMap((name) =>
          ['en', 'he'].map((lang) => (
            <PreviewCard
              key={`${name}-${lang}`}
              name={name}
              lang={lang}
              entry={style.styles[name]}
              sample={samples[name][lang]}
              globalSettings={style.global}
              patterns={style.patterns}
            />
          )),
        )}
      </div>
    </main>
  )
}
