import { useI18n } from '../i18n'

export function ShareBar({ onJpg, onMp4, exporting, progress = 0, onCancel, className = '' }) {
  const { t } = useI18n()
  return (
    <div className={`share-bar ${className}`.trim()}>
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
  )
}
