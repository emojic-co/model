import { FEELINGS, resolveFeeling } from '../../../web/src/feelings.js'

export const FONT_HREF =
  'https://fonts.googleapis.com/css2?family=Anton&family=Archivo+Black&family=Barlow+Condensed:wght@700&family=Bitter:ital,wght@0,400;1,400&family=Bungee&family=Caveat:wght@700&family=Chewy&family=Fredoka:wght@600&family=Gochi+Hand&family=Griffy&family=Inter:wght@400&family=Luckiest+Guy&family=Noto+Color+Emoji&family=Oswald:wght@500&family=Playfair+Display:ital,wght@1,600&family=Poppins:wght@500&family=Quicksand:wght@500&family=Rubik:wght@600&family=Schoolbell&family=Shadows+Into+Light&family=Shantell+Sans:wght@500&family=Spectral:ital,wght@0,400;1,400&family=Work+Sans:wght@600&display=swap'

const FAMILIES = Array.from(
  new Set([
    'Noto Color Emoji',
    ...Object.keys(FEELINGS).flatMap((name) =>
      [...resolveFeeling(name).font.matchAll(/"([^"]+)"/g)].map((m) => m[1]),
    ),
  ]),
)

let loaded: Promise<void> | undefined

// Injects the Google Fonts stylesheet once and waits for every card family.
export function loadFonts() {
  loaded ??= new Promise<void>((resolve) => {
    const link = document.createElement('link')
    link.rel = 'stylesheet'
    link.href = FONT_HREF
    link.onload = async () => {
      await Promise.all(
        FAMILIES.map((f) => document.fonts.load(`600 32px "${f}"`, 'Ag').catch(() => undefined)),
      )
      await document.fonts.ready
      resolve()
    }
    link.onerror = () => resolve()
    document.head.appendChild(link)
  })
  return loaded
}
