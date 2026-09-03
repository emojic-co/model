import { FEELINGS } from '../../../web/src/feelings.js'

export const FONT_HREF =
  'https://fonts.googleapis.com/css2?family=Anton&family=Archivo+Black&family=Barlow+Condensed:wght@700&family=Bitter:ital,wght@0,400;1,400&family=Bungee&family=Caveat:wght@700&family=Chewy&family=Fredoka:wght@600&family=Gochi+Hand&family=Griffy&family=Inter:wght@400&family=Luckiest+Guy&family=Noto+Color+Emoji&family=Oswald:wght@500&family=Playfair+Display:ital,wght@1,600&family=Poppins:wght@500&family=Quicksand:wght@500&family=Rubik:wght@600&family=Schoolbell&family=Shadows+Into+Light&family=Shantell+Sans:wght@500&family=Spectral:ital,wght@0,400;1,400&family=Work+Sans:wght@600&display=swap'

export const FAMILIES = Array.from(
  new Set([
    'Noto Color Emoji',
    ...Object.values(FEELINGS).flatMap((f: { font: string }) =>
      [...f.font.matchAll(/"([^"]+)"/g)].map((m) => m[1]),
    ),
  ]),
)
