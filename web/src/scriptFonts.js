export const LATIN = 'latin'

export const LANG_SCRIPT = {
  ar: 'arabic',
  bg: 'cyrillic',
  el: 'greek',
  he: 'hebrew',
  hi: 'devanagari',
  mr: 'devanagari',
  ja: 'japanese',
  ko: 'korean',
  ru: 'cyrillic',
  th: 'thai',
  uk: 'cyrillic',
  zh: 'chinese',
  'zh-Hant': 'chinese',
}

export function scriptForLang(lang) {
  return LANG_SCRIPT[lang] ?? LATIN
}

const SCRIPT_FONTS = {
  cyrillic: {
    anger: '"Russo One", sans-serif',
    joy: '"M PLUS Rounded 1c", sans-serif',
    play: '"Pacifico", cursive',
    calm: '"Comfortaa", sans-serif',
    tender: '"Marck Script", cursive',
    drive: '"Unbounded", sans-serif',
    sad: '"Cormorant Garamond", serif',
    anxiety: '"Amatic SC", cursive',
    reflective: '"PT Serif", serif',
  },
  greek: {
    anger: '"Tektur", sans-serif',
    joy: '"Arima", sans-serif',
    play: '"Mansalva", cursive',
    calm: '"Comfortaa", sans-serif',
    tender: '"Mynerve", cursive',
    drive: '"Syne", sans-serif',
    sad: '"Cardo", serif',
    anxiety: '"Fira Sans Extra Condensed", sans-serif',
    reflective: '"Noto Sans", sans-serif',
  },
  arabic: {
    anger: '"Blaka", cursive',
    joy: '"Marhey", sans-serif',
    play: '"Lemonada", sans-serif',
    calm: '"Mada", sans-serif',
    tender: '"Playpen Sans Arabic", cursive',
    drive: '"Alexandria", sans-serif',
    sad: '"Amiri", serif',
    anxiety: '"Jomhuria", cursive',
    reflective: '"Noto Sans Arabic", sans-serif',
  },
  hebrew: {
    anger: '"Suez One", serif',
    joy: '"Varela Round", sans-serif',
    play: '"Gveret Levin", cursive',
    calm: '"M PLUS Rounded 1c", sans-serif',
    tender: '"Secular One", sans-serif',
    drive: '"Miriam Libre", sans-serif',
    sad: '"Frank Ruhl Libre", serif',
    anxiety: '"Amatic SC", cursive',
    reflective: '"Heebo", sans-serif',
  },
  devanagari: {
    anger: '"Bakbak One", sans-serif',
    joy: '"Modak", cursive',
    play: '"Ranga", cursive',
    calm: '"Palanquin", sans-serif',
    tender: '"Kalam", cursive',
    drive: '"Rajdhani", sans-serif',
    sad: '"Tiro Devanagari Hindi", serif',
    anxiety: '"Khand", sans-serif',
    reflective: '"Noto Sans Devanagari", sans-serif',
  },
  thai: {
    anger: '"Srisakdi", sans-serif',
    joy: '"Chonburi", cursive',
    play: '"Itim", cursive',
    calm: '"Mitr", sans-serif',
    tender: '"Charmonman", cursive',
    drive: '"Chakra Petch", sans-serif',
    sad: '"Taviraj", serif',
    anxiety: '"Sriracha", cursive',
    reflective: '"Sarabun", sans-serif',
  },
  japanese: {
    anger: '"Dela Gothic One", sans-serif',
    joy: '"Kosugi Maru", sans-serif',
    play: '"Hachi Maru Pop", cursive',
    calm: '"Zen Maru Gothic", sans-serif',
    tender: '"Klee One", cursive',
    drive: '"RocknRoll One", sans-serif',
    sad: '"Shippori Mincho", serif',
    anxiety: '"DotGothic16", sans-serif',
    reflective: '"Noto Sans JP", sans-serif',
  },
  korean: {
    anger: '"Black Han Sans", sans-serif',
    joy: '"Jua", sans-serif',
    play: '"Gaegu", cursive',
    calm: '"Gowun Dodum", sans-serif',
    tender: '"Nanum Pen Script", cursive',
    drive: '"Do Hyeon", sans-serif',
    sad: '"Nanum Myeongjo", serif',
    anxiety: '"Single Day", cursive',
    reflective: '"Gothic A1", sans-serif',
  },
  chinese: {
    anger: '"WDXL Lubrifont SC", sans-serif',
    joy: '"ZCOOL KuaiLe", sans-serif',
    play: '"Liu Jian Mao Cao", cursive',
    calm: '"ZCOOL XiaoWei", sans-serif',
    tender: '"Zhi Mang Xing", cursive',
    drive: '"ZCOOL QingKe HuangYou", sans-serif',
    sad: '"Noto Serif SC", serif',
    anxiety: '"Long Cang", cursive',
    reflective: '"Noto Sans SC", sans-serif',
  },
}

const FALLBACK_FONT = '"Noto Sans", ui-sans-serif, system-ui, sans-serif'

export function fontForScript(script, cluster) {
  return SCRIPT_FONTS[script]?.[cluster] ?? FALLBACK_FONT
}

export const SCRIPT_FONT_QUERY = {
  cyrillic:
    'family=Russo+One&family=M+PLUS+Rounded+1c:wght@700&family=Pacifico&family=Comfortaa:wght@500&family=Marck+Script&family=Unbounded:wght@700&family=Cormorant+Garamond:ital,wght@1,500&family=Amatic+SC:wght@400&family=PT+Serif:ital,wght@0,400',
  greek:
    'family=Tektur:wght@700&family=Arima:wght@600&family=Mansalva&family=Comfortaa:wght@500&family=Mynerve&family=Syne:wght@700&family=Cardo:ital,wght@1,400&family=Fira+Sans+Extra+Condensed:wght@300&family=Noto+Sans:ital,wght@0,400',
  arabic:
    'family=Blaka&family=Marhey:wght@500&family=Lemonada:wght@600&family=Mada:wght@400&family=Playpen+Sans+Arabic:wght@400&family=Alexandria:wght@700&family=Amiri:ital,wght@1,400&family=Jomhuria&family=Noto+Sans+Arabic:wght@400',
  hebrew:
    'family=Suez+One&family=Varela+Round&family=Gveret+Levin&family=M+PLUS+Rounded+1c:wght@400&family=Secular+One&family=Miriam+Libre:wght@700&family=Frank+Ruhl+Libre:wght@300&family=Amatic+SC:wght@400&family=Heebo:wght@400',
  devanagari:
    'family=Bakbak+One&family=Modak&family=Ranga:wght@700&family=Palanquin:wght@400&family=Kalam:wght@400&family=Rajdhani:wght@600&family=Tiro+Devanagari+Hindi:ital,wght@1,400&family=Khand:wght@500&family=Noto+Sans+Devanagari:wght@400',
  thai: 'family=Srisakdi:wght@700&family=Chonburi&family=Itim&family=Mitr:wght@400&family=Charmonman:wght@400&family=Chakra+Petch:ital,wght@0,600&family=Taviraj:ital,wght@1,400&family=Sriracha&family=Sarabun:ital,wght@0,400',
  japanese:
    'family=Dela+Gothic+One&family=Kosugi+Maru&family=Hachi+Maru+Pop&family=Zen+Maru+Gothic:wght@400&family=Klee+One:wght@400&family=RocknRoll+One&family=Shippori+Mincho:wght@500&family=DotGothic16&family=Noto+Sans+JP:wght@400',
  korean:
    'family=Black+Han+Sans&family=Jua&family=Gaegu:wght@400&family=Gowun+Dodum&family=Nanum+Pen+Script&family=Do+Hyeon&family=Nanum+Myeongjo:wght@400&family=Single+Day&family=Gothic+A1:wght@400',
  chinese:
    'family=WDXL+Lubrifont+SC&family=ZCOOL+KuaiLe&family=Liu+Jian+Mao+Cao&family=ZCOOL+XiaoWei&family=Zhi+Mang+Xing&family=ZCOOL+QingKe+HuangYou&family=Noto+Serif+SC:wght@300&family=Long+Cang&family=Noto+Sans+SC:wght@400',
}

const loadedScripts = new Set()

export function ensureScriptFontsLoaded(script) {
  if (script === LATIN || loadedScripts.has(script) || typeof document === 'undefined') return
  const query = SCRIPT_FONT_QUERY[script]
  if (!query) return
  loadedScripts.add(script)
  const link = document.createElement('link')
  link.rel = 'stylesheet'
  link.href = `https://fonts.googleapis.com/css2?${query}&display=swap`
  document.head.appendChild(link)
}
