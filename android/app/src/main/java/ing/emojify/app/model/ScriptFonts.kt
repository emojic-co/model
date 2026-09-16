package ing.emojify.app.model

const val LATIN = "latin"

private val LANG_SCRIPT = mapOf(
    "ar" to "arabic",
    "bg" to "cyrillic",
    "el" to "greek",
    "he" to "hebrew",
    "hi" to "devanagari",
    "mr" to "devanagari",
    "ja" to "japanese",
    "ko" to "korean",
    "ru" to "cyrillic",
    "th" to "thai",
    "uk" to "cyrillic",
    "zh" to "chinese",
)

fun scriptForLang(lang: String?): String = LANG_SCRIPT[lang] ?: LATIN

private val HEBREW_RE = Regex("[\\u0590-\\u05FF]")

fun langForText(text: String): String = if (HEBREW_RE.containsMatchIn(text)) "he" else "en"

private val SCRIPT_FONTS: Map<String, Map<String, String>> = mapOf(
    "cyrillic" to mapOf(
        "anger" to "Russo One", "joy" to "M PLUS Rounded 1c", "play" to "Pacifico",
        "calm" to "Comfortaa", "tender" to "Marck Script", "drive" to "Unbounded",
        "sad" to "Cormorant Garamond", "anxiety" to "Amatic SC", "reflective" to "PT Serif",
    ),
    "greek" to mapOf(
        "anger" to "Tektur", "joy" to "Arima", "play" to "Mansalva",
        "calm" to "Comfortaa", "tender" to "Mynerve", "drive" to "Syne",
        "sad" to "Cardo", "anxiety" to "Fira Sans Extra Condensed", "reflective" to "Noto Sans",
    ),
    "arabic" to mapOf(
        "anger" to "Blaka", "joy" to "Marhey", "play" to "Lemonada",
        "calm" to "Mada", "tender" to "Playpen Sans Arabic", "drive" to "Alexandria",
        "sad" to "Amiri", "anxiety" to "Jomhuria", "reflective" to "Noto Sans Arabic",
    ),
    "hebrew" to mapOf(
        "anger" to "Suez One", "joy" to "Varela Round", "play" to "Gveret Levin",
        "calm" to "M PLUS Rounded 1c", "tender" to "Secular One", "drive" to "Miriam Libre",
        "sad" to "Frank Ruhl Libre", "anxiety" to "Amatic SC", "reflective" to "Heebo",
    ),
    "devanagari" to mapOf(
        "anger" to "Bakbak One", "joy" to "Modak", "play" to "Ranga",
        "calm" to "Palanquin", "tender" to "Kalam", "drive" to "Rajdhani",
        "sad" to "Tiro Devanagari Hindi", "anxiety" to "Khand", "reflective" to "Noto Sans Devanagari",
    ),
    "thai" to mapOf(
        "anger" to "Srisakdi", "joy" to "Chonburi", "play" to "Itim",
        "calm" to "Mitr", "tender" to "Charmonman", "drive" to "Chakra Petch",
        "sad" to "Taviraj", "anxiety" to "Sriracha", "reflective" to "Sarabun",
    ),
    "japanese" to mapOf(
        "anger" to "Dela Gothic One", "joy" to "Kosugi Maru", "play" to "Hachi Maru Pop",
        "calm" to "Zen Maru Gothic", "tender" to "Klee One", "drive" to "RocknRoll One",
        "sad" to "Shippori Mincho", "anxiety" to "DotGothic16", "reflective" to "Noto Sans JP",
    ),
    "korean" to mapOf(
        "anger" to "Black Han Sans", "joy" to "Jua", "play" to "Gaegu",
        "calm" to "Gowun Dodum", "tender" to "Nanum Pen Script", "drive" to "Do Hyeon",
        "sad" to "Nanum Myeongjo", "anxiety" to "Single Day", "reflective" to "Gothic A1",
    ),
    "chinese" to mapOf(
        "anger" to "WDXL Lubrifont SC", "joy" to "ZCOOL KuaiLe", "play" to "Liu Jian Mao Cao",
        "calm" to "ZCOOL XiaoWei", "tender" to "Zhi Mang Xing", "drive" to "ZCOOL QingKe HuangYou",
        "sad" to "Noto Serif SC", "anxiety" to "Long Cang", "reflective" to "Noto Sans SC",
    ),
)

private const val FALLBACK_FONT = "Noto Sans"

fun fontForScript(script: String, cluster: String): String = SCRIPT_FONTS[script]?.get(cluster) ?: FALLBACK_FONT
