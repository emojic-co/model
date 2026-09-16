package ing.emojify.model

import kotlin.math.abs
import kotlin.math.cbrt
import kotlin.math.max
import kotlin.math.min
import kotlin.math.pow

fun normalize(text: String, charSet: Set<Char>): String {
    val collapsedWhitespace = text.lowercase().replace(Regex("\\s+"), " ").trim()
    val cappedRepeats = Regex("(.)\\1{2,}").replace(collapsedWhitespace) { m -> m.groupValues[1].repeat(2) }
    val sb = StringBuilder()
    for (c in cappedRepeats) if (c in charSet) sb.append(c)
    return sb.toString()
}

fun encode(text: String, meta: Meta, char2idx: Map<Char, Int>): LongArray {
    val norm = normalize(text, char2idx.keys).take(meta.max_text_len)
    val ids = LongArray(meta.max_text_len) { meta.pad_idx.toLong() }
    for (i in norm.indices) ids[i] = (char2idx[norm[i]] ?: meta.pad_idx).toLong()
    return ids
}

data class Palette(val bg1: String, val bg2: String, val textColor: String)

private fun clampByte(v: Double): Int = max(0.0, min(255.0, Math.round(v).toDouble())).toInt()

private fun toHex(r: Double, g: Double, b: Double): String {
    fun h(v: Double) = clampByte(v).toString(16).padStart(2, '0')
    return "#${h(r)}${h(g)}${h(b)}"
}

fun decodeColors(c: FloatArray): Palette = Palette(
    bg1 = toHex(c[0].toDouble(), c[1].toDouble(), c[2].toDouble()),
    bg2 = toHex(c[3].toDouble(), c[4].toDouble(), c[5].toDouble()),
    textColor = toHex(c[6].toDouble(), c[7].toDouble(), c[8].toDouble()),
)

fun decodeColorList(flat: FloatArray): List<Palette> {
    val out = mutableListOf<Palette>()
    var i = 0
    while (i + 9 <= flat.size) {
        out.add(decodeColors(flat.copyOfRange(i, i + 9)))
        i += 9
    }
    return out
}

private fun hexToRgb(hex: String): Triple<Double, Double, Double> {
    val n = hex.removePrefix("#").toLong(16)
    return Triple(((n shr 16) and 0xFF).toDouble(), ((n shr 8) and 0xFF).toDouble(), (n and 0xFF).toDouble())
}

private fun rgbToHex(rgb: Triple<Double, Double, Double>): String = toHex(rgb.first, rgb.second, rgb.third)

private fun srgbToLinear(c: Double): Double = if (c <= 0.04045) c / 12.92 else ((c + 0.055) / 1.055).pow(2.4)
private fun linearToSrgb(c: Double): Double {
    val v = if (c <= 0.0031308) 12.92 * c else 1.055 * max(c, 0.0).pow(1.0 / 2.4) - 0.055
    return v * 255
}

fun srgbToOklab(rgb: Triple<Double, Double, Double>): Triple<Double, Double, Double> {
    val r = srgbToLinear(rgb.first / 255)
    val g = srgbToLinear(rgb.second / 255)
    val b = srgbToLinear(rgb.third / 255)
    val l = cbrt(0.4122214708 * r + 0.5363325363 * g + 0.0514459929 * b)
    val m = cbrt(0.2119034982 * r + 0.6806995451 * g + 0.1073969566 * b)
    val s = cbrt(0.0883024619 * r + 0.2817188376 * g + 0.6299787005 * b)
    return Triple(
        0.2104542553 * l + 0.793617785 * m - 0.0040720468 * s,
        1.9779984951 * l - 2.428592205 * m + 0.4505937099 * s,
        0.0259040371 * l + 0.7827717662 * m - 0.808675766 * s,
    )
}

fun oklabToSrgb(lab: Triple<Double, Double, Double>): Triple<Double, Double, Double> {
    val (L, a, b) = lab
    val l = (L + 0.3963377774 * a + 0.2158037573 * b).pow(3)
    val m = (L - 0.1055613458 * a - 0.0638541728 * b).pow(3)
    val s = (L - 0.0894841775 * a - 1.291485548 * b).pow(3)
    return Triple(
        linearToSrgb(4.0767416621 * l - 3.3077115913 * m + 0.2309699292 * s),
        linearToSrgb(-1.2684380046 * l + 2.6097574011 * m - 0.3413193965 * s),
        linearToSrgb(-0.0041960863 * l - 0.7034186147 * m + 1.707614701 * s),
    )
}

private fun relLuminance(rgb: Triple<Double, Double, Double>): Double =
    0.2126 * srgbToLinear(rgb.first / 255) + 0.7152 * srgbToLinear(rgb.second / 255) + 0.0722 * srgbToLinear(rgb.third / 255)

fun contrastRatio(hexA: String, hexB: String): Double {
    val la = relLuminance(hexToRgb(hexA))
    val lb = relLuminance(hexToRgb(hexB))
    return (max(la, lb) + 0.05) / (min(la, lb) + 0.05)
}

const val CONTRAST_MIN = 3.0

private fun minMargin(fg: String, bg1: String, bg2: String): Double =
    min(contrastRatio(fg, bg1), contrastRatio(fg, bg2))

fun fixContrast(palette: Palette, minContrast: Double = CONTRAST_MIN): Palette {
    val ok = { fg: String -> minMargin(fg, palette.bg1, palette.bg2) >= minContrast }
    if (ok(palette.textColor)) return palette

    val (l0, a, b) = srgbToOklab(hexToRgb(palette.textColor))
    val step = 0.02
    var best: String? = null
    var bestCost = Double.POSITIVE_INFINITY
    for (dir in listOf(-1, 1)) {
        var l = l0 + dir * step
        while (l in 0.0..1.0) {
            val cand = rgbToHex(oklabToSrgb(Triple(l, a, b)))
            if (ok(cand)) {
                if (abs(l - l0) < bestCost) {
                    best = cand
                    bestCost = abs(l - l0)
                }
                break
            }
            l += dir * step
        }
    }
    val resolved = best ?: if (minMargin("#000000", palette.bg1, palette.bg2) >= minMargin("#ffffff", palette.bg1, palette.bg2)) "#000000" else "#ffffff"
    return palette.copy(textColor = resolved)
}

fun mixColors(hexA: String, hexB: String, t: Double = 0.5): String {
    val (l1, a1, b1) = srgbToOklab(hexToRgb(hexA))
    val (l2, a2, b2) = srgbToOklab(hexToRgb(hexB))
    return rgbToHex(oklabToSrgb(Triple(l1 + (l2 - l1) * t, a1 + (a2 - a1) * t, b1 + (b2 - b1) * t)))
}

fun patternTint(bg1: String, bg2: String): String {
    val (l, a, b) = srgbToOklab(hexToRgb(mixColors(bg1, bg2)))
    return rgbToHex(oklabToSrgb(Triple(max(0.94, l), a, b)))
}

fun argmax(arr: FloatArray): Int {
    var best = 0
    for (i in 1 until arr.size) if (arr[i] > arr[best]) best = i
    return best
}

fun softmax(arr: FloatArray): FloatArray {
    val m = arr.max()
    val exps = arr.map { Math.exp((it - m).toDouble()) }
    val sum = exps.sum()
    return exps.map { (it / sum).toFloat() }.toFloatArray()
}

fun sigmoid(arr: FloatArray): FloatArray = arr.map { (1.0 / (1.0 + Math.exp(-it.toDouble()))).toFloat() }.toFloatArray()
