package ing.emojify.model

import com.charleskorn.kaml.Yaml
import kotlinx.serialization.Serializable

@Serializable
data class GlobalSettings(
    val referencePx: Int,
    val maxPatternOpacity: Float,
    val padRatio: Float,
    val gapRatio: Float,
    val emojiRatio: Float,
    val emojiDyRatio: Float,
    val textBoxPadXRatio: Float,
    val textBoxPadYRatio: Float,
    val textLineHeight: Float,
    val textMinRatio: Float,
    val textMaxRatio: Float,
    val maxLines: Int,
    val watermarkPxRatio: Float,
    val watermarkOpacity: Float,
    val watermarkMarginRatio: Float,
)

@Serializable
data class StylePattern(
    val name: String,
    val widthRatio: Float,
    val heightRatio: Float,
)

@Serializable
data class StyleEntry(
    val cluster: String,
    val font: String,
    val fontWeight: Int,
    val italic: Boolean,
    val uppercase: Boolean,
    val letterSpacingEm: Float? = null,
    val opacity: Float = 1f,
    val entrance: String,
    val emoji: String,
    val entranceMs: Int,
    val emojiMs: Int,
    val pattern: StylePattern,
)

@Serializable
data class StyleFile(
    val exportedAt: String,
    val global: GlobalSettings,
    val patterns: Map<String, String>,
    val styles: Map<String, StyleEntry>,
)

fun parseStyleFile(yaml: String): StyleFile = Yaml.default.decodeFromString(StyleFile.serializer(), yaml)
