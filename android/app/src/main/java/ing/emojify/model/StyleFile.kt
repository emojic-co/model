package ing.emojify.model

import com.charleskorn.kaml.Yaml
import com.charleskorn.kaml.YamlConfiguration
import kotlinx.serialization.Serializable

@Serializable
data class GlobalSettings(
    val referencePx: Int,
    val maxPatternOpacity: Float,
    val padRatio: Float,
    val padYRatio: Float,
    val emojiRatio: Float,
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
data class FontSpec(
    val family: String,
    val generic: String,
    val weight: Int? = null,
    val italic: Boolean = false,
)

@Serializable
data class FontsFile(
    val fallback: FontSpec,
    val latin: Map<String, FontSpec>,
    val scripts: Map<String, Map<String, FontSpec>>,
)

@Serializable
data class StyleFile(
    val exportedAt: String,
    val global: GlobalSettings,
    val textAnimations: TextAnimations? = null,
    val shimmer: ShimmerSpec? = null,
    val clip: ClipSpec? = null,
    val fonts: FontsFile? = null,
    val i18n: Map<String, Map<String, String>>? = null,
    val patterns: Map<String, String>,
    val styles: Map<String, StyleEntry>,
)

// Lenient so a style.yml with sections this build doesn't know yet (e.g. `shimmer`) still loads.
private val styleYaml = Yaml(configuration = YamlConfiguration(strictMode = false))

fun parseStyleFile(yaml: String): StyleFile = styleYaml.decodeFromString(StyleFile.serializer(), yaml)
