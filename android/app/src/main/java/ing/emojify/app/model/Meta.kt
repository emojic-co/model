package ing.emojify.app.model

import kotlinx.serialization.Serializable

@Serializable
data class ModelMetaInfo(
    val sha: String? = null,
    val dirty: Boolean? = null,
    val generated: String? = null,
    val config: List<String> = emptyList(),
    val train_sha: String? = null,
    val stage: String? = null,
)

@Serializable
data class Meta(
    val chars: String,
    val pad_idx: Int,
    val max_text_len: Int,
    val emojis: List<String>,
    val styles: List<String>,
    val exported_at: String? = null,
    val model_meta: ModelMetaInfo? = null,
)

@Serializable
data class Config(
    val max_text_len: Int,
)

fun Meta.charToIndex(): Map<Char, Int> =
    chars.withIndex().associate { (i, c) -> c to i }
