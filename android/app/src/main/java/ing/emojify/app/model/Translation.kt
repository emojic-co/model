package ing.emojify.app.model

import com.google.mlkit.nl.languageid.LanguageIdentification
import com.google.mlkit.nl.translate.TranslateLanguage
import com.google.mlkit.nl.translate.Translation
import com.google.mlkit.nl.translate.Translator
import com.google.mlkit.nl.translate.TranslatorOptions
import kotlinx.coroutines.tasks.await

data class TranslationResult(
    val text: String,
    val lang: String,
    val detectedLang: String?,
    val translated: Boolean,
)

private val languageIdentifier by lazy { LanguageIdentification.getClient() }
private val translatorCache = mutableMapOf<String, Translator>()

private fun getTranslator(sourceCode: String): Translator = translatorCache.getOrPut(sourceCode) {
    val options = TranslatorOptions.Builder()
        .setSourceLanguage(sourceCode)
        .setTargetLanguage(TranslateLanguage.ENGLISH)
        .build()
    Translation.getClient(options)
}

private suspend fun detectLanguage(text: String): String? = try {
    val code = languageIdentifier.identifyLanguage(text).await()
    if (code == "und") null else code
} catch (e: Exception) {
    null
}

suspend fun detectAndTranslate(text: String): TranslationResult {
    val detectedLang = detectLanguage(text)
    if (detectedLang == null || detectedLang == "en") {
        return TranslationResult(text, "en", detectedLang, translated = false)
    }
    val sourceCode = TranslateLanguage.fromLanguageTag(detectedLang)
        ?: return TranslationResult(text, detectedLang, detectedLang, translated = false)
    return try {
        val translator = getTranslator(sourceCode)
        translator.downloadModelIfNeeded().await()
        val translated = translator.translate(text).await()
        TranslationResult(translated, detectedLang, detectedLang, translated = true)
    } catch (e: Exception) {
        TranslationResult(text, detectedLang, detectedLang, translated = false)
    }
}
