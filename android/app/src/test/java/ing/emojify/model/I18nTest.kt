package ing.emojify.model

import java.io.File
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class I18nTest {
    private val table = parseStyleFile(File("src/main/assets/style.yml").readText()).i18n!!
    private val en = Strings("en", table)
    private val he = Strings("he", table)

    private fun placeholders(s: String) = Regex("\\{(\\w+)\\}").findAll(s).map { it.groupValues[1] }.sorted().toList()

    @Test
    fun `en and he have identical keys and placeholders`() {
        assertEquals(table.getValue("en").keys, table.getValue("he").keys)
        for ((k, v) in table.getValue("en")) assertEquals(k, placeholders(v), placeholders(table.getValue("he").getValue(k)))
    }

    @Test
    fun `every style has a feeling label in both languages`() {
        for (id in parseStyleFile(File("src/main/assets/style.yml").readText()).styles.keys) {
            assertTrue(id, table.getValue("en").containsKey("feeling.$id"))
            assertTrue(id, table.getValue("he").containsKey("feeling.$id"))
        }
    }

    @Test
    fun `substitutes placeholders and leaves unknown ones`() {
        assertEquals("Max emojis shown: 7", en.t("settings.maxEmojis", mapOf("n" to 7)))
        assertEquals("color {n}", en.t("color.aria"))
        assertEquals("{n} saved ✓", en.t("toast.saved", mapOf("format" to "{n}", "n" to "x")))
    }

    @Test
    fun `falls back to en then key, feeling falls back to id`() {
        assertEquals("no.such.key", he.t("no.such.key"))
        assertEquals("Brand New", he.feeling("Brand New"))
        assertEquals("שמח", he.feeling("Joyful"))
    }

    @Test
    fun `literals jpg gif mp4 are kept in hebrew`() {
        assertTrue(he.t("card.copyJpg").contains("jpg"))
        assertTrue(he.t("settings.exportRes").contains("jpg / mp4"))
    }

    @Test
    fun `language resolution prefers saved, then device, then en`() {
        assertEquals("en", LanguagePrefs.resolve("en", "he"))
        assertEquals("he", LanguagePrefs.resolve(null, "he"))
        assertEquals("he", LanguagePrefs.resolve(null, "iw"))
        assertEquals("en", LanguagePrefs.resolve("fr", "fr"))
        assertEquals("en", LanguagePrefs.resolve(null, "de"))
    }
}
