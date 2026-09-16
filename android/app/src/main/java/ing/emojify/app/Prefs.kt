package ing.emojify.app

import android.content.Context
import androidx.datastore.preferences.core.booleanPreferencesKey
import androidx.datastore.preferences.core.edit
import androidx.datastore.preferences.preferencesDataStore
import kotlinx.coroutines.flow.map

private val Context.dataStore by preferencesDataStore(name = "prefs")
private val CONTRAST_FIX_KEY = booleanPreferencesKey("contrastFix")

class Prefs(private val context: Context) {
    val contrastFix = context.dataStore.data.map { it[CONTRAST_FIX_KEY] ?: true }

    suspend fun setContrastFix(value: Boolean) {
        context.dataStore.edit { it[CONTRAST_FIX_KEY] = value }
    }
}
