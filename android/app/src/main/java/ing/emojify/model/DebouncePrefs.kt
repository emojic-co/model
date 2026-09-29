package ing.emojify.model

object DebouncePrefs {
    const val FILE = "debounce"
    const val KEY_DEBOUNCE_MS = "debounce_ms"
    const val DEFAULT_DEBOUNCE_MS = 600
    const val MIN_DEBOUNCE_MS = 0
    const val MAX_DEBOUNCE_MS = 2000
    const val STEP_MS = 50
}
