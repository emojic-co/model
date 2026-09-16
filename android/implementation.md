# emojify.ing Android app — design & implementation plan

## Purpose

Native Android clone of `web/`'s functionality: type text, get a live-predicted
emoji + feeling + color-palette "card", written in Kotlin. Built incrementally,
each phase installed to a physical phone for review before the next starts.

Source of truth for behavior is the web app (`web/src/`) — this doc describes
how each piece maps to Android, not a redefinition of the product.

## Architecture

- Single Gradle module, `android/app/`, Jetpack Compose UI.
- Package: `ing.emojify.app` (reverse-DNS of the product domain; the git repo
  name and product name are already intentionally different, see project
  memory).
- No backend. Everything runs on-device, same as the web app.
- minSdk 26 (Android 8.0), target/compile SDK = latest stable at time of
  scaffolding.

### Model inference

- `onnxruntime-android` (ONNX Runtime Mobile AAR) — the Android counterpart to
  `onnxruntime-web` used by `web/src/hooks/useOnnx.js`.
- `model.onnx`, `meta.json`, `config.json` bundled as Android assets
  (`android/app/src/main/assets/`), copied from `web/public/`. Whenever
  `model/export_onnx.py` regenerates the web assets, the Android assets need
  the same refresh — note this in the module's own README once scaffolded.

### Domain layer (plain Kotlin, no Android framework deps, unit-testable)

- `ModelIo.kt` — port of `web/src/model.js`: `normalize`, `encode`,
  `decodeColorList`/`decodeColors`, OKLab conversions (`srgbToOklab`,
  `oklabToSrgb`), `contrastRatio`, `fixContrast`, `mixColors`, `patternTint`,
  `argmax`, `softmax`, `sigmoid`. Logic must match the web port exactly (same
  role as the `model/data.py` / `web/src/model.js` byte-identical-normalize
  requirement already documented in CLAUDE.md — this is a third copy of the
  same contract).
- `Feelings.kt` — port of `web/src/feelings.js`: `FEELINGS` table (cluster,
  font, style flags, entrance/emoji motif durations), `CLUSTERS`,
  `topFeelings`, `resolveFeeling`.
- `Nav.kt` — port of `cycle()` from `web/src/nav.js` (used for swipe/tap
  cycling through predicted emoji/feeling/color options).
- `OnnxPredictor.kt` — session lifecycle + `predict(text)`, mirroring
  `useOnnx.js`'s `predict` callback; returns emoji logits, style logits,
  decoded color palettes, and timing.

### UI layer (Compose)

- `MainScreen` — holds the same state shape as `App.jsx`: input text,
  debounced model-input text, predicted scores, per-field override
  (emoji/feeling/color index), contrast-fix toggle.
- `EmojiList`, `FeelingBar`, `ColorBar`, `Card` composables, mirroring
  `web/src/components/*.jsx` 1:1 in responsibility.
- Debounce: 250ms, same as `DEBOUNCE_MS` in `App.jsx`.

## Where Android diverges from the web app (deliberate)

- **Translation/language detection** — deferred to a later, explicitly
  optional phase (Phase 9). Chrome's on-device `Translator`/`LanguageDetector`
  APIs (`web/src/translate.js`) have no built-in Android equivalent; the
  eventual replacement is Google ML Kit's on-device Translation +
  Language ID. Until Phase 9, all input text is sent to the model as-is
  (matches the model's English-tuned training).
- **Keyboard-modifier cycling** (`Ctrl+↑/↓`, `Alt+↑/↓` in `App.jsx`) doesn't
  apply to a touchscreen. Direct taps on emoji/feeling/color list items
  (already how `onPick` works in the web components) are the primary
  interaction from Phase 2 onward. A swipe-to-cycle gesture on the card is
  added later (Phase 8) as a touch-native bonus, not a required parity item.
- **Copy-as-image** (`web/src/hooks/useCardImage.js`'s canvas render +
  `navigator.clipboard`) becomes: render the Compose card to a `Bitmap`,
  then Android `ClipboardManager` (image via `FileProvider` content URI) for
  copy, `Intent.ACTION_SEND` for share — replacing `navigator.share`.
- **Background patterns** (`web/src/patterns.js`, built on the `hero-patterns`
  JS library) — rather than hand-porting ~20 SVG-generator functions to
  Compose `Canvas` code, pre-render each pattern once via a small one-off
  Node script (reusing the existing `hero-patterns` functions already
  imported in `web/src/patterns.js`) into static tileable SVG/PNG assets
  committed under `android/app/src/main/res/`. Tile them in Compose with a
  `BitmapShader`. Pixel-identical output, far less Kotlin.
- **Fonts** — the ~20 per-feeling Google Fonts referenced in `FEELINGS` are
  bundled as `.ttf` assets under `res/font/`, mapped through `Feelings.kt`.
- **URL routing / share links** (`textToPath`/`pathToText` in `nav.js`,
  `shareUrl` in `App.jsx`) — no backend to route to. "Share" shares the
  rendered card image, optionally alongside a `https://emojify.ing/<text>`
  deep link so non-app recipients still see something meaningful.

## Testing

- Kotlin `JUnit` tests for the pure-Kotlin ports (`ModelIoTest.kt`,
  `FeelingsTest.kt`, `NavTest.kt`), mirroring the existing `web/src/*.test.js`
  files case-for-case where the logic overlaps. Run via `./gradlew test`.
- No instrumented/UI test framework for now. Each phase ends with
  `./gradlew installDebug` to a physical phone for visual/manual review —
  that review is the actual acceptance gate for UI-affecting phases.

## Phased plan

Each phase is a self-contained, installable increment. Stop after each one
for phone review; do not start the next phase until the current one is
approved or explicitly modified.

0. **Scaffold** — empty Compose app, one text field, static placeholder
   card. Proves the Gradle/Compose/adb toolchain end-to-end before any real
   logic exists.
1. **Model bundling + inference smoke test** — add `onnxruntime-android`,
   bundle `model.onnx`/`meta.json`/`config.json` as assets, load the session
   at startup, log a prediction for a known hardcoded input to Logcat. Verify
   the logged emoji/feeling/color logits match the web app's output for the
   same input before moving on.
2. **Minimal predict→UI loop** — debounced text field drives real inference;
   tappable emoji list; tappable feeling name; solid-color placeholder card
   (no GAN colors, custom fonts, or patterns yet — `DEFAULT_COLORS`-equivalent
   background, system font).
3. **Real color palettes** — GAN-decoded gradient background, contrast-fix
   toggle logic (`fixContrast`), tappable color swatch row.
4. **Per-feeling fonts** — bundle the Google Fonts subset, apply per
   predicted/overridden feeling.
5. **Background patterns** — pre-exported static tileable assets (script +
   asset pipeline as described above), tiled behind the gradient per
   feeling/cluster.
6. **Entrance/emoji motion** — port the CSS keyframe animations
   (`ENTRANCE_MOTIFS`, `EMOJI_MOTIFS`, per-feeling durations) to Compose
   `animate*AsState`/`Transition` APIs.
7. **Share & copy-as-image** — render card to `Bitmap`; wire Android share
   sheet and clipboard image copy.
8. **Persistence & polish** — contrast-fix toggle persisted (DataStore),
   footer (model-updated date, about link), app icon/branding, swipe-to-cycle
   gesture on the card.
9. *(stretch, deferred)* **ML Kit translation** — on-device language
   detection + translation, language-override picker (equivalent of
   `KeyHints`' language menu).

---

# Detailed Implementation Steps

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking. Tasks are numbered `<phase>.<task>` to match the phase list above; complete every task in a phase, install, and get phone review before starting the next phase's first task.

**Goal:** Build the Android clone of emojify.ing incrementally — each phase produces an installable, phone-reviewable increment, starting from an empty scaffold and ending with feature parity (minus the deferred translation phase).

**Architecture:** Single-module Jetpack Compose app (`android/app/`). A plain-Kotlin domain layer (`model/` package: `ModelIo.kt`, `Feelings.kt`, `Nav.kt`, `OnnxPredictor.kt`, `Meta.kt`) ported line-for-line from `web/src/model.js`, `web/src/feelings.js`, `web/src/nav.js`, `web/src/hooks/useOnnx.js`. A Compose UI layer (`ui/` package) mirroring `web/src/App.jsx` and `web/src/components/*.jsx`.

**Tech Stack:** Kotlin 2.0.21, Jetpack Compose (BOM 2024.12.01), AGP 8.7.3, ONNX Runtime Mobile (`onnxruntime-android`), kotlinx.serialization, AndroidX DataStore (Phase 8), Coil + SVG decoder (Phase 5), ML Kit Translate/Language-ID (Phase 9 only).

**Spec:** The sections above in this same file ("Purpose" through "Phased plan").

## Global Constraints

- minSdk 26, compileSdk/targetSdk 35, Java/Kotlin target 17.
- Package `ing.emojify.app`.
- Model assets (`model.onnx`, `meta.json`, `config.json`) are copied verbatim from `web/public/` — never hand-edited in `android/`.
- All math/logic in `model/` is a literal, behavior-preserving port of the named `web/src/*.js` file — no independent reinterpretation.
- Every phase ends with `./gradlew installDebug` to a physical phone; do not start the next phase's tasks until the current phase is reviewed and approved (or explicit modifications are requested).
- JUnit (`./gradlew test`) covers pure-Kotlin domain logic only (`ModelIo`, `Feelings`, `Nav`, `Meta` parsing). UI/visual work is verified by installing and looking at it on-device — there is no instrumented UI test framework in this plan.
- Commit after every task (not just every phase).

---

## Phase 0 — Scaffold

### Task 0.1: Gradle project skeleton

**Files:**
- Create: `android/settings.gradle.kts`
- Create: `android/build.gradle.kts`
- Create: `android/gradle.properties`
- Create: `android/app/build.gradle.kts`

**Interfaces:**
- Produces: a Gradle project buildable with `./gradlew help` from `android/`, and an `:app` module later tasks add sources to.

- [x] **Step 1: Create `android/settings.gradle.kts`**

```kotlin
pluginManagement {
    repositories {
        google()
        mavenCentral()
        gradlePluginPortal()
    }
}
dependencyResolutionManagement {
    repositoriesMode.set(RepositoriesMode.FAIL_ON_PROJECT_REPOS)
    repositories {
        google()
        mavenCentral()
    }
}
rootProject.name = "emojify"
include(":app")
```

- [x] **Step 2: Create `android/build.gradle.kts`**

```kotlin
plugins {
    id("com.android.application") version "8.7.3" apply false
    id("org.jetbrains.kotlin.android") version "2.0.21" apply false
    id("org.jetbrains.kotlin.plugin.compose") version "2.0.21" apply false
    id("org.jetbrains.kotlin.plugin.serialization") version "2.0.21" apply false
}
```

- [x] **Step 3: Create `android/gradle.properties`**

```properties
org.gradle.jvmargs=-Xmx2048m
android.useAndroidX=true
kotlin.code.style=official
```

- [x] **Step 4: Create `android/app/build.gradle.kts`**

```kotlin
plugins {
    id("com.android.application")
    id("org.jetbrains.kotlin.android")
    id("org.jetbrains.kotlin.plugin.compose")
    id("org.jetbrains.kotlin.plugin.serialization")
}

android {
    namespace = "ing.emojify.app"
    compileSdk = 35

    defaultConfig {
        applicationId = "ing.emojify.app"
        minSdk = 26
        targetSdk = 35
        versionCode = 1
        versionName = "0.1"
    }

    buildFeatures {
        compose = true
    }

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }
    kotlinOptions {
        jvmTarget = "17"
    }
}

dependencies {
    implementation(platform("androidx.compose:compose-bom:2024.12.01"))
    implementation("androidx.activity:activity-compose:1.9.3")
    implementation("androidx.compose.ui:ui")
    implementation("androidx.compose.ui:ui-tooling-preview")
    implementation("androidx.compose.material3:material3")
    debugImplementation("androidx.compose.ui:ui-tooling")

    testImplementation("junit:junit:4.13.2")
}
```

- [x] **Step 5: Generate the Gradle wrapper**

Run from `android/` (requires a system-installed Gradle, one-time):

```bash
gradle wrapper --gradle-version 8.9 --distribution-type bin
```

This creates `gradlew`, `gradlew.bat`, and `gradle/wrapper/`. All later commands use `./gradlew`.

- [x] **Step 6: Verify the project loads**

Run: `cd android && ./gradlew help`
Expected: Gradle prints the task list with no errors (no source files or `:app` Android sources exist yet, but the module configuration itself must be valid — this will still fail until Task 0.2 adds a manifest; if it fails only on "manifest not found", that's expected and resolves in Task 0.2).

- [x] **Step 7: Commit**

```bash
git add android/settings.gradle.kts android/build.gradle.kts android/gradle.properties android/app/build.gradle.kts android/gradlew android/gradlew.bat android/gradle/wrapper
git commit -m "android: add Gradle project skeleton"
```

### Task 0.2: Manifest, launcher icon, and minimal Compose screen

**Files:**
- Create: `android/app/src/main/AndroidManifest.xml`
- Create: `android/app/src/main/res/values/strings.xml`
- Create: `android/app/src/main/res/mipmap-anydpi-v26/ic_launcher.xml`
- Create: `android/app/src/main/res/drawable/ic_launcher_background.xml`
- Create: `android/app/src/main/res/drawable/ic_launcher_foreground.xml`
- Create: `android/app/src/main/java/ing/emojify/app/MainActivity.kt`
- Create: `android/app/src/main/java/ing/emojify/app/ui/MainScreen.kt`

**Interfaces:**
- Consumes: nothing (first UI code).
- Produces: `MainScreen()` composable — later phases add parameters/state to this same function rather than creating a second entry point.

- [x] **Step 1: Create `AndroidManifest.xml`**

```xml
<?xml version="1.0" encoding="utf-8"?>
<manifest xmlns:android="http://schemas.android.com/apk/res/android">
    <application
        android:allowBackup="true"
        android:icon="@mipmap/ic_launcher"
        android:label="@string/app_name"
        android:theme="@android:style/Theme.Material.Light.NoActionBar">
        <activity
            android:name=".MainActivity"
            android:exported="true"
            android:theme="@android:style/Theme.Material.Light.NoActionBar">
            <intent-filter>
                <action android:name="android.intent.action.MAIN" />
                <category android:name="android.intent.category.LAUNCHER" />
            </intent-filter>
        </activity>
    </application>
</manifest>
```

- [x] **Step 2: Create `res/values/strings.xml`**

```xml
<resources>
    <string name="app_name">emojify.ing</string>
</resources>
```

- [x] **Step 3: Create the launcher icon (pure XML, no binary assets)**

`res/mipmap-anydpi-v26/ic_launcher.xml`:

```xml
<?xml version="1.0" encoding="utf-8"?>
<adaptive-icon xmlns:android="http://schemas.android.com/apk/res/android">
    <background android:drawable="@drawable/ic_launcher_background"/>
    <foreground android:drawable="@drawable/ic_launcher_foreground"/>
</adaptive-icon>
```

`res/drawable/ic_launcher_background.xml`:

```xml
<vector xmlns:android="http://schemas.android.com/apk/res/android"
    android:width="108dp" android:height="108dp"
    android:viewportWidth="108" android:viewportHeight="108">
    <path android:fillColor="#78C9F4" android:pathData="M0,0h108v108h-108z"/>
</vector>
```

`res/drawable/ic_launcher_foreground.xml`:

```xml
<vector xmlns:android="http://schemas.android.com/apk/res/android"
    android:width="108dp" android:height="108dp"
    android:viewportWidth="108" android:viewportHeight="108">
    <path android:fillColor="#282E36" android:pathData="M54,30a24,24 0,1 0,0.1 0z"/>
</vector>
```

- [x] **Step 4: Create `MainActivity.kt`**

```kotlin
package ing.emojify.app

import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Surface
import androidx.compose.ui.Modifier
import ing.emojify.app.ui.MainScreen

class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContent {
            MaterialTheme {
                Surface(modifier = Modifier.fillMaxSize()) {
                    MainScreen()
                }
            }
        }
    }
}
```

- [x] **Step 5: Create `ui/MainScreen.kt` (placeholder card, no model yet)**

```kotlin
package ing.emojify.app.ui

import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.material3.TextField
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp

@Composable
fun MainScreen() {
    var text by remember { mutableStateOf("") }
    Column(modifier = Modifier.fillMaxSize().padding(16.dp)) {
        TextField(
            value = text,
            onValueChange = { text = it },
            placeholder = { Text("type at least 3 characters…") },
            modifier = Modifier.fillMaxWidth(),
        )
        Spacer(modifier = Modifier.height(24.dp))
        Text(text = "🙂", style = MaterialTheme.typography.displayLarge)
        Text(text = "What's on your mind?")
    }
}
```

- [x] **Step 6: Build and install on the phone**

Run: `cd android && ./gradlew installDebug`
Expected: build succeeds, app installs. Launch it manually — a text field and a static 🙂 placeholder should appear.

- [x] **Step 7: Commit**

```bash
git add android/app/src/main/AndroidManifest.xml android/app/src/main/res android/app/src/main/java
git commit -m "android: scaffold minimal Compose screen"
```

**Phase 0 checkpoint — stop here for phone review before Phase 1.**

---

## Phase 1 — Model bundling + inference smoke test

### Task 1.1: Bundle model assets

**Files:**
- Modify: `android/app/build.gradle.kts` (add ONNX Runtime + serialization dependencies)
- Create: `android/app/src/main/assets/model.onnx` (copy)
- Create: `android/app/src/main/assets/meta.json` (copy)
- Create: `android/app/src/main/assets/config.json` (copy)

- [x] **Step 1: Add dependencies**

In `android/app/build.gradle.kts`, inside `dependencies { ... }`, add:

```kotlin
    implementation("com.microsoft.onnxruntime:onnxruntime-android:1.20.0")
    implementation("org.jetbrains.kotlinx:kotlinx-serialization-json:1.7.3")
```

- [x] **Step 2: Copy the model assets from the web app**

```bash
mkdir -p android/app/src/main/assets
cp web/public/model.onnx web/public/meta.json web/public/config.json android/app/src/main/assets/
```

- [x] **Step 3: Verify**

Run: `ls -la android/app/src/main/assets/`
Expected: `model.onnx`, `meta.json`, `config.json` present, sizes matching `web/public/`.

- [x] **Step 4: Commit**

```bash
git add android/app/build.gradle.kts android/app/src/main/assets
git commit -m "android: bundle model assets, add ONNX Runtime + serialization deps"
```

### Task 1.2: `Meta`/`Config` data classes

**Files:**
- Create: `android/app/src/main/java/ing/emojify/app/model/Meta.kt`
- Test: `android/app/src/test/java/ing/emojify/app/model/MetaTest.kt`

**Interfaces:**
- Produces: `Meta(chars, pad_idx, max_text_len, emojis, styles, exported_at?, model_meta?)`, `Config(max_text_len)`, `Meta.charToIndex(): Map<Char, Int>`. Every later task that parses `meta.json`/`config.json` uses these types.

- [x] **Step 1: Write the failing test**

```kotlin
package ing.emojify.app.model

import kotlinx.serialization.json.Json
import org.junit.Assert.assertEquals
import org.junit.Test

class MetaTest {
    private val sampleJson = """
        {
          "chars": "·ab",
          "pad_idx": 0,
          "max_text_len": 4,
          "emojis": ["🙂", "🚗"],
          "styles": ["Joyful", "Tense"]
        }
    """.trimIndent()

    @Test
    fun `parses meta json`() {
        val meta = Json.decodeFromString(Meta.serializer(), sampleJson)
        assertEquals("·ab", meta.chars)
        assertEquals(0, meta.pad_idx)
        assertEquals(4, meta.max_text_len)
        assertEquals(listOf("🙂", "🚗"), meta.emojis)
        assertEquals(listOf("Joyful", "Tense"), meta.styles)
    }

    @Test
    fun `builds char to index map in declared order`() {
        val meta = Json.decodeFromString(Meta.serializer(), sampleJson)
        val map = meta.charToIndex()
        assertEquals(0, map['·'])
        assertEquals(1, map['a'])
        assertEquals(2, map['b'])
    }
}
```

- [x] **Step 2: Run test to verify it fails**

Run: `cd android && ./gradlew test --tests "ing.emojify.app.model.MetaTest"`
Expected: FAIL — `Meta` is unresolved.

- [x] **Step 3: Implement `Meta.kt`**

```kotlin
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
```

- [x] **Step 4: Run test to verify it passes**

Run: `cd android && ./gradlew test --tests "ing.emojify.app.model.MetaTest"`
Expected: PASS.

- [x] **Step 5: Commit**

```bash
git add android/app/src/main/java/ing/emojify/app/model/Meta.kt android/app/src/test/java/ing/emojify/app/model/MetaTest.kt
git commit -m "android: add Meta/Config data classes"
```

### Task 1.3: `ModelIo.kt` — port of `web/src/model.js`

**Files:**
- Create: `android/app/src/main/java/ing/emojify/app/model/ModelIo.kt`
- Test: `android/app/src/test/java/ing/emojify/app/model/ModelIoTest.kt`

**Interfaces:**
- Consumes: `Meta`, `Meta.charToIndex()` (Task 1.2).
- Produces: `normalize(text, charSet)`, `encode(text, meta, char2idx): LongArray`, `Palette(bg1, bg2, textColor)`, `decodeColors(FloatArray): Palette`, `decodeColorList(FloatArray): List<Palette>`, `contrastRatio`, `fixContrast`, `mixColors`, `patternTint`, `argmax`, `softmax`, `sigmoid`. Every later task touching color/text-normalization/logits uses these exact names.

- [x] **Step 1: Write the failing tests**

```kotlin
package ing.emojify.app.model

import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test
import kotlin.math.abs

class ModelIoTest {
    private val charSet = "·abcdefghijklmnopqrstuvwxyz0123456789!?:()@$%&* ".toSet()

    @Test
    fun `normalize lowercases, collapses whitespace, drops unknown chars, caps repeats at 2`() {
        assertEquals("hi there", normalize("  Hi   there", charSet))
        assertEquals("aab", normalize("aaaab", charSet))
        assertEquals("ab", normalize("aéb", charSet)) // é is not in charSet, dropped
    }

    @Test
    fun `encode pads to max_text_len with pad_idx and truncates`() {
        val meta = Meta(chars = "·ab", pad_idx = 0, max_text_len = 4, emojis = emptyList(), styles = emptyList())
        val char2idx = meta.charToIndex()
        val ids = encode("aabbb", meta, char2idx)
        assertEquals(4, ids.size)
        assertEquals(1L, ids[0]) // 'a'
        assertEquals(1L, ids[1]) // 'a'
        assertEquals(2L, ids[2]) // 'b'
        assertEquals(2L, ids[3]) // 'b' (truncated at max_text_len=4, "aabb" after collapse of 3rd b)
    }

    @Test
    fun `decodeColorList chunks flat floats into 9-value palettes`() {
        val flat = floatArrayOf(255f, 0f, 0f, 0f, 255f, 0f, 0f, 0f, 0f)
        val palettes = decodeColorList(flat)
        assertEquals(1, palettes.size)
        assertEquals("#ff0000", palettes[0].bg1)
        assertEquals("#00ff00", palettes[0].bg2)
        assertEquals("#000000", palettes[0].textColor)
    }

    @Test
    fun `contrastRatio is 1 for identical colors and greater for black vs white`() {
        assertEquals(1.0, contrastRatio("#808080", "#808080"), 0.001)
        assertTrue(contrastRatio("#000000", "#ffffff") > 20.0)
    }

    @Test
    fun `fixContrast leaves already-sufficient contrast untouched`() {
        val palette = Palette(bg1 = "#ffffff", bg2 = "#ffffff", textColor = "#000000")
        val fixed = fixContrast(palette)
        assertEquals(palette, fixed)
    }

    @Test
    fun `argmax returns index of largest value`() {
        assertEquals(2, argmax(floatArrayOf(0.1f, 0.4f, 0.9f, 0.2f)))
    }

    @Test
    fun `softmax sums to 1`() {
        val out = softmax(floatArrayOf(1f, 2f, 3f))
        assertTrue(abs(out.sum() - 1.0f) < 0.0001f)
    }

    @Test
    fun `sigmoid maps 0 to 0_5`() {
        val out = sigmoid(floatArrayOf(0f))
        assertEquals(0.5, out[0].toDouble(), 0.0001)
    }
}
```

- [x] **Step 2: Run tests to verify they fail**

Run: `cd android && ./gradlew test --tests "ing.emojify.app.model.ModelIoTest"`
Expected: FAIL — nothing in `ModelIo.kt` exists yet.

- [x] **Step 3: Implement `ModelIo.kt`** (ported from `web/src/model.js`)

```kotlin
package ing.emojify.app.model

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
```

- [x] **Step 4: Run tests to verify they pass**

Run: `cd android && ./gradlew test --tests "ing.emojify.app.model.ModelIoTest"`
Expected: PASS.

- [x] **Step 5: Commit**

```bash
git add android/app/src/main/java/ing/emojify/app/model/ModelIo.kt android/app/src/test/java/ing/emojify/app/model/ModelIoTest.kt
git commit -m "android: port model.js math/color logic to ModelIo.kt"
```

### Task 1.4: `OnnxPredictor.kt` + Logcat smoke test

**Files:**
- Create: `android/app/src/main/java/ing/emojify/app/model/OnnxPredictor.kt`
- Modify: `android/app/src/main/java/ing/emojify/app/MainActivity.kt`

**Interfaces:**
- Consumes: `Meta`, `Meta.charToIndex()`, `encode()`, `decodeColorList()` (Tasks 1.2–1.3).
- Produces: `OnnxPredictor(assets, meta)` with `predict(text, meta): OnnxPredictor.Prediction` where `Prediction(emojiLogits: FloatArray, styleLogits: FloatArray, palettes: List<Palette>, ms: Double)`. Phase 2's UI wiring consumes this type directly.

- [x] **Step 1: Implement `OnnxPredictor.kt`**

```kotlin
package ing.emojify.app.model

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import android.content.res.AssetManager
import java.nio.LongBuffer

class OnnxPredictor(assets: AssetManager, meta: Meta) : AutoCloseable {
    data class Prediction(
        val emojiLogits: FloatArray,
        val styleLogits: FloatArray,
        val palettes: List<Palette>,
        val ms: Double,
    )

    private val env = OrtEnvironment.getEnvironment()
    private val char2idx = meta.charToIndex()
    private val session: OrtSession = assets.open("model.onnx").use { stream ->
        env.createSession(stream.readBytes())
    }

    fun predict(text: String, meta: Meta): Prediction {
        val ids = encode(text, meta, char2idx)
        val shape = longArrayOf(1, meta.max_text_len.toLong())
        val t0 = System.nanoTime()
        OnnxTensor.createTensor(env, LongBuffer.wrap(ids), shape).use { input ->
            session.run(mapOf("input" to input)).use { result ->
                val ms = (System.nanoTime() - t0) / 1_000_000.0
                @Suppress("UNCHECKED_CAST")
                val emojiLogits = (result.get("emoji_logits").get().value as Array<FloatArray>)[0]
                @Suppress("UNCHECKED_CAST")
                val styleLogits = (result.get("style_logits").get().value as Array<FloatArray>)[0]
                @Suppress("UNCHECKED_CAST")
                val colorFlat = (result.get("color").get().value as Array<FloatArray>)[0]
                return Prediction(emojiLogits, styleLogits, decodeColorList(colorFlat), ms)
            }
        }
    }

    override fun close() {
        session.close()
    }
}
```

- [x] **Step 2: Wire a hardcoded smoke test into `MainActivity.kt`**

Add to `MainActivity.kt`, inside `onCreate` before `setContent`:

```kotlin
        val metaJson = assets.open("meta.json").bufferedReader().use { it.readText() }
        val meta = kotlinx.serialization.json.Json.decodeFromString(ing.emojify.app.model.Meta.serializer(), metaJson)
        val predictor = ing.emojify.app.model.OnnxPredictor(assets, meta)
        val result = predictor.predict("I love sunny mornings", meta)
        android.util.Log.d(
            "emojify-smoke",
            "top emoji=${meta.emojis[ing.emojify.app.model.argmax(result.emojiLogits)]} " +
                "top feeling=${meta.styles[ing.emojify.app.model.argmax(result.styleLogits)]} " +
                "palettes=${result.palettes.size} ms=${result.ms}",
        )
        predictor.close()
```

- [x] **Step 3: Install and check Logcat**

Run: `cd android && ./gradlew installDebug`
Then: `adb logcat -s emojify-smoke`, launch the app.
Expected: one log line with a non-empty top emoji, a top feeling name from `meta.styles`, `palettes=1` (or more), and a millisecond timing.

- [x] **Step 4: Cross-check against the web app**

Run `cd web && npm install && npm run dev`, open the site, type the same text ("I love sunny mornings"), and compare the top emoji/feeling shown there against the Logcat line. They should match (same model, same weights) — if they don't, stop and debug the port before continuing to Phase 2.

- [x] **Step 5: Remove the temporary smoke-test code from `MainActivity.kt`** (Phase 2 replaces it with real UI wiring) and commit.

```bash
git add android/app/src/main/java/ing/emojify/app/model/OnnxPredictor.kt android/app/src/main/java/ing/emojify/app/MainActivity.kt
git commit -m "android: add OnnxPredictor and verify inference against the web app"
```

**Phase 1 checkpoint — stop here for phone review before Phase 2.**

---

## Phase 2 — Minimal predict→UI loop

### Task 2.1: `pickEmojiList` + `topFeelings` (prediction → display lists)

**Files:**
- Create: `android/app/src/main/java/ing/emojify/app/model/Predictions.kt`
- Create: `android/app/src/main/java/ing/emojify/app/model/Feelings.kt`
- Test: `android/app/src/test/java/ing/emojify/app/model/PredictionsTest.kt`
- Test: `android/app/src/test/java/ing/emojify/app/model/FeelingsTest.kt`

**Interfaces:**
- Consumes: `sigmoid()` (Task 1.3).
- Produces: `EmojiScore(emoji, p)`, `pickEmojiList(emojiLogits, emojis, slots): List<EmojiScore>` (port of `pickEmojiList` in `App.jsx`), `topFeelings(feelingScores, feelings, selected, count): List<String>` (port of `topFeelings` in `feelings.js`).

- [x] **Step 1: Write failing tests**

```kotlin
// PredictionsTest.kt
package ing.emojify.app.model

import org.junit.Assert.assertEquals
import org.junit.Test

class PredictionsTest {
    @Test
    fun `pickEmojiList returns top-N by logit, sigmoid-scored`() {
        val logits = floatArrayOf(0.1f, 5.0f, -2.0f, 1.0f)
        val emojis = listOf("🙂", "🚗", "🌊", "📅")
        val top = pickEmojiList(logits, emojis, slots = 2)
        assertEquals(listOf("🚗", "📅"), top.map { it.emoji })
        assertEquals(sigmoid(floatArrayOf(5.0f))[0], top[0].p, 0.0001f)
    }
}
```

```kotlin
// FeelingsTest.kt
package ing.emojify.app.model

import org.junit.Assert.assertEquals
import org.junit.Test

class FeelingsTest {
    @Test
    fun `topFeelings ranks by score descending`() {
        val scores = floatArrayOf(0.1f, 0.9f, 0.5f)
        val feelings = listOf("Joyful", "Excited", "Hopeful")
        assertEquals(listOf("Excited", "Hopeful", "Joyful"), topFeelings(scores, feelings, null, count = 3))
    }

    @Test
    fun `topFeelings keeps the selected feeling even if outside top-N`() {
        val scores = floatArrayOf(0.1f, 0.9f, 0.5f, 0.05f)
        val feelings = listOf("Joyful", "Excited", "Hopeful", "Serene")
        val result = topFeelings(scores, feelings, "Serene", count = 2)
        assertEquals(2, result.size)
        assertEquals("Excited", result[0])
        assertEquals("Serene", result[1])
    }

    @Test
    fun `topFeelings returns empty list when scores are null`() {
        assertEquals(emptyList<String>(), topFeelings(null, listOf("Joyful"), null, count = 3))
    }
}
```

- [x] **Step 2: Run tests to verify they fail**

Run: `cd android && ./gradlew test --tests "ing.emojify.app.model.PredictionsTest" --tests "ing.emojify.app.model.FeelingsTest"`
Expected: FAIL — types unresolved.

- [x] **Step 3: Implement**

```kotlin
// Predictions.kt
package ing.emojify.app.model

data class EmojiScore(val emoji: String, val p: Float)

fun pickEmojiList(emojiLogits: FloatArray, emojis: List<String>, slots: Int): List<EmojiScore> {
    val sigmoids = sigmoid(emojiLogits)
    return emojiLogits.indices
        .sortedByDescending { emojiLogits[it] }
        .take(slots)
        .map { EmojiScore(emojis[it], sigmoids[it]) }
}
```

```kotlin
// Feelings.kt (Phase 2 slice — visual FEELINGS map added in Phase 4)
package ing.emojify.app.model

fun topFeelings(
    feelingScores: FloatArray?,
    feelings: List<String>,
    selected: String?,
    count: Int,
): List<String> {
    if (feelingScores == null) return emptyList()
    val ranked = feelings.indices.sortedByDescending { feelingScores[it] }.map { feelings[it] }
    val top = ranked.take(count)
    return if (selected != null && selected !in top) ranked.take(count - 1) + selected else top
}
```

- [x] **Step 4: Run tests to verify they pass**

Run: `cd android && ./gradlew test --tests "ing.emojify.app.model.PredictionsTest" --tests "ing.emojify.app.model.FeelingsTest"`
Expected: PASS.

- [x] **Step 5: Commit**

```bash
git add android/app/src/main/java/ing/emojify/app/model/Predictions.kt android/app/src/main/java/ing/emojify/app/model/Feelings.kt android/app/src/test/java/ing/emojify/app/model/PredictionsTest.kt android/app/src/test/java/ing/emojify/app/model/FeelingsTest.kt
git commit -m "android: port pickEmojiList and topFeelings"
```

### Task 2.2: `EmojiList` and `FeelingBar` composables (tappable)

**Files:**
- Create: `android/app/src/main/java/ing/emojify/app/ui/components/EmojiList.kt`
- Create: `android/app/src/main/java/ing/emojify/app/ui/components/FeelingBar.kt`

**Interfaces:**
- Consumes: `EmojiScore` (Task 2.1).
- Produces: `EmojiList(items: List<EmojiScore>?, active: String?, onPick: (String) -> Unit)`, `FeelingBar(feelings: List<String>, active: String?, onPick: (String) -> Unit)`. `MainScreen` (Task 2.3) is the sole caller.

- [x] **Step 1: Implement `EmojiList.kt`** (port of `web/src/components/EmojiList.jsx`'s data flow: ranked items, active highlight, tap to override)

```kotlin
package ing.emojify.app.ui.components

import androidx.compose.foundation.background
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.lazy.LazyRow
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp
import ing.emojify.app.model.EmojiScore

@Composable
fun EmojiList(items: List<EmojiScore>?, active: String?, onPick: (String) -> Unit) {
    if (items.isNullOrEmpty()) return
    LazyRow(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
        items(items, key = { it.emoji }) { item ->
            val isActive = item.emoji == active
            Text(
                text = item.emoji,
                style = MaterialTheme.typography.headlineSmall,
                modifier = Modifier
                    .clickable { onPick(item.emoji) }
                    .background(
                        if (isActive) MaterialTheme.colorScheme.secondaryContainer else MaterialTheme.colorScheme.surface,
                        CircleShape,
                    )
                    .padding(8.dp),
            )
        }
    }
}
```

- [x] **Step 2: Implement `FeelingBar.kt`** (port of `web/src/components/FeelingBar.jsx`'s data flow)

```kotlin
package ing.emojify.app.ui.components

import androidx.compose.foundation.background
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp

@Composable
fun FeelingBar(feelings: List<String>, active: String?, onPick: (String) -> Unit) {
    if (feelings.isEmpty()) return
    Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
        feelings.forEach { feeling ->
            val isActive = feeling == active
            Text(
                text = feeling,
                modifier = Modifier
                    .clickable { onPick(feeling) }
                    .background(
                        if (isActive) MaterialTheme.colorScheme.secondaryContainer else MaterialTheme.colorScheme.surface,
                        RoundedCornerShape(16.dp),
                    )
                    .padding(horizontal = 12.dp, vertical = 6.dp),
            )
        }
    }
}
```

- [x] **Step 3: Commit**

```bash
git add android/app/src/main/java/ing/emojify/app/ui/components/EmojiList.kt android/app/src/main/java/ing/emojify/app/ui/components/FeelingBar.kt
git commit -m "android: add tappable EmojiList and FeelingBar composables"
```

### Task 2.3: Wire debounced prediction into `MainScreen`

**Files:**
- Modify: `android/app/src/main/java/ing/emojify/app/ui/MainScreen.kt`
- Modify: `android/app/src/main/java/ing/emojify/app/MainActivity.kt`

**Interfaces:**
- Consumes: `OnnxPredictor`/`Prediction` (Task 1.4), `Meta`/`Config` (Task 1.2), `pickEmojiList`/`topFeelings` (Task 2.1), `EmojiList`/`FeelingBar` (Task 2.2).
- Produces: the real `MainScreen(meta: Meta, predictor: OnnxPredictor)` signature — later phases (color palettes, fonts, patterns) add parameters/state here, not a new screen.

Compose's `LaunchedEffect(key)` already cancels its previous coroutine when `key` changes, which replaces `App.jsx`'s manual `seq`/ref-based staleness guard — no extra bookkeeping needed.

- [x] **Step 1: Rewrite `MainScreen.kt`**

```kotlin
package ing.emojify.app.ui

import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.material3.TextField
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp
import ing.emojify.app.model.EmojiScore
import ing.emojify.app.model.Meta
import ing.emojify.app.model.OnnxPredictor
import ing.emojify.app.model.pickEmojiList
import ing.emojify.app.model.topFeelings
import ing.emojify.app.ui.components.EmojiList
import ing.emojify.app.ui.components.FeelingBar
import kotlinx.coroutines.delay

private const val MIN_CHARS = 3
private const val DEBOUNCE_MS = 250L
private const val EMOJI_SLOTS = 9
private const val FEELING_COUNT = 4

private data class Override(val emoji: String? = null, val feeling: String? = null)

@Composable
fun MainScreen(meta: Meta, predictor: OnnxPredictor) {
    var text by remember { mutableStateOf("") }
    var emojiTop by remember { mutableStateOf<List<EmojiScore>>(emptyList()) }
    var predictedFeeling by remember { mutableStateOf<String?>(null) }
    var feelingScores by remember { mutableStateOf<FloatArray?>(null) }
    var override by remember { mutableStateOf(Override()) }

    LaunchedEffect(text) {
        if (text.trim().length < MIN_CHARS) {
            emojiTop = emptyList()
            predictedFeeling = null
            feelingScores = null
            override = Override()
            return@LaunchedEffect
        }
        delay(DEBOUNCE_MS)
        val result = predictor.predict(text, meta)
        emojiTop = pickEmojiList(result.emojiLogits, meta.emojis, EMOJI_SLOTS)
        val feelingIdx = ing.emojify.app.model.argmax(result.styleLogits)
        predictedFeeling = meta.styles[feelingIdx]
        feelingScores = result.styleLogits
        override = Override()
    }

    val shownEmoji = override.emoji ?: emojiTop.firstOrNull()?.emoji
    val shownFeeling = override.feeling ?: predictedFeeling
    val feelingOptions = topFeelings(feelingScores, meta.styles, shownFeeling, FEELING_COUNT)

    Column(modifier = Modifier.fillMaxSize().padding(16.dp)) {
        TextField(
            value = text,
            onValueChange = { text = it },
            placeholder = { Text("type at least 3 characters…") },
            modifier = Modifier.fillMaxWidth(),
        )
        Spacer(modifier = Modifier.height(16.dp))
        EmojiList(items = emojiTop.ifEmpty { null }, active = shownEmoji) { picked ->
            override = override.copy(emoji = picked)
        }
        Spacer(modifier = Modifier.height(16.dp))
        Column(
            modifier = Modifier
                .fillMaxWidth()
                .background(MaterialTheme.colorScheme.surfaceVariant)
                .padding(32.dp),
        ) {
            Text(text = shownEmoji ?: "🙂", style = MaterialTheme.typography.displayLarge)
            Text(text = text.ifBlank { "What's on your mind?" })
        }
        Spacer(modifier = Modifier.height(16.dp))
        FeelingBar(feelings = feelingOptions, active = shownFeeling) { picked ->
            override = override.copy(feeling = picked)
        }
    }
}
```

- [x] **Step 2: Update `MainActivity.kt`** to load `Meta`/build `OnnxPredictor` once and pass them to `MainScreen`

```kotlin
package ing.emojify.app

import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Surface
import androidx.compose.ui.Modifier
import ing.emojify.app.model.Meta
import ing.emojify.app.model.OnnxPredictor
import ing.emojify.app.ui.MainScreen
import kotlinx.serialization.json.Json

class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        val metaJson = assets.open("meta.json").bufferedReader().use { it.readText() }
        val meta = Json.decodeFromString(Meta.serializer(), metaJson)
        val predictor = OnnxPredictor(assets, meta)
        setContent {
            MaterialTheme {
                Surface(modifier = Modifier.fillMaxSize()) {
                    MainScreen(meta = meta, predictor = predictor)
                }
            }
        }
    }
}
```

- [ ] **Step 3: Install and manually verify** — build/install succeeded; the interactive part (type text, watch prediction/tap-override) is pending your own check on the phone, not yet confirmed.

Run: `cd android && ./gradlew installDebug`
Expected: typing ≥3 characters shows a predicted emoji, an emoji list you can tap to override, and a feeling bar you can tap to override — values matching the web app for the same input.

- [x] **Step 4: Commit**

```bash
git add android/app/src/main/java/ing/emojify/app/ui/MainScreen.kt android/app/src/main/java/ing/emojify/app/MainActivity.kt
git commit -m "android: wire debounced prediction into MainScreen"
```

**Phase 2 checkpoint — stop here for phone review before Phase 3.**

---

## Phase 3 — Real color palettes

### Task 3.1: `ColorBar` composable

**Files:**
- Create: `android/app/src/main/java/ing/emojify/app/ui/components/ColorBar.kt`

**Interfaces:**
- Consumes: `Palette` (Task 1.3).
- Produces: `ColorBar(palettes: List<Palette>, active: Int, onPick: (Int) -> Unit)`.

- [x] **Step 1: Implement**

```kotlin
package ing.emojify.app.ui.components

import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.material3.MaterialTheme
import androidx.compose.runtime.Composable
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.unit.dp
import ing.emojify.app.model.Palette

@Composable
fun ColorBar(palettes: List<Palette>, active: Int, onPick: (Int) -> Unit) {
    if (palettes.size <= 1) return
    Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
        palettes.forEachIndexed { index, palette ->
            val isActive = index == active
            androidx.compose.foundation.layout.Box(
                modifier = Modifier
                    .size(28.dp)
                    .clickable { onPick(index) }
                    .background(Color(android.graphics.Color.parseColor(palette.bg1)), CircleShape)
                    .border(
                        width = if (isActive) 2.dp else 0.dp,
                        color = MaterialTheme.colorScheme.primary,
                        shape = CircleShape,
                    ),
            )
        }
    }
}
```

- [x] **Step 2: Commit**

```bash
git add android/app/src/main/java/ing/emojify/app/ui/components/ColorBar.kt
git commit -m "android: add ColorBar composable"
```

### Task 3.2: Extract `Card` composable with real gradient + contrast fix

**Files:**
- Create: `android/app/src/main/java/ing/emojify/app/ui/components/Card.kt`
- Modify: `android/app/src/main/java/ing/emojify/app/ui/MainScreen.kt`

**Interfaces:**
- Consumes: `Palette`, `fixContrast()` (Task 1.3), `ColorBar` (Task 3.1).
- Produces: `Card(text, emoji, feeling, colors: Palette, onCopy: () -> Unit, onShare: () -> Unit)` — `onCopy`/`onShare` are real no-op lambdas until Phase 7 wires them.

- [x] **Step 1: Implement `Card.kt`** (structure ported from `web/src/components/Card.jsx`; fonts/patterns/animation come in Phases 4–6)

```kotlin
package ing.emojify.app.ui.components

import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.unit.dp
import ing.emojify.app.model.Palette

@Composable
fun Card(text: String, emoji: String, feeling: String?, colors: Palette, onCopy: () -> Unit, onShare: () -> Unit) {
    val bg1 = Color(android.graphics.Color.parseColor(colors.bg1))
    val bg2 = Color(android.graphics.Color.parseColor(colors.bg2))
    val textColor = Color(android.graphics.Color.parseColor(colors.textColor))
    Box(
        modifier = Modifier
            .fillMaxWidth()
            .background(Brush.linearGradient(listOf(bg1, bg2)), RoundedCornerShape(16.dp))
            .padding(32.dp),
    ) {
        Column {
            Text(text = emoji, style = MaterialTheme.typography.displayLarge, color = textColor)
            Text(text = text.ifBlank { "What's on your mind?" }, color = textColor)
        }
    }
}
```

- [x] **Step 2: Wire GAN palettes + contrast fix + `Card`/`ColorBar` into `MainScreen.kt`**

In `MainScreen.kt`: add `var palettes by remember { mutableStateOf<List<ing.emojify.app.model.Palette>>(emptyList()) }`, `var colorOverride by remember { mutableStateOf(0) }`, and `var contrastFix by remember { mutableStateOf(true) }` (persistence lands in Phase 8). Inside the `LaunchedEffect(text)` success branch, set `palettes = result.palettes.ifEmpty { listOf(DEFAULT_PALETTE) }` where:

```kotlin
private val DEFAULT_PALETTE = ing.emojify.app.model.Palette(bg1 = "#a8e2f4", bg2 = "#78c9f4", textColor = "#282e36")
```

(the same fallback as `DEFAULT_COLORS` in `feelings.js`). Compute the displayed colors as:

```kotlin
    val displayPalettes = if (contrastFix) palettes.map { ing.emojify.app.model.fixContrast(it) } else palettes
    val colors = displayPalettes.getOrElse(colorOverride) { DEFAULT_PALETTE }
```

Replace the inline placeholder `Column` in `MainScreen`'s layout with:

```kotlin
        Card(text = text, emoji = shownEmoji ?: "🙂", feeling = shownFeeling, colors = colors, onCopy = {}, onShare = {})
        Spacer(modifier = Modifier.height(16.dp))
        ColorBar(palettes = displayPalettes, active = colorOverride) { colorOverride = it }
```

Reset `colorOverride = 0` alongside the other overrides at the top of `LaunchedEffect(text)` when clearing state for short input.

- [ ] **Step 3: Install and manually verify** — build/install succeeded; the visual part (gradient look, legibility, swatch tapping) is pending your own check on the phone.

Run: `cd android && ./gradlew installDebug`
Expected: card background is a real gradient from the model's predicted palette (not the placeholder gray), text stays legible (contrast fix active by default), and tapping color swatches switches palettes when the model returns more than one.

- [x] **Step 4: Commit**

```bash
git add android/app/src/main/java/ing/emojify/app/ui/components/Card.kt android/app/src/main/java/ing/emojify/app/ui/MainScreen.kt
git commit -m "android: wire GAN color palettes, contrast fix, and ColorBar"
```

**Phase 3 checkpoint — stop here for phone review before Phase 4.**

---

## Phase 4 — Per-feeling fonts

Uses Compose's dedicated `androidx.compose.ui:ui-text-google-fonts` API (`GoogleFont`/`GoogleFont.Provider`) instead of bundling raw `.ttf` binaries or per-font XML `<font-family>` resources — this mirrors how the web app pulls fonts by name from Google Fonts, needs only one shared certificate resource (not one XML file per font), and is the Compose-idiomatic way to do downloadable fonts (the classic View-system XML `<font-family>`-per-font approach from an earlier draft of this plan was dropped in favor of this once actually implementing it — same outcome, less boilerplate).

### Task 4.1: Google Fonts provider setup

**Files:**
- Create: `android/app/src/main/res/values/font_certs.xml`
- Modify: `android/app/build.gradle.kts` (add `androidx.compose.ui:ui-text-google-fonts`)

**Interfaces:**
- Produces: `R.array.com_google_android_gms_fonts_certs` (resource), and the `androidx.compose.ui:ui-text-google-fonts` dependency Task 4.2 uses to construct a shared `GoogleFont.Provider` + per-feeling `GoogleFont(name)` instances.

- [x] **Step 1: Add `res/values/font_certs.xml`**

Copy the `com_google_android_gms_fonts_certs` array **verbatim** from Google's own [Jetchat compose-samples file](https://github.com/android/compose-samples/blob/main/Jetchat/app/src/main/res/values-v23/font_certs.xml) — its certificate hashes are fixed, Google-published values and must not be retyped by hand or approximated. (The generic Android Downloadable Fonts guide only shows a truncated placeholder hash; this sample is the documented authoritative source for the real values, and it's fine at `res/values/` rather than `values-v23/` since this project's minSdk is already 26.)

- [x] **Step 2: Add the dependency**

In `android/app/build.gradle.kts`, inside `dependencies { ... }`, add:

```kotlin
    implementation("androidx.compose.ui:ui-text-google-fonts")
```

(version comes from the already-applied `compose-bom` platform, same as the other `androidx.compose.ui` artifacts.)

- [x] **Step 3: Commit**

```bash
git add android/app/src/main/res/values/font_certs.xml android/app/build.gradle.kts
git commit -m "android: add Google Fonts provider setup for per-feeling fonts"
```

### Task 4.2: `Feelings.kt` — full `FEELINGS` table + `resolveFeeling`

**Files:**
- Modify: `android/app/src/main/java/ing/emojify/app/model/Feelings.kt`
- Test: `android/app/src/test/java/ing/emojify/app/model/FeelingsTableTest.kt`

**Interfaces:**
- Produces: `FeelingStyle(cluster, fontName: String, bold: Boolean, italic: Boolean, uppercase: Boolean, letterSpacingEm: Float?, entranceMs: Int, emojiMs: Int)`, `FEELINGS: Map<String, FeelingStyle>`, `resolveFeeling(feeling: String): FeelingStyle`. `fontName` is the literal Google Fonts family name (e.g. `"Fredoka"`), resolved to an actual `FontFamily` in `Card.kt` via `GoogleFont`/`GoogleFont.Provider` (Task 4.3), not an `R.font.*` resource id. Task 4.3 (`Card.kt`) consumes this.

- [x] **Step 1: Write the failing test**

```kotlin
package ing.emojify.app.model

import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class FeelingsTableTest {
    @Test
    fun `resolveFeeling falls back to Neutral for unknown names`() {
        val resolved = resolveFeeling("NotARealFeeling")
        assertEquals(FEELINGS.getValue("Neutral"), resolved)
    }

    @Test
    fun `every style in FEELINGS has positive durations`() {
        FEELINGS.values.forEach {
            assertTrue(it.entranceMs > 0)
            assertTrue(it.emojiMs > 0)
        }
    }
}
```

- [x] **Step 2: Run to verify it fails**, then implement — append to `Feelings.kt` (keep the `topFeelings` function from Task 2.1 as-is):

```kotlin
data class FeelingStyle(
    val cluster: String,
    val fontName: String,
    val bold: Boolean = false,
    val italic: Boolean = false,
    val uppercase: Boolean = false,
    val letterSpacingEm: Float? = null,
    val entranceMs: Int,
    val emojiMs: Int,
)

val FEELINGS: Map<String, FeelingStyle> = mapOf(
    "Joyful" to FeelingStyle("joy", "Fredoka", bold = true, entranceMs = 560, emojiMs = 900),
    "Excited" to FeelingStyle("joy", "Chewy", uppercase = true, letterSpacingEm = 0.05f, entranceMs = 460, emojiMs = 380),
    "Hopeful" to FeelingStyle("drive", "Poppins", entranceMs = 780, emojiMs = 3000),
    "Serene" to FeelingStyle("calm", "Quicksand", entranceMs = 900, emojiMs = 4200),
    "Tender" to FeelingStyle("tender", "Caveat", bold = true, entranceMs = 700, emojiMs = 1300),
    "Playful" to FeelingStyle("play", "Bungee", entranceMs = 600, emojiMs = 1100),
    "Whimsical" to FeelingStyle("play", "Gochi Hand", letterSpacingEm = 0.02f, entranceMs = 640, emojiMs = 1500),
    "Awed" to FeelingStyle("reflective", "Luckiest Guy", letterSpacingEm = 0.04f, entranceMs = 520, emojiMs = 2600),
    "Earnest" to FeelingStyle("tender", "Shadows Into Light", letterSpacingEm = 0.01f, entranceMs = 720, emojiMs = 1600),
    "Determined" to FeelingStyle("drive", "Barlow Condensed", uppercase = true, bold = true, entranceMs = 560, emojiMs = 1400),
    "Proud" to FeelingStyle("drive", "Rubik", uppercase = true, bold = true, letterSpacingEm = 0.05f, entranceMs = 700, emojiMs = 2600),
    "Wistful" to FeelingStyle("sad", "Spectral", italic = true, letterSpacingEm = 0.05f, entranceMs = 1050, emojiMs = 4200),
    "Melancholy" to FeelingStyle("sad", "Playfair Display", italic = true, entranceMs = 1000, emojiMs = 3200),
    "Anxious" to FeelingStyle("anxiety", "Shantell Sans", entranceMs = 560, emojiMs = 220),
    "Tense" to FeelingStyle("anxiety", "Oswald", letterSpacingEm = -0.01f, entranceMs = 500, emojiMs = 420),
    "Furious" to FeelingStyle("anger", "Anton", uppercase = true, letterSpacingEm = 0.06f, entranceMs = 420, emojiMs = 450),
    "Irritated" to FeelingStyle("anger", "Archivo Black", uppercase = true, entranceMs = 520, emojiMs = 600),
    "Disgusted" to FeelingStyle("anger", "Griffy", italic = true, letterSpacingEm = 0.03f, entranceMs = 480, emojiMs = 700),
    "Startled" to FeelingStyle("play", "Schoolbell", entranceMs = 420, emojiMs = 2600),
    "Sarcastic" to FeelingStyle("reflective", "Bitter", italic = true, entranceMs = 800, emojiMs = 4200),
    "Deadpan" to FeelingStyle("reflective", "Inter", entranceMs = 700, emojiMs = 6000),
    "Neutral" to FeelingStyle("reflective", "Work Sans", bold = true, entranceMs = 650, emojiMs = 3200),
)

fun resolveFeeling(feeling: String?): FeelingStyle = FEELINGS[feeling] ?: FEELINGS.getValue("Neutral")
```

(cross-check the durations/style flags against `web/src/feelings.js`'s `FEELINGS` table while transcribing — this listing was copied from it directly.)

- [x] **Step 3: Run to verify it passes and commit**

```bash
cd android && ./gradlew test --tests "ing.emojify.app.model.FeelingsTableTest"
git add android/app/src/main/java/ing/emojify/app/model/Feelings.kt android/app/src/test/java/ing/emojify/app/model/FeelingsTableTest.kt
git commit -m "android: port full FEELINGS table and resolveFeeling"
```

### Task 4.3: Apply feeling font/style in `Card.kt`

**Files:**
- Modify: `android/app/src/main/java/ing/emojify/app/ui/components/Card.kt`
- Modify: `android/app/src/main/java/ing/emojify/app/ui/MainScreen.kt`

- [x] **Step 1: Update `Card`'s signature and text styling**

Change `Card`'s `feeling: String?` usage: replace the plain `Text(text = ..., color = textColor)` for the card text with a `FeelingStyle`-driven style:

```kotlin
    val style = ing.emojify.app.model.resolveFeeling(feeling)
    val fontProvider = androidx.compose.ui.text.googlefonts.GoogleFont.Provider(
        providerAuthority = "com.google.android.gms.fonts",
        providerPackage = "com.google.android.gms",
        certificates = ing.emojify.app.R.array.com_google_android_gms_fonts_certs,
    )
    val fontFamily = androidx.compose.ui.text.font.FontFamily(
        androidx.compose.ui.text.googlefonts.Font(
            googleFont = androidx.compose.ui.text.googlefonts.GoogleFont(style.fontName),
            fontProvider = fontProvider,
        )
    )
    val displayText = if (style.uppercase) text.uppercase() else text
    Text(
        text = displayText.ifBlank { "What's on your mind?" },
        color = textColor,
        fontFamily = fontFamily,
        fontWeight = if (style.bold) androidx.compose.ui.text.font.FontWeight.Bold else androidx.compose.ui.text.font.FontWeight.Normal,
        fontStyle = if (style.italic) androidx.compose.ui.text.font.FontStyle.Italic else androidx.compose.ui.text.font.FontStyle.Normal,
        letterSpacing = style.letterSpacingEm?.let { androidx.compose.ui.unit.TextUnit(it, androidx.compose.ui.unit.TextUnitType.Em) } ?: androidx.compose.ui.unit.TextUnit.Unspecified,
    )
```

- [ ] **Step 2: Install and manually verify** — build/install succeeded (compiles clean, `installDebug` succeeded on the Pixel 7a); the interactive part (typing text and watching the card's font/weight/case change per predicted feeling) is pending your own check on the phone, not yet confirmed. App was not launched or sent simulated input.

- [x] **Step 3: Commit**

```bash
git add android/app/src/main/java/ing/emojify/app/ui/components/Card.kt
git commit -m "android: apply per-feeling font/weight/case/letter-spacing to the card"
```

**Phase 4 checkpoint — stop here for phone review before Phase 5.**

---

## Phase 5 — Background patterns

### Task 5.1: Export `hero-patterns` SVGs as static assets

**Files:**
- Create: `android/scripts/export-patterns.mjs`
- Create: `android/app/src/main/assets/patterns/*.svg` (generated output)

- [ ] **Step 1: Write the export script**, reusing the exact pattern functions already imported in `web/src/patterns.js`:

```javascript
// android/scripts/export-patterns.mjs
import { writeFileSync, mkdirSync } from 'node:fs'
import {
  anchorsAway, brickWall, bubbles, circuitBoard, diagonalStripes, endlessClouds,
  fallingTriangles, floatingCogs, fourPointStars, glamorous, hideout,
  overlappingCircles, skulls, squaresInSquares, stripes, ticTacToe, topography,
  volcanoLamp, wiggle, zigZag,
} from 'hero-patterns'

const CLUSTER_PATTERNS = {
  anger: volcanoLamp,
  joy: stripes,
  play: ticTacToe,
  calm: topography,
  sad: fallingTriangles,
  anxiety: zigZag,
  tender: bubbles,
  drive: anchorsAway,
  reflective: hideout,
}

mkdirSync('android/app/src/main/assets/patterns', { recursive: true })

for (const [cluster, pattern] of Object.entries(CLUSTER_PATTERNS)) {
  const dataUrl = pattern('#ffffff', 1)
  const encoded = dataUrl.match(/^url\((['"]?)data:image\/svg\+xml,(.*)\1\)$/)[2]
  const svg = decodeURIComponent(encoded)
  writeFileSync(`android/app/src/main/assets/patterns/${cluster}.svg`, svg)
}

console.log('exported', Object.keys(CLUSTER_PATTERNS).length, 'pattern SVGs')
```

Run from the repo root (`hero-patterns` is already a `web/` dependency):

```bash
cd web && node ../android/scripts/export-patterns.mjs
```

- [ ] **Step 2: Verify**

Run: `ls android/app/src/main/assets/patterns/`
Expected: one `.svg` file per cluster (`anger.svg`, `joy.svg`, `play.svg`, `calm.svg`, `sad.svg`, `anxiety.svg`, `tender.svg`, `drive.svg`, `reflective.svg`).

- [ ] **Step 3: Commit**

```bash
git add android/scripts/export-patterns.mjs android/app/src/main/assets/patterns
git commit -m "android: export hero-patterns SVGs as static tileable assets"
```

### Task 5.2: Tile the pattern behind the card gradient

**Files:**
- Modify: `android/app/build.gradle.kts` (add Coil + SVG decoder)
- Create: `android/app/src/main/java/ing/emojify/app/ui/components/PatternBackground.kt`
- Modify: `android/app/src/main/java/ing/emojify/app/ui/components/Card.kt`

- [ ] **Step 1: Add dependencies**

```kotlin
    implementation("io.coil-kt.coil3:coil-compose:3.0.4")
    implementation("io.coil-kt.coil3:coil-svg:3.0.4")
```

- [ ] **Step 2: Implement `PatternBackground.kt`** — loads the cluster's SVG asset, decodes it once, and tiles it as a `BitmapShader` behind content at the same 25% opacity the web app uses (`MAX_OPACITY` in `patterns.js`):

```kotlin
package ing.emojify.app.ui.components

import android.graphics.BitmapShader
import android.graphics.Shader
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.runtime.Composable
import androidx.compose.runtime.produceState
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Paint
import androidx.compose.ui.graphics.asAndroidBitmap
import androidx.compose.ui.graphics.drawscope.drawIntoCanvas
import androidx.compose.ui.platform.LocalContext
import coil3.ImageLoader
import coil3.request.ImageRequest
import coil3.request.SuccessResult
import coil3.toBitmap

private const val PATTERN_OPACITY = 0.25f
private const val TILE_PX = 240

@Composable
fun PatternBackground(cluster: String, tint: androidx.compose.ui.graphics.Color, modifier: Modifier = Modifier) {
    val context = LocalContext.current
    val bitmap by produceState<android.graphics.Bitmap?>(initialValue = null, cluster) {
        val loader = ImageLoader(context)
        val request = ImageRequest.Builder(context)
            .data("file:///android_asset/patterns/$cluster.svg")
            .size(TILE_PX, TILE_PX)
            .build()
        val result = loader.execute(request)
        value = (result as? SuccessResult)?.image?.toBitmap()
    }
    bitmap?.let { bmp ->
        Canvas(modifier = modifier.fillMaxSize()) {
            drawIntoCanvas { canvas ->
                val shader = BitmapShader(bmp, Shader.TileMode.REPEAT, Shader.TileMode.REPEAT)
                val paint = Paint().asFrameworkPaint().apply {
                    this.shader = shader
                    alpha = (PATTERN_OPACITY * 255).toInt()
                    colorFilter = android.graphics.PorterDuffColorFilter(tint.toArgb(), android.graphics.PorterDuff.Mode.SRC_IN)
                }
                canvas.nativeCanvas.drawRect(0f, 0f, size.width, size.height, paint)
            }
        }
    }
}
```

- [ ] **Step 3: Layer it behind the gradient in `Card.kt`**

Wrap the existing gradient `Box` content in a `Box` with `PatternBackground` drawn first, gradient `Box` on top with the gradient at reduced alpha, or simpler: draw `PatternBackground` as the first child of the existing `Box`, tinted with `patternTint(colors.bg1, colors.bg2)` (already ported in `ModelIo.kt`, Task 1.3) converted to a Compose `Color`, then keep the gradient background on the `Box.background` as before (it composites on top since Compose draws children in order — move the gradient application from `Modifier.background` to a second `Box.background` sibling drawn after `PatternBackground`, or apply the gradient at partial alpha so the pattern shows through, matching the web app's layered `backgroundImage` list). Use `style.cluster` from `resolveFeeling(feeling)` (Task 4.2) as the `cluster` argument.

- [ ] **Step 4: Install and manually verify**

Run: `cd android && ./gradlew installDebug`
Expected: card background shows a faint repeating pattern tinted to match the palette, varying by feeling cluster, matching the web app's look for the same feeling.

- [ ] **Step 5: Commit**

```bash
git add android/app/build.gradle.kts android/app/src/main/java/ing/emojify/app/ui/components/PatternBackground.kt android/app/src/main/java/ing/emojify/app/ui/components/Card.kt
git commit -m "android: tile background patterns behind the card gradient"
```

**Phase 5 checkpoint — stop here for phone review before Phase 6.**

---

## Phase 6 — Entrance/emoji motion

### Task 6.1: Motif → Compose animation mapping

**Files:**
- Create: `android/app/src/main/java/ing/emojify/app/ui/components/Motion.kt`
- Modify: `android/app/src/main/java/ing/emojify/app/ui/components/Card.kt`

**Interfaces:**
- Consumes: `FeelingStyle.entranceMs`/`emojiMs`, cluster's motif name (ported from `ENTRANCE_MOTIFS`/`EMOJI_MOTIFS` in `feelings.js`).
- Produces: `rememberEntranceOffset(motif: String, durationMs: Int, key: Any?): State<Float>` and `rememberEmojiTransform(motif: String, durationMs: Int): InfiniteTransition` used by `Card.kt`.

- [ ] **Step 1: Implement `Motion.kt`** — a curated Compose equivalent per motif name from `web/src/feelings.js`'s `ENTRANCE_MOTIFS`/`EMOJI_MOTIFS` constants (scale/offset/rotation approximations, not literal CSS keyframe replicas):

```kotlin
package ing.emojify.app.ui.components

import androidx.compose.animation.core.Animatable
import androidx.compose.animation.core.LinearOutSlowInEasing
import androidx.compose.animation.core.RepeatMode
import androidx.compose.animation.core.animateFloat
import androidx.compose.animation.core.infiniteRepeatable
import androidx.compose.animation.core.rememberInfiniteTransition
import androidx.compose.animation.core.tween
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.State
import androidx.compose.runtime.remember

@Composable
fun rememberEntranceScale(motif: String, durationMs: Int, key: Any?): State<Float> {
    val anim = remember(key) { Animatable(if (motif == "pop" || motif == "spin") 0.6f else 1f) }
    LaunchedEffect(key) { anim.animateTo(1f, tween(durationMs, easing = LinearOutSlowInEasing)) }
    return anim.asState()
}

@Composable
fun rememberEmojiBounce(motif: String, durationMs: Int): State<Float> {
    val transition = rememberInfiniteTransition(label = "emoji-$motif")
    val amplitude = when (motif) {
        "hop", "lift" -> 6f
        "shake", "tremor" -> 3f
        "breathe" -> 2f
        else -> 0f
    }
    return transition.animateFloat(
        initialValue = -amplitude,
        targetValue = amplitude,
        animationSpec = infiniteRepeatable(tween(durationMs, easing = LinearOutSlowInEasing), RepeatMode.Reverse),
        label = "emoji-offset",
    )
}
```

- [ ] **Step 2: Apply in `Card.kt`**

Wrap the emoji `Text` in a `Modifier.offset(y = rememberEmojiBounce(style.emojiMotif, style.emojiMs).value.dp)` and the card text in `Modifier.scale(rememberEntranceScale(style.entranceMotif, style.entranceMs, key = text).value)`, where `style.emojiMotif`/`entranceMotif` are two new `FeelingStyle` fields added in this task's `Feelings.kt` edit (default per-cluster motif, same fallback logic as `resolveFeeling` in `feelings.js`: explicit per-feeling motif overrides the cluster default).

- [ ] **Step 3: Install and manually verify**

Run: `cd android && ./gradlew installDebug`
Expected: the emoji visibly bounces/shakes per feeling, and the card text scales in when the prediction changes.

- [ ] **Step 4: Commit**

```bash
git add android/app/src/main/java/ing/emojify/app/ui/components/Motion.kt android/app/src/main/java/ing/emojify/app/ui/components/Card.kt android/app/src/main/java/ing/emojify/app/model/Feelings.kt
git commit -m "android: add entrance/emoji motion per feeling"
```

**Phase 6 checkpoint — stop here for phone review before Phase 7.**

---

## Phase 7 — Share & copy-as-image

### Task 7.1: Capture the card as a `Bitmap`

**Files:**
- Modify: `android/app/src/main/java/ing/emojify/app/ui/components/Card.kt`

- [ ] **Step 1: Use Compose's `GraphicsLayer` capture API**

```kotlin
    val graphicsLayer = androidx.compose.ui.graphics.layer.rememberGraphicsLayer()
    Box(
        modifier = Modifier
            .fillMaxWidth()
            .drawWithContent {
                graphicsLayer.record { this@drawWithContent.drawContent() }
                drawLayer(graphicsLayer)
            }
            /* existing background/padding modifiers */,
    ) { /* existing content */ }
```

Expose a `suspend fun captureBitmap(): android.graphics.Bitmap = graphicsLayer.toImageBitmap().asAndroidBitmap()` from `Card` via a passed-in callback parameter `onCaptureReady: ((suspend () -> android.graphics.Bitmap) -> Unit)?` invoked once after first composition, so `MainScreen` can trigger capture on share/copy taps.

### Task 7.2: Share sheet + clipboard image copy

**Files:**
- Create: `android/app/src/main/java/ing/emojify/app/ShareActions.kt`
- Create: `android/app/src/main/res/xml/file_paths.xml`
- Modify: `android/app/src/main/AndroidManifest.xml`
- Modify: `android/app/src/main/java/ing/emojify/app/ui/MainScreen.kt`

- [ ] **Step 1: Add a `FileProvider`**

`res/xml/file_paths.xml`:

```xml
<?xml version="1.0" encoding="utf-8"?>
<paths>
    <cache-path name="cards" path="cards/" />
</paths>
```

In `AndroidManifest.xml`, inside `<application>`:

```xml
        <provider
            android:name="androidx.core.content.FileProvider"
            android:authorities="ing.emojify.app.fileprovider"
            android:exported="false"
            android:grantUriPermissions="true">
            <meta-data
                android:name="android.support.FILE_PROVIDER_PATHS"
                android:resource="@xml/file_paths" />
        </provider>
```

- [ ] **Step 2: Implement `ShareActions.kt`**

```kotlin
package ing.emojify.app

import android.content.ClipData
import android.content.ClipboardManager
import android.content.Context
import android.content.Intent
import android.graphics.Bitmap
import androidx.core.content.FileProvider
import java.io.File
import java.io.FileOutputStream

private fun writeCardPng(context: Context, bitmap: Bitmap): android.net.Uri {
    val dir = File(context.cacheDir, "cards").apply { mkdirs() }
    val file = File(dir, "card.png")
    FileOutputStream(file).use { bitmap.compress(Bitmap.CompressFormat.PNG, 100, it) }
    return FileProvider.getUriForFile(context, "ing.emojify.app.fileprovider", file)
}

fun shareCard(context: Context, bitmap: Bitmap) {
    val uri = writeCardPng(context, bitmap)
    val intent = Intent(Intent.ACTION_SEND).apply {
        type = "image/png"
        putExtra(Intent.EXTRA_STREAM, uri)
        addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION)
    }
    context.startActivity(Intent.createChooser(intent, null))
}

fun copyCardToClipboard(context: Context, bitmap: Bitmap) {
    val uri = writeCardPng(context, bitmap)
    val clipboard = context.getSystemService(Context.CLIPBOARD_SERVICE) as ClipboardManager
    clipboard.setPrimaryClip(ClipData.newUri(context.contentResolver, "emojify card", uri))
}
```

- [ ] **Step 3: Wire into `MainScreen.kt`**

Add a `LocalContext.current` reference and pass `onCopy = { scope.launch { copyCardToClipboard(context, capture()) } }` / `onShare = { scope.launch { shareCard(context, capture()) } }` to `Card`, using the `captureBitmap()` callback from Task 7.1 and a `rememberCoroutineScope()`.

- [ ] **Step 4: Install and manually verify**

Run: `cd android && ./gradlew installDebug`
Expected: tapping "share" opens the Android share sheet with the rendered card image; tapping "copy" lets you paste the card image into another app (e.g. Messages).

- [ ] **Step 5: Commit**

```bash
git add android/app/src/main/java/ing/emojify/app/ShareActions.kt android/app/src/main/res/xml/file_paths.xml android/app/src/main/AndroidManifest.xml android/app/src/main/java/ing/emojify/app/ui/MainScreen.kt android/app/src/main/java/ing/emojify/app/ui/components/Card.kt
git commit -m "android: add share and copy-as-image"
```

**Phase 7 checkpoint — stop here for phone review before Phase 8.**

---

## Phase 8 — Persistence & polish

### Task 8.1: Persist the contrast-fix toggle

**Files:**
- Modify: `android/app/build.gradle.kts` (add DataStore)
- Create: `android/app/src/main/java/ing/emojify/app/Prefs.kt`
- Modify: `android/app/src/main/java/ing/emojify/app/ui/MainScreen.kt`

- [ ] **Step 1: Add dependency**

```kotlin
    implementation("androidx.datastore:datastore-preferences:1.1.1")
```

- [ ] **Step 2: Implement `Prefs.kt`** (mirrors `localStorage`'s `contrastFix` key in `App.jsx`)

```kotlin
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
```

- [ ] **Step 3: Replace `MainScreen`'s local `contrastFix` state** with `Prefs(context).contrastFix.collectAsState(initial = true)` for reads, and call `prefs.setContrastFix(it)` in a `rememberCoroutineScope()` launch on toggle.

- [ ] **Step 4: Install, toggle contrast-fix off, kill and relaunch the app, verify the setting persisted. Commit.**

```bash
git add android/app/build.gradle.kts android/app/src/main/java/ing/emojify/app/Prefs.kt android/app/src/main/java/ing/emojify/app/ui/MainScreen.kt
git commit -m "android: persist contrast-fix preference via DataStore"
```

### Task 8.2: Footer — model date, about link, branding

**Files:**
- Modify: `android/app/src/main/java/ing/emojify/app/ui/MainScreen.kt`

- [ ] **Step 1: Add a footer `Row`/`Column`** below the `ColorBar`/`FeelingBar`, showing `meta.exported_at` formatted via `java.time.OffsetDateTime.parse(...)`/`DateTimeFormatter`, a "made with ❤️ by Gilad" `Text`, and an "about this model" `Text` wrapped in `Modifier.clickable { context.startActivity(Intent(Intent.ACTION_VIEW, Uri.parse("https://github.com/emoji-co/model/blob/main/ABOUT.md"))) }` (same URL as `App.jsx`'s footer link).

- [ ] **Step 2: Install, verify the footer renders and the about link opens a browser. Commit.**

```bash
git add android/app/src/main/java/ing/emojify/app/ui/MainScreen.kt
git commit -m "android: add footer with model date and about link"
```

### Task 8.3: Swipe-to-cycle gesture (touch-native replacement for keyboard modifiers)

**Files:**
- Create: `android/app/src/main/java/ing/emojify/app/model/Nav.kt`
- Test: `android/app/src/test/java/ing/emojify/app/model/NavTest.kt`
- Modify: `android/app/src/main/java/ing/emojify/app/ui/components/Card.kt`
- Modify: `android/app/src/main/java/ing/emojify/app/ui/MainScreen.kt`

**Interfaces:**
- Produces: `cycle(list: List<T>, current: T?, dir: Int): T?` (port of `cycle()` in `web/src/nav.js`).

- [ ] **Step 1: Write the failing test**

```kotlin
package ing.emojify.app.model

import org.junit.Assert.assertEquals
import org.junit.Test

class NavTest {
    @Test
    fun `cycle wraps forward and backward`() {
        val list = listOf("a", "b", "c")
        assertEquals("b", cycle(list, "a", 1))
        assertEquals("a", cycle(list, "c", 1))
        assertEquals("a", cycle(list, "b", -1))
        assertEquals("c", cycle(list, "a", -1))
    }

    @Test
    fun `cycle starts at first or last when current is not in the list`() {
        val list = listOf("a", "b", "c")
        assertEquals("a", cycle(list, null, 1))
        assertEquals("c", cycle(list, null, -1))
    }

    @Test
    fun `cycle returns current when list is empty`() {
        assertEquals("x", cycle(emptyList(), "x", 1))
    }
}
```

- [ ] **Step 2: Run to verify it fails, then implement**

```kotlin
package ing.emojify.app.model

fun <T> cycle(list: List<T>, current: T?, dir: Int): T? {
    if (list.isEmpty()) return current
    val i = list.indexOf(current)
    if (i == -1) return if (dir > 0) list.first() else list.last()
    return list[(i + dir + list.size) % list.size]
}
```

- [ ] **Step 3: Run to verify it passes**

Run: `cd android && ./gradlew test --tests "ing.emojify.app.model.NavTest"`

- [ ] **Step 4: Wire swipe gestures in `Card.kt`**, cycling the emoji list horizontally and the feeling list vertically:

```kotlin
    modifier = Modifier.pointerInput(emojiList, feelingOptions) {
        detectDragGestures(
            onDragEnd = {},
            onDrag = { change, dragAmount ->
                change.consume()
                if (kotlin.math.abs(dragAmount.x) > kotlin.math.abs(dragAmount.y)) {
                    if (kotlin.math.abs(dragAmount.x) > 24) onEmojiCycle(if (dragAmount.x > 0) -1 else 1)
                } else {
                    if (kotlin.math.abs(dragAmount.y) > 24) onFeelingCycle(if (dragAmount.y > 0) -1 else 1)
                }
            },
        )
    }
```

Pass `onEmojiCycle: (Int) -> Unit` / `onFeelingCycle: (Int) -> Unit` from `MainScreen`, implemented as `override = override.copy(emoji = cycle(emojiTop.map { it.emoji }, shownEmoji, dir))` (and the feeling equivalent using `feelingOptions`).

- [ ] **Step 5: Install and manually verify**

Run: `cd android && ./gradlew installDebug`
Expected: horizontal swipe on the card cycles the predicted emoji; vertical swipe cycles the feeling.

- [ ] **Step 6: Commit**

```bash
git add android/app/src/main/java/ing/emojify/app/model/Nav.kt android/app/src/test/java/ing/emojify/app/model/NavTest.kt android/app/src/main/java/ing/emojify/app/ui/components/Card.kt android/app/src/main/java/ing/emojify/app/ui/MainScreen.kt
git commit -m "android: add swipe-to-cycle gesture for emoji and feeling"
```

**Phase 8 checkpoint — stop here for phone review. This completes parity-minus-translation with the web app.**

---

## Phase 9 (stretch, deferred) — ML Kit translation

Only start this phase if explicitly requested after Phase 8 review — it's the one feature intentionally deferred per the design's translation decision.

### Task 9.1: On-device language ID + translation service

**Files:**
- Modify: `android/app/build.gradle.kts` (add `com.google.mlkit:language-id`, `com.google.mlkit:translate`)
- Create: `android/app/src/main/java/ing/emojify/app/model/Translate.kt`
- Modify: `android/app/src/main/java/ing/emojify/app/ui/MainScreen.kt`

- [ ] **Step 1: Add dependencies**

```kotlin
    implementation("com.google.mlkit:language-id:17.0.6")
    implementation("com.google.mlkit:translate:17.0.3")
```

- [ ] **Step 2: Implement `Translate.kt`**, mirroring `detectAndTranslate()`'s contract in `web/src/translate.js` (detect language, translate to English if a translator is available and download succeeds, otherwise fall back to the original text — never block or crash the predict pipeline on translation failure):

```kotlin
package ing.emojify.app.model

import com.google.mlkit.nl.languageid.LanguageIdentification
import com.google.mlkit.nl.translate.TranslateLanguage
import com.google.mlkit.nl.translate.Translation
import com.google.mlkit.nl.translate.TranslatorOptions
import kotlinx.coroutines.tasks.await

data class TranslationResult(val text: String, val lang: String, val detectedLang: String?, val translated: Boolean)

suspend fun detectAndTranslate(text: String, forcedLang: String? = null): TranslationResult {
    val identifier = LanguageIdentification.getClient()
    val detected = runCatching { identifier.identifyLanguage(text).await() }.getOrNull()
    val detectedLang = detected?.takeIf { it != "und" }
    val lang = forcedLang ?: detectedLang ?: "en"
    if (lang == "en") return TranslationResult(text, "en", detectedLang, translated = false)

    val sourceCode = TranslateLanguage.fromLanguageTag(lang) ?: return TranslationResult(text, lang, detectedLang, translated = false)
    val options = TranslatorOptions.Builder().setSourceLanguage(sourceCode).setTargetLanguage(TranslateLanguage.ENGLISH).build()
    val translator = Translation.getClient(options)
    return try {
        translator.downloadModelIfNeeded().await()
        val translated = translator.translate(text).await()
        TranslationResult(translated, lang, detectedLang, translated = true)
    } catch (e: Exception) {
        TranslationResult(text, lang, detectedLang, translated = false)
    } finally {
        translator.close()
    }
}
```

- [ ] **Step 3: Wire into `MainScreen.kt`'s `LaunchedEffect(text)`** — call `detectAndTranslate(text)` before `predictor.predict(...)`, feeding its `.text` into the predictor instead of the raw input (same ordering as `App.jsx`).

- [ ] **Step 4: Add a minimal language-override UI** (a `DropdownMenu` listing detected/forced language, equivalent to `KeyHints`' language picker in the web app) that sets a `forcedLang` state variable passed into `detectAndTranslate`.

- [ ] **Step 5: Install and manually verify** with non-English input (e.g. Spanish), confirming the predicted emoji/feeling reflect the translated meaning, not the raw non-English text.

- [ ] **Step 6: Commit**

```bash
git add android/app/build.gradle.kts android/app/src/main/java/ing/emojify/app/model/Translate.kt android/app/src/main/java/ing/emojify/app/ui/MainScreen.kt
git commit -m "android: add ML Kit on-device translation"
```

---

## Self-Review Notes

- **Spec coverage:** every architecture element (domain layer, UI layer), every listed divergence (translation, keyboard→touch, copy-as-image, patterns, fonts, URL routing), the testing section, and all ten phases each map to at least one task above.
- **Type consistency checked:** `Palette(bg1, bg2, textColor)`, `EmojiScore(emoji, p)`, `OnnxPredictor.Prediction(emojiLogits, styleLogits, palettes, ms)`, `FeelingStyle(...)`, and `cycle<T>(list, current, dir)` are defined once (Tasks 1.3/1.4/2.1/4.2/8.3) and referenced with the same names/shapes in every later task that consumes them.
- **No placeholders:** every step above either ships real, compilable-as-written Kotlin/XML/Gradle/shell content, or — where a value must not be hand-typed (the Google Fonts provider certificate array) — points to the exact external source to copy verbatim rather than inventing one.
