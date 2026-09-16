# Publishing emojify.ing to Google Play

Status as of 2026-09-16: the app (`android/`) has feature parity with the web
app through Phase 8 of `android/implementation.md`. Nothing below is done yet
— this is the checklist to actually ship it.

Package: **`ing.emojify`** (renamed from the scaffold-era `ing.emojify.app`
to match the product domain `emojify.ing`, same convention as the reverse-DNS
note in `android/implementation.md`).

## 0. Play Developer API access

`/home/gilad/Work/appy.fyi/` turned out not to have `androidpublisher`
credentials (it has a Google Search Console key and an unauthenticated
scraper dependency — different API, no Play Console access), so that wasn't
reusable. A real one was created instead:

- `play/service-account.json` — service account
  `play-console@sage-collector-504503-i1.iam.gserviceaccount.com`, scoped for
  `androidpublisher`. Gitignored (`play/*service-account*.json` in the repo
  `.gitignore`) — never commit it.
- Verified 2026-09-16: it authenticates against Google's OAuth endpoint and
  the Play Developer API correctly recognizes it (a call against package
  `ing.emojify` returns `404 Package not found`, not an auth/permission
  error — the credential itself is good).

**What it can't do:** create the app listing itself. The Play Developer API
has no "create a new app" call — `ing.emojify` has to be created once through
the Play Console UI (app name + default language, accept the Developer
Distribution Agreement for this title) before any API call against it will
work. Do that first:

1. Play Console → **Create app** → name it, set default language, declare
   free/paid, accept the agreement.
2. Confirm the service account has API access granted to it: Play Console →
   **Setup → API access** → the `play-console@...` account should be listed;
   if not, link/invite it there and grant permissions (e.g. "Release to
   testing tracks", "View app information", "Edit store listing").

Once that shell exists, the API can drive everything else from this repo —
upload the AAB, set store listing text/graphics from `play/assets/`, create
tracks, add testers — without touching the Console UI again for routine
releases.

## 1. Developer account

- Sign up at [play.google.com/console](https://play.google.com/console/) —
  $25 one-time fee, government ID verification (can take a few days for a
  new account).
- New accounts are required by Google policy to run a **closed testing
  track with ≥12 opted-in testers for 14 continuous days** before the
  Production track unlocks. This is calendar-gated, not effort-gated —
  start it as soon as you have a working build, even before the rest of
  this checklist is done.

## 2. Release signing (not set up yet)

`android/app/build.gradle.kts` currently has no `signingConfigs` — builds
are debug-signed only. Before a release build:

1. Generate a keystore (once, keep it forever — losing it means you can
   never update the app under this listing again unless you're enrolled in
   Play App Signing, see below):
   ```bash
   keytool -genkeypair -v -keystore play/release.jks \
     -alias emojify -keyalg RSA -keysize 2048 -validity 10000
   ```
   `play/release.jks` — gitignored via `android/.gitignore`'s `*.jks` rule.
   Back it up somewhere outside this repo (password manager / secure
   storage); it is not recoverable if lost.
2. Add `android/keystore.properties` (gitignored, `android/.gitignore`
   already has the `keystore.properties` rule):
   ```properties
   storeFile=../play/release.jks
   storePassword=<your password>
   keyAlias=emojify
   keyPassword=<your password>
   ```
3. Add a `signingConfigs`/`release` build type to
   `android/app/build.gradle.kts` that reads `keystore.properties` (standard
   Gradle pattern — ask Claude to wire this up when you're ready, it's a
   small, mechanical addition).
4. Enroll in **Play App Signing** on first upload (Play Console prompts for
   this) — Google then holds the signing key that ends up on users' devices,
   and your local `release.jks` becomes an *upload* key only. This is the
   default and recommended path; it means a lost/rotated upload key is
   recoverable through Google, a lost app signing key would not be.

## 3. Build the release bundle

Google Play requires the AAB format for new apps, not a raw APK:

```bash
cd android
./gradlew bundleRelease
# output: android/app/build/outputs/bundle/release/app-release.aab
```

Bump `versionCode` (integer, must strictly increase every upload) and
`versionName` in `android/app/build.gradle.kts` before each release build.
Current: `versionCode = 1`, `versionName = "0.1"`.

## 4. Store listing assets

Generated from the same brand art as the web app's `web/public/og.png`
(warm-yellow smiley + `emojify.ing` wordmark), so the Play listing matches
the site instead of introducing a new look:

- `play/assets/icon-512.png` — 512×512 app icon (Play Console → Store
  presence → Main store listing → App icon).
- `play/assets/feature-graphic-1024x500.png` — 1024×500 feature graphic
  (same page → Feature graphic).

**Not generated — need a real device:**

- **Screenshots** (2–8 required, phone size, 16:9 or 9:16, 320–3840px per
  side, JPEG/PNG). These have to be genuine captures of the running app —
  Play reviews for listings that misrepresent the product, and there's no
  emulator available in this environment to fake them credibly anyway. Once
  you have a signed build on a phone (`./gradlew installDebug` or install
  the release AAB via `bundletool`), grab:
  1. Empty/placeholder state (matches the current `MainScreen` initial
     view).
  2. A card result for a clear, appealing phrase — something that shows off
     a good emoji + feeling + gradient combo.
  3. A second card result with a different feeling cluster/font, to show
     visual range.
  4. (Optional) the settings screen (`SettingsScreen.kt`) showing the
     contrast-fix toggle.
- **App icon inside the APK itself**: `android/app/src/main/res/mipmap-anydpi-v26/ic_launcher.xml`
  currently points at a placeholder (`ic_launcher_background.xml` is a flat
  blue square, `ic_launcher_foreground.xml` is a plain dark circle — this is
  the Phase 0 scaffold icon, never replaced). This is a real gap: the Play
  *listing* icon (`icon-512.png`, generated above) will show the branded
  smiley, but the icon that actually lands on the user's home screen after
  install is still the placeholder. This is also the still-open "app
  icon/branding" item under Phase 8 in `android/implementation.md`. Worth
  fixing (adaptive icon foreground/background built from the same smiley
  art) before submitting, not after — a mismatched listing-vs-launcher icon
  looks broken and reviewers/users notice immediately.

## 5. Store listing text

Short description (≤80 chars):
```
Type a phrase, get the emoji, feeling, and colors that fit.
```

Full description (≤4000 chars) — reusing the web app's own framing
(`web/index.html`'s meta description) rather than writing new copy:
```
emojify.ing turns a short phrase into a card: the emoji that fits, a one-word
feeling, and a color palette to match — picked by a small model that runs
entirely on your device.

• Type anything — a mood, a sentence, a name.
• See ranked emoji and feeling suggestions; tap to pick a different one if
  the top guess isn't quite it.
• Every result comes with a generated color palette and matching font.
• Share or copy the result as an image.

No account, no ads, no tracking — nothing you type ever leaves your device.
```

- **Category**: Tools, or Entertainment — either fits; Tools is the safer
  default for a utility-style app with no ongoing content feed.
- **Contact email**: gilad.kutiel@gmail.com (or a dedicated address if you'd
  rather not expose a personal one publicly on the listing).

## 6. Privacy policy

Required even for a no-data-collection app — Play won't let you publish
without a URL. Added `web/public/privacy.html`, deployed via the existing
`.github/workflows/deploy-pages.yml` on next push to `main`, so it will be
live at:

```
https://emojify.ing/privacy.html
```

Paste that URL into Play Console → **Policy → App content → Privacy
policy**.

## 7. Data safety form

Play Console → **Policy → App content → Data safety**. Since the app has no
backend, no `INTERNET` permission, and no third-party SDKs (verified:
`android/app/src/main/AndroidManifest.xml` declares zero `<uses-permission>`
elements), the honest answer throughout is **"No data collected"** — every
sub-question ("Is data collected or shared?") should be No. Do not accept
the form's default suggestions without checking; this only holds as long as
the app stays offline-only.

## 8. App content questionnaire

Also under Policy → App content:

- **Content rating (IARC questionnaire)**: answer as a text/utility app with
  no user-generated content shown to others, no violence/gambling/etc. —
  should land on "Everyone."
- **Target audience & content**: not primarily designed for children;
  pick the adult/general age ranges. If you later want a children's
  audience, that pulls in COPPA-style Families Policy requirements this app
  doesn't currently meet (e.g. stricter ad/data rules) — out of scope for
  now.
- **Ads**: declare "No ads" (none present).
- **Government app / financial features / health**: No to all.

## 9. Submit

1. Play Console → **Testing → Closed testing**, create a track, upload the
   AAB from step 3, add ≥12 testers, start the 14-day clock.
2. Fill in steps 4–8 (Main store listing, Privacy policy, Data safety, App
   content) — Play blocks Production release until all are complete.
3. After the 14-day closed test, promote the same build (or a newer one) to
   **Production**.

## Ongoing

- Every model retrain that changes `web/public/model.onnx` needs
  `android/app/src/main/assets/model.onnx` (+ `meta.json`/`config.json`)
  refreshed the same way — see `android/implementation.md`'s "Model
  inference" section — then a new versioned release through this same
  pipeline.
- Bump `versionCode`/`versionName` every release; Play rejects a re-upload
  with a `versionCode` it's already seen.
