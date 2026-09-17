# Publishing emojify.ing to Google Play

Status as of 2026-09-17: the app (`android/`) has feature parity with the web
app through Phase 8 of `android/implementation.md`. The `ing.emojify` app
shell already exists in Play Console with the service account granted API
access, and store listing text, icon, feature graphic, and 3 screenshots are
already live there (steps 0, 4, 5 below are done — this file lagged the
actual state). Release signing (step 2) and a signed AAB (step 3) are also
done locally. What's left is entirely manual, Console-UI-only steps: the
developer account itself, the closed-testing tester list, and the Data
safety / content rating questionnaires (no public API for those) — see the
per-section notes below.

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

## 2. Release signing (done 2026-09-17)

`play/release.jks` — a PKCS12 keystore, alias `emojify`, generated locally.
`.gitignore` now has `play/*.jks` at the repo root (the old note that
`android/.gitignore`'s `*.jks` rule covered it was wrong — that rule only
applies inside `android/`, not sibling `play/`; fixed alongside the keystore
generation). `android/keystore.properties` (gitignored) holds the generated
password. `android/app/build.gradle.kts` has a `signingConfigs["release"]`
reading from it, wired into `buildTypes.release`.

**Back up `play/release.jks` and `android/keystore.properties` outside this
repo (password manager / secure storage) — this is not something Claude Code
can do, and losing both means you can never update this listing again unless
already enrolled in Play App Signing.**

Enroll in **Play App Signing** on first Console upload — Google then holds
the signing key that ends up on users' devices, and the local `release.jks`
becomes an *upload* key only (lost/rotated upload key recoverable through
Google; a lost app signing key would not be).

## 3. Build the release bundle (done 2026-09-17)

```bash
cd android
./gradlew bundleRelease
# output: android/app/build/outputs/bundle/release/app-release.aab
```

Built and signed with the keystore above (`versionCode = 1`,
`versionName = "0.1"`). Bump both before every subsequent release build —
Play rejects a re-upload with a `versionCode` it's already seen.

## 4. Store listing assets (done — uploaded to Play Console)

Generated from the same brand art as the web app's `web/public/og.png`
(warm-yellow smiley + `emojify.ing` wordmark), so the Play listing matches
the site instead of introducing a new look:

- `play/assets/icon-512.png` — 512×512 app icon. Uploaded.
- `play/assets/feature-graphic-1024x500.png` — 1024×500 feature graphic.
  Uploaded.
- `play/assets/screenshots/{1-empty,2-grateful,3-party}.png` — 3 genuine
  on-device captures (empty state, two card results with different
  feeling/font combos). Uploaded. 2 is the Play minimum; more can be added
  later but isn't required to submit.
- Launcher icon (`android/app/src/main/res/drawable/ic_launcher_{background,foreground}.xml`)
  is now the branded smiley adaptive icon, not the Phase 0 scaffold
  placeholder — listing icon and installed-app icon match.

## 5. Store listing text (done — already set in Play Console)

Title, short/full description, contact website, and contact email are
already committed there (`defaultLanguage: en-US`, contact email is
`gilad@appy.fyi`, not the `gilad.kutiel@gmail.com` suggested below — change
it in Console → Store presence if you'd rather use a different address).
Text kept here for reference:

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

## 9. Submit — remaining steps, all manual/Console-only

The Play Developer API (`androidpublisher` v3) has no endpoint for Data
safety, content rating, or privacy-policy-URL declarations — those three are
Console-UI-only regardless of API access. Concretely, what's left:

1. **Developer account**: sign up + $25 fee + government ID verification
   (step 1) — has to be you, Claude Code can't do KYC.
2. **Privacy policy URL**: paste `https://emojify.ing/privacy.html` (already
   live) into Policy → App content → Privacy policy.
3. **Data safety** and **App content questionnaire** (steps 7–8): fill in
   by hand using the honest answers already written out there ("No data
   collected", Everyone rating, no ads).
4. **Upload the signed AAB and start closed testing**: the bundle is built
   (`android/app/build/outputs/bundle/release/app-release.aab`) and the API
   credentials can upload it and attach it to the `alpha` track directly
   from this repo — but doing so is a real submission against your Play
   Console account, so Claude Code won't do it without you asking in the
   moment. Either:
   - ask Claude Code to run it (the upload+commit script), or
   - Play Console → **Testing → Closed testing** → upload the AAB by hand.

   Either way you still need to supply **≥12 real opted-in tester emails**
   (or a Google Group) — that's a list only you can produce — then start the
   14-day clock.
5. After the 14-day closed test, promote the same build (or a newer one) to
   **Production**.

## Ongoing

- Every model retrain that changes `web/public/model.onnx` needs
  `android/app/src/main/assets/model.onnx` (+ `meta.json`/`config.json`)
  refreshed the same way — see `android/implementation.md`'s "Model
  inference" section — then a new versioned release through this same
  pipeline.
- Bump `versionCode`/`versionName` every release; Play rejects a re-upload
  with a `versionCode` it's already seen.
