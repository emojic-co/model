---
description: Release a new Android version to Google Play (bump, tag, notes, upload)
allowed-tools: [Read, Edit, Write, Bash]
---

# Release

Argument: `$ARGUMENTS` = new version (one zero-padded integer, e.g. `0004`; must sort after every `v*` tag and the current `versionName`). If empty, pick the next number.

1. `uv run tools/play/release.py status` — see what Play has live.
2. Write `release/<version>-notes.txt` (en-US, ≤500 chars, user-facing bullets from `git log` since the last `v*` tag). Update `play/listing/*/` if the app's features changed. Commit these.
3. `uv run tools/play/release.py prepare <version>` — bumps `versionCode`/`versionName`, runs unit tests, builds the signed AAB. Nothing is uploaded.
4. **Ask the user to confirm** (track + notes), then `uv run tools/play/release.py publish <version> --track <alpha|production>` — syncs listing, uploads AAB, sets notes, validates, commits the Play edit, commits `Release <version>` and tags `v<version>`.
5. Tell the user to `git push origin HEAD v<version>`.

Never run `publish` without explicit user approval in the current conversation. Data safety / content rating / privacy URL are Console-only (see `play/publish.md`).
