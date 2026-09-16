---
name: implementing-android-plan
description: Use when picking up or continuing work on the Android app in android/ — asked to "implement the next step", continue android/implementation.md, work on the Android port, or check what's left to build for the Android clone of the web app.
---

# Implementing the Android Plan

## Overview

`android/implementation.md` is a phased, checkbox-tracked implementation plan (design spec above the `---`, file-level tasks below it). Progress lives entirely in that file's `- [ ]` / `- [x]` checkboxes — there is no separate tracker. This skill finds the next unchecked task, implements it, verifies it the way its own steps say to, checks it off, and stops at the right boundary.

## Finding "next"

1. Read `android/implementation.md` in full — the phase list near the top and the task detail in "Detailed Implementation Steps" are both load-bearing. Don't work from a prior read earlier in the conversation; re-read fresh, the file may have changed.
2. Scan top to bottom for the first `- [ ]` (unchecked step). The task containing it is "next."
3. If every step in the current phase is checked but the next unit would be the next phase's first task, **stop** — the `**Phase N checkpoint**` line between them means the user reviews that phase on their phone first. Report what's ready for review and wait; do not start the next phase's tasks.
4. If no `- [ ]` remains anywhere, the plan is complete — say so, don't invent new work.

## Implementing one unit

A **task** (`### Task N.M: ...`), not a single micro-step, is the unit of work per invocation — its Files/Interfaces header and its steps form one self-contained, testable, commit-ending deliverable. Work through its steps in order:

1. Before starting, sanity-check the task's declared **Consumes** interfaces actually exist as written (read/grep the file) — if an earlier task's checkbox is checked but the code it promised isn't there, stop and say so rather than building on a false foundation.
2. Implement each step's content as written — the plan's code/XML/shell blocks are meant to be used as given, not reinvented from scratch. Check off (`- [x]`) a step only after actually running its stated verification and confirming the expected result (see superpowers:verification-before-completion). Never check a box because the code "looks right." Read each step's "Expected:" text literally — it sometimes describes one specific, deliberate failure as the correct outcome at that point in the plan (a later task resolves it); don't treat "any failure" as disqualifying, and don't treat "the command ran" as passing if the actual output doesn't match what "Expected:" says.
3. If a step's command needs a tool this machine doesn't have (system `gradle` for the wrapper, a connected/emulated device for `installDebug`, an Android SDK install), don't skip the step silently or fabricate a result — stop and tell the user exactly what's missing and which step is blocked on it.
4. Step text repeats verbatim across tasks (e.g. "Run test to verify it fails" appears in several tasks) — anchor each Edit on the full step heading line plus enough surrounding content (or the task's unique file paths) that the replacement can't land in the wrong task's checkbox.
5. A step that requires looking at the running app on the phone (an `installDebug` + an "Expected: ..." visual description) is not something you can verify yourself. Build and install it, then stop and ask the user to confirm what they actually see before checking that step off — a successful build alone is not enough to check it off.
6. The task's final step is always a commit — run it exactly as written (its own `git add` file list, its own message) once every preceding step in the task is checked off. Don't fold multiple tasks into one commit, and don't `git push`.
7. Push to the phone after every task, not only tasks whose own steps already call for `installDebug`. After the commit, run `cd android && ./gradlew installDebug` (check `adb devices` first — if nothing's attached, say so instead of silently skipping). This is in addition to the plan's own steps, not a replacement for a task's own build/verify step.

## After finishing a task

Report which task/steps you completed and what's checked off now. If the task you just finished was the last one before a `**Phase N checkpoint**` line, say so explicitly and ask for the phone review before continuing — don't start the next phase's first task in the same turn.

## Common mistakes

- Treating "implement the next step" as license to blitz through an entire phase unattended — stop at task boundaries and at phase checkpoints.
- Checking off a step because the code was written, without running that step's own verification command.
- Marking an `installDebug`/visual step done without the user actually having looked at the phone.
- Losing track of which occurrence of a repeated step string an Edit landed on.
- Re-reading `android/implementation.md` from stale conversation memory instead of fresh off disk.

## When NOT to use

- The user is designing a *new* phase or changing the plan's scope — that's editing the plan, not implementing it.
- No `android/implementation.md` exists yet — that's the writing-plans skill's job, not this one.
