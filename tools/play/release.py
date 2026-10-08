# /// script
# requires-python = ">=3.11"
# dependencies = ["google-auth>=2.30", "requests>=2.32"]
# ///
"""Release the Android app to Google Play. See `.claude/commands/release.md`.

    uv run tools/play/release.py status
    uv run tools/play/release.py prepare <version>   # bump, build, validate; no upload
    uv run tools/play/release.py publish <version> [--track internal|alpha|production]

Version = one zero-padded integer (e.g. 0004): lexicographically increasing, used
verbatim as versionName, int(version) as versionCode, `v<version>` as git tag.
Release notes live in release/<version>-notes.txt (<=500 chars, en-US).
Listing text lives in play/listing/<locale>/{title,short,full}.txt, optional video.txt
(YouTube id) and phone/*.png screenshots.
`publish` is the only command that touches Play.
"""
import argparse
import re
import subprocess
import sys
from pathlib import Path

import google.auth.transport.requests
from google.oauth2 import service_account

ROOT = Path(__file__).resolve().parents[2]
PKG = "ing.emojify"
GRADLE = ROOT / "android/app/build.gradle.kts"
AAB = ROOT / "android/app/build/outputs/bundle/release/app-release.aab"
KEY = ROOT / "play/service-account.json"
API = f"https://androidpublisher.googleapis.com/androidpublisher/v3/applications/{PKG}"
UPLOAD = f"https://androidpublisher.googleapis.com/upload/androidpublisher/v3/applications/{PKG}"
NOTES_MAX = 500
TRACKS = ("internal", "alpha", "production")


def sh(*cmd, cwd=ROOT, out=False):
    r = subprocess.run(cmd, cwd=cwd, check=True, text=True, capture_output=out)
    return r.stdout.strip() if out else None


def session():
    creds = service_account.Credentials.from_service_account_file(
        KEY, scopes=["https://www.googleapis.com/auth/androidpublisher"]
    )
    return google.auth.transport.requests.AuthorizedSession(creds)


def call(s, method, url, **kw):
    r = s.request(method, url, **kw)
    if not r.ok:
        sys.exit(f"{method} {url} -> {r.status_code}\n{r.text}")
    return r.json() if r.text else {}


def current_version():
    return re.search(r'versionName = "([^"]+)"', GRADLE.read_text()).group(1)


def check_version(version):
    if not re.fullmatch(r"\d{4}", version):
        sys.exit(f"version must be a zero-padded integer like 0004, got {version!r}")
    tags = sh("git", "tag", "-l", "v*", out=True).split()
    newest = max([f"v{current_version()}"] + tags, default="v0")
    if f"v{version}" <= newest:
        sys.exit(f"version v{version} must sort after {newest}")
    if f"v{version}" in tags:
        sys.exit(f"tag v{version} already exists")


def notes_for(version):
    p = ROOT / f"release/{version}-notes.txt"
    if not p.exists():
        sys.exit(f"missing {p.relative_to(ROOT)} — write the release notes first")
    text = p.read_text().strip()
    if not text or len(text) > NOTES_MAX:
        sys.exit(f"{p.name}: must be 1..{NOTES_MAX} chars, got {len(text)}")
    return text


def listings():
    out = {}
    for d in sorted((ROOT / "play/listing").glob("*")):
        f = {k: (d / f"{k}.txt").read_text().strip() for k in ("title", "short", "full")}
        if len(f["title"]) > 30 or len(f["short"]) > 80 or len(f["full"]) > 4000:
            sys.exit(f"{d.name}: listing exceeds Play limits (30/80/4000)")
        out[d.name] = {"title": f["title"], "shortDescription": f["short"], "fullDescription": f["full"]}
        video = d / "video.txt"
        if video.exists():  # YouTube video id
            out[d.name]["video"] = f"https://www.youtube.com/watch?v={video.read_text().strip()}"
    return out


def status(_):
    s = session()
    edit = call(s, "POST", f"{API}/edits", json={})
    for t in call(s, "GET", f"{API}/edits/{edit['id']}/tracks").get("tracks", []):
        for r in t.get("releases", []):
            print(t["track"], r.get("name"), r.get("status"), r.get("versionCodes"))
    for lang in call(s, "GET", f"{API}/edits/{edit['id']}/listings").get("listings", []):
        print("listing", lang["language"], "|", lang.get("title"))
    call(s, "DELETE", f"{API}/edits/{edit['id']}")
    print("local versionName:", current_version(), "| tags:", sh("git", "tag", "-l", "v*", out=True).split())


def prepare(args):
    version = args.version
    check_version(version)
    notes_for(version)
    listings()
    if sh("git", "status", "--porcelain", out=True):
        sys.exit("git tree is dirty — commit or stash first")
    text = GRADLE.read_text()
    text = re.sub(r"versionCode = \d+", f"versionCode = {int(version)}", text)
    text = re.sub(r'versionName = "[^"]+"', f'versionName = "{version}"', text)
    GRADLE.write_text(text)
    sh("./gradlew", "testDebugUnitTest", "bundleRelease", "--console=plain", "-q", cwd=ROOT / "android")
    print(f"built {AAB} for {version}; version bump left uncommitted — `publish` commits and tags")


def publish(args):
    version = args.version
    if current_version() != version:
        sys.exit(f"run `prepare {version}` first (build.gradle.kts is at {current_version()})")
    notes = notes_for(version)
    s = session()
    sh("git", "add", str(GRADLE), str(ROOT / f"release/{version}-notes.txt"))
    sh("git", "commit", "-m", f"Release {version}")
    edit = call(s, "POST", f"{API}/edits", json={})["id"]
    for lang, body in listings().items():
        call(s, "PUT", f"{API}/edits/{edit}/listings/{lang}", json=body)
    for lang in listings():
        shots = sorted((ROOT / f"play/listing/{lang}/phone").glob("*.png"))
        if not shots:
            continue
        call(s, "DELETE", f"{API}/edits/{edit}/listings/{lang}/phoneScreenshots")
        for shot in shots:
            with open(shot, "rb") as f:
                call(s, "POST", f"{UPLOAD}/edits/{edit}/listings/{lang}/phoneScreenshots?uploadType=media",
                     data=f, headers={"Content-Type": "image/png"})
    with open(AAB, "rb") as f:
        bundle = call(s, "POST", f"{UPLOAD}/edits/{edit}/bundles?uploadType=media", data=f,
                      headers={"Content-Type": "application/octet-stream"}, timeout=600)
    assert bundle["versionCode"] == int(version), bundle
    release = {"name": version, "versionCodes": [str(bundle["versionCode"])], "status": "completed",
               "releaseNotes": [{"language": "en-US", "text": notes}]}
    call(s, "PUT", f"{API}/edits/{edit}/tracks/{args.track}", json={"track": args.track, "releases": [release]})
    call(s, "POST", f"{API}/edits/{edit}:validate")
    call(s, "POST", f"{API}/edits/{edit}:commit")
    sh("git", "tag", "-a", f"v{version}", "-m", f"Release {version}")
    print(f"published {version} to {args.track}; push with: git push origin HEAD v{version}")


p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
sub = p.add_subparsers(required=True)
sub.add_parser("status").set_defaults(fn=status)
for name, fn in (("prepare", prepare), ("publish", publish)):
    sp = sub.add_parser(name)
    sp.add_argument("version")
    if name == "publish":
        sp.add_argument("--track", choices=TRACKS, default="alpha")
    sp.set_defaults(fn=fn)
a = p.parse_args()
a.fn(a)
