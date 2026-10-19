"""Render the Homebrew formula for a release from GitHub's asset digests.

    python packaging/homebrew/render.py 0.9.0 > Formula/furnace.rb

Reads the release's assets with `gh api` (GitHub records each asset's SHA-256), fills
furnace.rb.in, and fails if an asset the formula needs is missing.
"""
import json
import pathlib
import re
import subprocess
import sys

REPO = "Gilfeather/furnace"


def main():
    version = sys.argv[1]
    release = json.loads(subprocess.run(
        ["gh", "api", f"repos/{REPO}/releases/tags/v{version}"],
        check=True, capture_output=True, text=True).stdout)
    digests = {}
    for asset in release["assets"]:
        digest = asset.get("digest") or ""
        if digest.startswith("sha256:"):
            digests[asset["name"]] = digest.removeprefix("sha256:")

    template = (pathlib.Path(__file__).parent / "furnace.rb.in").read_text(encoding="utf-8")

    def sha(match):
        name = match.group(1)
        if name not in digests:
            sys.exit(f"release v{version} has no SHA-256 for {name}")
        return digests[name]

    formula = re.sub(r"@SHA_([^@]+)@", sha, template.replace("@VERSION@", version))
    sys.stdout.write(formula)


if __name__ == "__main__":
    main()
