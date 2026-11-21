"""
TruthLens Documentation Link Checker.
Validates that all internal markdown links in docs/ and README.md resolve to existing files.
"""

import os
import re
import sys
import urllib.parse
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

LINK_PATTERN = re.compile(r"\[([^\]]+)\]\(([^)]+)\)")


def check_links() -> int:
    md_files = list(REPO_ROOT.glob("docs/**/*.md")) + [REPO_ROOT / "README.md"]
    broken_count = 0
    total_links = 0

    for md_file in md_files:
        if not md_file.exists():
            continue

        try:
            content = md_file.read_text(encoding="utf-8")
        except Exception:
            content = md_file.read_text(encoding="latin-1")

        matches = LINK_PATTERN.findall(content)

        for text, target in matches:
            total_links += 1
            target = target.strip()

            # Ignore external URLs, mailto, anchor-only links
            if target.startswith(("http://", "https://", "mailto:", "#")):
                continue

            # Strip anchor fragment and unquote URL encoding (e.g. %20 -> space)
            path_part = target.split("#")[0]
            if not path_part:
                continue

            path_part = urllib.parse.unquote(path_part)

            # Resolve target path relative to the md file directory
            resolved = (md_file.parent / path_part).resolve()

            # Also check if it's relative to REPO_ROOT
            resolved_root = (REPO_ROOT / path_part).resolve()

            if not resolved.exists() and not resolved_root.exists():
                print(f"[BROKEN LINK] In {md_file.relative_to(REPO_ROOT)}: [{text}]({target}) -> Target not found")
                broken_count += 1

    print(f"\nScanned {len(md_files)} files, checked {total_links} links.")
    if broken_count > 0:
        print(f"FAILED: Found {broken_count} broken links.")
        return 1
    else:
        print("SUCCESS: All internal markdown links are valid.")
        return 0


if __name__ == "__main__":
    sys.exit(check_links())
