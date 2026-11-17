"""Privacy-conscious aggregate image inventory; does not infer labels or rights."""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = False
EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff", ".gif"}


def revision(root: Path) -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return None


def worktree_fingerprint(root: Path) -> dict:
    try:
        status = subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=all"], cwd=root, text=True, stderr=subprocess.DEVNULL)
        entries = []
        for line in status.splitlines():
            rel = line[3:]
            if rel.replace("\\", "/").startswith("experiments/results/"):
                continue
            path = root / rel
            content = hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else "missing-or-nonfile"
            entries.append(f"{line}\0{content}")
        return {"dirty": bool(status.strip()), "status_sha256": hashlib.sha256(status.encode()).hexdigest(), "worktree_content_sha256": hashlib.sha256("\n".join(entries).encode()).hexdigest()}
    except Exception:
        return {"dirty": None, "status_sha256": None}


def dhash(path: Path) -> int:
    with Image.open(path) as image:
        image = image.convert("L").resize((9, 8))
    px = list(image.tobytes())
    bits = 0
    for y in range(8):
        for x in range(8):
            bits = (bits << 1) | int(px[y * 9 + x] > px[y * 9 + x + 1])
    return bits


def hamming64(left: int, right: int) -> int:
    return (left ^ right).bit_count()


def near_candidate_pairs(values: list[int], radius: int = 2) -> int:
    """BK-tree count of prior-image dHash pairs within a small Hamming radius."""
    if not values:
        return 0
    frequencies = Counter(values)
    iterator = iter(frequencies.items())
    first, first_count = next(iterator)
    tree = {"value": first, "count": first_count, "children": {}}
    pairs = first_count * (first_count - 1) // 2
    for value, count in iterator:
        stack, prior_candidates = [tree], 0
        while stack:
            node = stack.pop()
            distance = hamming64(value, node["value"])
            if distance <= radius:
                prior_candidates += node["count"]
            for edge, child in node["children"].items():
                if distance - radius <= edge <= distance + radius:
                    stack.append(child)
        pairs += count * prior_candidates
        node = tree
        while True:
            distance = hamming64(value, node["value"])
            child = node["children"].get(distance)
            if child is None:
                node["children"][distance] = {"value": value, "count": count, "children": {}}
                break
            node = child
    return pairs


def audit(root: Path) -> dict:
    root = root.resolve()
    files = sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in EXTS)
    dims, formats, sizes, invalid = Counter(), Counter(), [], 0
    hashes: dict[str, int] = defaultdict(int)
    dhashes: dict[str, int] = defaultdict(int)
    dhash_values: list[int] = []
    for path in files:
        try:
            raw_hash = hashlib.sha256(path.read_bytes()).hexdigest()
            hashes[raw_hash] += 1
            sizes.append(path.stat().st_size)
            with Image.open(path) as im:
                im.verify()
            with Image.open(path) as im:
                dims[f"{im.width}x{im.height}"] += 1
                formats[im.format or "unknown"] += 1
            dval = dhash(path)
            dhashes[f"{dval:016x}"] += 1
            dhash_values.append(dval)
        except Exception:
            invalid += 1
    exact_dup_extra = sum(n - 1 for n in hashes.values() if n > 1)
    phash_dup_extra = sum(n - 1 for n in dhashes.values() if n > 1)
    return {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "repository_revision": revision(root),
        "worktree": worktree_fingerprint(root),
        "root_label": root.name,
        "python": platform.python_version(),
        "files_found": len(files),
        "valid_images": len(files) - invalid,
        "invalid_or_unreadable": invalid,
        "formats": dict(sorted(formats.items())),
        "dimension_counts": dict(sorted(dims.items())),
        "bytes": {"total": sum(sizes), "min": min(sizes) if sizes else None, "max": max(sizes) if sizes else None},
        "exact_duplicate_extra_copies": exact_dup_extra,
        "dhash_exact_candidate_extra_copies": phash_dup_extra,
        "dhash_hamming_le_2_candidate_pairs": near_candidate_pairs(dhash_values, radius=2),
        "duplicate_caveat": "dHash equality is only a candidate signal; it does not establish semantic near-duplicates. Filenames, image content, labels, source, license, and train/test independence are not inferred.",
        "labels_available": False,
        "license_reviewed": False,
        "research_ready": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = audit(args.root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
