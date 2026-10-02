"""
TruthLens — Image hashing service.

Provides:
- SHA-256 cryptographic hash (exact match)
- pHash perceptual hash (near-duplicate detection)
- Optional local database matching for known images

SHA-256 is computed during upload. pHash is computed here for
comparison / deduplication workflows.
"""

from __future__ import annotations

import hashlib
import io
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import imagehash
from PIL import Image

from backend.app.core.logging import get_logger

log = get_logger("hashing")


@dataclass
class HashResult:
    """Image hash computation result."""
    sha256: str = ""
    phash: str = ""
    ahash: str = ""
    dhash: str = ""
    whash: str = ""
    runtime_ms: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "sha256": self.sha256,
            "phash": self.phash,
            "ahash": self.ahash,
            "dhash": self.dhash,
            "whash": self.whash,
            "runtime_ms": self.runtime_ms,
        }


@dataclass
class HashMatch:
    """A match found in the local database."""
    matched_hash: str = ""
    match_type: str = ""  # exact | near_duplicate
    hamming_distance: int = 0
    matched_analysis_id: str = ""
    similarity: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "matched_hash": self.matched_hash,
            "match_type": self.match_type,
            "hamming_distance": self.hamming_distance,
            "matched_analysis_id": self.matched_analysis_id,
            "similarity": round(self.similarity, 4),
        }


@dataclass
class HashMatchResult:
    """Complete hash matching result."""
    hashes: HashResult = field(default_factory=HashResult)
    matches: List[HashMatch] = field(default_factory=list)
    has_exact_match: bool = False
    has_near_duplicate: bool = False
    runtime_ms: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "hashes": self.hashes.to_dict(),
            "matches": [m.to_dict() for m in self.matches],
            "has_exact_match": self.has_exact_match,
            "has_near_duplicate": self.has_near_duplicate,
            "runtime_ms": self.runtime_ms,
        }


def compute_hashes(image_data: bytes) -> HashResult:
    """
    Compute all hashes for an image.

    Args:
        image_data: Raw image bytes

    Returns:
        HashResult with SHA-256 and perceptual hashes
    """
    t0 = time.time()

    # SHA-256
    sha256 = hashlib.sha256(image_data).hexdigest()

    # Perceptual hashes
    try:
        img = Image.open(io.BytesIO(image_data))
        phash = str(imagehash.phash(img))
        ahash = str(imagehash.average_hash(img))
        dhash = str(imagehash.dhash(img))
        whash = str(imagehash.whash(img))
    except Exception as e:
        log.warning("Perceptual hash computation failed: %s", e)
        phash = ahash = dhash = whash = ""

    runtime_ms = int((time.time() - t0) * 1000)

    return HashResult(
        sha256=sha256,
        phash=phash,
        ahash=ahash,
        dhash=dhash,
        whash=whash,
        runtime_ms=runtime_ms,
    )


def compute_hamming_distance(hash1: str, hash2: str) -> int:
    """Compute Hamming distance between two hex hash strings."""
    if not hash1 or not hash2 or len(hash1) != len(hash2):
        return -1

    try:
        h1 = imagehash.hex_to_hash(hash1)
        h2 = imagehash.hex_to_hash(hash2)
        return int(h1 - h2)
    except Exception:
        return -1


def find_matches(
    image_data: bytes,
    known_hashes: List[Dict[str, str]],
    near_duplicate_threshold: int = 10,
) -> HashMatchResult:
    """
    Compute hashes and check against a list of known hashes.

    Args:
        image_data: Raw image bytes
        known_hashes: List of dicts with "sha256", "phash", "analysis_id"
        near_duplicate_threshold: Max Hamming distance for near-duplicate

    Returns:
        HashMatchResult with any matches found
    """
    t0 = time.time()

    hashes = compute_hashes(image_data)
    matches: List[HashMatch] = []
    has_exact = False
    has_near = False

    for known in known_hashes:
        # Check exact match (SHA-256)
        if known.get("sha256") and known["sha256"] == hashes.sha256:
            matches.append(HashMatch(
                matched_hash=hashes.sha256,
                match_type="exact",
                hamming_distance=0,
                matched_analysis_id=known.get("analysis_id", ""),
                similarity=1.0,
            ))
            has_exact = True
            continue

        # Check near-duplicate (pHash)
        known_phash = known.get("phash", "")
        if known_phash and hashes.phash:
            distance = compute_hamming_distance(hashes.phash, known_phash)
            if 0 <= distance <= near_duplicate_threshold:
                # Convert distance to similarity (0-1)
                max_bits = len(hashes.phash) * 4  # Each hex char = 4 bits
                similarity = 1.0 - (distance / max_bits) if max_bits > 0 else 0.0
                matches.append(HashMatch(
                    matched_hash=known_phash,
                    match_type="near_duplicate",
                    hamming_distance=distance,
                    matched_analysis_id=known.get("analysis_id", ""),
                    similarity=similarity,
                ))
                has_near = True

    runtime_ms = int((time.time() - t0) * 1000)

    return HashMatchResult(
        hashes=hashes,
        matches=matches,
        has_exact_match=has_exact,
        has_near_duplicate=has_near,
        runtime_ms=runtime_ms,
    )
