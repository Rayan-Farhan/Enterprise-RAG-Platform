"""Multi-level deduplication and boilerplate detection engine (Master Plan §10)."""

import collections
import hashlib
import re
from collections.abc import Sequence

from app.db.models.element import Element

# Common boilerplate pattern heuristics
BOILERPLATE_REGEX_PATTERNS = [
    re.compile(r"^page\s+\d+(\s+of\s+\d+)?$", re.IGNORECASE),
    re.compile(r"^\d+\s*/\s*\d+$"),
    re.compile(r"^(confidential|strictly confidential|internal use only|proprietary|all rights reserved).*$", re.IGNORECASE),
    re.compile(r"^copyright\s+(©|\(c\))?\s*\d{4}.*$", re.IGNORECASE),
    re.compile(r"^draft\s*-\s*not\s*for\s*distribution$", re.IGNORECASE),
]


def compute_file_sha256(content: bytes) -> str:
    """Compute SHA-256 hash for raw file payload."""
    return hashlib.sha256(content).hexdigest()


def compute_simhash(text: str, hash_bits: int = 64) -> int:
    """Compute a 64-bit SimHash fingerprint for near-duplicate text detection (Master §10)."""
    # Normalize and extract tokens (shingles of 3-4 words or simple word tokens)
    tokens = re.findall(r"\w+", text.lower())
    if not tokens:
        return 0

    v = [0] * hash_bits
    for token in tokens:
        token_hash = int(hashlib.md5(token.encode("utf-8")).hexdigest(), 16)
        for i in range(hash_bits):
            bitmask = 1 << i
            if token_hash & bitmask:
                v[i] += 1
            else:
                v[i] -= 1

    fingerprint = 0
    for i in range(hash_bits):
        if v[i] > 0:
            fingerprint |= 1 << i

    return fingerprint


def simhash_similarity(hash1: int, hash2: int, hash_bits: int = 64) -> float:
    """Compute cosine-like similarity from Hamming distance between two SimHash values."""
    if hash1 == 0 and hash2 == 0:
        return 1.0
    if hash1 == 0 or hash2 == 0:
        return 0.0

    xor_val = hash1 ^ hash2
    hamming_dist = bin(xor_val).count("1")
    return 1.0 - (hamming_dist / float(hash_bits))


class BoilerplateDetector:
    """Detects repeated boilerplate elements (headers, footers, page numbering, legal notices)."""

    def __init__(self, recurrence_threshold: float = 0.60, min_pages_for_recurrence: int = 3) -> None:
        self.recurrence_threshold = recurrence_threshold
        self.min_pages_for_recurrence = min_pages_for_recurrence

    def detect_and_flag(self, elements: Sequence[Element], total_pages: int = 1) -> list[Element]:
        """Analyze elements across the document, flag boilerplate in-place, and return elements."""
        if not elements:
            return list(elements)

        # 1. Count occurrences of normalized text strings across different pages
        text_to_pages: dict[str, set[int]] = collections.defaultdict(set)
        for el in elements:
            norm_text = el.text_content.strip().lower()
            if norm_text and len(norm_text) < 300:  # Short phrases only
                text_to_pages[norm_text].add(el.page_number)

        # 2. Flag elements
        for el in elements:
            norm_text = el.text_content.strip().lower()
            if not norm_text:
                continue

            # Heuristic A: Explicit parser header/footer element types
            if el.element_type in ("header", "footer"):
                el.is_boilerplate = True
                el.boilerplate_reason = f"Parser classified as {el.element_type}"
                continue

            # Heuristic B: Regex matching standard disclaimers/page numbers
            for pattern in BOILERPLATE_REGEX_PATTERNS:
                if pattern.match(norm_text):
                    el.is_boilerplate = True
                    el.boilerplate_reason = "Matched boilerplate pattern regex"
                    break
            if el.is_boilerplate:
                continue

            # Heuristic C: Recurring text across pages (e.g. Header or Footer on every page)
            if total_pages >= self.min_pages_for_recurrence:
                pages_present = len(text_to_pages[norm_text])
                recurrence_ratio = pages_present / float(total_pages)

                if recurrence_ratio >= self.recurrence_threshold:
                    el.is_boilerplate = True
                    el.boilerplate_reason = f"Recurring text across {pages_present}/{total_pages} pages ({recurrence_ratio:.0%})"
                    continue

            # Heuristic D: Positional heuristics for headers/footers with bounding box
            if el.bounding_box and total_pages > 1:
                y0 = el.bounding_box.get("y0", 0)
                y1 = el.bounding_box.get("y1", 1000)

                # Top 60 points or bottom 60 points in standard 842 pt page (A4)
                is_top_margin = y1 <= 60.0
                is_bottom_margin = y0 >= 780.0

                if (is_top_margin or is_bottom_margin) and len(text_to_pages[norm_text]) >= 2:
                    el.is_boilerplate = True
                    margin_type = "header margin" if is_top_margin else "footer margin"
                    el.boilerplate_reason = f"Positional {margin_type} repeated across pages"
                    continue

            if not getattr(el, "is_boilerplate", False):
                el.is_boilerplate = False

        return list(elements)


