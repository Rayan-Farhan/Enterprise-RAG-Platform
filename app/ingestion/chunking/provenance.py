"""Provenance prefixing shared by the contextual strategies (Task 5.3).

A chunk lifted out of a 90-page policy manual loses everything that told a reader
which policy it belonged to. Prefixing the document title and section path is
what makes a vector encode "sick leave, in the staff handbook" rather than
"sick leave" — measured at +0.13 recall@5 on this corpus, which carries three
documents restating the same policies.

Extracted rather than duplicated because two strategies now apply it. If
``contextual`` and ``hierarchical_contextual`` prefixed differently, a comparison
between them would be measuring the difference in prefixing as though it were a
difference in hierarchy.
"""

from __future__ import annotations

from app.db.models.element import Element
from app.ingestion.chunking.base import estimate_tokens
from app.ingestion.chunking.segmentation import segment

#: Separator between the provenance prefix and the body. A blank line keeps the
#: prefix from reading as the first sentence of the policy.
PREFIX_SEPARATOR = "\n\n"

#: Metadata keys promoted into the prefix, in the order they appear.
PREFIXED_METADATA_KEYS = ("policy_id", "effective_date", "version_label")


def build_prefix(title: str, section_path: list[str], metadata: dict[str, object]) -> str:
    """Build the provenance line, omitting parts the document does not have.

    An empty prefix is returned rather than one full of placeholders when nothing
    is known: "Document: unknown" is noise in every embedding and tells a reader
    nothing a blank would not.
    """
    parts: list[str] = []
    if title:
        parts.append(f"Document: {title}")
    if section_path:
        parts.append(f"Section: {' > '.join(section_path)}")

    for key in PREFIXED_METADATA_KEYS:
        value = metadata.get(key)
        if value:
            parts.append(f"{key.replace('_', ' ').title()}: {value}")

    return " | ".join(parts)


def reserve_tokens(title: str, elements: list[Element], metadata: dict[str, object]) -> int:
    """Tokens to hold back so body plus prefix still fits the nominal chunk size.

    Without reserving, the prefix silently pushes chunks over budget — measured
    at 147 of 826 on the real corpus — and the strategy ends up compared against
    the others at a larger effective chunk size, crediting the prefix for gains
    that came from carrying more text.

    The reserve is the worst-case prefix for the document rather than the exact
    per-chunk one, so shallow sections give up a few tokens they did not need to.
    That is the safe direction to be wrong in.
    """
    deepest: list[str] = []
    for section in segment(elements):
        if len(build_prefix(title, section.section_path, metadata)) > len(
            build_prefix(title, deepest, metadata)
        ):
            deepest = section.section_path

    prefix = build_prefix(title, deepest, metadata)
    return estimate_tokens(prefix + PREFIX_SEPARATOR) if prefix else 0
