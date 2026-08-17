"""Versioned prompt artifact registry (Task 3.4, ADR-047).

Prompts are files on disk, not string literals, so that a prompt change is a
reviewable diff with a version identifier. Every generated answer records the
prompt version it used, and the content hash makes an undeclared edit to a
"locked" prompt version detectable after the fact.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

TEMPLATE_DIR = Path(__file__).parent / "templates"


@dataclass(frozen=True)
class PromptTemplate:
    """An immutable, versioned prompt artifact."""

    version: str
    text: str
    content_hash: str

    def render(self, **values: object) -> str:
        """Substitute ``{placeholder}`` values, leaving unknown braces intact."""
        rendered = self.text
        for key, value in values.items():
            rendered = rendered.replace(f"{{{key}}}", str(value))
        return rendered


class PromptNotFoundError(KeyError):
    """Raised when a requested prompt version has no artifact on disk."""


@lru_cache(maxsize=64)
def load_prompt(directory: Path, version: str) -> PromptTemplate:
    """Load a prompt artifact from any template directory.

    Stage 4's judge prompts live in their own directory but need the identical
    version-plus-content-hash guarantee: a judge whose prompt drifts without a
    version bump makes every score before and after it incomparable.
    """
    # Guard against traversal: prompt versions are flat identifiers, never paths.
    if not version or "/" in version or "\\" in version or version.startswith("."):
        raise PromptNotFoundError(f"Invalid prompt version identifier: '{version}'")

    path = directory / f"{version}.md"
    if not path.is_file():
        available = ", ".join(sorted(p.stem for p in directory.glob("*.md")))
        raise PromptNotFoundError(
            f"Prompt version '{version}' not found in {directory}. Available: {available}"
        )

    text = path.read_text(encoding="utf-8").strip()
    return PromptTemplate(
        version=version,
        text=text,
        content_hash=hashlib.sha256(text.encode("utf-8")).hexdigest(),
    )


def get_prompt(version: str) -> PromptTemplate:
    """Load a generation prompt artifact by version identifier, e.g. ``answer_v1``."""
    return load_prompt(TEMPLATE_DIR, version)


def list_prompt_versions(directory: Path | None = None) -> list[str]:
    """Return every available prompt version identifier in a template directory."""
    return sorted(p.stem for p in (directory or TEMPLATE_DIR).glob("*.md"))
