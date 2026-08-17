"""Pre-commit guard: refuse to commit a .env file (ADR-033).

Secrets come from the environment. `.env.example` documents the keys; the real
`.env` must never enter version control. This runs as a pre-commit hook and is
also safe to invoke directly with a list of paths.
"""

from __future__ import annotations

import sys
from pathlib import Path


def main(argv: list[str]) -> int:
    offenders = [path for path in argv if Path(path).name == ".env"]
    if not offenders:
        return 0

    print("Refusing to commit environment files (ADR-033: secrets come from the environment):")
    for path in offenders:
        print(f"  - {path}")
    print("\nDocument new keys in .env.example instead, and keep .env git-ignored.")
    return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
