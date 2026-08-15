"""Unit test enforcing provider isolation boundary (Task 0.7, ADR-046)."""

import ast
from pathlib import Path

FORBIDDEN_MODULES = {
    "google.generativeai",
    "google.genai",
    "groq",
    "openai",
}


def test_provider_isolation_ast_scan() -> None:
    """Verify that no file outside app/models/providers/ imports direct provider SDKs."""
    root_dir = Path(__file__).resolve().parent.parent.parent
    app_dir = root_dir / "app"
    providers_dir = app_dir / "models" / "providers"

    violations: list[str] = []

    for py_file in app_dir.rglob("*.py"):
        # Skip files inside app/models/providers/
        if providers_dir in py_file.parents:
            continue

        try:
            tree = ast.parse(py_file.read_text(encoding="utf-8"), filename=str(py_file))
        except Exception as e:
            violations.append(f"Failed to parse {py_file}: {e}")
            continue

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    for forbidden in FORBIDDEN_MODULES:
                        if alias.name == forbidden or alias.name.startswith(f"{forbidden}."):
                            violations.append(
                                f"{py_file.relative_to(root_dir)}:{node.lineno} imports forbidden module '{alias.name}'"
                            )
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    for forbidden in FORBIDDEN_MODULES:
                        if node.module == forbidden or node.module.startswith(f"{forbidden}."):
                            violations.append(
                                f"{py_file.relative_to(root_dir)}:{node.lineno} imports from forbidden module '{node.module}'"
                            )

    assert not violations, (
        "Found direct provider SDK imports outside app/models/providers/ (ADR-046 violation):\n"
        + "\n".join(violations)
    )
