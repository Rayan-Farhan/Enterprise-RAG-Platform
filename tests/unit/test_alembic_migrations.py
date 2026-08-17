"""Unit tests for Alembic configuration and migration scripts (Task 2.2, ADR-034)."""

from __future__ import annotations

from pathlib import Path

import pytest
from alembic.config import Config
from alembic.script import ScriptDirectory

from app.db.models import Base

MIGRATIONS_DIR = Path("alembic/versions")


@pytest.fixture(scope="module")
def script_dir() -> ScriptDirectory:
    alembic_ini_path = Path("alembic.ini")
    assert alembic_ini_path.exists(), "alembic.ini missing"
    return ScriptDirectory.from_config(Config(str(alembic_ini_path)))


def test_alembic_configuration(script_dir: ScriptDirectory) -> None:
    """Verify alembic.ini is present and can locate the script directory and revisions."""
    revisions = list(script_dir.walk_revisions())
    assert len(revisions) >= 2
    assert revisions[-1].revision == "0001_initial_schema"


def test_revision_chain_is_linear_with_a_single_head(script_dir: ScriptDirectory) -> None:
    """Multiple heads mean a merge is required before the schema can be applied."""
    assert len(script_dir.get_heads()) == 1


def test_every_revision_defines_upgrade_and_downgrade() -> None:
    """Stage 2's gate requires downgrade to work one step."""
    for path in MIGRATIONS_DIR.glob("[0-9]*.py"):
        source = path.read_text(encoding="utf-8")
        assert "def upgrade()" in source, f"{path.name} has no upgrade()"
        assert "def downgrade()" in source, f"{path.name} has no downgrade()"


def test_models_metadata_registered_tables() -> None:
    """Verify all canonical hierarchy tables are registered in SQLAlchemy Base.metadata."""
    table_names = set(Base.metadata.tables.keys())
    expected_tables = {
        "documents",
        "document_versions",
        "document_metadata",
        "pages",
        "elements",
        "chunks",
    }
    assert expected_tables <= table_names


@pytest.mark.parametrize(
    "table",
    ["documents", "document_versions", "document_metadata", "pages", "elements", "chunks"],
)
def test_every_model_column_appears_in_some_migration(table: str) -> None:
    """Catch model/migration drift.

    A column added to a model without a corresponding migration is invisible until
    a fresh database is built from migrations alone — exactly what the Stage 2 exit
    gate requires to work. This is a static check, so it runs without a database.
    """
    migration_source = "\n".join(
        path.read_text(encoding="utf-8") for path in sorted(MIGRATIONS_DIR.glob("[0-9]*.py"))
    )

    missing = [
        column.name
        for column in Base.metadata.tables[table].columns
        if f'"{column.name}"' not in migration_source
    ]
    assert not missing, f"Columns on '{table}' have no migration: {missing}"


def test_stage3_chunk_columns_are_migrated() -> None:
    """The Stage 3 chunk fields specifically (Tasks 3.1/3.2)."""
    source = (MIGRATIONS_DIR / "0002_stage3_chunk_fields.py").read_text(encoding="utf-8")
    for column in (
        "document_id",
        "chunk_type",
        "chunking_version",
        "section_path",
        "page_span",
        "embedding_version",
    ):
        assert f'"{column}"' in source

    # The uniqueness constraint is the final idempotency backstop (ADR-036).
    assert "uq_chunks_version_chunking_index" in source
