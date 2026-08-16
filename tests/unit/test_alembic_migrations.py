"""Unit tests for Alembic configuration and initial migration script (Task 2.2, ADR-034)."""

from pathlib import Path

from alembic.config import Config
from alembic.script import ScriptDirectory

from app.db.models import Base


def test_alembic_configuration() -> None:
    """Verify alembic.ini is present and can locate script directory and revisions."""
    alembic_ini_path = Path("alembic.ini")
    assert alembic_ini_path.exists(), "alembic.ini missing"

    alembic_cfg = Config(str(alembic_ini_path))
    script_dir = ScriptDirectory.from_config(alembic_cfg)

    revisions = list(script_dir.walk_revisions())
    assert len(revisions) >= 1, "Expected at least 1 migration revision"
    assert revisions[-1].revision == "0001_initial_schema"


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
    for expected in expected_tables:
        assert expected in table_names, f"Table '{expected}' not registered in Base.metadata"
