"""Initial database schema for canonical document hierarchy (ADR-002, ADR-005, ADR-034).

Revision ID: 0001_initial_schema
Revises:
Create Date: 2026-08-17 00:00:00.000000

"""
from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "0001_initial_schema"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # 1. documents table
    op.create_table(
        "documents",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("external_id", sa.String(length=255), nullable=True),
        sa.Column("title", sa.String(length=512), nullable=False),
        sa.Column("mime_type", sa.String(length=128), nullable=False),
        sa.Column("file_size_bytes", sa.BigInteger(), nullable=False, server_default="0"),
        sa.Column("file_hash", sa.String(length=64), nullable=False),
        sa.Column("storage_key", sa.String(length=512), nullable=False),
        sa.Column("duplicate_group_id", sa.Uuid(), nullable=True),
        sa.Column("source_priority", sa.Integer(), nullable=False, server_default="1"),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("now()")),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("now()")),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_documents_external_id", "documents", ["external_id"])
    op.create_index("ix_documents_title", "documents", ["title"])
    op.create_index("ix_documents_file_hash", "documents", ["file_hash"], unique=True)
    op.create_index("ix_documents_duplicate_group_id", "documents", ["duplicate_group_id"])

    # 2. document_versions table
    op.create_table(
        "document_versions",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("document_id", sa.Uuid(), nullable=False),
        sa.Column("version_number", sa.Integer(), nullable=False, server_default="1"),
        sa.Column("status", sa.String(length=32), nullable=False, server_default="active"),
        sa.Column("effective_from", sa.DateTime(timezone=True), nullable=True),
        sa.Column("effective_until", sa.DateTime(timezone=True), nullable=True),
        sa.Column("supersedes_id", sa.Uuid(), nullable=True),
        sa.Column("authority", sa.String(length=255), nullable=True),
        sa.Column("total_pages", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("total_elements", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("parser_name", sa.String(length=64), nullable=False),
        sa.Column("parsing_duration_ms", sa.Float(), nullable=False, server_default="0.0"),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("now()")),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("now()")),
        sa.ForeignKeyConstraint(["document_id"], ["documents.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["supersedes_id"], ["document_versions.id"], ondelete="SET NULL"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_document_versions_document_id", "document_versions", ["document_id"])
    op.create_index("ix_document_versions_status", "document_versions", ["status"])
    op.create_index("ix_document_versions_effective_from", "document_versions", ["effective_from"])
    op.create_index("ix_document_versions_effective_until", "document_versions", ["effective_until"])

    # 3. document_metadata table
    op.create_table(
        "document_metadata",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("version_id", sa.Uuid(), nullable=False),
        sa.Column("department", sa.String(length=128), nullable=True),
        sa.Column("policy_type", sa.String(length=128), nullable=True),
        sa.Column("policy_status", sa.String(length=64), nullable=True),
        sa.Column("country", sa.String(length=64), nullable=True),
        sa.Column("location", sa.String(length=128), nullable=True),
        sa.Column("employee_type", sa.String(length=64), nullable=True),
        sa.Column("grade", sa.String(length=64), nullable=True),
        sa.Column("confidentiality", sa.String(length=64), nullable=True),
        sa.Column("audience", sa.String(length=128), nullable=True),
        sa.Column("custom_attributes", sa.JSON(), nullable=False, server_default="{}"),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("now()")),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("now()")),
        sa.ForeignKeyConstraint(["version_id"], ["document_versions.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("version_id"),
    )
    op.create_index("ix_document_metadata_department", "document_metadata", ["department"])
    op.create_index("ix_document_metadata_policy_type", "document_metadata", ["policy_type"])
    op.create_index("ix_document_metadata_policy_status", "document_metadata", ["policy_status"])
    op.create_index("ix_document_metadata_country", "document_metadata", ["country"])
    op.create_index("ix_document_metadata_location", "document_metadata", ["location"])
    op.create_index("ix_document_metadata_employee_type", "document_metadata", ["employee_type"])
    op.create_index("ix_document_metadata_grade", "document_metadata", ["grade"])
    op.create_index("ix_document_metadata_confidentiality", "document_metadata", ["confidentiality"])
    op.create_index("ix_document_metadata_audience", "document_metadata", ["audience"])

    # 4. pages table
    op.create_table(
        "pages",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("version_id", sa.Uuid(), nullable=False),
        sa.Column("page_number", sa.Integer(), nullable=False),
        sa.Column("width", sa.Float(), nullable=False, server_default="595.0"),
        sa.Column("height", sa.Float(), nullable=False, server_default="842.0"),
        sa.Column("content_hash", sa.String(length=64), nullable=False),
        sa.Column("page_image_key", sa.String(length=512), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("now()")),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("now()")),
        sa.ForeignKeyConstraint(["version_id"], ["document_versions.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_pages_version_id", "pages", ["version_id"])
    op.create_index("ix_pages_page_number", "pages", ["page_number"])

    # 5. elements table
    op.create_table(
        "elements",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("version_id", sa.Uuid(), nullable=False),
        sa.Column("page_id", sa.Uuid(), nullable=False),
        sa.Column("page_number", sa.Integer(), nullable=False),
        sa.Column("element_id", sa.String(length=128), nullable=False),
        sa.Column("parent_id", sa.String(length=128), nullable=True),
        sa.Column("element_type", sa.String(length=32), nullable=False),
        sa.Column("sequence_index", sa.Integer(), nullable=False),
        sa.Column("text_content", sa.Text(), nullable=False, server_default=""),
        sa.Column("content_hash", sa.String(length=64), nullable=False),
        sa.Column("bounding_box", sa.JSON(), nullable=True),
        sa.Column("table_data", sa.JSON(), nullable=True),
        sa.Column("asset_storage_key", sa.String(length=512), nullable=True),
        sa.Column("source_uri", sa.String(length=512), nullable=True),
        sa.Column("is_boilerplate", sa.Boolean(), nullable=False, server_default=sa.text("false")),
        sa.Column("boilerplate_reason", sa.String(length=255), nullable=True),
        sa.Column("extra_metadata", sa.JSON(), nullable=False, server_default="{}"),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("now()")),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("now()")),
        sa.ForeignKeyConstraint(["version_id"], ["document_versions.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["page_id"], ["pages.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_elements_version_id", "elements", ["version_id"])
    op.create_index("ix_elements_page_id", "elements", ["page_id"])
    op.create_index("ix_elements_page_number", "elements", ["page_number"])
    op.create_index("ix_elements_element_id", "elements", ["element_id"])
    op.create_index("ix_elements_element_type", "elements", ["element_type"])
    op.create_index("ix_elements_sequence_index", "elements", ["sequence_index"])
    op.create_index("ix_elements_is_boilerplate", "elements", ["is_boilerplate"])

    # 6. chunks table (ready for Stage 3 & 5)
    op.create_table(
        "chunks",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("version_id", sa.Uuid(), nullable=False),
        sa.Column("chunk_index", sa.Integer(), nullable=False),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("token_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("element_ids", sa.JSON(), nullable=False, server_default="[]"),
        sa.Column("primary_page_number", sa.Integer(), nullable=False, server_default="1"),
        sa.Column("bounding_box", sa.JSON(), nullable=True),
        sa.Column("embedding_id", sa.String(length=128), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("now()")),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("now()")),
        sa.ForeignKeyConstraint(["version_id"], ["document_versions.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_chunks_version_id", "chunks", ["version_id"])
    op.create_index("ix_chunks_chunk_index", "chunks", ["chunk_index"])
    op.create_index("ix_chunks_embedding_id", "chunks", ["embedding_id"])


def downgrade() -> None:
    op.drop_table("chunks")
    op.drop_table("elements")
    op.drop_table("pages")
    op.drop_table("document_metadata")
    op.drop_table("document_versions")
    op.drop_table("documents")
