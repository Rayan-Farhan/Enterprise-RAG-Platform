"""Base schemas and protocol for document parsers (Task 1.3, ADR-004, ADR-005)."""

from enum import StrEnum
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel, Field


class ElementType(StrEnum):
    """Canonical element types."""

    HEADING = "heading"
    PARAGRAPH = "paragraph"
    TABLE = "table"
    FIGURE = "figure"
    IMAGE = "image"
    LIST = "list"
    FORMULA = "formula"
    HEADER = "header"
    FOOTER = "footer"


class BoundingBox(BaseModel):
    """Normalized or absolute bounding box coordinates."""

    x0: float = Field(description="Left coordinate")
    y0: float = Field(description="Top coordinate")
    x1: float = Field(description="Right coordinate")
    y1: float = Field(description="Bottom coordinate")
    page_number: int = Field(description="1-indexed page number")
    unit: str = Field(default="pt", description="Coordinate unit (pt, px, normalized)")

    @property
    def width(self) -> float:
        return max(0.0, self.x1 - self.x0)

    @property
    def height(self) -> float:
        return max(0.0, self.y1 - self.y0)


class ParsedElement(BaseModel):
    """An individual extracted structural element."""

    element_id: str
    element_type: ElementType
    text: str
    page_number: int
    bounding_box: BoundingBox | None = None
    level: int | None = Field(default=None, description="Heading level (1=H1, 2=H2, etc.)")
    sequence_index: int = 0
    parent_id: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class ParsedTable(BaseModel):
    """Structured table element."""

    table_id: str
    title: str | None = None
    page_number: int
    num_rows: int
    num_cols: int
    headers: list[str] = Field(default_factory=list)
    cells: list[list[str]] = Field(default_factory=list)
    bounding_box: BoundingBox | None = None
    markdown: str = ""


class ParsedFigure(BaseModel):
    """Extracted figure or image asset."""

    figure_id: str
    caption: str | None = None
    page_number: int
    bounding_box: BoundingBox | None = None
    image_bytes: bytes | None = None
    format: str = "png"


class ParsedPage(BaseModel):
    """Page representation containing its constituent elements."""

    page_number: int
    width: float = 595.0
    height: float = 842.0
    elements: list[ParsedElement] = Field(default_factory=list)
    tables: list[ParsedTable] = Field(default_factory=list)
    figures: list[ParsedFigure] = Field(default_factory=list)
    raw_text: str = ""


class ParsedDocument(BaseModel):
    """Canonical parser intermediate output decoupling parsers from database models."""

    filename: str
    file_type: str
    total_pages: int
    pages: list[ParsedPage] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
    parser_name: str
    parsing_duration_ms: float = 0.0

    @property
    def all_elements(self) -> list[ParsedElement]:
        """Flattened list of all elements across all pages."""
        return [el for page in self.pages for el in page.elements]

    @property
    def all_tables(self) -> list[ParsedTable]:
        """Flattened list of all tables across all pages."""
        return [tbl for page in self.pages for tbl in page.tables]

    @property
    def all_figures(self) -> list[ParsedFigure]:
        """Flattened list of all figures across all pages."""
        return [fig for page in self.pages for fig in page.figures]


@runtime_checkable
class DocumentParser(Protocol):
    """Uniform parser adapter interface."""

    parser_name: str

    def parse(self, file_path: Path | str, mime_type: str | None = None) -> ParsedDocument:
        """Parse a document file into a canonical ParsedDocument representation."""
        ...
