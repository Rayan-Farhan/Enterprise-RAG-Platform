"""Ingestion parsers package."""

from app.ingestion.parsers.base import (
    BoundingBox,
    DocumentParser,
    ElementType,
    ParsedDocument,
    ParsedElement,
    ParsedFigure,
    ParsedPage,
    ParsedTable,
)
from app.ingestion.parsers.docling_parser import DoclingParser
from app.ingestion.parsers.office_parser import OfficeParser
from app.ingestion.parsers.opendataloader_parser import OpenDataLoaderParser
from app.ingestion.parsers.pymupdf_parser import PyMuPDFParser
from app.ingestion.parsers.router import FormatRouter

__all__ = [
    "BoundingBox",
    "DocumentParser",
    "DoclingParser",
    "ElementType",
    "FormatRouter",
    "OfficeParser",
    "OpenDataLoaderParser",
    "ParsedDocument",
    "ParsedElement",
    "ParsedFigure",
    "ParsedPage",
    "ParsedTable",
    "PyMuPDFParser",
]
