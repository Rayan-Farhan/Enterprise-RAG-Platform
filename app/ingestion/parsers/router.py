"""Format Router for document intelligence parsing (Task 1.5, ADR-004).

Routes files by format to the optimal primary parser and manages automatic fallback chains:
  - PDF: DoclingParser (Primary) -> OpenDataLoaderParser (Fallback) -> PyMuPDFParser (Fast Baseline)
  - DOCX, XLSX, PPTX, TXT, MD: OfficeParser
"""

from pathlib import Path

import structlog

from app.ingestion.parsers.base import DocumentParser, ParsedDocument
from app.ingestion.parsers.docling_parser import DoclingParser
from app.ingestion.parsers.office_parser import OfficeParser
from app.ingestion.parsers.opendataloader_parser import OpenDataLoaderParser
from app.ingestion.parsers.pymupdf_parser import PyMuPDFParser

logger = structlog.get_logger(__name__)


class FormatRouter:
    """Intelligent multi-format document parser router with fallback guarantees."""

    def __init__(
        self,
        pdf_primary: DocumentParser | None = None,
        pdf_fallbacks: list[DocumentParser] | None = None,
        office_parser: DocumentParser | None = None,
    ) -> None:
        self.pdf_primary = pdf_primary or DoclingParser()
        self.pdf_fallbacks = pdf_fallbacks or [OpenDataLoaderParser(), PyMuPDFParser()]
        self.office_parser = office_parser or OfficeParser()

    def route_and_parse(
        self, file_path: Path | str, mime_type: str | None = None
    ) -> ParsedDocument:
        """Route input file to the appropriate parser pipeline and execute with fallback."""
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        ext = path.suffix.lower()
        log = logger.bind(filename=path.name, extension=ext)

        # 1. Office / Text Documents
        if ext in (
            ".docx",
            ".doc",
            ".xlsx",
            ".xls",
            ".pptx",
            ".ppt",
            ".txt",
            ".md",
            ".json",
            ".csv",
        ):
            log.info("routing_to_office_parser", parser=self.office_parser.parser_name)
            return self.office_parser.parse(path, mime_type=mime_type)

        # 2. PDF Documents with Fallback Pipeline
        if ext == ".pdf":
            pipeline = [self.pdf_primary] + self.pdf_fallbacks
            errors: list[str] = []

            for parser in pipeline:
                try:
                    log.info("attempting_pdf_parse", parser=parser.parser_name)
                    parsed_doc = parser.parse(path, mime_type=mime_type)

                    # Sanity check: Ensure at least some elements or pages were retrieved
                    if parsed_doc.total_pages > 0 and (
                        parsed_doc.all_elements or parsed_doc.all_tables
                    ):
                        parsed_doc.metadata["fallback_chain_attempts"] = [
                            p.parser_name for p in pipeline[: pipeline.index(parser) + 1]
                        ]
                        if errors:
                            parsed_doc.metadata["prior_parser_errors"] = errors
                            log.warning(
                                "parsed_with_fallback",
                                successful_parser=parser.parser_name,
                                prior_errors=errors,
                            )
                        return parsed_doc

                    err_msg = f"Parser '{parser.parser_name}' produced empty output."
                    log.warning("parser_produced_empty_output", parser=parser.parser_name)
                    errors.append(err_msg)

                except Exception as exc:
                    err_msg = f"Parser '{parser.parser_name}' failed with error: {str(exc)}"
                    log.warning("parser_failed", parser=parser.parser_name, error=str(exc))
                    errors.append(err_msg)

            # If all parsers in pipeline failed, raise runtime exception with collected trace
            raise RuntimeError(
                f"All PDF parser candidates failed for file '{path.name}'. Errors: {'; '.join(errors)}"
            )

        # 3. Default fallback for unknown file types
        log.warning("unknown_file_type_fallback_to_office", extension=ext)
        return self.office_parser.parse(path, mime_type=mime_type)
