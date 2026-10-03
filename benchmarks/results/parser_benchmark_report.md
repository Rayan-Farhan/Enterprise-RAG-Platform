# Parser Benchmark Report (ADR-004)

**Generated:** 2026-10-04 00:27 · `python -m benchmarks.parser_benchmark`

## Coverage

* **Full-text reference:** 8 PDFs, 370 pages, 159,879 words — every page, OCR of rendered page images (tesseract v5.4.0.20240606, 300 dpi).
* **Hand annotation:** 23 pages across all 8 PDFs: 59 headings, 11 tables / 131 cells, and 11 pages with every paragraph verbatim (3,434 words).
* **Reference error margin:** the OCR reference recovers 99.6% of the words on the 11 verbatim hand pages. Text recall near that ceiling is as good as this reference can show; the hand-page column has no OCR in it.

## Parsers

* `pymupdf-layout` — PyMuPDF + typography heuristics (production primary)
* `pymupdf-columns` — PyMuPDF + column-sorted blocks (production fallback)
* `pymupdf` — PyMuPDF text blocks, no typing (baseline)
* `docling` — Docling layout + TableFormer models, OCR off
* `opendataloader` — OpenDataLoader PDF, rule-based (Java)

## Scorecard

| metric | `pymupdf-layout` | `pymupdf-columns` | `pymupdf` | `docling` | `opendataloader` |
|---|---:|---:|---:|---:|---:|
| Text recall vs OCR (micro) | 98.4% | 98.4% | 98.5% | 98.2% | 98.1% |
| Text recall vs OCR (macro, per doc) | 95.5% | 95.5% | 95.8% | 95.6% | 95.1% |
| Text recall vs hand pages | 99.9% | 99.9% | 99.9% | 99.6% | 98.8% |
| Text precision (1 − extra words) | 98.8% | 98.8% | 94.2% | 98.6% | 99.1% |
| Reading-order similarity | 96.9% | 94.0% | 96.1% | 96.7% | 97.0% |
| Paragraphs intact (≥95% in order) | 99.5% | 99.5% | 99.5% | 98.4% | 93.1% |
| Heading precision | 25.0% | 80.0% | 88.9% | 89.5% | 85.7% |
| Heading recall | 32.2% | 13.6% | 13.6% | 86.4% | 50.8% |
| Heading F1 | 28.1% | 23.2% | 23.5% | 87.9% | 63.8% |
| Elements typed heading (corpus) | 29.1% | 3.8% | 3.3% | 16.1% | 9.4% |
| Tables found (annotated pages) | 90.9% | 90.9% | 90.9% | 81.8% | 90.9% |
| Table precision (annotated pages) | 83.3% | 83.3% | 90.9% | 69.2% | 100.0% |
| Cell recall (multiset) | 78.6% | 78.6% | 78.6% | 62.6% | 80.2% |
| Cell recall (by position) | 66.4% | 66.4% | 66.4% | 59.5% | 83.2% |
| Page + box provenance | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |
| Speed (ms/page) | 221 | 235 | 235 | 3,631 | 95 |

## Per document (micro within the document)

**text recall**

| document | `pymupdf-layout` | `pymupdf-columns` | `pymupdf` | `docling` | `opendataloader` |
|---|---:|---:|---:|---:|---:|
| `bcbs_dental_booklet_2026.pdf` | 96.0% | 96.0% | 96.0% | 95.9% | 95.8% |
| `bcbs_health_booklet_2026.pdf` | 97.7% | 97.7% | 97.7% | 97.5% | 97.5% |
| `dental_plan_at_a_glance_2026.pdf` | 80.7% | 80.7% | 82.3% | 82.1% | 80.9% |
| `health_plan_at_a_glance_2026.pdf` | 93.3% | 93.3% | 93.5% | 93.2% | 91.5% |
| `organizational_structure.pdf` | 97.9% | 97.9% | 97.9% | 97.9% | 96.8% |
| `staff_handbook.pdf` | 99.4% | 99.4% | 99.6% | 99.4% | 99.4% |
| `una-faculty-handbook-2026-27-initial-version.8-1-26.pdf` | 99.9% | 99.9% | 99.9% | 99.3% | 99.8% |
| `university_employee_policy_manual_and_handbook.pdf` | 99.1% | 99.1% | 99.5% | 99.2% | 98.7% |

**heading F1**

| document | `pymupdf-layout` | `pymupdf-columns` | `pymupdf` | `docling` | `opendataloader` |
|---|---:|---:|---:|---:|---:|
| `bcbs_dental_booklet_2026.pdf` | 69.2% | 37.5% | 37.5% | 96.3% | 100.0% |
| `bcbs_health_booklet_2026.pdf` | 82.4% | 50.0% | 50.0% | 100.0% | 88.9% |
| `dental_plan_at_a_glance_2026.pdf` | 0.0% | 28.6% | 28.6% | 66.7% | 40.0% |
| `health_plan_at_a_glance_2026.pdf` | n/a | n/a | n/a | 85.7% | 50.0% |
| `organizational_structure.pdf` | 0.0% | n/a | n/a | 0.0% | 0.0% |
| `staff_handbook.pdf` | 5.1% | n/a | n/a | 96.0% | 14.3% |
| `una-faculty-handbook-2026-27-initial-version.8-1-26.pdf` | 18.2% | 25.0% | 28.6% | 100.0% | 90.9% |
| `university_employee_policy_manual_and_handbook.pdf` | 7.4% | n/a | n/a | 71.4% | n/a |

**cell recall by position**

| document | `pymupdf-layout` | `pymupdf-columns` | `pymupdf` | `docling` | `opendataloader` |
|---|---:|---:|---:|---:|---:|
| `bcbs_dental_booklet_2026.pdf` | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |
| `bcbs_health_booklet_2026.pdf` | 81.2% | 81.2% | 81.2% | 100.0% | 100.0% |
| `dental_plan_at_a_glance_2026.pdf` | 53.3% | 53.3% | 53.3% | 0.0% | 100.0% |
| `health_plan_at_a_glance_2026.pdf` | 80.6% | 80.6% | 80.6% | 38.7% | 100.0% |
| `organizational_structure.pdf` | n/a | n/a | n/a | n/a | n/a |
| `staff_handbook.pdf` | 0.0% | 0.0% | 0.0% | 100.0% | 0.0% |
| `una-faculty-handbook-2026-27-initial-version.8-1-26.pdf` | n/a | n/a | n/a | n/a | n/a |
| `university_employee_policy_manual_and_handbook.pdf` | n/a | n/a | n/a | n/a | n/a |

## How to read this

* All parsers read the same born-digital PDFs; none runs OCR. Text recall below the OCR ceiling is text the parser lost or garbled.
* Text precision below 100% means words the reference does not have on that page: duplicated blocks, hidden or overprinted text in the PDF, or words OCR missed.
* Heading scores cover the annotated pages only (complete inventories). A parser heading that matches an annotated table cell is neutral.
* Speed is wall-clock for one sequential parse per document on the development laptop (Intel i3-1215U, 8 GB, no GPU), including model loading for the first Docling document.
