"""Validate ground-truth annotation structure and corpus references (Task 1.2)."""

import json
import sys
from pathlib import Path

GROUND_TRUTH_PATH = Path(
    r"d:\Projects & Certificates\Projects\Enterprise-RAG-Platform\benchmarks\ground_truth\annotations.json"
)
CORPUS_DIR = Path(r"d:\Projects & Certificates\Projects\Enterprise-RAG-Platform\benchmarks\corpus")


def validate_ground_truth() -> bool:
    """Check that annotations JSON is valid and every document exists in the corpus."""
    if not GROUND_TRUTH_PATH.exists():
        print(f"ERROR: Ground truth file not found at {GROUND_TRUTH_PATH}")
        return False

    try:
        data = json.loads(GROUND_TRUTH_PATH.read_text(encoding="utf-8"))
    except Exception as exc:
        print(f"ERROR: Failed to parse JSON: {exc}")
        return False

    documents = data.get("documents", [])
    if not documents:
        print("ERROR: No documents found in ground truth.")
        return False

    errors = 0
    total_annotations = 0

    for doc in documents:
        filename = doc.get("filename")
        if not filename:
            print("ERROR: Document entry missing 'filename'.")
            errors += 1
            continue

        file_path = CORPUS_DIR / filename
        if not file_path.exists():
            print(f"ERROR: Referenced file '{filename}' does not exist in corpus directory.")
            errors += 1

        annotations = doc.get("annotations", [])
        total_annotations += len(annotations)

        for page in annotations:
            p_num = page.get("page_number")
            if p_num is None or p_num < 1:
                print(f"ERROR: Invalid page number '{p_num}' in {filename}.")
                errors += 1

            for table in page.get("tables", []):
                num_rows = table.get("num_rows", 0)
                cells = table.get("cells", [])
                if len(cells) != num_rows:
                    print(
                        f"ERROR: Table '{table.get('title')}' in {filename} p{p_num} declared {num_rows} rows but has {len(cells)} cells."
                    )
                    errors += 1

    if errors == 0:
        print(
            f"SUCCESS: Ground truth valid. {len(documents)} documents, {total_annotations} annotated pages verified."
        )
        return True
    else:
        print(f"FAILED: Found {errors} validation errors in ground truth annotations.")
        return False


if __name__ == "__main__":
    success = validate_ground_truth()
    sys.exit(0 if success else 1)
