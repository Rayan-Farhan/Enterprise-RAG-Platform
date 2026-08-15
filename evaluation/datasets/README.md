# Evaluation Datasets (ADR-028, Master Plan §48–49)

This directory holds the versioned golden datasets and evaluation splits.

## Dataset Splits

1. **`golden_dataset_dev_v1.jsonl`**: ~100–200 curated questions used for configuration tuning and iterative prompt/retrieval optimization.
2. **`golden_dataset_validation_v1.jsonl`**: Held-out split used for selecting pipeline winners without data contamination.
3. **`golden_dataset_test_v1.jsonl`**: Locked test set, evaluated strictly once at Stage 14 for final readiness qualification.

## Required Question Types (10 Categories)

- `FACTUAL`: Single-hop factual lookup.
- `EXACT_RETRIEVAL`: Precise clause numbers, acronyms, or specific dates.
- `MULTI_HOP`: Requires combining evidence from multiple sections or documents.
- `AMBIGUOUS`: Underspecified questions requiring clarification rather than assumptions.
- `NEGATIVE_UNSUPPORTED`: Questions with no supporting evidence in the corpus (must trigger abstention).
- `TEMPORAL`: Version-dependent questions ("What was the policy in 2024?").
- `CONFLICTING_VERSIONS`: Questions involving superseded vs. active policies.
- `CALCULATION`: Date and numerical policy calculations.
- `MULTIMODAL`: Table/chart/figure-dependent questions.
- `ADVERSARIAL`: Prompt injection attempts, jailbreaks, and unauthorized data extraction.
