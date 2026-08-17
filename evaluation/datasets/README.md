# Evaluation Datasets (Task 4.1, ADR-028, master §48–49)

Curated question/answer pairs over the **real** HR corpus in `benchmarks/corpus/`
(8 documents, 370 pages, ~3,300 canonical elements). These files are the
measuring instrument for every stage from 5 onward.

## Splits

| File | Questions | Purpose |
|---|---:|---|
| `golden_dataset_dev_v1.jsonl` | 100 | Tuning. Every experiment from Stage 5 runs here first. |
| `golden_dataset_validation_v1.jsonl` | 41 | Held out. Selects between configurations that already look good on dev. |
| `golden_dataset_test_v1.jsonl` | 40 | **Locked.** Opened exactly once, at Stage 14. |

The test split refuses to load without an explicit unlock token
(`app.evaluation.dataset.TEST_SPLIT_UNLOCK_TOKEN`). Tuning against it would make
the final readiness number a measurement of the tuning rather than of the system.

## Type distribution

All ten master §49 types are represented in all three splits.

| Type | dev | validation | test |
|---|---:|---:|---:|
| `factual` | 20 | 8 | 8 |
| `exact_retrieval` | 12 | 5 | 5 |
| `multi_hop` | 10 | 4 | 4 |
| `negative_unsupported` | 10 | 4 | 4 |
| `multimodal` | 9 | 3 | 3 |
| `ambiguous` | 8 | 4 | 4 |
| `calculation` | 8 | 3 | 3 |
| `temporal` | 8 | 4 | 3 |
| `adversarial` | 8 | 3 | 3 |
| `conflicting_versions` | 7 | 3 | 3 |
| **Total** | **100** | **41** | **40** |

Abstention cases (`negative_unsupported` + `adversarial` + anything with
`must_abstain`): 18 dev, 7 validation, 7 test. These carry no
`expected_evidence` — they are scored on abstention, not on recall.

## Expected failures

Questions carry `expected_to_fail_until_stage` when Stage 3's pipeline
structurally cannot answer them. This is deliberate: it turns "Stage 9 improved
multimodal" from an impression into a delta against a recorded number.

| Marked for | Why Stage 3 cannot answer it | Fixed by |
|---|---|---|
| `multimodal`, and table-dependent `calculation` / `conflicting_versions` | Benefit matrices and the org chart are flattened to pipe-delimited text or lost; the answer lives in the visual structure. | Stage 9 |
| `temporal`, `conflicting_versions` | Three documents published within weeks of each other restate the same policies; nothing in Stage 3 ranks by version or effective date. | Stage 10 |
| `multi_hop` | Stage 3 retrieves one dense neighbourhood per query and never decomposes. | Stage 10 |

## Authoring and regeneration

Questions and reference answers are hand-authored in
`scripts/golden_questions.py`. Document UUIDs, version UUIDs, page numbers, and
section paths are **not** hand-written — they are resolved from the live corpus
by `scripts/build_golden_dataset.py`:

```bash
python -m scripts.ingest_corpus                    # rebuild the corpus
python -m scripts.build_golden_dataset --strict    # rebuild the datasets
make eval-validate SPLIT=dev                       # confirm against PostgreSQL
```

`--strict` fails when any evidence element belongs to no chunk. An element that
is parsed but never chunked cannot be retrieved by *any* configuration, so a
question resting on one measures the chunker rather than the retriever. Cover
pages are the usual culprit — the "Effective March 01, 2026" line on the plan
summaries is parsed and then dropped by fixed-size chunking, which is why the
dating questions point at the matrix footers instead.

Evidence is recorded at **element** granularity, never chunk granularity
(ADR-036): Stage 5 re-chunks the whole corpus, and a dataset keyed on chunk IDs
would be invalidated by the very experiment it exists to measure. `section_path`
is a denormalisation of the chunks that currently contain each element and is
refreshed by a rebuild; `element_ids` are the authoritative key.
