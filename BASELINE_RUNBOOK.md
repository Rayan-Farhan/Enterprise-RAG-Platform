# Baseline Runbook — `experiment-001-baseline`

**Task 4.5.** Produce the Stage 4 baseline: the Stage 3 pipeline measured over
the full 100-question dev split, committed to `evaluation/results/`, and cited by
every stage from 5 onward.

This is a **multi-day** procedure. Not because the pipeline is slow — a question
takes about 10 seconds — but because the free inference tiers meter by the day
and the run costs more tokens than one day allows. The runner checkpoints after
every question, so the job is: run until the quota stops you, come back
tomorrow, run the same command again.

> **Status (2026-08-19):** the baseline is **done and committed** —
> `evaluation/results/experiment-001-baseline.json`, 100/100 questions,
> `failure_rate` 0.010, evaluated in a single day. It needed one day of quota,
> not the two or three estimated below: a full daily allowance goes further than
> the partial one available when that estimate was made. The resume path is still
> correct — the ceiling is real, it simply was not reached this time.
>
> **What remains:** the judged subset (§5c) and the gate demonstration (§5d).
> The §5d run is already **39/100 banked** and resumes with the same command
> once the daily quota resets.

---

## 1. Why this takes several days

Measured, not estimated — these come from partial runs on 2026-08-18:

| Provider | Role | Free-tier limit | Notes |
|---|---|---|---|
| Gemini | generation (primary) | **20 requests / day** | Exhausts in the first fifth of a run, then the gateway falls back to Groq. |
| Groq | generation (fallback), LLM judge | **200,000 tokens / day**, 8,000 / minute | The binding constraint. |
| Jina | embeddings | not a practical limit | One embed per question. |

**Cost per question:** ~5.4k tokens for generation (a ~4k-token evidence block
plus prompt and completion), plus ~7k more if the LLM judge runs.

**Therefore:**

| Run | Questions | Approx. tokens | Days of Groq budget |
|---|---:|---:|---:|
| Baseline, no judge | 100 | ~540,000 | **~3** |
| Judged subset (`--per-type 2`) | 20 | ~250,000 | ~1.5 |

Raising `EVAL_CONCURRENCY` does not help. Tokens per minute is the constraint,
not round trips; above ~2 the run just spends its time waiting out 429s. Leave it
at the default.

---

## 2. Prerequisites (once)

```bash
make up                                    # postgres, qdrant, minio, redis, rabbitmq, opensearch
make migrate                               # schema to head (0003 = experiment tracking)
python -m scripts.ingest_corpus            # all 8 corpus PDFs: ingest, chunk, index
python -m scripts.build_golden_dataset --strict
make eval-validate SPLIT=dev               # must print "Every expected_evidence pointer resolves"
```

`ingest_corpus` is idempotent — re-running re-embeds nothing. It should report
**8/8 documents**. Confirm with:

```bash
python -m scripts.ingest_corpus --list     # expect 8 documents
```

`.env` needs `GEMINI_API_KEY`, `GROQ_API_KEY`, `JINA_API_KEY`, and
`POSTGRES_PORT=5433` (a host PostgreSQL service occupies 5432 on this machine).

> **Do not run the baseline with `INFERENCE_PROFILE=stub`.** The CLI refuses, and
> that refusal is deliberate: stub output is canned text, and scoring it produces
> numbers that look like evaluation data without being it.

---

## 3. The baseline run

Run this **same command** each day until it exits 0:

```bash
python -m app.evaluation.cli run \
  --name experiment-001-baseline \
  --split dev \
  --no-judge \
  --description "Stage 3 thin RAG: fixed chunking, dense-only Qdrant retrieval, no reranking. Layer 0 and Layer 1 metrics over the full dev split." \
  --notes "Baseline for the whole roadmap. Layer 2 recorded separately in experiment-001-baseline-judged."
```

### What each exit code means

| Exit | Meaning | What to do |
|---:|---|---|
| **0** | Run complete. `evaluation/results/experiment-001-baseline.json` written, checkpoint cleared. | Go to §5. |
| **3** | Provider quota exhausted. Progress is checkpointed. | Wait for the quota to reset, re-run the identical command. |
| **2** | Refused to start (stub profile, or dataset/split problem). | Read the message; fix the cause. Nothing was spent. |
| 1 | Genuine error. | Read the traceback. |

Exit 3 prints how far it got and where the checkpoint is:

```
PROVIDER QUOTA EXHAUSTED — 37/100 questions evaluated.
Progress is checkpointed at evaluation/results/.checkpoints/experiment-001-baseline.jsonl.
Re-run the same command when the quota resets to continue from here.
```

Expect roughly **35–55 questions per day**, so **2–3 days**.

### Checking progress between days

```bash
wc -l evaluation/results/.checkpoints/experiment-001-baseline.jsonl   # questions banked
```

---

## 4. How checkpointing works (read before deviating)

- Every completed question is **appended immediately** to
  `evaluation/results/.checkpoints/experiment-001-baseline.jsonl`, flushed per
  line. Killing the process loses at most the question in flight.
- Re-running the same `--name` **resumes**: banked questions are skipped and
  their latencies and token counts are replayed into the aggregates, so the final
  percentiles describe the whole run rather than the last day.
- **Quota refusals are never banked.** A question the provider declined was not
  measured; it stays in the queue for tomorrow.
- After two consecutive quota refusals the run **stops** rather than marking the
  remaining questions as failures. A 60%-failure run would look like a
  catastrophic quality regression and would really be a billing limit.
- Checkpoints are gitignored. The finished `.json` beside them is what gets
  committed.
- `--restart` discards the checkpoint and starts over. **Only** use this if the
  corpus, the dataset, or the pipeline config changed mid-run — in which case the
  banked questions were measured against a different system and mixing them would
  be silently wrong.

### The one thing to watch

A multi-day run can span a provider's model change. The run file records
`evaluated_at` per question and the CLI prints a note when a run touched more
than one day:

```
NOTE: this run was evaluated across 3 days (2026-08-18, 2026-08-19, 2026-08-20).
```

That is expected here. If the note lists more than ~4 days, consider
`--restart`: the wider the window, the weaker the claim that one system was
measured.

---

## 5. After the baseline completes

### 5a. Sanity-check before committing

```bash
python -m app.evaluation.cli list
```

Check the summary line, and open the JSON for:

- `dataset_size` = **100**
- `system_metrics.failure_rate` — should be near **0.0**. Anything above ~0.05
  means questions failed for non-quota reasons; investigate before committing,
  because this number is the baseline every later stage is compared against.
- `metrics_by_type` — the `multimodal`, `temporal`, `conflicting_versions`, and
  `multi_hop` cells are **expected to be poor**. Those questions carry
  `expected_to_fail_until_stage` and are what Stages 9 and 10 will be measured
  by. Do not "fix" them.
- `evaluation_days` — note it in the commit message.

### 5b. Commit it

```bash
git add evaluation/results/experiment-001-baseline.json
git commit -m "feat(evaluation): record experiment-001-baseline (Task 4.5)"
```

### 5c. The judged subset (Layer 2 numbers)

A separate ~1.5-day run, over a stratified 20-question subset:

```bash
python -m app.evaluation.cli run \
  --name experiment-001-baseline-judged \
  --split dev --per-type 2 \
  --description "Layer 2 LLM-judge reference over two questions of each type." \
  --notes "Subset because the judge costs ~7k tokens per question on top of generation."
```

`--per-type 2` takes two questions of **each** of the ten types. Do not
substitute `--limit 20`: question IDs sort by type, so `--limit 20` yields every
adversarial and ambiguous question and no factual ones at all.

This also resumes on exit 3. Commit it the same way.

### 5d. Prove the regression gate fails (Task 4.6 "done when")

The roadmap requires demonstrating that deliberately degrading retrieval fails
CI. Needs about a third of a day's quota.

```bash
# A deliberately crippled retrieval configuration.
# Already 39/100 banked — re-running the identical command resumes.
RETRIEVAL_TOP_K=1 python -m app.evaluation.cli run \
  --name experiment-001x-topk1 --split dev --no-judge --no-db

make eval-gate RUN_A=experiment-001-baseline RUN_B=experiment-001x-topk1
echo "exit: $?"      # MUST be 1
```

The gate refuses a candidate of a different size, so this run must reach all
100 questions before it can be compared. A partial 39-question record is not a
regression, it is a different measurement, and the gate says so rather than
raising a false alarm.

Then **delete the degraded run** — do not commit it:

```bash
rm evaluation/results/experiment-001x-topk1.json
```

Leaving it in place would make it the newest committed experiment, and CI gates
the newest against the baseline, so every later commit would go red.

---

## 6. Stage 4 exit gate

| Item | State |
|---|---|
| Golden dataset, ten types, dev/validation/locked-test | done — 100 / 41 / 40, every pointer resolves |
| Retrieval, generation (3 layers), system metrics compute | done — 426 tests green |
| Experiment diff CLI works | done |
| CI fails on evaluation regression | job written; everyday "nothing to gate" path verified (exit 0); **the failure case in §5d still to demonstrate** |
| `experiment-001-baseline` committed | done — commit `d6f267d`, recall@5 0.354, failure_rate 0.010 |
| Known-failing categories documented | done — `docs/STAGE_4_EVALUATION.md`, `evaluation/datasets/README.md` |

Stage 5 may now start: the number it must beat exists. It is aimed at `recall@5`
0.354, `mrr` 0.266, `context_precision` 0.257. Weak retrieval starves grounding,
so the pipeline abstains on answerable questions (`answered_when_expected` 0.424)
— that is the gap Stages 5 and 6 exist to close.

---

## 7. If you want it faster

The multi-day procedure exists only because the tiers are free. Any of these
collapses it to a single afternoon:

- A paid Groq or Gemini tier (raises tokens/day by orders of magnitude).
- An OpenRouter key as a third generation provider, spreading the load.
- `INFERENCE_PROFILE=local` with vLLM + TEI — unmetered, and the profile the
  production deployment targets anyway (ADR-045/051). This is the only option
  that also removes the privacy deviation of sending the HR corpus to a hosted
  model.

Switching profile changes what the baseline measures — a local model is a
different generator — so if you take that route, do it **before** producing the
baseline, not after.
