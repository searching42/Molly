# OLED Gold admission preflight (PR-AB)

## Purpose

PR-AB consumes one exact PR-AA facet adjudication and derives candidate-only
Gold admission records from observations whose human decisions are exactly:

```text
scientific_consistency = consistent
confidence_sufficiency = sufficient
```

It does not create or publish Gold records, write a curated dataset, or enable
training.

## Exact input and eligibility replay

The controlled file entry records the supplied PR-AA file SHA-256, validates
its complete embedded PR-U/PR-T/ledger chain, and embeds the validated
adjudication in the output.

PR-AB independently replays every adjudicated observation. Blocked evidence is
counted but never converted into a candidate. It separately reports:

- scientific inconsistency;
- scientific source-check requirements;
- insufficient confidence evidence; and
- confidence source-check requirements.

An empty eligible roster produces an explicit no-eligible-evidence artifact,
not an empty Gold publication.

## Candidate contents

Each candidate preserves:

- reviewed-evidence entry, projection, and source-claim IDs;
- exact PR-AA adjudicated-observation digest;
- Material Registry ID, complete entry, entry digest, and full-payload digest;
- property ID, source label, causal layer, reported literal/precision/unit, and
  normalized value/unit;
- complete or explicitly not-required comparison context;
- PDF SHA-256, page, table, row, column, source-cell digest, and evidence refs;
- human reviewer, timestamp, note, and both categorical facet outcomes.

Candidates are deterministic and candidate-only. They remain eligible only for
a later Gold publication preflight.

## Numeric-confidence incompatibility

The legacy `OledGoldDatasetRecord` contract expects a numeric confidence
assessment. PR-U and PR-AA intentionally use categorical sufficiency because
the project has no calibrated probability model. PR-AB therefore records:

```text
categorical_confidence_only = true
numeric_confidence_score_assigned = false
legacy_numeric_confidence_record_constructed = false
```

It must not recreate the historical arbitrary `0.5` fallback merely to satisfy
the older schema. A later Gold writer should consume this exact categorical
candidate contract or introduce a separately reviewed schema migration.

## CLI

```bash
PYTHONPATH=src .venv/bin/python \
  -m ai4s_agent.oled_gold_admission_preflight \
  --facet-adjudication /operator/local/pr-aa-adjudication.json \
  --output /operator/local/pr-ab-gold-admission-preflight.json
```

The output must be fresh and distinct from the input. Symbolic paths, input
overwrite, changed output parents, timestamp reversal, candidate/count/status
tamper, and partial publication fail closed. CLI failures expose only a stable
error code and exception type.

## Explicitly false after PR-AB

- numeric confidence assignment;
- legacy numeric-confidence Gold record construction;
- Gold creation or publication;
- dataset or training eligibility;
- reviewed-evidence, Registry, or alias mutation;
- source PDF reads; and
- network, external-service, LLM, or MinerU calls.

## Next boundary

The next safe step is a separately authorized immutable Gold candidate
publication boundary. It must not silently translate categorical confidence
into a number. Dataset views, splits, leakage checks, and training remain later
consumers of published Gold rather than side effects of admission.
