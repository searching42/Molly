# OLED reviewed-evidence facet adjudication (PR-AA)

## Purpose

PR-AA records one exact-bound human decision for every observation in a PR-U
facet review request. It closes only two categorical review facets:

- scientific consistency; and
- confidence sufficiency for later Gold consideration.

It does not create a Gold record, mutate the reviewed-evidence ledger, write a
dataset, enable training, or invent a numeric confidence score.

## Exact inputs and complete roster

The controlled file entry consumes:

1. one exact `oled_reviewed_evidence_facet_review_request.v1` file; and
2. one complete `oled_reviewed_evidence_facet_decision_manifest.v1`.

The decision manifest binds the request file SHA-256 and semantic digest, the
carried PR-T file SHA-256 and semantic digest, reviewer identity, and a
timezone-aware review timestamp. Every decision copies:

- review group ID and group digest;
- ledger entry ID and observation digest;
- one scientific-consistency disposition;
- one confidence-sufficiency disposition; and
- a required reviewer note.

Decision coverage must equal the complete PR-U observation roster. Missing,
extra, duplicated, reordered, stale-group, or stale-observation decisions fail
closed.

## Decision semantics

Scientific consistency allows:

```text
consistent | inconsistent | needs_source_check
```

Confidence sufficiency allows:

```text
sufficient | insufficient | needs_source_check
```

Only the exact pair below becomes eligible for a later Gold admission
preflight:

```text
scientific_consistency = consistent
confidence_sufficiency = sufficient
```

All other combinations retain explicit blockers:

- `scientific_consistency_inconsistent`;
- `scientific_consistency_source_check_required`;
- `confidence_evidence_insufficient`; or
- `confidence_source_check_required`.

This eligibility flag is not direct Gold admission. A later boundary must
replay the exact source/Registry/reviewed-evidence/facet chain and construct any
Gold candidate separately.

## CLI

```bash
PYTHONPATH=src .venv/bin/python \
  -m ai4s_agent.oled_reviewed_evidence_facet_adjudication \
  --request-artifact /operator/local/pr-u-request.json \
  --decision-manifest /operator/local/pr-aa-decisions.json \
  --output /operator/local/pr-aa-adjudication.json
```

Inputs must be distinct regular files and the output must be fresh. Symbolic
input/output paths, input overwrite, changed output parents, exact-byte
binding changes, timestamp reversal, partial rosters, and digest tampering
fail closed. CLI failures expose only a stable error code and exception type.

## Explicitly false after PR-AA

- reviewed-evidence mutation;
- direct Gold admission or Gold record creation;
- numeric confidence assignment;
- dataset or training eligibility;
- Registry or alias mutation;
- device-only admission; and
- network, external-service, or LLM calls.

## Next boundary

The next safe step is a Gold admission preflight over only the observations
whose two human facets passed. It must preserve blocked observations as
reviewed evidence, not delete or reinterpret them, and must keep Gold
publication separate from dataset-view generation.
