# OLED real-paper vertical resume runner

PR-AJ adds one operational entry point over the existing PR-AA through PR-AI
contracts. It does not replace their validators or publication boundaries.

The runner accepts:

- one exact reviewed-evidence facet review request;
- one complete, human-confirmed facet decision manifest;
- either an existing categorical Gold snapshot or an explicit Gold Registry ID
  from which to create the deterministic generation-0 snapshot; and
- an existing, empty output directory.

It then invokes the existing stages in order:

1. facet adjudication;
2. Gold admission preflight;
3. immutable Gold candidate publication and verification;
4. categorical Gold successor preflight, publication, and verification;
5. categorical dataset admission; and
6. versioned dataset-view materialization, material-group split, and mean
   baseline smoke execution.

No scientific decision is synthesized by this runner. Calling it without a
decision manifest performs a read-only readiness inspection and exits with code
`3` when human facet decisions remain missing:

```bash
PYTHONPATH=src python -m ai4s_agent.oled_real_paper_vertical_run \
  --facet-review-request /path/to/reviewed_evidence_facet_review_request.json
```

A first Gold publication uses an explicit genesis Registry identity:

```bash
mkdir /path/to/empty-run-output
PYTHONPATH=src python -m ai4s_agent.oled_real_paper_vertical_run \
  --facet-review-request /path/to/reviewed_evidence_facet_review_request.json \
  --facet-decisions /path/to/facet_decisions.json \
  --gold-registry-id oled-categorical-gold:production \
  --output-root /path/to/empty-run-output
```

Later runs replace `--gold-registry-id` with
`--current-gold-snapshot /path/to/categorical_gold_snapshot.json`.

The output directory contains all intermediate exact-bound artifacts, both
immutable publication directories, the PR-AI dataset directory, and
`run_summary.json`. The directory must be empty so an operator cannot
accidentally blend evidence from separate invocations.

## Real paper016 status

The operator-local paper016 artifact was inspected with this runner. Its request
validates as seven review groups and 35 eligible observations. No facet decision
manifest is currently present, so the exact readiness result is:

- `status=blocked_on_human_facet_decisions`;
- `missing_decision_count=35`; and
- no downstream Gold or dataset publication was attempted.

This is a real-input readiness result, not a claim that the 35 observations are
scientifically consistent, confidence-sufficient, Gold-eligible, or suitable
for benchmark evaluation.

## Boundary

PR-AJ is orchestration for a useful vertical capability. Every semantic check,
compare-and-swap, immutable publication, post-write verification, admitted-only
dataset routing, and no-replace dataset write remains owned by its original
stage. The runner does not add schema fields, mutate the Material Registry,
perform structure recognition from images, invent numeric confidence scores,
promote a model, or claim training readiness.
