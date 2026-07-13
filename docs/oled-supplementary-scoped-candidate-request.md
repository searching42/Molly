# OLED supplementary scoped candidate-proposal request MVP

This stage converts accepted supplementary table locators into a narrowly
scoped, offline request for later semantic candidate proposals. It is a
request-context stage only. It does not accept an LLM response, assign a
property ID, create a schema candidate, or admit data.

## Inputs and binding

The file entry point requires both artifacts from the preceding review chain:

- the complete PR-E supplementary locator review artifact, which contains the
  literal matched table; and
- the PR-F supplementary locator adjudication artifact, which contains the
  human locator decision and any semantic note.

Both files are read as stable regular files with `O_NOFOLLOW`, bounded byte
sizes, duplicate-key rejection, and non-finite JSON rejection. The builder
then verifies:

- matching paper and run identities;
- the PR-F exact-byte SHA-256 and canonical digest binding to PR-E;
- upstream execution, locator-manifest, and preflight bindings;
- exact decision coverage of every PR-E review item; and
- every review-item, source PDF, parsed document, parser backend, locator,
  warning, table ID, page, and table-content digest binding.

Only items with an accepted exact locator and
`eligible_for_later_scoped_candidate_proposal=true` become request scopes.
Rejected and `needs_source_check` items are not copied. Mixed adjudications
are supported, but a run with no eligible item fails closed.

## Request content

Each request scope contains:

- stable source, review-item, and table bindings;
- the complete matched caption, headers, string-valued rows, footnotes, page,
  bounding box, and table-content digest from PR-E;
- the PR-F `semantic_note` and matching `semantic_review_required` flag;
- the molecule/interaction-only dataset scope;
- a versioned, pinned snapshot of the current ontology for those two layers;
  and
- fixed proposal instructions requiring exact row, column, and cell evidence.

Reported labels and cell strings must remain literal. Signs and trailing zeros
such as `0.030` are not normalized. A semantic note is an unresolved issue,
not a correction. In particular, a note about HOMO/LUMO does not authorize the
request stage to swap, reinterpret, or map those columns.

The request also tells a later proposer not to infer canonical identity,
SMILES, or a material role from a row label, not to force unknown properties
into an existing ontology entry, and not to treat unproposed cells as absent.
Device-only records remain outside the current dataset scope.

The source PDF remains authoritative. The copied MinerU/PR-E table is literal
request context but is not marked authoritative because its transcription and
scientific content have not been validated.

## Boundary flags

The artifact is `ready_for_semantic_proposal`, but it remains request-only:

- `response_received=false`;
- `response_validation_implemented=false`;
- `schema_mapping_performed=false`;
- `schema_candidates_created=false`;
- `physical_semantics_validated=false`;
- `candidate_regenerated=false`;
- `automatic_candidate_merge=false`;
- `reviewed_evidence_staging=false`;
- `device_only_admitted=false`;
- `gold_records_created=false`; and
- `dataset_written=false`.

The builder does not read PDFs or parsed output, access the network, call an
external service, call an LLM, or call MinerU. Copying the bound table into the
request does not validate its transcription, scientific content, completeness,
or physical meaning.

## paper016 canary

The real paper016 Supplementary Table S1 chain produced one request scope bound
to the accepted PR-F decision. The request preserved the 7-row, 8-column table
and all 49 numeric cell strings. Checks included trailing-zero examples `2.80`,
`3.30`, and `0.1280`, plus the unchanged semantic note:

```text
HOMO/LUMO labels are preserved as reported but require semantic review
```

No property mapping or candidate object was produced by this canary.

## Run the request builder

```bash
PYTHONPATH=src .venv/bin/python \
  -m ai4s_agent.oled_supplementary_scoped_candidate_request \
  --review-artifact /operator/local/supplementary_locator_review.json \
  --adjudication-artifact /operator/local/supplementary_locator_adjudication.json \
  --output /operator/local/supplementary_scoped_candidate_request.json
```

The output must be fresh and cannot overwrite either input. CLI output is a
redacted status/count summary; it does not print local paths, table content, or
semantic notes.

## Next boundary

A later PR-H may define a separate, exact-byte-bound response contract and
validate externally produced semantic proposals. That stage must keep every
proposal pending human review, preserve source cell strings and semantic notes,
and continue to prohibit automatic merge, evidence staging, device-only
admission, gold creation, and dataset writing.
