# Exact-bound supplementary material-identity candidate request

## Purpose

This stage converts the intersection accepted by PR-I and PR-J into a bounded
request for later material-identity evidence. It answers only:

> Which paper-local table row must eventually be identified before these
> already reviewed property cells can be materialized?

It does not answer which canonical molecule the row denotes. A reported row
label is an alias in one paper, not structure evidence, a canonical name, a
SMILES string, an InChIKey, or permission to merge records across papers.

The artifact version is
`oled_supplementary_material_identity_candidate_request.v1`. A successful
artifact has status `ready_for_material_identity_evidence_proposal`.

## Exact input chain

The builder consumes all nine JSON artifacts from PR-G through PR-J:

1. scoped candidate request;
2. external response manifest;
3. validated response artifact;
4. semantic review packet;
5. semantic decision manifest;
6. semantic adjudication;
7. source-transcription review packet;
8. source-transcription decision manifest; and
9. source-transcription adjudication.

It validates every model, byte SHA-256, canonical digest, run/paper identity,
scope, table, row, source-cell roster, human decision, and timestamp binding.
It then deterministically rebuilds the PR-I and PR-J packets and adjudications.
Copied readiness flags are not trusted.

This stage does not reopen the PDF or parsed document and does not call
Poppler, MinerU, an LLM, a network service, or executable correction code.

## Eligibility and row partition

The input roster is the exact set intersection:

```text
PR-I cells accepted as known-property mappings
AND
PR-J cells inside an accepted bounded source transcription
```

The two independently recorded digest sets must be exactly equal. Missing,
extra, duplicated, moved, ontology-pending, source-check, excluded, rejected,
or otherwise blocked cells fail closed.

Every eligible cell is assigned exactly once to a paper-local row key:

```text
(scope_id, table_id, table_content_digest, row_index,
 reported_subject_text, subject-header binding)
```

`reported_subject_text` is preserved byte-for-byte, but it is not used alone
as a key. Two rows with the same spelling remain two independent groups. No
case folding, punctuation removal, alias normalization, fuzzy matching, or
cross-paper merge is allowed.

Each group binds:

- the authoritative source PDF identity already attested by PR-J;
- the selected table, one-based PDF page, and bounded table digest;
- the exact zero-based row and subject-column binding;
- the reported subject literal;
- the complete, sorted roster of identity-dependent source-cell digests; and
- the PR-I decision item responsible for each member cell.

For a blank first source header, a parser key such as `column_1` remains an
internal positional placeholder. It is rendered as “no explicit source header”
and must not be presented as a source-reported material label.

## Scientific boundary

The request asks a later stage to provide source-located identity evidence for
each row. Acceptable future evidence must be bound to authoritative source
bytes and an exact page/figure/scheme/table/text locator. A structure proposal
must later be independently parsed, canonicalized, cross-checked, and reviewed.

This stage deliberately keeps all of the following false:

- source structure evidence included or validated;
- material identity resolved;
- canonical SMILES or InChIKey assigned;
- cross-paper or automatic identity merge;
- scientific or physical semantics validated;
- schema candidates or reviewed evidence created;
- direct or device-only admission;
- gold records or dataset writes; and
- network, external-service, LLM, or MinerU calls.

Consequently, a successful request is not a dataset record and is not training
eligible.

## paper016 canary

For accepted Supplementary Table S1, the exact result is:

```text
1 accepted source scope
49 bounded transcription-validated numeric cells
35 identity-dependent known-property cells
14 ontology-pending cells excluded from this request
7 paper-local row identity groups
5 dependent cells per group
0 device-only admitted cells
```

The seven groups retain source row order:

1. `TDBA`
2. `TDBA-Ph`
3. `mTDBA-Ph`
4. `mTDBA-2Ph`
5. `TDBA-Si`
6. `mTDBA-Si`
7. `mTDBA-2Si`

Each group contains only the accepted HOMO, LUMO, S1, T1, and Delta-EST cells.
The seven HOMO-LUMO-gap and seven oscillator-strength cells remain in the
ontology-pending count and never enter an identity group.

The unusual reported HOMO/LUMO ordering is preserved as reported. Neither this
request nor a later identity decision can resolve that separate physical
semantics issue.

## CLI

Build a fresh request while replaying the complete chain:

```bash
PYTHONPATH=src .venv/bin/python \
  -m ai4s_agent.oled_supplementary_material_identity_candidate_request build \
  --request-artifact /operator/local/scoped_candidate_request.json \
  --response-manifest /operator/local/response_manifest.json \
  --response-artifact /operator/local/validated_response.json \
  --semantic-review-packet /operator/local/semantic_review_packet.json \
  --semantic-decision-manifest /operator/local/semantic_decisions.json \
  --semantic-adjudication /operator/local/semantic_adjudication.json \
  --transcription-review-packet /operator/local/transcription_review_packet.json \
  --transcription-decision-manifest /operator/local/transcription_decisions.json \
  --transcription-adjudication /operator/local/transcription_adjudication.json \
  --output /operator/local/material_identity_candidate_request.json
```

Render a human-readable summary:

```bash
PYTHONPATH=src .venv/bin/python \
  -m ai4s_agent.oled_supplementary_material_identity_candidate_request render \
  --request-artifact /operator/local/material_identity_candidate_request.json \
  --output-markdown /operator/local/material_identity_candidate_request.md
```

All inputs must be bounded regular files and all outputs must be fresh. The
runner rejects symlinks, duplicate JSON keys, non-finite values, stale or
inconsistent chains, output/input collisions, and publication races. CLI error
output contains only a stable error code and exception type.

## Next boundary

PR-L, documented in
`docs/oled-supplementary-material-identity-evidence-response.md`, validates an
externally supplied evidence response against this exact request and the bound
PR-J source packet. It requires complete group/cell coverage, structured
locators inside the already-bound supplementary PDF, explicit producer
provenance, and deterministic chemistry-tool checks for any proposed graph.

PR-L may retain anchor-only and unresolved outcomes. It must not accept an
LLM-generated structure merely because it is chemically parseable: RDKit can
validate a candidate graph but cannot establish that the graph matches the
source. A separate PR-M PDF-backed human review and adjudication remains
required before any identity resolution, Registry write, schema/observation
materialization, Gold admission, or dataset/training use.
