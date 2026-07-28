# Exact-bound supplementary material-identity evidence response

## Purpose

PR-L is the offline validation boundary between the PR-K material-identity
candidate request and a later human material-identity review. It answers only:

> Did an external responder return one safe, complete, exact-group-bound
> identity-evidence disposition for every PR-K paper-local row group; are all
> positive dispositions source-located; and are any proposed chemical graphs
> deterministically parseable?

It does not decide that a proposed graph is the molecule shown by the paper.
Chemical parseability, source support, material identity, and cross-paper
identity are separate claims.

A successful validated artifact has status
`ready_for_human_material_identity_review`. This status means that the response
is safe and complete enough to review; it does not mean that any identity has
been accepted.

## Exact inputs

The validator consumes three independently supplied JSON inputs:

1. the exact PR-K
   `oled_supplementary_material_identity_candidate_request.v1` artifact;
2. the exact PR-J source-transcription review packet already named by the PR-K
   request; and
3. an external material-identity evidence response manifest.

The PR-K request binds the paper-local row groups and dependent cells. The PR-J
packet supplies the already verified source ID, PDF SHA-256, byte size, page
count, and source-evidence digest needed to bound evidence locators. The
validator must check the exact input-file hashes, canonical digests, run/paper
identity, and PR-K-to-PR-J binding rather than trusting copied source metadata.

PR-L accepts evidence locators only in a source PDF already bound by this
chain. A response cannot introduce a main-paper PDF, database record, URL,
local path, or other source. If the needed evidence exists only outside the
bound supplementary PDF, the response must remain unresolved until that source
passes a separate intake and binding workflow.

PR-L does not render or semantically inspect the PDF. The later PR-M human
review must reopen the exact bound PDF and present the cited pages.

## External response manifest

The response manifest must bind:

- manifest version, run ID, and paper ID;
- exact PR-K request file SHA-256 and material-identity request digest;
- exact PR-J review-packet file SHA-256 and source-PDF evidence digest;
- response production timestamp and prompt contract/version/hash;
- external producer provenance; and
- `response_complete=true` plus exactly one `group_results` entry for every
  PR-K identity group.

Producer provenance must distinguish the execution client from the actual
model provider and immutable model snapshot. For example, an operator may use
`claude_cli` as the execution client while the provider/model fields identify a
pinned DeepSeek V4 Pro endpoint. That is an illustrative shape, not a hardcoded
client or model. A client name must never be used to imply that its built-in
model produced the response.

The producer `kind` is exactly `human` or `external_llm_assisted`. The concrete
provenance fields are `client_id`, `model_provider_id`, `model_snapshot_id`,
`prompt_contract_version`, `prompt_sha256`, and `produced_at`. Their causal
order is enforced as
`request.generated_at <= producer.produced_at <= artifact.generated_at`.

Every group response must replay the exact:

- identity group ID and digest;
- scope, table, table digest, and zero-based row;
- reported subject literal and subject-header binding; and
- complete sorted roster of identity-dependent source-cell digests.

Missing, duplicate, extra, moved, merged, or split groups fail closed. The
response must not merge groups because their aliases or proposed structures
look similar.

## Dispositions and evidence anchors

The response contract supports exactly these outcomes:

- `propose_structure_candidate`: propose a chemical graph for later review;
- `record_structure_anchor_only`: preserve useful source-located name, scheme,
  figure, or text evidence without proposing a graph;
- `needs_source_check`: the bound source is insufficient or unreadable;
- `ambiguous_identity`: the evidence supports more than one assignment; and
- `exclude_identity_group`: the row cannot represent an admitted material
  identity, with a bounded reason.

`record_structure_anchor_only`, `needs_source_check`, and
`ambiguous_identity` are complete and valid response outcomes. PR-L must not
require a fixed number of structure proposals and must not reward guessing.

Every evidence anchor contains:

- the already-bound source ID and PDF SHA-256;
- a one-based PDF page within the PR-J page-count boundary;
- an `anchor_kind` of `figure`, `scheme`, `table`, `text`, or
  `structure_diagram`;
- one compatible `singleton_locator`, never a range or list;
- an optional exact `panel_label`;
- one or both sorted, unique evidence roles: `structure_representation` and
  `subject_to_structure_link`;
- an explicit `source_representation_kind` of `authored_description`,
  `smiles_literal`, or `inchi_literal`;
- a required bounded `source_representation`, validated according to that kind;
  and
- a bounded `source_excerpt`, which is required for a `text` anchor.

A positive structure-candidate or anchor-only disposition must include at least
one anchor that carries both evidence roles and repeats the exact reported row
subject as its `panel_label`. Roles split across unrelated anchors, a subject
substring inside `source_representation`, or a broad `source_excerpt` cannot
establish the link. This is still a response-shape claim, not confirmation that
the PDF actually contains or supports it.

URLs, local paths, credentials, executable content, invented source IDs, and
out-of-range pages are invalid. A locator is a response claim for later human
inspection; validating its shape and source binding does not validate that its
content proves the proposed identity.

## Chemical-graph validation boundary

A structure proposal must use one exact `candidate_origin`:

- `source_reported_smiles`;
- `source_reported_inchi`;
- `diagram_derived`; or
- `systematic_name_derived`.

`source_reported_smiles` requires a SMILES encoding and an exact literal match
to its same-anchor `source_representation`; `source_reported_inchi` has the same
rule for standard InChI. `diagram_derived` and `systematic_name_derived` require
SMILES plus, respectively, a subject-bound diagram or text anchor. Legitimate
SMILES `/` and `\\` bond-direction syntax and multi-fragment dots are admitted
only when explicitly typed as `smiles_literal` and strictly replayed by RDKit;
they are not exemptions in the general authored-description safety filter.

The original reported literal, when present, and the derived proposal must be
kept separate. Model-derived SMILES must never be described as source-reported
text.

PR-L requires the pinned RDKit/InChI runtime profile. Every proposal must supply
`structure_encoding_kind`, `structure_candidate_text`,
`canonical_isomeric_smiles_candidate`, and `inchikey_candidate`. RDKit parses
and sanitizes the candidate, derives isomeric canonical SMILES, and recomputes
the InChIKey; both supplied candidate identifiers must match. Invalid graphs or
inconsistent identifiers fail closed.

The profile records findings for multi-fragment structures, formal charge,
unassigned atom or bond stereochemistry, InChI warnings, and a changed
standard-InChI round trip. It does not silently neutralize structures or apply
tautomer standardization, and an InChIKey collision is never an automatic
identity merge. `chemistry_tool_called` is true only when at least one structure
candidate is evaluated, even though runtime provenance is recorded on every
artifact.

The validated fields must remain named as candidates, for example
`canonical_isomeric_smiles_candidate` and `inchikey_candidate`. RDKit validates
that a representation denotes a parseable chemical graph. It does not validate
that the graph matches a diagram, name, row label, or molecule in the source
PDF.

## Validated artifact and status

The validated artifact should record:

- exact request, source packet, response manifest, and producer bindings;
- one validated disposition per identity group;
- all evidence-anchor bindings and validation findings;
- deterministic chemistry-tool profile/version evidence and derived graph
  candidates where applicable;
- exact group and dependent-cell counts plus `structure_candidate_count`,
  `structure_anchor_only_count`, `source_check_count`,
  `ambiguous_identity_count`, `exclusion_proposal_count`,
  `evidence_anchor_count`, `chemistry_validated_candidate_count`, and
  `collision_finding_count`; and
- a canonical validated-response digest.

Complete safe coverage yields
`ready_for_human_material_identity_review`, including when every group is
anchor-only or unresolved. Malformed or incomplete input must fail without
publishing an output.

The artifact must keep these claims false:

- source-to-structure semantic match validated;
- human identity review completed;
- material or canonical identity resolved;
- canonical SMILES or InChIKey assigned;
- alias or cross-paper identity merge;
- Registry or schema write;
- reviewed-evidence staging or direct admission;
- scientific or physical semantics validated;
- Gold, dataset, feature, or training output; and
- device-only admission.

The output remains proposal evidence only and is never training eligible.

Standalone artifact loading rechecks its canonical digest, internally derived
counts, candidate chemistry under the recorded runtime, and fixed downstream
boundaries. It cannot independently reconstruct the PR-K upstream partition or
the authoritative PR-J PDF page count from the PR-L artifact alone. Those
claims must always be revalidated jointly with the exact PR-K request, PR-J
packet, and response manifest. The artifact records this limitation explicitly
with `joint_exact_input_revalidation_required=true` and both standalone
upstream-revalidation capability flags set to false.

## Controlled runner

Validate one externally prepared response manifest:

```bash
PYTHONPATH=src .venv/bin/python \
  -m ai4s_agent.oled_supplementary_material_identity_evidence_response \
  --request-artifact /operator/local/material_identity_candidate_request.json \
  --transcription-review-packet /operator/local/transcription_review_packet.json \
  --response-manifest /operator/local/material_identity_evidence_response_manifest.json \
  --output /operator/local/material_identity_evidence_response.json
```

All inputs must be bounded regular files and the output must be fresh. The
runner rejects symlinks, duplicate JSON keys, non-finite values, input/output
collisions, stale bindings, and publication races. CLI failures expose only a
stable error code and exception type.

## paper016 canary

The real canary must preserve the complete PR-K partition:

```text
7 paper-local identity groups
35 identity-dependent known-property cells
14 ontology-pending cells excluded
0 device-only cells admitted
```

There must be exactly seven group responses and exactly five dependent cells
in each group. The number of structure proposals may be anywhere from zero to
seven; every remaining group must carry an explicit anchor-only, source-check,
ambiguous, or excluded outcome. All evidence locators must use the exact bound
paper016 supplementary PDF and its validated page range.

Supplementary Fig. S27 and the systematic-name material around S48-S51 are
useful future source-anchor candidates, but they are not hardcoded identity
truth and this stage must not freeze any unreviewed paper016 SMILES or InChIKey
as a test fixture. `S48-S51` is descriptive prose here: an actual response must
split evidence into individual singleton locators and pages. In addition, the
mTDBA-Si HRMS formula/value shown around SI p51/S51 appears visually internally
inconsistent. Formula or HRMS agreement must therefore not automatically
confirm identity. That apparent conflict is reserved for explicit human
source-conflict review and does not by itself invalidate the separate S27 or
systematic-name anchors.

Tests may use synthetic, known-valid structures to exercise deterministic
RDKit behavior. The current suite uses a paper016-shaped upstream fixture. Once
a real paper016 response artifact is generated, it may prove only exact response
coverage, source-locator binding, and honest unresolved handling until PR-M
review.

## Next boundary

PR-M may consume the exact PR-K request, PR-J source packet, external response
manifest, successful PR-L validated artifact, and an operator-local PDF whose
SHA-256 matches PR-J. It must reopen that PDF, render every cited full page, and
display the asserted panel label. Because PR-L carries no bounding boxes, PR-M
may crop a panel only if it adds and validates a separate crop locator. It then
generates a compact human review packet plus a separately bound
decision/adjudication artifact.

Only that human stage may decide whether a source anchor supports a proposed
graph or paper-local identity. Registry mutation, cross-paper alias resolution,
schema/observation materialization, Gold admission, and dataset/training writes
remain later explicit stages.
