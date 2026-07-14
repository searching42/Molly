# Exact-bound supplementary material-identity human review

## Purpose

PR-M is the PDF-backed human review boundary after the validated PR-L
material-identity evidence response. It answers only:

> For each exact PR-K paper-local row group, what does the human reviewer
> conclude about the cited source anchors and, when one was proposed, whether
> the exact candidate graph is supported by the reviewed paper-local evidence?

The source PDF remains authoritative. PR-L proves response shape, source
allowlisting, complete group coverage, and deterministic chemical
parseability; it does not prove that a cited page contains the asserted
evidence or that a proposed graph matches the paper. PR-M reopens the exact PDF
to perform that missing source check.

An accepted result is paper-local evidence only. It is not a Registry record,
a cross-paper alias merge, a globally canonical identity, a schema or
observation write, a Gold record, or dataset/training admission.

## Exact input chain

Packet generation consumes five independently supplied inputs:

1. the exact PR-K material-identity candidate request;
2. the exact PR-J source-transcription review packet bound by PR-K;
3. the original external PR-L response manifest;
4. the successful PR-L validated-response artifact; and
5. an operator-local copy of the authoritative supplementary PDF.

The builder must jointly replay the PR-L request, source-packet, response, and
validated-artifact bindings rather than trusting copied readiness flags. It
must check every input file SHA-256, canonical digest, run and paper identity,
source ID, source-PDF hash, identity group, row, subject literal, dependent
cell roster, response disposition, evidence-anchor digest, chemistry result,
and collision finding.

The operator-local PDF must be opened without following symlinks as a stable
regular file. Its complete SHA-256, byte size, and page count must match the
PR-J source-PDF evidence. A changed, substituted, truncated, out-of-range, or
concurrently modified PDF fails closed. Local paths are never copied into the
canonical packet or printed by the CLI.

## Source-page render set

The PR-M source-page set is derived exactly as:

```text
all PR-K group table-context pages
UNION
all PR-L evidence-anchor pages
```

Pages are sorted and deduplicated before rendering. A table-context page must
be present for every group even when the group has no evidence anchor. It
shows where the reported row literal and its property cells originated, but
the packet must label it explicitly as **table context, not material-identity
evidence**. It cannot satisfy an anchor check or a candidate-graph check unless
the PR-L response independently cited that same page as an evidence anchor.

Every page in the union is rendered from the exact PDF as a full-page, 200 dpi,
8-bit RGB PNG under a pinned Poppler runtime contract. The packet records the
PDF hash, one-based page, renderer/profile identity, executable evidence,
dimensions, PNG byte size and SHA-256, and a deterministic page-asset digest.
The asset bundle must have exactly one PNG per page in the derived union.

PR-L v1 supplies no bounding boxes. PR-M therefore does not infer, OCR-locate,
or crop a panel, structure, caption, or text passage. The reviewer sees the
full page together with the claimed singleton locator and panel label. If that
is insufficient to locate or read the asserted evidence, the result is
`not_checked`/`needs_source_check`, not a guessed crop. A future crop feature
would require a separately validated crop locator and crop digest while still
retaining the full page; it is outside this contract.

The PDF is authoritative and the PNG is its bound review projection. At
adjudication time the runner must reopen the PDF, rederive the page union,
rerender every page, and require byte equality with the packet asset bundle.
Merely accepting previously generated PNG files is insufficient.

## Exact-bound candidate depictions

Every `propose_structure_candidate` group also receives one deterministic
RDKit 2D PNG derived from that group's exact PR-L `structure_candidate_text`.
The depiction is a reviewer aid, never source evidence. Its binding records at
least:

- identity group ID and digest;
- validated PR-L result ID and digest;
- candidate origin and encoding kind;
- exact candidate text and candidate chemistry-validation digest;
- recorded RDKit/InChI runtime identity;
- fixed depiction profile and dimensions; and
- PNG byte size, SHA-256, and deterministic depiction-asset digest.

The depiction must be regenerated under the exact recorded chemistry/runtime
boundary and must remain bound to the same candidate graph. An image supplied
by the external responder, a depiction generated from a different normalized
string, or a stale image from another group is invalid. Anchor-only,
source-check, ambiguous, and exclusion-proposal groups do not receive a
candidate depiction.

The Markdown labels every RDKit image as **untrusted candidate depiction - not
source evidence**. A visually plausible drawing, successful RDKit parse, or
matching InChIKey candidate does not establish that the paper reports that
graph.

## Reviewer-facing Markdown

The packet is evidence-first and compact. It must use this order:

1. **Boundary summary** - paper/run, exact input hashes and digests, source
   identity, PDF hash/page count, producer client/provider/model provenance,
   the complete PR-K/PR-L counts, and a prominent proposal-only warning.
2. **Source-page gallery** - each deduplicated full-page PNG exactly once,
   ordered by page number. Each entry lists whether it is table context,
   identity evidence, or both, plus all referring group labels, singleton
   locators, and asserted panel labels.
3. **Group overview** - one row per paper-local identity group showing a stable
   human label, exact reported subject, PR-L disposition, anchor count,
   candidate presence, chemistry/collision warning state, and allowed human
   outcomes.
4. **Group evidence sections** - exact row/table/group binding, the dependent
   property-header summary and cell count, then every evidence anchor with its
   anchor digest, page link, kind, singleton locator, panel label, roles,
   representation kind/text, excerpt, and response note or reason.
5. **Untrusted candidate section** - only after all source evidence for that
   group, show candidate origin, original candidate text, canonical isomeric
   SMILES candidate, InChIKey candidate, deterministic chemistry findings,
   collision findings, and the exact-bound RDKit 2D PNG.

The UI never presents a candidate before the relevant source pages, never
calls a candidate a canonical identity, never displays model confidence as
evidence, and never implies that two equal candidate keys authorize a merge.
Reported subject literals stay byte-for-byte unchanged.

## One group decision with per-anchor checks

There is exactly one human decision entry per identity group, not one decision
per dependent property cell. The entry nevertheless contains one tri-state
result for every exact PR-L evidence-anchor digest:

```text
supports_claim
does_not_support_claim
not_checked
```

`supports_claim` means the reviewer found the cited source content on the
bound full page and confirmed that it supports the asserted paper-local
subject-to-structure link represented by that anchor. `does_not_support_claim` means the
page is readable but the asserted locator, panel label, representation, or
link does not agree. `not_checked` means the full-page evidence is
insufficient or unreadable. Missing, extra, duplicated, stale, or reordered
anchor results fail closed.

A structure-candidate group has one additional graph check:

```text
matches_source
does_not_match_source
not_checked
not_applicable
```

This check compares the exact PR-L candidate and its RDKit depiction with the
verified source evidence. It is not applicable to a group without a structure
candidate.

The compatible group outcomes are:

- `accept_structure_candidate`: allowed only for a structure candidate when
  every asserted anchor is `supports_claim` and the candidate graph check is
  `matches_source`;
- `accept_structure_anchor_only`: allowed only for an anchor-only response
  when every asserted anchor is `supports_claim`; it retains
  useful paper-local evidence but does not resolve a graph;
- `confirm_source_check`: confirms the exact PR-L source-check outcome without
  making it positive;
- `confirm_ambiguous_identity`: confirms the exact PR-L ambiguous outcome
  without making it positive;
- `accept_identity_exclusion`: allowed only for a PR-L
  `exclude_identity_group` proposal under its separately compatible evidence
  checks; it is the only outcome in this stage that confirms that negative
  group disposition;
- `needs_source_check`: retains the group for a new or better exact-bound
  source review and requires an explanatory note; and
- `reject_response_evidence`: rejects the supplied response evidence as
  unsupported or incorrect and requires an explanatory note.

`reject_response_evidence` is **not** an identity-group exclusion. It says the
external response cannot be relied on; it returns the group to an unresolved
state and cannot silently remove the paper row from later consideration.
Likewise, `confirm_source_check` and `confirm_ambiguous_identity` cannot upgrade
an unresolved PR-L disposition,
and a negative or unresolved response cannot be converted into positive
identity evidence by editing the decision manifest.

The decision manifest binds the exact review-packet file SHA-256 and canonical
digest, every page and candidate asset digest, reviewer identity, timezone-aware
timestamp, one complete decision per group, all anchor results, and the graph
check when applicable. No positive outcome is preselected in the packet.

## Adjudication boundary

Adjudication replays the complete PR-K/PR-J/PR-L chain, revalidates the human
manifest, reopens and rerenders the exact PDF, regenerates every candidate
depiction, and checks complete group/anchor coverage before producing an
artifact.

A positive adjudication may establish only that specified evidence anchors,
and optionally one exact candidate graph, were accepted for the exact
paper-local row. It may retain the accepted candidate SMILES/InChIKey as
human-reviewed candidate representations, but it does not assign a global
canonical identity.

All of the following remain false or disabled:

- Registry mutation or canonical material-ID assignment;
- alias normalization or cross-paper identity merge;
- automatic merge based on candidate SMILES or InChIKey collisions;
- ontology extension, schema-candidate creation, or observation
  materialization;
- reviewed-evidence staging or direct/device-only admission;
- Gold record creation, curated-dataset writing, feature generation, or
  training eligibility; and
- network, external-service, LLM, MinerU, or model-generated script execution.

These are later explicit boundaries, not side effects of PR-M acceptance.

## Controlled workflow

Generate the exact-bound JSON packet and all review assets:

```bash
PYTHONPATH=src .venv/bin/python \
  -m ai4s_agent.oled_supplementary_material_identity_review packet \
  --request-artifact /operator/local/material_identity_candidate_request.json \
  --transcription-review-packet /operator/local/transcription_review_packet.json \
  --response-manifest /operator/local/material_identity_evidence_response_manifest.json \
  --response-artifact /operator/local/material_identity_evidence_response.json \
  --source-pdf /operator/local/supplementary_information.pdf \
  --poppler-bin-dir /operator/trusted/poppler/bin \
  --asset-dir /operator/local/material_identity_review/assets \
  --output /operator/local/material_identity_review/review_packet.json
```

Render the validated reviewer-facing Markdown beside its `assets` directory:

```bash
PYTHONPATH=src .venv/bin/python \
  -m ai4s_agent.oled_supplementary_material_identity_review render \
  --review-packet /operator/local/material_identity_review/review_packet.json \
  --asset-dir /operator/local/material_identity_review/assets \
  --output-markdown /operator/local/material_identity_review/review_packet.md
```

After completing every group decision, build the adjudication artifact:

```bash
PYTHONPATH=src .venv/bin/python \
  -m ai4s_agent.oled_supplementary_material_identity_review adjudicate \
  --request-artifact /operator/local/material_identity_candidate_request.json \
  --transcription-review-packet /operator/local/transcription_review_packet.json \
  --response-manifest /operator/local/material_identity_evidence_response_manifest.json \
  --response-artifact /operator/local/material_identity_evidence_response.json \
  --source-pdf /operator/local/supplementary_information.pdf \
  --poppler-bin-dir /operator/trusted/poppler/bin \
  --review-packet /operator/local/material_identity_review/review_packet.json \
  --decision-manifest /operator/local/material_identity_review/decisions.json \
  --asset-dir /operator/local/material_identity_review/assets \
  --output /operator/local/material_identity_review/adjudication.json
```

The runner rejects unsafe or stale inputs without publishing partial packet,
asset, Markdown, or adjudication outputs.

Each file entry pins the output and asset-parent directories with descriptors
before beginning PDF or RDKit work. Final directory creation, file linking,
asset reads, validation, and owned-output rollback remain relative to those
same descriptors. If an ancestor or the named parent is renamed, replaced, or
redirected through a symbolic link during the operation, publication fails;
the runner does not follow the replacement and does not leave output in the
redirected location.

## paper016 canary boundary

The automated canary must remain paper016-shaped and preserve:

```text
7 paper-local identity groups
35 identity-dependent known-property cells
14 ontology-pending cells excluded
0 device-only cells admitted
```

Every group contributes its Table S1 context page, so the shared PDF page 38
is rendered once and labeled non-identity context. Synthetic PR-L responses
may cite page 27/Supplementary Fig. S27 and use known-valid synthetic
structures to exercise anchor tri-states, exact-bound RDKit depictions,
candidate checks, collisions, and unresolved outcomes. Such tests prove packet
partitioning and binding behavior, not the identity of any paper016 material.

The operator-local paper016 supplementary PDF is available for a real render
feasibility check and is bound as:

```text
SHA-256: b1d775a3eb59969ed170a81ea5e72d40a1c87833d1370a369807c9bb30d6f59b
page count: 54
byte size: 5266901
```

Its PDF pages 27, 48, 50, and 51 can support a future response-shaped render
check for Fig. S27 and systematic-name evidence, while page 38 supplies the
table context. Those pages are not hardcoded identity truth; the actual page
union must always be derived from the supplied PR-L anchors. The apparent
mTDBA-Si formula/HRMS conflict around page 51 must remain an explicit source
conflict and must never be converted into automatic confirmation.

There is not yet a real paper016 PR-L response manifest and validated artifact.
Therefore PR-M may currently claim only a paper016-shaped automated canary and
real-PDF rendering feasibility. It must not claim a completed real paper016
end-to-end identity adjudication until those exact PR-L inputs exist and a
human completes this review.

## Fail-closed acceptance criteria

Packet generation or adjudication fails without publishing output when:

- any PR-K, PR-J, PR-L, group, row, cell, response, or digest binding changes;
- the PDF hash, byte size, page count, or stable-file identity differs;
- the rendered page set differs from the exact context-plus-anchor union;
- a page is cropped, an unbound preview replaces the full page, or a locator
  or bbox is inferred;
- a page or RDKit depiction asset is missing, extra, stale, substituted, or
  has different bytes;
- an output or asset-parent path is replaced or redirected while packet,
  Markdown, or adjudication work is in progress;
- a candidate appears without its exact-bound RDKit 2D PNG;
- a group, anchor tri-state, candidate graph check, or decision is missing,
  duplicated, incompatible, or stale;
- acceptance lacks a verified supporting anchor or, for a graph candidate, a
  verified graph match;
- response rejection is treated as identity exclusion;
- a collision performs or implies an automatic merge; or
- any Registry, cross-paper, schema, Gold, dataset, training, or device-only
  boundary is crossed.

## Next boundary

After PR-M, a separate material-identity/Registry stage may decide how an
accepted paper-local candidate relates to existing canonical entities and
aliases. That stage must define collision, conflict, versioning, and
cross-paper merge rules explicitly. Only later observation-materialization and
Gold-admission gates may connect a resolved Registry identity to the already
reviewed property cells.
