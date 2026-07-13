# OLED Supplementary-Source Recovery Planning

## Purpose

This module turns a validated OLED needs_source_check result for missing
supplementary information into a narrow, review-only recovery plan. It makes
the evidence gap explicit before anyone searches for, downloads, parses, or
uses a supplementary file.

The plan is not an acquisition request, a review decision, a schema candidate,
or a staging input.

## Inputs and Binding

The planner accepts:

1. an OledLLMPaperMappingRequest, including the supplied document context and
   source packets;
2. an OledLLMContextMappingResult bound to that request's paper id, packet
   set, and request digest.

It considers only packet results whose action is needs_source_check and whose
missing-evidence reason includes supplementary_information.

An explicit target is emitted only when:

- the source packet itself contains the reference; and
- a directly bound element in the supplied document context contains the same
  reference.

The plan records the packet id, source candidate hash/anchor, context element
id/source hash/page, exact matched reference text, and match offsets. It also
records deterministic candidate ids only when they have the same source
candidate hash as the packet.

For this planner, a context element is directly bound only through an exact
source-anchor match, an exact source-hash match, or canonical full-text
equality with one complete source-packet text part. Canonical equality decodes
HTML entities, removes markup tags, normalizes Unicode/case, and removes
whitespace-only parser differences; a substring or a nearby citation is not a
binding.

This prevents a citation elsewhere in the same paper from being silently
assigned to an unrelated source-check packet.

## Locator Status

explicit_reference_found means a directly bound main-text reference identifies
one target, for example:

- Supplementary Table S1
- Supplementary Fig. S27
- Supporting Information Table S1

The resulting item still only asks a reviewer to provide an approved local
supplementary source. It does not fetch anything.

manual_locator_required means that the missing-source issue is real but the
target cannot be safely identified from the supplied evidence. This includes
generic Supplementary Information, unnumbered supplementary tables/figures,
or a bare Table S1 or Fig. S1 on either side of the packet/context binding,
and source-check packets without a bound main-text reference. The planner never
derives a table number from an LLM question.

A range or list such as Supplementary Table S1-S3 or Supplementary Table S1,
S2, and S3 is also manual-only. Its complete cited expression is retained as
an evidence anchor, but the planner does not split it into inferred locators.

Explicit and manual items are independent: if one packet cites Supplementary
Table S1 and also a bare Fig. S2 or generic Supplementary Information, the S1
item is explicit while the unresolved citation remains a separate manual item.
An explicit citation never resolves or suppresses another citation in the same
packet.

## Offline Artifact

Create a file artifact only after the LLM request and mapping result already
exist:

    PYTHONPATH=src .venv/bin/python -m ai4s_agent.oled_supplementary_evidence_recovery \
      --llm-context-request runs/<run_id>/review/oled_llm_context_request.json \
      --llm-context-result runs/<run_id>/review/oled_llm_context_result.json \
      --output runs/<run_id>/review/oled_supplementary_evidence_recovery.json \
      --run-id <recovery-run-id>

The output binds the source request digest, mapping-result digest, document
context digest, and a stable plan digest. The artifact repeats the context
digest and device-only exclusion flag at its top level. It does not store input
paths in the plan.

## Safety Boundary

Every plan and artifact is fixed to:

- review_only=true
- executable=false
- offline_only=true
- no network, external-service, LLM, MinerU, or download side effect
- no automatic candidate merge, reviewed-evidence staging, device-only
  admission, gold creation, or dataset write

The next permitted action is human review and provision or approval of a local
supplementary source through the
[local-source intake gate](oled-supplementary-source-intake.md). A future
acquisition feature must be separately gated and must validate official
provenance, access policy, redirects, content type, PDF bytes, size, page
count, and file hash.

## D01-Style Example

If a source packet and its bound main-text paragraph both state that calculated
values are reported in Supplementary Table S1, the planner emits one table item
for S1 with both evidence bindings. It does not infer any individual
material/value row from that citation; row-level extraction remains a later,
separate source-parse and human-review action.
