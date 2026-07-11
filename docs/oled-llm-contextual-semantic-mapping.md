# OLED LLM Contextual Semantic Mapping

This module is the review-only LLM layer between deterministic MinerU semantic mapping and human dataset-admission review. It addresses tables whose meaning cannot be recovered from one cell or header alone, while preserving the deterministic compiler and human gate as the execution authority.

## Design Decision

Use a hybrid model:

- keep a stable, versioned OLED property ontology for supported properties;
- let an LLM propose evidence-bound schema candidates for ambiguous context;
- let an LLM propose ontology extensions for genuinely unsupported properties;
- never let an LLM generate and execute temporary extraction scripts;
- never merge LLM proposals into compiled, gold, curated, or training data automatically.

This avoids trying to enumerate every paper-specific header in advance without turning free-form model output into executable code or accepted data.

## Inputs

`build_oled_llm_paper_mapping_request(...)` accepts:

- all semantic mapping packets for one paper;
- the in-memory `ParsedDocument` payload;
- the optional deterministic semantic mapping report.

The request contains the complete supplied document elements without automatic truncation, the table/text packets, the current ontology snapshot, deterministic candidates and findings, and a content digest. The builder does not read PDFs or files and does not call an LLM.

The request also fixes the active dataset scope to `molecule_interaction_properties_only`. Measurement- or device-only observations remain QA context even when they are scientifically valid OLED metrics.

One request covers one paper so the provider receives the full document context once rather than once per table cell.

## Structured Response

Every source packet must receive exactly one result with:

- an action: keep, supplement, replace, no eligible property, or source check;
- a scope classification: property-bearing, device-only, or no eligible property;
- optional schema candidate proposals;
- optional ontology extension proposals;
- evidence references and rationale.

`replace` results must list the exact `superseded_deterministic_candidate_ids`; deterministic candidates not listed there remain preserved. Table-derived candidate proposals must bind an exact `row_index` and matching source cell.

`needs_source_check` results must identify evidence that is genuinely absent from the supplied full text using one or more structured reasons: supplementary information, an unavailable figure/image, an external reference, unresolved identity/abbreviation, or a missing method definition. Generic requests to re-check the already supplied PDF are invalid.

`needs_ontology_review` is separate from missing-source review. Use it when the supplied evidence is complete but a molecule/interaction property is not represented in the current ontology. A `supplement` result may contain both candidates for known properties and extension proposals for additional unsupported properties from the same packet.

When a text packet contains explicit eV property signals such as HOMO/LUMO or S1/T1/Delta-EST but is still marked `no_eligible_property`, the response must record one structured exclusion reason: external/background evidence, duplication of an existing candidate, or ambiguous identity/assignment.

Known properties may become `OledSchemaCandidate` proposals only when the property id and causal layer are allowed. Unknown properties must remain ontology extension proposals and cannot be materialized as schema candidates.

## Fail-Closed Validation

`run_oled_llm_context_mapping(...)` rejects the complete response when:

- the paper id or packet set does not match the request;
- packet results are missing, duplicated, or unknown;
- a candidate uses a property outside the packet ontology;
- a candidate cites evidence outside the packet or supplied ParsedDocument;
- a candidate does not cite its source packet;
- a table candidate does not cite an exact matching row and cell;
- an ontology extension duplicates an existing property;
- ontology extension proposals repeat a proposed property id;
- an ontology extension is device/measurement-only under the current dataset scope;
- device-only or no-property content emits schema candidates;
- a measurement/device-only deterministic result is labelled `property_bearing`;
- a `replace` result omits or invents superseded deterministic candidate ids;
- a source-check request merely asks to re-check supplied PDF text;
- complete unsupported evidence is incorrectly routed through source-check instead of ontology review;
- explicit eV property evidence is excluded without a structured exclusion reason;
- materialized candidates fail existing semantic validation.

Valid candidate proposals are always marked `needs_llm`, carry the request digest and source packet id, and require human review. Valid ontology extensions are preserved as proposals only; the ontology is not mutated.

## Provider Boundary

The module uses the existing `LLMProvider` abstraction. A provider must be passed explicitly. Building a request does not call any external service, and the default literature workflow remains deterministic.

Use `StubLLMProvider` for tests and offline contract validation. Before using an external OpenAI-compatible endpoint with private or unpublished papers, obtain explicit user approval for the payload and provider.

## Offline Request Artifact

Generate a content-bound request without calling any provider:

```bash
PYTHONPATH=src python -m ai4s_agent.oled_llm_context_request \
  --parsed-document runs/<run_id>/parsed_documents/<paper>.json \
  --oled-candidates runs/<run_id>/extraction/oled_candidates.json \
  --output runs/<run_id>/review/oled_llm_context_request.json \
  --run-id <run_id>
```

The artifact includes the request digest, full supplied document context, semantic packets, deterministic candidates/findings, ontology snapshot, and explicit `llm_called=false` metadata. It can be inspected before any approved provider call.

## Current Dataset Scope

Device-only packets cannot emit admission candidates through this layer. Device context may still support a molecular or interaction property record, but a device-only result remains QA context and outside the current dataset.

## Non-Goals

This layer does not:

- run MinerU or read PDFs/images;
- execute model-generated scripts or tools;
- modify the ontology;
- replace deterministic candidates automatically;
- compile layered records;
- adjudicate human decisions;
- create gold records, curated datasets, or training data.
