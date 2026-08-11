# BR2 contextual mapping v1 acceptance

This directory records the fresh real-provider acceptance for PR #50,
`M3.5-BR2-MAPPING — Contextual mapping and evidence binding`. The run consumed
the verified ParsedDocument published by the PR #49 MinerU runtime acceptance
for `oled-paper-018` and stopped at the review-only candidate raw dataset.

## Reviewed code and input

- Base SHA: `bd51ba4585ccf8e9beebde006d0b21ffcafc96e5`
- Acceptance code HEAD: `2e5ba55cf1c658d07c50b527c79435247fc13d57`
- Paper logical identity: `oled-paper-018`
- ParsedDocument logical identity: `parsed_document`
- ParsedDocument SHA-256: `sha256:8927d670f97cf16a417476da2a8200b9e71ebca8046b0d68eb1fb1a33737632f`

The acceptance used an isolated copy of the existing user LLM profile with
only the request timeout extended for this long real-document call. It did
not modify durable settings. No PDF bytes, private paths, host information,
credentials, or raw paper content are stored in this evidence.

## Real mapping result

- Provider class: `openai_compatible`
- Model: `deepseek-v4-flash`
- Prompt version: `oled.contextual_semantic_mapping.v5`
- Real provider call: yes
- Mapping status: `ready_for_human_review`
- Deterministic candidates / semantic packets / deterministic schema candidates: `32 / 32 / 31`
- Deterministic findings: `101`
- Structured LLM schema candidates: `96`
- Final review candidate records: `31`
- Property observations: `96`
- Evidence coverage: `96 / 96`
- Layered-schema compilation: ran; `0` compiled, `31` partial, `0` rejected
- Unresolved review items: `16`
- Device-only observations excluded from dataset scope: `15`

The contextual response, packet bindings, semantic constraints, evidence
references, and layered candidate compilation all passed the current
contracts. LLM proposals remain proposals; they are not scientific truth or
confirmation authority.

## Boundary

The package is explicitly review-only: `confirmed=false`,
`gold_records_created=false`, `human_confirmation_required=true`, and
`ontology_mutated=false`. No human confirmation, Gate approval, training,
generation, prediction, or ranking occurred. Conversation-driven acceptance
and `WAITING_USER` remain PR #51.

See [acceptance_manifest.json](acceptance_manifest.json) for privacy-safe
identities and [mapping_summary.json](mapping_summary.json) for the compact
outcome summary.
