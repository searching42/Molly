# BR2 frozen domain-request replay binding v1

This contract closes the BR2 boundary between contextual mapping and the
review-only candidate assembler. It does not change the OLED ontology,
provider, prompt, response validation, semantic validation, or candidate
compiler.

## Request identity

`OledLLMPaperMappingRequest.request_digest` remains the existing SHA-256 over
the canonical JSON bytes of:

```text
request.model_dump(mode="python", exclude={"metadata"})
```

The canonicalizer recursively normalizes nested Pydantic models and enums,
sorts mapping keys, preserves list order, retains explicit `null`, and emits
compact UTF-8 JSON. The top-level request `metadata` field is excluded. This
preserves existing request identities; the frozen artifact does not introduce a
second domain-request digest.

## Frozen artifacts

The provider task creates the following private runtime artifacts before
dispatch:

```text
br2_domain_mapping_request.v1
br2_provider_invocation_manifest.v1
contextual_mapping_result.json
```

The domain artifact contains the complete `OledLLMPaperMappingRequest`, the
request digest, and the source-candidate snapshot needed by candidate
assembly. Its source snapshot has a separate `source_candidates_digest`
because those MinerU candidate fields are not part of the pre-existing
provider-facing request model. Candidate hashes, anchors, types, text, table
headers/rows, and nearby context are cross-checked against the request
packets; path/image/runtime metadata is excluded from that evidence digest.

The invocation manifest contains only safe linkage metadata: request digest,
invocation digest, provider/model/prompt identity, schema version, and counts or
presence flags. Raw provider response material remains in the existing private
mapping-result artifact. The invocation digest is computed over the complete
persisted `LLMInvocationRecord` using the shared canonical JSON rules.

Both frozen artifacts are create-only at the writer boundary. Every write is
followed by reread and model validation. Any request digest, source evidence
digest, invocation digest, roster, schema-version, paper, or linkage mismatch
fails closed.

## Replay boundary

Candidate assembly loads and verifies:

```text
Frozen domain request
  -> request_digest
Frozen provider invocation manifest
  -> request_digest + invocation_digest
Validated mapping result
  -> request_digest + invocation record
```

It then passes the hydrated request and the frozen source-candidate snapshot to
the existing assembly function. It does not rebuild the request from
`ParsedDocument`/evidence and it never parses provider messages back into a
domain request. Exact replay therefore needs no MinerU run, packet rebuild,
deterministic candidate rebuild, prompt rebuild, or provider call.

The legacy `oled_llm_context_request.v6` artifact remains readable for the
existing supplementary-evidence recovery helpers. It is not used as the BR2
provider/candidate replay authority unless it is explicitly upgraded into the
new verified frozen contract.
