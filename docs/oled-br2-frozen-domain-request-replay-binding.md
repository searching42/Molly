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

The provider task publishes the domain request before dispatch and publishes
the validated result linkage only after the external effect returns:

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

Both frozen artifacts are no-replace publications at the writer boundary. An
identical identity resolves to the existing immutable artifact; different
bytes or a different identity are a conflict. Publication uses a fully fsynced
temporary file followed by an atomic hard-link commit, so concurrent processes
cannot pass an `exists()` check and then overwrite each other. Every publish is
followed by reread and model validation. Any request digest, source evidence
digest, invocation digest, roster, schema-version, paper, or linkage mismatch
fails closed.

## Attempt publication lifecycle

BR2 uses the shared `AttemptPublication` primitive under the task's private
runtime directory:

```text
RESERVED
  -> REQUEST_FROZEN
  -> EFFECT_STARTED
  -> RESULT_COMMITTED
  -> COMPLETE
```

The reservation binds the run, request digest, and source-candidate digest.
The request marker binds the exact public request bytes. `EFFECT_STARTED` is
durably committed immediately before the provider call. Result and invocation
manifest bytes are committed without replacement and bound into
`RESULT_COMMITTED`; `COMPLETE` binds both request and result manifests.

Retries follow these invariants:

- failure before `EFFECT_STARTED` may resume the identical frozen request;
- `COMPLETE` replays verified artifacts without resolving or calling a
  provider;
- a verified mapping result written before a publication crash may be used to
  reconstruct a missing safe invocation manifest and finish deterministically;
- an interrupted effect with no verified result is `UNKNOWN` and must not call
  the provider again;
- a known provider failure may be retried only when provider semantics
  explicitly mark it retryable. BR2 currently defaults to non-retryable when
  that guarantee is absent.

Exact provider-invocation artifacts use the same no-replace file primitive.
An empty or payload-only digest directory created by a pre-effect crash is
resumable; a conflicting or tampered payload/manifest remains fail-closed.

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
