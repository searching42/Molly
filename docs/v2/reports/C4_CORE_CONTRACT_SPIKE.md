# C4 Core contract spike

Status: `PASS`

## Scope

This is a deliberately small, offline, non-production validation of the
approved Core v2 authority/data model. It does not implement `src/molly/`, an
AgentLoop, acquisition, a scheduler, UI/API integration, an LLM provider, a
generic plugin framework, BR1 integration, remote compute, GPU work, or error
propagation research.

The spike boundary is explicitly:

```text
NOT_PRODUCTION
NOT_INSTALLED
NO_NETWORK
NO_LLM
NO_REMOTE_COMPUTE
NO_GPU
```

It does not import `ai4s_agent` and uses only the Python standard library.

## Files

- `prototypes/core_v2_contract_spike/NOT_PRODUCTION.md`
- `prototypes/core_v2_contract_spike/__init__.py`
- `prototypes/core_v2_contract_spike/contract.py`
- `tests/test_core_v2_contract_spike.py`

The implementation contains `RunRequest`, `ToolSpec`, `ToolRegistry`,
`ToolPolicy`, `ApprovalRecord`, `ArtifactRecord`, `ArtifactStore`, `RunLedger`,
`ArtifactLineage`, one deterministic JSON example tool, and a single execution
function for the spike.

## Test command and result

```text
.venv/bin/python -m pytest -q tests/test_core_v2_contract_spike.py
13 passed
```

The focused tests cover:

1. known tool plus valid policy execution;
2. unknown-tool fail-closed behavior;
3. disallowed-tool fail-closed behavior;
4. missing exact approval fail-closed behavior;
5. stale or mismatched approval digest rejection;
6. successful exact approval binding;
7. deterministic output identity;
8. SHA-256 artifact identity and immutable collision rejection;
9. append-only hash-chained ledger behavior;
10. input/output lineage recording;
11. process-restart reload of artifact and ledger state; and
12. absence of network, LLM, remote, GPU, shell, credential, and legacy-package
    authority imports, plus privileged tool-capability rejection.

## Design findings

- A request digest that excludes the approval field gives an approval a stable
  binding while the request digest including the approval binds the executed
  request to the exact approval record.
- A closed registry and a policy allowlist fail closed before the tool handler
  runs. Approval is an exact binding to run, tool, request, and policy digests.
- Artifact identity is `sha256:<content digest>`. Existing bytes and metadata
  are checked before reuse, and a conflicting content type or byte collision is
  rejected rather than silently replacing an immutable artifact.
- The ledger is append-only JSONL with sequence numbers and a previous-record
  digest. Reload validates the chain, making accidental truncation or mutation
  visible.
- Lineage is explicit in both the result and the ledger payload, linking input
  artifact IDs to output artifact IDs for one tool execution.

## Limitations

This spike is not a production implementation, security proof, persistence
engine, distributed lock, network client, parser, scientific tool, LLM
adapter, or remote worker. It uses a local filesystem and a single-process
append path. It does not claim real BR1 parity, GPU validation, remote restart
acceptance, or experimental scientific validity.

## Decision

The spike validates the approved authority/data model without revealing a
structural contradiction and without requiring a new architecture. C4 is
`PASS`. Production implementations remain separately gated by the complete
C0-C7 readiness state and by their milestone-specific tests.
