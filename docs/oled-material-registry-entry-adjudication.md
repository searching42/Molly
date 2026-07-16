# OLED local Material Registry entry adjudication (PR-W)

PR-W applies one complete human decision manifest to one exact PR-V local
Material Registry entry proposal request. It produces deterministic,
human-approved Registry-entry candidates for a later write preflight. It does
not reserve a material ID, create an authoritative Registry entry, or mutate
the Registry.

## Exact inputs and causal order

The file entry consumes the PR-V request JSON and the human decision-manifest
JSON as distinct, non-symlinked inputs. It records both construction-time file
SHA-256 values and binds the request artifact, Registry snapshot, review
contract, request item, and human decision semantic digests. Human review must
not predate the PR-V request, and adjudication generation must not predate the
human review.

Standalone model validation embeds and replays the complete PR-V semantic
request and reconstructs the semantic human decision manifest from the
adjudicated items. The original external input byte sequences are not embedded,
so `standalone_input_bytes_revalidation_supported` is fixed to `false`.

## Approval contract

For `approve_local_registry_entry_candidate`, the reviewer must explicitly
confirm the single-entity scope, exact proposed material ID, exact proposed
preferred name, exact proposed alias list, and the bound review contract. PR-W
does not permit an approval response to rewrite any of these values. A new or
different name/alias requires a new source-supported request.

Every existing-name hint, Registry-snapshot conflict, and within-batch conflict
must be acknowledged by exact digest. A within-batch conflict blocks approval;
it can only be kept unresolved, deferred for entity-policy review, or routed
back to existing-Registry resolution.

Approved chemistry is rebuilt with the generic Registry-entry builder and must
reproduce the PR-O accepted standard InChI and InChIKey. The production logic
is paper-agnostic; paper016 is a bounded seven-entry canary, not a hard-coded
shape or naming policy.

## Explicit boundaries

After PR-W:

- the proposed material ID is human-approved but not reserved or assigned;
- the Registry entry is a candidate, not an authoritative created entry;
- the later writer must recheck the current Registry and compare-and-swap
  preconditions immediately before committing;
- no observations, reviewed evidence, Gold records, datasets, or training data
  are written;
- no PDF, network, LLM, MinerU, or external service is accessed.

## File entry

```bash
PYTHONPATH=src .venv/bin/python \
  -m ai4s_agent.oled_material_registry_entry_adjudication \
  --request-artifact /absolute/path/to/entry_proposal_request.json \
  --decision-manifest /absolute/path/to/entry_decisions.json \
  --output /absolute/fresh/path/to/entry_adjudication.json
```

The output must be fresh and cannot overlap either input. Publication uses the
repository's pinned-parent, atomic no-replace path. CLI errors expose only a
stable error code and exception type.

