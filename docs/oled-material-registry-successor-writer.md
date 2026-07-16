# OLED Material Registry successor snapshot writer (PR-Y)

PR-Y consumes one exact PR-X successor preflight and the exact current Material
Registry snapshot bytes bound by that preflight. It publishes the PR-X expected
successor snapshot together with a publication/write receipt as one fresh,
atomic directory unit.

PR-Y publishes an immutable snapshot version; it does not update or create a
mutable default Registry head. It also does not enter the observation,
reviewed-evidence, Gold, dataset, or training chains.

## Exact inputs and compare-and-swap

The writer accepts two distinct, non-symlinked JSON files:

1. one `oled_material_registry_successor_preflight.v1` artifact;
2. the exact current `oled_material_registry_snapshot.v1` file used by PR-X.

The current snapshot file SHA-256 must equal PR-X's
`current_registry_snapshot_sha256`. Its validated model and semantic digest
must also equal PR-X's embedded current snapshot. A semantically equivalent but
reformatted file is a different compare-and-swap parent and fails closed.

Both inputs are read again immediately before publication. Any byte or parsed
payload change after initial derivation aborts publication.

## Exact successor transition

PR-Y does not regenerate or reinterpret the successor. The published snapshot
must equal PR-X's `expected_successor_snapshot` in full and reproduce its exact
snapshot digest. The writer independently checks that:

- every prior entry remains byte-for-model identical;
- the successor entry roster is exactly prior entries plus PR-X planned entries;
- every planned Registry entry is unchanged;
- added material IDs and entry digests are sorted, unique, and complete; and
- prior, added, dependent-cell, and final counts match PR-X.

After successful publication, the material IDs, canonical names, alias lists,
and Registry entries exist in this immutable successor snapshot. The prior
snapshot remains untouched. No global head is activated.

## Atomic publication

The writer creates a private temporary sibling directory and writes exactly:

```text
<fresh-output-dir>/
  material_registry_successor_write.json
  material_registry_snapshot.json
```

Each file is created with no-follow/exclusive semantics and fsynced. The writer
then fsyncs the temporary directory and publishes it with Linux
`renameat2(RENAME_NOREPLACE)` or macOS `renameatx_np(RENAME_EXCL)`. Unsupported
runtimes fail closed.

The temporary directory's device/inode is pinned before publication and checked
through rename. After publication, the still-open descriptor revalidates the
exact two filenames and exact bytes, then the parent directory is fsynced.
Cleanup removes only files and directories whose inodes still belong to the
current invocation.

The protocol makes the published version append-only and non-overwriting. It
does not claim an operating-system immutable-file attribute.

## Receipt boundary

The publication receipt binds:

- construction-time PR-X file SHA-256 and semantic digest;
- exact prior snapshot SHA-256, semantic digest, and complete model;
- exact published successor snapshot and serialized file SHA-256;
- successor Registry version and snapshot digest;
- added material IDs, entry digests, and counts; and
- the fixed output filenames and publication safety assertions.

Standalone model validation replays semantic transition and serialized
successor bytes, but cannot recover the original external input bytes. Thus
`standalone_input_bytes_revalidation_supported=false` remains explicit.

## Explicitly false after PR-Y

- Registry head activated;
- activation receipt created;
- prior Registry snapshot mutated;
- observation materialization or admission;
- reviewed-evidence staging;
- Gold records, dataset views, or training eligibility;
- device-only admission; and
- network, external service, LLM, or MinerU calls.

## CLI

```bash
PYTHONPATH=src .venv/bin/python \
  -m ai4s_agent.oled_material_registry_successor_writer \
  --successor-preflight /absolute/path/to/registry_successor_preflight.json \
  --current-registry-snapshot /absolute/path/to/current_registry_snapshot.json \
  --output-dir /absolute/fresh/path/to/registry-successor-publication
```

CLI failures expose only a stable error code and exception type.

## Next boundary

PR-Z must independently read the publication receipt and separately published
snapshot. It must replay prior-entry preservation, exact planned additions,
file and semantic hashes, counts, ordering, version, and lineage without
trusting PR-Y's verification booleans. Only a PR-Z-verified snapshot may be
supplied to a new PR-N Registry resolution request.
