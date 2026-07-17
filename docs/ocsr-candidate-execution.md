# OCSR candidate execution

This command executes MolScribe over an exact batch of molecular-structure
images and records candidate molecular graphs. It is an execution capability,
not an identity decision or publication step.

For each image, the request binds a candidate ID, the paper-reported alias, the
local image filename, and the exact SHA-256 of the image bytes. The runner
rejects symlink inputs, re-checks the bytes before inference, runs MolScribe,
and independently parses, sanitizes, canonicalizes, and derives a standard
InChI and InChIKey with RDKit.

An invalid or empty MolScribe graph is retained as `candidate_rejected`; it is
never silently promoted. A valid graph is retained as `candidate_ready` with
its raw SMILES, canonical isomeric SMILES, InChIKey, and optional model
confidence.

The output deliberately keeps all of the following false:

- source-to-graph match validated;
- material identity resolved;
- Registry mutation;
- Gold publication; and
- dataset publication.

## Request

```json
{
  "schema_version": "ocsr_candidate_request.v1",
  "run_id": "paper018-figure1-ocsr",
  "items": [
    {
      "candidate_id": "paper018-CBP-1",
      "reported_alias": "CBP-1",
      "image_file": "CBP-1.png",
      "image_sha256": "sha256:<64 lowercase hex characters>"
    }
  ]
}
```

Items must be sorted by unique `candidate_id`. Relative image paths are
resolved against the request JSON directory.

## Run

Run this command on a host with MolScribe, RDKit, PyTorch, the official
MolScribe checkpoint, and an appropriate accelerator:

```bash
PYTHONPATH=src python -m ai4s_agent.ocsr_candidate_execution \
  --request /operator/ocsr/request.json \
  --checkpoint /operator/models/swin_base_char_aux_1m.pth \
  --output /operator/ocsr/candidates.json \
  --device cuda
```

The checkpoint is opened without following symlinks, streamed into an
invocation-owned private regular file while its SHA-256 is computed, and
fsynced. MolScribe loads that exact owned inode through a file-descriptor path;
the inode, size, timestamps, and SHA-256 are rechecked after model loading.
Checkpoint paths containing symlink components are rejected.

Before any request or checkpoint input is read, the output parent is opened
component-by-component without following symlinks and pinned by directory file
descriptor. The output is created relative to that descriptor with
`O_NOFOLLOW | O_CREAT | O_EXCL`, written to completion even across short
writes, and fsynced together with its parent. Publication then rechecks the
parent and output inode, exact bytes, length, and artifact model. Cleanup only
removes an inode created by the current invocation, so a concurrent
replacement is never deleted.

## Image preparation

Use the deterministic
[OCSR crop preprocessing boundary](ocsr-crop-preprocessing.md) to create the
input request. It binds explicit operator-authored pixel boxes, removes only
explicitly declared non-structure regions, applies replayable transforms, and
withholds the OCSR request when its crop-quality gate fails. Reaction arrows,
reagent text, compound aliases, atom-position callouts, crystal annotations,
and neighboring structures can otherwise produce a chemically parseable but
source-incorrect graph.

The crop gate establishes pixel-input quality, not chemical correctness.
RDKit validation establishes only that the candidate is a valid graph; neither
boundary establishes that MolScribe copied the source diagram correctly.

Low confidence, rejected graphs, and even high-confidence parseable graphs must
remain candidate evidence until an independent source-to-graph review confirms
the exact structure.

This stage does not itself crop figures, call MinerU, download model checkpoints,
build material-identity evidence responses, or write Registry, Gold, or dataset
artifacts. Those remain explicit downstream steps.

Source-copy accuracy is evaluated separately by the exact-bound
[OCSR real-corpus benchmark](ocsr-real-corpus-benchmark.md). A chemically valid
candidate remains untrusted until that boundary compares it with independently
reviewed source truth.
