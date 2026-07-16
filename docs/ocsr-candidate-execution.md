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

The checkpoint SHA-256 and installed MolScribe version are derived at runtime.
The output file is created with no-replace semantics.

## Image preparation

Use a tight, high-resolution crop containing one complete molecular diagram.
Exclude reaction arrows, reagent text, compound numbers, aliases, atom-position
callouts, crystal annotations, and neighboring structures where possible.
These elements can produce a chemically parseable but source-incorrect graph.
RDKit validation establishes only that the candidate is a valid graph; it does
not establish that MolScribe copied the source diagram correctly.

Low confidence, rejected graphs, and even high-confidence parseable graphs must
remain candidate evidence until an independent source-to-graph review confirms
the exact structure.

This stage does not crop figures, call MinerU, download model checkpoints,
build material-identity evidence responses, or write Registry, Gold, or dataset
artifacts. Those remain explicit downstream steps.
