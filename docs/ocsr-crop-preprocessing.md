# Deterministic OCSR crop preprocessing

This boundary turns an exact page raster plus operator-authored pixel boxes
into a replayable batch of structure-only OCSR images. It does not locate
figures with a model, infer chemistry, or make an identity decision.

## Input contract

An `ocsr_crop_preprocessing_request.v1` binds every candidate to:

- one exact page-raster filename and SHA-256;
- a source document ID and source locator;
- one explicit pixel crop box; and
- zero or more explicit exclusion boxes with a reason such as
  `reported_alias`, `atom_number_annotation`, or `neighboring_fragment`.

Every pixel box uses half-open coordinates: `[left, right) x [top, bottom)`.
The exclusion transform builds an exact binary mask with that same convention,
and `exclusion_pixel_count` is counted from the applied mask rather than
inferred from authored dimensions.

Request items are sorted by unique candidate ID. Repeated exact evidence
bindings are rejected. Source paths must be one relative filename; symbolic
links and symbolic path components are rejected.

Exclusion masks are deliberately operator-authored. Automatically deleting
text-shaped pixels is unsafe for chemical diagrams because genuine `N`, `O`,
element, charge, and stereochemical labels are also text-shaped.

## Deterministic profile and quality gate

The `deterministic_structure_crop.v1` profile performs only fixed transforms:

1. verify and decode one exact, single-frame raster;
2. convert it to grayscale and apply the declared white exclusion boxes;
3. threshold ink and find four-connected components;
4. retain substantial components and cluster them by vertical center;
5. select the cluster with the greatest ink, add fixed proportional padding;
6. resize the longest edge to 768 pixels with Pillow LANCZOS; and
7. encode a deterministic grayscale PNG.

The artifact records the Pillow version and every numeric profile parameter.
The crop fails closed when the likely structure touches an input edge, is too
small, has ambiguous competing component clusters, contains too many selected
components, or has an implausible final ink fraction. A rejected batch still
publishes diagnostic crop images and `crop_artifact.json`, but it does not
publish `ocsr_request.json`.

This is an input-quality gate, not proof that the selected pixels encode the
correct molecule. A human-reviewed source-to-graph benchmark remains required
after OCSR inference.

## Publication

```bash
PYTHONPATH=src .venv/bin/python \
  -m ai4s_agent.ocsr_crop_preprocessing \
  --request /operator/paper018/crop-request.json \
  --output-dir /operator/paper018/crop-bundle
```

The published no-replace directory contains:

- the exact `crop_request.json` bytes;
- `crop_artifact.json` with input, output, metric, and digest bindings;
- one deterministic PNG per request item; and
- `ocsr_request.json` only when every crop is ready.

The output parent is opened component by component without following
symbolic links and pinned by directory descriptor. Files are completely
written and fsynced in an invocation-owned temporary directory. Publication
uses an atomic no-replace directory rename, fsyncs the parent, rechecks parent
and directory inode ownership, reads back exact file bytes, and revalidates
the artifact and both request models. Cleanup removes only the invocation-owned
inode. Concurrent targets and output-parent replacement therefore fail closed.

The boundary never performs OCSR inference, validates a source-to-graph match,
resolves identity, or writes Registry, Gold, or dataset state.
