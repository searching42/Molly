# paper018 OCSR real-paper canary — 2026-07-17

This evidence records the first source-bound real-paper execution of the OCSR
benchmark boundary. It is a one-paper, four-structure canary and is explicitly
not a corpus-scale accuracy claim.

## Exact inputs

| Input | Role | SHA-256 |
| --- | --- | --- |
| paper018 main article | source diagrams, p2 Figure 1 | `b0005bbdb14c572404bb441aa819c96c56a73a9c408a785f31b01bb4d3e75e22` |
| paper018 supporting information | systematic names and characterization | `09f028ec956f63f94af9735be7532f41cc040901c44e16effc7ab2b4670c37dc` |
| hardened OCSR candidate artifact | four MolScribe results | `7e2d3b50e592f596b3e2bc1dc2bccc8ccd0ede05a4f096dc470a81e3c73a8ec2` |
| reviewed ground-truth manifest | four exact candidate bindings | `cbd13f55918790841d2cfd7f21de7844700cc1f21a2fa456812432e5e4032894` |
| benchmark report | immutable report bytes | `e666114dfed1a690521f2cb9544cab97d531f8f8a7a9b008d32b85ba9ad4d7a0` |
| verification artifact | exact-input replay receipt | `113ef1006ba383f8157493411d7ade8547f3469e968f7d44afec12cf03c42d93` |

Additional provenance:

| Field | Value |
| --- | --- |
| candidate run ID | `paper018-figure1-ocsr-20260717` |
| candidate artifact digest | `sha256:d5d0bdad776daf7e7582763afc922418e41ac0feaea4d2be6f0f6bb6036cfbb1` |
| MolScribe version | `1.1.1` |
| checkpoint SHA-256 | `sha256:6f0df56fa32b5ffc21f8c7f311ef333da522f590bf5622e966c6bcb1f2d9ea1d` |
| truth benchmark ID | `paper018-figure1-ocsr-canary-20260717` |
| truth manifest digest | `sha256:06337e7f2284d3d7378d88a8365aa2ad689364f76eb7807c5d3743be94a89698` |
| truth resolver | OPSIN web service `2.9.0` |
| report digest | `sha256:5f846717f99f0dd66be460e348ebaad4cb6f164eda69d219d86f93600e498f3c` |
| verification digest | `sha256:4732fec8f60d97bc4a02840075b0be5c1b198e708af410b4866b15e3c3ac4bb3` |
| exact-input replay | confirmed |
| benchmark scope | `bounded_real_paper_canary` |

The source PDFs, image crops, truth JSON, candidate JSON, and report JSON remain
outside git. Only this evidence summary is committed.

## Ground-truth review

Each systematic name was taken from the supporting information, resolved to a
graph with OPSIN 2.9.0, canonicalized with RDKit, and independently compared
with the molecular diagram in main-paper Figure 1. Formula and characterization
evidence in the supporting information were used as cross-checks.

| Alias | Reviewed InChIKey | Reviewed formula | Reference location |
| --- | --- | --- | --- |
| CBP-1 | `AWNQKZDWLDGQQN-UHFFFAOYSA-N` | `C50H32N2O2` | SI p3 |
| CCO-1 | `AHESUVKREFCROS-UHFFFAOYSA-N` | `C50H30N2O3` | SI p4 |
| CCO-2 | `NDQAPBNMEZKOSW-UHFFFAOYSA-N` | `C54H38N2O3` | SI p5–6 |
| CCO-3 | `OUEPTZZOIDWYIN-UHFFFAOYSA-N` | `C58H46N2O3` | SI p6–7 |

The final CCO-3 heading on SI p7 appears to omit one tert-butyl substituent.
The reviewed truth uses the two-substituent graph because it is supported by
all three independent source signals: the main-paper Figure 1 depiction, the
3,5-di-tert-butyl precursor-17 name on SI p6, and the final HRMS formula
`C58H46N2O3`. This discrepancy is recorded in the per-sample review note rather
than silently discarded.

## Results

| Metric | Value |
| --- | ---: |
| paper count | 1 |
| metadata sample count | 4 |
| distinct source-document SHA count | 1 |
| distinct crop count | 4 |
| distinct source locator count | 4 |
| distinct source evidence count | 4 |
| candidate ready | 2 |
| candidate rejected | 2 |
| exact InChIKey matches | 0 |
| wrong ready graphs | 2 |
| false rejections | 2 |
| exact InChIKey accuracy | 0% |
| ready rate | 50% |
| rejection rate | 50% |
| false-ready rate among ready candidates | 100% |

| Alias | Candidate result | Ground-truth comparison |
| --- | --- | --- |
| CBP-1 | rejected: invalid SMILES | false rejection |
| CCO-1 | ready, `GGRZFEUMKVLVAM-UHFFFAOYSA-N`, `C50H28N2O3`, confidence 0.466010 | wrong graph; truth is `C50H30N2O3` |
| CCO-2 | rejected: unsupported atoms | false rejection |
| CCO-3 | ready, `DQFCHNPFLKAWNF-UHFFFAOYSA-N`, `C59H48N2O2`, confidence 0.212375 | wrong graph; truth is `C58H46N2O3` |

Neither ready candidate was correct, so confidence did not discriminate a
usable graph in this canary.

The post-write verifier independently re-read the report, truth manifest,
candidate artifact, main article, and supporting information; rebuilt every
result with the persisted report timestamp; and confirmed exact report model
equality before publishing the verification artifact.

## Failure analysis and next decision

Visual inspection of the exact crops found deterministic input-quality
problems:

- CBP-1 includes heading text, red atom markers, and atom-number callouts;
- CCO-1 includes its alias beneath the graph;
- CCO-2 clips the carbonyl oxygen and includes a neighboring fragment; and
- CCO-3 clips the carbonyl oxygen and includes a neighboring crystal fragment.

These observations explain why expanding the same crop path immediately to
twenty or more structures would mostly measure preprocessing defects. The
scientifically useful next boundary is deterministic structure-diagram crop
preprocessing and crop-quality validation, followed by replaying paper018. The
three-paper/twenty-structure campaign should resume only after that replay
shows the source diagrams reach the model intact.

No candidate was promoted to material identity, Registry, Gold, or dataset
state by this run.
