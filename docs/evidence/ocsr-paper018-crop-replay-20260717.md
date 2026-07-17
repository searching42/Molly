# paper018 deterministic-crop OCSR replay - 2026-07-17

This evidence replays the four-structure paper018 canary after introducing the
deterministic crop-preprocessing boundary. It tests whether exact, complete,
structure-only inputs improve MolScribe source-copy accuracy. It remains a
one-paper bounded canary, not a corpus-scale accuracy claim.

## Exact inputs and artifacts

| Input or artifact | SHA-256 |
| --- | --- |
| paper018 main article | `b0005bbdb14c572404bb441aa819c96c56a73a9c408a785f31b01bb4d3e75e22` |
| paper018 supporting information | `09f028ec956f63f94af9735be7532f41cc040901c44e16effc7ab2b4670c37dc` |
| fixed page-2 raster | `f6bf65d6f8b805cf9a31ca471269d764652872515887cc462c8a4649691ae433` |
| crop request | `194cc1925724c1af50d826dcbea32648a201508748a89f605c754c004602cce6` |
| crop artifact | `44c1739d5109c40477dc700250d92b40ad0b0fb76885794fa142b68b10188bdb` |
| OCSR request | `b8b311c3518fc1bad04e0c87880d6db3f23e7c3bf7f0bfaae08b3dc4834d96d2` |
| OCSR candidate artifact | `ef4a20bb6606f9c95c71d822032d2e327f73e60eaca49490751ee8dc510b55a5` |
| reviewed ground-truth manifest | `1def7693afd377bab96f3b0026f3295262df1efec1b101cda07de89235c11308` |
| benchmark report | `548d645d284a6e972423fc484b05ac8d5c96c7e056c5ac183554b544faa73454` |
| benchmark verification | `540e8ec50ca4e22e4f49707fe873d6dd5518fc75ded649298e7eb6da02b40fb1` |

The page raster was generated from the bound main article with Poppler:

```bash
pdftoppm -f 2 -l 2 -r 150 -png -singlefile \
  papers/paper018.pdf paper018-page-002
```

Additional provenance:

| Field | Value |
| --- | --- |
| crop artifact digest | `sha256:76ee199a33a21cc2b35f6bec4243f44b91ba89e37f170e93d9d99f9b2a7349aa` |
| crop request digest | `sha256:f8f7ca6d61ab42c73ad577486ab3ff5189984c73f7e8b3e94b4278fa20ed8cc9` |
| crop-ready count | 4/4 |
| candidate artifact digest | `sha256:5d3e30ae24b36540fc922e3d0ceb14f700410e1eecd3618d52362ed056d4b0d0` |
| MolScribe version | `1.1.1` |
| checkpoint SHA-256 | `sha256:6f0df56fa32b5ffc21f8c7f311ef333da522f590bf5622e966c6bcb1f2d9ea1d` |
| inference device | CPU on workstation2/node45 |
| truth manifest digest | `sha256:74b6834883bffb8100af681216755497432d025626122da9be902a61786c857c` |
| report digest | `sha256:f4449ce1323b796a69ef58bdfb2a611aaa66c3ca6b5b07692eca209bea80456f` |
| verification digest | `sha256:eb6ee0e839024316524b09546bae3aa68d1e96b95c536dd839b8b884afb1df44` |
| exact-input replay | confirmed |

The RTX 5090 could not execute this checkpoint because the installed PyTorch
build does not support CUDA capability `sm_120`; the failed GPU invocation did
not publish an artifact. The completed run used the same privately copied and
SHA-bound checkpoint on CPU.

PDFs, page rasters, crop bundles, candidate artifacts, truth, and benchmark
JSON remain outside git. Only this evidence summary is committed.

## Crop review

All four deterministic crops passed the recorded quality gate. Visual review
confirmed that each contains one complete molecular diagram and no neighboring
row or crystal-structure fragment. Explicit authored masks remove the compound
aliases. The prior clipped carbonyl oxygens in CCO-2 and CCO-3 are present in
full. CBP-1's source-authored C6/C27 position labels remain visible because
generic text deletion could also erase genuine atom labels; the request makes
that choice explicit rather than silently altering chemistry.

| Alias | Output image SHA-256 | Edge clearance (L/T/R/B px) | Final ink fraction |
| --- | --- | --- | ---: |
| CBP-1 | `d2e6a870dde40fb9ce47f5c0e0d214fed4e51af2fd62fc5de051ace62457065c` | 23/20/17/15 | 0.216646 |
| CCO-1 | `7e49b2c4c59f252ce79e56efb1867c6428f0d87f727273da5fc4865e003a66ef` | 23/9/17/7 | 0.197144 |
| CCO-2 | `c2f6bb3390747f6f58fcc11010356c1ff9ca8eabfe3dd1aa84e091e834379e02` | 9/30/16/19 | 0.189868 |
| CCO-3 | `c0214ccfcd3b0c35ada7a0be25dfcbba1c5a2068603902a10f0d453ed22979a1` | 13/24/14/7 | 0.190381 |

## Benchmark comparison

| Metric | PR-AL original crops | PR-AM deterministic crops |
| --- | ---: | ---: |
| crop-quality gate ready | not available | 4/4 |
| candidate ready | 2/4 | 2/4 |
| candidate rejected | 2/4 | 2/4 |
| exact InChIKey matches | 0/4 | 0/4 |
| wrong ready graphs | 2 | 2 |
| false rejections | 2 | 2 |
| exact InChIKey accuracy | 0% | 0% |

| Alias | PR-AM outcome | Detail |
| --- | --- | --- |
| CBP-1 | false rejection | MolScribe returned invalid SMILES |
| CCO-1 | wrong graph | `GGRZFEUMKVLVAM-UHFFFAOYSA-N`, `C50H28N2O3`, confidence 0.571226; same wrong graph as PR-AL |
| CCO-2 | false rejection | MolScribe returned invalid SMILES |
| CCO-3 | wrong graph | `NVOXPLOBAZZLLY-UHFFFAOYSA-N`, `C58H50N2O3`, confidence 0.071648; elemental/carbon count improved but exact graph remains wrong |

The independently replayed benchmark confirms that deterministic cropping
fixed the known input defects but did not improve exact-graph accuracy on this
canary. This is a useful negative result: the current bottleneck is now the
MolScribe model/domain fit, not the previously clipped and contaminated crop
path. Expanding immediately to a three-paper/twenty-structure campaign would
measure a model already known to have 0% exact accuracy here. The next mainline
step should evaluate a stronger OCSR checkpoint/model or a bounded adaptation
experiment before scaling the corpus benchmark.

No result was promoted to material identity, Registry, Gold, or dataset state.
