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
| crop artifact | `bf845e069a529f8aea6a1b88346a0f1f71b4906790c6714701e4b41051a4aca8` |
| OCSR request | `ccf35fdea224214677d7f6cd98ced71aaa65a5a6560ddfde7ee9b651b9eb5378` |
| OCSR candidate artifact | `9f057c9bce21be07571e9239f8ee0959fefca48cc6fef64bae3210a000cb9b93` |
| reviewed ground-truth manifest | `0f6e5df028093b64667c30b66f67a0979e5b8801ba9045d4005a145e3e379b79` |
| benchmark report | `f3007228ef33ecfdb51087db71b5773dc46394bb6e54b286c9bc6c76a5fb56ea` |
| benchmark verification | `fe729241738145110ce5d5c76b16bfcb3a655060de9f71453f49e71172031d15` |

The page raster was generated from the bound main article with Poppler:

```bash
pdftoppm -f 2 -l 2 -r 150 -png -singlefile \
  papers/paper018.pdf paper018-page-002
```

Additional provenance:

| Field | Value |
| --- | --- |
| crop artifact digest | `sha256:ebe2fe9fea3482a2b2715dfb33a748b51a2764045f59d41ff097c3291f660096` |
| crop request digest | `sha256:f8f7ca6d61ab42c73ad577486ab3ff5189984c73f7e8b3e94b4278fa20ed8cc9` |
| crop-ready count | 4/4 |
| candidate artifact digest | `sha256:8c38614d9d02c6cebb46f25a48b5b956911e1667db9ee2e98bb2c59dd7605f07` |
| MolScribe version | `1.1.1` |
| checkpoint SHA-256 | `sha256:6f0df56fa32b5ffc21f8c7f311ef333da522f590bf5622e966c6bcb1f2d9ea1d` |
| inference device | CPU on workstation2/node45 |
| truth manifest digest | `sha256:271be2e413aad990d50b1f03399a33a99694d0111a93a2f3f9b5d15468971eeb` |
| report digest | `sha256:4f1940aa631b3b480be8741fb49b384400c049c10fc711d316084a3a4c656b06` |
| verification digest | `sha256:cbd40060189a1d2ec81babe81e1310fe5dfb6fb6cd89da32d87fc860c44701f2` |
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

The final replay applies every exclusion as the authored half-open box
`[left, right) x [top, bottom)`. The recorded exclusion count is measured from
the exact applied binary mask. This prevents Pillow's inclusive rectangle
endpoint from erasing the adjacent right-hand column or bottom row.

| Alias | Output image SHA-256 | Edge clearance (L/T/R/B px) | Final ink fraction |
| --- | --- | --- | ---: |
| CBP-1 | `56584b5888b022e13fdb6ec843fbda00866bbb46470cdc0a258a404fba0683b7` | 23/20/17/15 | 0.216833 |
| CCO-1 | `d729d818b22be53612b93042bda59eee597313d9c7c6b30d43bcd8d863937437` | 23/9/17/7 | 0.197331 |
| CCO-2 | `c2f6bb3390747f6f58fcc11010356c1ff9ca8eabfe3dd1aa84e091e834379e02` | 9/30/16/19 | 0.189868 |
| CCO-3 | `5deb6435e6a5f2b257917951552479df1211972ac6311bd833a4ddf482743e7c` | 13/24/14/7 | 0.190786 |

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
| CCO-1 | wrong graph | `GGRZFEUMKVLVAM-UHFFFAOYSA-N`, `C50H28N2O3`, confidence 0.568372; same wrong graph as PR-AL |
| CCO-2 | false rejection | MolScribe returned invalid SMILES |
| CCO-3 | wrong graph | `NVOXPLOBAZZLLY-UHFFFAOYSA-N`, `C58H50N2O3`, confidence 0.069470; elemental/carbon count improved but exact graph remains wrong |

The independently replayed benchmark confirms that deterministic cropping
fixed the known input defects but did not improve exact-graph accuracy on this
canary. This is a useful negative result: the current bottleneck is now the
MolScribe model/domain fit, not the previously clipped and contaminated crop
path. Expanding immediately to a three-paper/twenty-structure campaign would
measure a model already known to have 0% exact accuracy here. The next mainline
step should evaluate a stronger OCSR checkpoint/model or a bounded adaptation
experiment before scaling the corpus benchmark.

No result was promoted to material identity, Registry, Gold, or dataset state.
