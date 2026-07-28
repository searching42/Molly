# PR-AN paper018 contextual-alias canary — 2026-07-17

## Scope

This canary tested the PR-AN candidate boundary against the four OLED aliases
from paper018. The supplementary PDF was re-extracted page by page with
pdfplumber 0.11.10 and visually checked on pages 3, 4, 5, and 7 against rendered
PDF pages. The resolver calls were live requests to the official OPSIN web
service; RDKit independently canonicalized every successful response.
Each alias request bound the visually reviewed page and exact global line span
plus the normalized heading SHA-256; no heading-start heuristic was used.

No Registry, Gold, or dataset state was read or written by the PR-AN runner.
The previously reviewed PR-AM ground-truth manifest was used only for the
comparison below, after candidate publication.

## Exact inputs and output

| Object | Bytes | SHA-256 |
|---|---:|---|
| `paper018_si.pdf` | 3,750,585 | `sha256:09f028ec956f63f94af9735be7532f41cc040901c44e16effc7ab2b4670c37dc` |
| deterministic parsed text | 157,115 | `sha256:1abbecc161ce05e277863f60ed8d11fd59b890e7f2f46a56b687378be7be6359` |
| PR-AN exact-span request JSON | 1,487 | `sha256:e28c60340ff85dbe55698876c7abbb190c7ad855b96e1eba403c0e098231deb9` |
| PR-AN exact-span artifact JSON | 89,742 | `sha256:7772663b0d2d8d7782c587779bfb1e98a1a7eec1236371e502328f1482ce7239` |
| PR-AM ground truth | 7,482 | `sha256:0f6e5df028093b64667c30b66f67a0979e5b8801ba9045d4005a145e3e379b79` |

Artifact digest:
`sha256:4b48ad0a6e3583e6d006ae3e9b25145085b708fc10f1fcef869591edd34a9428`.
All four aliases produced parseable candidate results; none was identity-admitted.

## Comparison

| Alias | SI heading span | OPSIN + RDKit candidate InChIKey | Reviewed reference InChIKey | Exact |
|---|---|---|---|---|
| CBP-1 | page 3, lines 164-165 | `AWNQKZDWLDGQQN-UHFFFAOYSA-N` | `AWNQKZDWLDGQQN-UHFFFAOYSA-N` | yes |
| CCO-1 | page 4, lines 247-248 | `AHESUVKREFCROS-UHFFFAOYSA-N` | `AHESUVKREFCROS-UHFFFAOYSA-N` | yes |
| CCO-2 | page 5, lines 331-332 | `NDQAPBNMEZKOSW-UHFFFAOYSA-N` | `NDQAPBNMEZKOSW-UHFFFAOYSA-N` | yes |
| CCO-3 | page 7, lines 422-423 | `NDQAPBNMEZKOSW-UHFFFAOYSA-N` | `OUEPTZZOIDWYIN-UHFFFAOYSA-N` | no |

The exact-match result is therefore **3/4**. This is the scientifically useful
outcome, not a pipeline failure hidden by correction logic.

## CCO-3 conflict

The SI heading for CCO-3 repeats the CCO-2 systematic name:

`5-(4-(tert-Butyl)phenyl)-11-(10H-spiro[acridine-9,9'-xanthen]-10-yl)chromeno[3,2-c]carbazol-8(5H)-one`

OPSIN and RDKit consistently resolve that exact reported heading to the CCO-2
InChIKey. The reviewed crystallographic/diagram reference for CCO-3 has a
different InChIKey. PR-AN correctly preserves this disagreement with
`source_match_validated=false` and does not infer or silently repair a different
name. A later evidence-review boundary may adjudicate the conflict.

## Acceptance conclusion

The canary demonstrates that contextual systematic-name resolution materially
improves alias-only coverage while retaining a fail-closed candidate boundary.
It also exposes a real source-level discrepancy that an alias-only or
resolver-only pipeline would otherwise misclassify as a resolved identity.
