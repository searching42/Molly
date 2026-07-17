# PR-AN paper018 contextual-alias canary — 2026-07-17

## Scope

This canary tested the PR-AN candidate boundary against the four OLED aliases
from paper018. The supplementary PDF was re-extracted page by page with
pdfplumber 0.11.10 and visually checked on pages 3, 4, 5, and 7 against rendered
PDF pages. The resolver calls were live requests to the official OPSIN web
service; RDKit independently canonicalized every successful response.

No Registry, Gold, or dataset state was read or written by the PR-AN runner.
The previously reviewed PR-AM ground-truth manifest was used only for the
comparison below, after candidate publication.

## Exact inputs and output

| Object | Bytes | SHA-256 |
|---|---:|---|
| `paper018_si.pdf` | 3,750,585 | `sha256:09f028ec956f63f94af9735be7532f41cc040901c44e16effc7ab2b4670c37dc` |
| deterministic parsed text | 157,115 | `sha256:1abbecc161ce05e277863f60ed8d11fd59b890e7f2f46a56b687378be7be6359` |
| PR-AN request JSON | 684 | `sha256:4bae97a36fb9ac66d5777b0c0c61cc84edc0a49a2a9c8bd87a694d61cb6f615e` |
| PR-AN artifact JSON | 88,896 | `sha256:c9465ae293cfbffeccd50f0052f303c82206ac1faa773a25e74588372ec2df1e` |
| PR-AM ground truth | 7,482 | `sha256:0f6e5df028093b64667c30b66f67a0979e5b8801ba9045d4005a145e3e379b79` |

Artifact digest:
`sha256:a9480801dcf65d8798d88ea2aa0f8d2ef1826491efa4740d07c2402291299adc`.
All four aliases produced parseable candidate results; none was identity-admitted.

## Comparison

| Alias | SI heading page | OPSIN + RDKit candidate InChIKey | Reviewed reference InChIKey | Exact |
|---|---:|---|---|---|
| CBP-1 | 3 | `AWNQKZDWLDGQQN-UHFFFAOYSA-N` | `AWNQKZDWLDGQQN-UHFFFAOYSA-N` | yes |
| CCO-1 | 4 | `AHESUVKREFCROS-UHFFFAOYSA-N` | `AHESUVKREFCROS-UHFFFAOYSA-N` | yes |
| CCO-2 | 5 | `NDQAPBNMEZKOSW-UHFFFAOYSA-N` | `NDQAPBNMEZKOSW-UHFFFAOYSA-N` | yes |
| CCO-3 | 7 | `NDQAPBNMEZKOSW-UHFFFAOYSA-N` | `OUEPTZZOIDWYIN-UHFFFAOYSA-N` | no |

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
