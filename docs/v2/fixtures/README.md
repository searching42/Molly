# Core v2 fixture and parity boundary

Status: `FROZEN_C6_FIXTURE_CONTRACT`

The JSON manifests in this directory are bounded offline contracts. They
identify public-safe synthetic parser inputs, the existing synthetic OLED
fixture source, and the BR1 parity stages/invariants. They do not claim a
fresh real literature acquisition, fresh Uni-Mol training, real REINVENT4
generation, GPU execution, remote execution, or experimental validity.

No PDF is included because the C6 audit found no reviewed, redistributable PDF
already present in the repository. A future real-literature manifest must
record DOI, canonical source, provider, license/access status, expected access
route, and expected content family before any content is copied or fetched.

The CORE-05 OLED evidence checkpoint additionally uses
`CORE05_OLED_EVIDENCE_FIXTURE_MANIFEST.json` and the derived public-safe
synthetic JATS source at
`tests/fixtures/v2/synthetic/minimal.oled.jats.xml`. The source is parsed by
the CORE-04 `DocumentParserRouter` into an exact `CanonicalDocument` before
candidate extraction. Its four values are contract fixtures only and are not
experimental, computational, or validated scientific claims.
