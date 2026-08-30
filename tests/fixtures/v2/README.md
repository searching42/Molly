# Core v2 offline fixtures

These fixtures are public-safe, synthetic contract inputs. They are not
literature claims, experimental measurements, model results, or fresh BR1
parity evidence.

- `synthetic/minimal.jats.xml` is a minimal synthetic JATS/XML parser input.
- `synthetic/minimal.html` is a minimal synthetic HTML parser input.
- No PDF is included: the repository had no reviewed, redistributable PDF
  fixture at the C6 audit, so no license was inferred.

The source fixture used by the OLED contract manifest is the existing,
explicitly synthetic `tests/fixtures/phase3_to_phase1/parsed_document.json`.
