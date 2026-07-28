from __future__ import annotations

from ai4s_agent.domains import (
    OledTextEvidenceCandidate,
    extract_oled_text_evidence_candidates_from_document as package_extract_oled_text_evidence_candidates_from_document,
)
from ai4s_agent.domains.oled_text_evidence_candidates import (
    extract_oled_text_evidence_candidates_from_document,
)
from ai4s_agent.schemas import ParsedDocument


def test_extracts_plqy_with_condition_and_compound_mentions() -> None:
    parsed = _parsed_document(
        [
            {
                "element_id": "el_p1_0001",
                "page": 1,
                "type": "paragraph",
                "text": "4CzIPN showed a photoluminescence quantum yield of 94 ± 2% in toluene.",
            }
        ]
    )

    candidates = extract_oled_text_evidence_candidates_from_document(parsed)
    package_candidates = package_extract_oled_text_evidence_candidates_from_document(parsed)

    assert len(candidates) == 1
    assert isinstance(package_candidates[0], OledTextEvidenceCandidate)
    assert package_candidates[0].candidate_id == candidates[0].candidate_id
    candidate = candidates[0]
    assert candidate.property_id == "plqy"
    assert candidate.property_label == "photoluminescence quantum yield"
    assert candidate.raw_value == "94 ± 2%"
    assert candidate.numeric_value == 94.0
    assert candidate.unit == "%"
    assert candidate.condition_text == "in toluene"
    assert candidate.compound_mentions == ["4CzIPN"]
    assert candidate.page == 1
    assert candidate.element_id == "el_p1_0001"
    assert candidate.evidence_span["start"] >= 0
    assert candidate.evidence_span["end"] <= len(candidate.evidence_text)
    assert candidate.provenance["paper_id"] == "paper-text"
    assert candidate.provenance["source_path"] == "/tmp/paper-text.pdf"
    assert candidate.extraction_method == "deterministic_oled_text_evidence_v1"


def test_extracts_emission_wavelength_eqe_and_delta_est() -> None:
    parsed = _parsed_document(
        [
            {
                "element_id": "el_p2_0001",
                "page": 2,
                "type": "paragraph",
                "text": "The emission maximum was observed at 507 nm in doped film.",
            },
            {
                "element_id": "el_p3_0001",
                "page": 3,
                "type": "paragraph",
                "text": "The device achieved a maximum EQE of 21.3% at 1000 cd m−2.",
            },
            {
                "element_id": "el_p4_0001",
                "page": 4,
                "type": "paragraph",
                "text": "The ΔEST value was estimated to be 0.03 eV.",
            },
        ]
    )

    candidates = extract_oled_text_evidence_candidates_from_document(parsed)

    by_property = {candidate.property_id: candidate for candidate in candidates}
    assert by_property["emission_wavelength_nm"].numeric_value == 507.0
    assert by_property["emission_wavelength_nm"].unit == "nm"
    assert by_property["emission_wavelength_nm"].condition_text == "in doped film"
    assert by_property["eqe_percent"].numeric_value == 21.3
    assert by_property["eqe_percent"].unit == "%"
    assert by_property["eqe_percent"].condition_text == "at 1000 cd m−2"
    assert by_property["delta_e_st_ev"].numeric_value == 0.03
    assert by_property["delta_e_st_ev"].unit == "eV"


def test_ignores_keywords_without_explicit_nearby_values() -> None:
    parsed = _parsed_document(
        [
            {
                "element_id": "el_p1_heading",
                "page": 1,
                "type": "title",
                "text": "Photoluminescence quantum yield and device performance",
            },
            {
                "element_id": "el_p1_compare",
                "page": 1,
                "type": "paragraph",
                "text": "The PLQY was higher than that of the reference emitter.",
            },
            {
                "element_id": "el_p1_header",
                "page": 1,
                "type": "paragraph",
                "text": "Max EQE (%) Turn on voltage (V) Current efficiency (cd A−1)",
            },
        ]
    )

    candidates = extract_oled_text_evidence_candidates_from_document(parsed)

    assert candidates == []


def test_ignores_reference_like_text() -> None:
    parsed = _parsed_document(
        [
            {
                "element_id": "ref_001",
                "page": 9,
                "type": "ref_text",
                "text": "[12] A. Author, Journal of OLEDs 2020, 10, 94% quantum yield.",
            }
        ]
    )

    candidates = extract_oled_text_evidence_candidates_from_document(parsed)

    assert candidates == []


def _parsed_document(elements: list[dict]) -> ParsedDocument:
    return ParsedDocument(
        paper_id="paper-text",
        source_path="/tmp/paper-text.pdf",
        parser_backend="mineru_api:hybrid-engine",
        metadata={"source_document_id": "paper-text-source"},
        pages=[{"page": 1}],
        elements=elements,
        tables=[],
    )
