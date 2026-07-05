from __future__ import annotations

from ai4s_agent.domains import (
    OledMineruCandidate,
    OledMineruCandidateSummary,
    OledMineruCandidateType,
    OledMineruRelevanceSignal,
    OledMineruSourceFormat,
    OledMineruTableParseStatus,
    extract_oled_mineru_candidates as package_extract_oled_mineru_candidates,
    extract_oled_mineru_candidates_from_document as package_extract_oled_mineru_candidates_from_document,
    summarize_oled_mineru_candidates as package_summarize_oled_mineru_candidates,
)
from ai4s_agent.domains.oled_mineru_candidates import (
    detect_oled_mineru_source_format,
    extract_oled_mineru_candidates,
    extract_oled_mineru_candidates_from_document,
    summarize_oled_mineru_candidates,
)


def test_flat_content_list_table_extraction_parses_headers_rows_and_anchor_fields() -> None:
    candidates = extract_oled_mineru_candidates_from_document(
        [
            {
                "type": "text",
                "text": "Acknowledgements and unrelated funding text.",
                "page_idx": 0,
            },
            {
                "type": "table",
                "table_caption": "Table 1. OLED device performance.",
                "table_body": (
                    "<table><tr><th>emitter</th><th>host</th><th>EQE (%)</th></tr>"
                    "<tr><td>D1</td><td>mCBP</td><td>22.1</td></tr></table>"
                ),
                "table_footnote": "Measured at 100 cd/m².",
                "bbox": [1.0, 2.0, 3.0, 4.0],
                "page_idx": 3,
            },
        ],
        paper_id="paper-003",
        source_path="/absolute/local/path/paper-003_content_list.json",
    )

    assert len(candidates) == 1
    candidate = candidates[0]
    assert candidate.paper_id == "paper-003"
    assert candidate.source_path == "/absolute/local/path/paper-003_content_list.json"
    assert candidate.source_format == OledMineruSourceFormat.CONTENT_LIST
    assert candidate.candidate_type == OledMineruCandidateType.TABLE
    assert candidate.page_index == 3
    assert candidate.block_index == 1
    assert candidate.block_id == "paper-003:content_list:p3:b1"
    assert candidate.section_title is None
    assert candidate.bbox == [1.0, 2.0, 3.0, 4.0]
    assert candidate.image_path is None
    assert candidate.caption == "Table 1. OLED device performance."
    assert candidate.html_table.startswith("<table>")
    assert candidate.markdown_table is None
    assert candidate.table_headers == ["emitter", "host", "EQE (%)"]
    assert candidate.table_rows == [{"emitter": "D1", "host": "mCBP", "EQE (%)": "22.1"}]
    assert candidate.table_parse_status == OledMineruTableParseStatus.PARSED
    assert candidate.nearby_text_before is None
    assert candidate.nearby_text_after is None
    assert candidate.evidence_anchor == "paper-003:content_list:p3:b1:table"
    assert len(candidate.candidate_hash) == 64
    assert OledMineruRelevanceSignal.PROPERTY_KEYWORD in candidate.relevance_signals
    assert OledMineruRelevanceSignal.MEASUREMENT_KEYWORD in candidate.relevance_signals
    assert {"eqe", "host", "oled"}.issubset(set(candidate.matched_terms))
    assert candidate.metadata["md_sidecar_policy"] == "nearby_context_only"

    same_candidates = extract_oled_mineru_candidates_from_document(
        [
            {
                "type": "table",
                "table_caption": "Table 1. OLED device performance.",
                "table_body": (
                    "<table><tr><th>emitter</th><th>host</th><th>EQE (%)</th></tr>"
                    "<tr><td>D1</td><td>mCBP</td><td>22.1</td></tr></table>"
                ),
                "table_footnote": "Measured at 100 cd/m².",
                "bbox": [1.0, 2.0, 3.0, 4.0],
                "page_idx": 3,
            }
        ],
        paper_id="paper-003",
        source_path="/different/machine/path/paper-003_content_list.json",
    )
    assert same_candidates[0].candidate_hash == candidate.candidate_hash


def test_nested_content_list_v2_table_extraction_reads_content_html_and_md_context() -> None:
    candidates = extract_oled_mineru_candidates_from_document(
        [
            [
                {
                    "type": "title",
                    "content": {"title_content": ["Device results"]},
                    "bbox": [0.0, 0.0, 1.0, 1.0],
                },
                {
                    "type": "table",
                    "content": {
                        "html": (
                            "<table><tr><th>PLQY</th><th>EQE</th></tr>"
                            "<tr><td>0.80</td><td>20%</td></tr></table>"
                        ),
                        "table_caption": ["Table S2. OLED device metrics for DACT-II emitters."],
                        "table_footnote": ["a Doped film, 10 wt%."],
                    },
                    "bbox": [2.0, 3.0, 4.0, 5.0],
                },
            ]
        ],
        paper_id="paper-003-v2",
        md_text=(
            "# Supporting information\n"
            "Before context names the host mCBP and device stack.\n"
            "Table S2. OLED device metrics for DACT-II emitters.\n"
            "After context reports luminance at 100 cd m-2.\n"
        ),
    )

    table_candidate = next(candidate for candidate in candidates if candidate.candidate_type == OledMineruCandidateType.TABLE)
    assert table_candidate.source_format == OledMineruSourceFormat.CONTENT_LIST_V2
    assert table_candidate.section_title == "Device results"
    assert table_candidate.page_index == 0
    assert table_candidate.block_index == 1
    assert table_candidate.html_table.startswith("<table>")
    assert table_candidate.table_headers == ["PLQY", "EQE"]
    assert table_candidate.table_rows == [{"PLQY": "0.80", "EQE": "20%"}]
    assert table_candidate.table_parse_status == OledMineruTableParseStatus.PARSED
    assert "host mCBP" in (table_candidate.nearby_text_before or "")
    assert "luminance" in (table_candidate.nearby_text_after or "")
    assert {"plqy", "eqe", "doped"}.issubset(set(table_candidate.matched_terms))


def test_text_and_title_candidates_are_extracted_when_relevant() -> None:
    candidates = extract_oled_mineru_candidates_from_document(
        [
            [
                {
                    "type": "title",
                    "content": {"title_content": ["OLED device fabrication"]},
                    "bbox": [0.0, 0.0, 1.0, 1.0],
                },
                {
                    "type": "paragraph",
                    "content": {
                        "paragraph_content": [
                            "The OLED PLQY of the DACT-II doped film was reported as 0.80."
                        ]
                    },
                    "bbox": [1.0, 1.0, 2.0, 2.0],
                },
            ]
        ],
        paper_id="paper-text",
    )

    assert [candidate.candidate_type for candidate in candidates] == [
        OledMineruCandidateType.TITLE,
        OledMineruCandidateType.TEXT,
    ]
    title_candidate = candidates[0]
    assert title_candidate.raw_text == "OLED device fabrication"
    assert OledMineruRelevanceSignal.FABRICATION_KEYWORD in title_candidate.relevance_signals
    text_candidate = candidates[1]
    assert text_candidate.section_title == "OLED device fabrication"
    assert "plqy" in text_candidate.matched_terms
    assert OledMineruRelevanceSignal.MATERIAL_ROLE_KEYWORD in text_candidate.relevance_signals


def test_image_and_chart_caption_candidates_preserve_paths_and_types() -> None:
    candidates = extract_oled_mineru_candidates_from_document(
        [
            {
                "type": "image",
                "image_caption": "Figure 2. EL spectra and EQE roll-off.",
                "img_path": "images/fig2.png",
                "page_idx": 5,
            },
            {
                "type": "chart",
                "chart_caption": "Chart 1. OLED current density and luminance curves.",
                "page_idx": 6,
            },
        ],
        paper_id="paper-figures",
    )

    assert [candidate.candidate_type for candidate in candidates] == [
        OledMineruCandidateType.FIGURE,
        OledMineruCandidateType.CHART,
    ]
    assert candidates[0].caption == "Figure 2. EL spectra and EQE roll-off."
    assert candidates[0].image_path == "images/fig2.png"
    assert OledMineruRelevanceSignal.PROPERTY_KEYWORD in candidates[0].relevance_signals
    assert OledMineruRelevanceSignal.MEASUREMENT_KEYWORD in candidates[1].relevance_signals


def test_malformed_and_complex_html_tables_are_emitted_without_crashing() -> None:
    candidates = extract_oled_mineru_candidates_from_document(
        [
            {
                "type": "table",
                "table_caption": "Table S3. OLED EQE values.",
                "table_body": "<table><tr><th>emitter</th><th>EQE</th></tr><tr><td>D1<td>22.1</tr>",
            },
            {
                "type": "table",
                "table_caption": "Table S4. OLED device stack.",
                "table_body": (
                    "<table><tr><th rowspan=\"2\">emitter</th><th>EQE</th></tr>"
                    "<tr><td>22.1</td></tr></table>"
                ),
            },
        ],
        paper_id="paper-bad-tables",
    )

    assert [candidate.table_parse_status for candidate in candidates] == [
        OledMineruTableParseStatus.MALFORMED,
        OledMineruTableParseStatus.COMPLEX_UNSUPPORTED,
    ]
    assert all(candidate.candidate_type == OledMineruCandidateType.TABLE for candidate in candidates)


def test_markdown_table_parsing() -> None:
    candidates = extract_oled_mineru_candidates_from_document(
        [
            {
                "type": "table",
                "table_caption": "Table 2. OLED markdown table.",
                "table_body": "| emitter | host | EQE (%) |\n| --- | --- | --- |\n| D1 | mCBP | 22.1 |",
                "page_idx": 2,
            }
        ],
        paper_id="paper-markdown-table",
    )

    candidate = candidates[0]
    assert candidate.markdown_table.startswith("| emitter")
    assert candidate.html_table is None
    assert candidate.table_headers == ["emitter", "host", "EQE (%)"]
    assert candidate.table_rows == [{"emitter": "D1", "host": "mCBP", "EQE (%)": "22.1"}]
    assert candidate.table_parse_status == OledMineruTableParseStatus.PARSED


def test_include_irrelevant_behavior() -> None:
    parsed_document = [
        {"type": "text", "text": "Acknowledgements and unrelated funding text.", "page_idx": 0},
        {"type": "text", "text": "OLED EQE was measured at 22%.", "page_idx": 1},
    ]

    relevant_only = extract_oled_mineru_candidates_from_document(parsed_document, paper_id="paper-filtered")
    all_candidates = extract_oled_mineru_candidates_from_document(
        parsed_document,
        paper_id="paper-filtered",
        include_irrelevant=True,
    )

    assert len(relevant_only) == 1
    assert relevant_only[0].raw_text == "OLED EQE was measured at 22%."
    assert len(all_candidates) == 2
    assert all_candidates[0].relevance_signals == []


def test_multi_document_api_uses_paper_id_mappings_and_source_paths() -> None:
    candidates = extract_oled_mineru_candidates(
        [
            {
                "paper_id": "paper-a",
                "content_list": [
                    {"type": "text", "text": "OLED EQE candidate.", "page_idx": 1},
                ],
            },
            [
                {"type": "text", "text": "OLED PLQY candidate.", "page_idx": 2},
            ],
        ],
        md_by_paper_id={"paper-a": "Before text. OLED EQE candidate. After text."},
        source_path_by_paper_id={"paper-a": "/tmp/paper-a_content_list.json", "paper-002": "/tmp/paper-b.json"},
    )

    assert [candidate.paper_id for candidate in candidates] == ["paper-a", "paper-002"]
    assert candidates[0].source_path == "/tmp/paper-a_content_list.json"
    assert candidates[1].source_path == "/tmp/paper-b.json"
    assert candidates[0].nearby_text_after == "After text."


def test_summary_counts_candidates_by_type_signal_and_paper() -> None:
    candidates = extract_oled_mineru_candidates_from_document(
        [
            {
                "type": "table",
                "table_caption": "Table 1. OLED device performance.",
                "table_body": "| emitter | host | EQE (%) |\n| --- | --- | --- |\n| D1 | mCBP | 22.1 |",
            },
            {"type": "text", "text": "OLED PLQY in doped films.", "page_idx": 1},
            {"type": "image", "image_caption": "Figure 1. OLED EQE roll-off.", "page_idx": 2},
            {"type": "chart", "chart_caption": "Chart 1. Luminance curves.", "page_idx": 3},
        ],
        paper_id="paper-summary",
    )

    summary = summarize_oled_mineru_candidates(candidates)

    assert summary.total_candidates == 4
    assert summary.relevant_candidate_count == 4
    assert summary.paper_ids == ["paper-summary"]
    assert summary.candidates_by_type[OledMineruCandidateType.TABLE] == 1
    assert summary.candidates_by_signal[OledMineruRelevanceSignal.OLED_KEYWORD] == 3
    assert summary.table_candidate_count == 1
    assert summary.text_candidate_count == 1
    assert summary.caption_candidate_count == 3
    assert summary.figure_candidate_count == 1
    assert summary.chart_candidate_count == 1


def test_public_api_is_exported_from_domain_package() -> None:
    candidates = package_extract_oled_mineru_candidates_from_document(
        [{"type": "text", "text": "OLED PLQY values are listed in the paragraph.", "page_idx": 2}],
        paper_id="paper-package-export",
    )
    multi_candidates = package_extract_oled_mineru_candidates(
        [[{"type": "text", "text": "OLED EQE candidate.", "page_idx": 1}]]
    )
    summary = package_summarize_oled_mineru_candidates([*candidates, *multi_candidates])

    assert isinstance(candidates[0], OledMineruCandidate)
    assert isinstance(summary, OledMineruCandidateSummary)
    assert candidates[0].candidate_type == OledMineruCandidateType.TEXT
    assert candidates[0].table_parse_status == OledMineruTableParseStatus.NOT_TABLE
    assert OledMineruRelevanceSignal.PROPERTY_KEYWORD in candidates[0].relevance_signals
    assert multi_candidates[0].paper_id == "paper-001"


def test_detects_supported_mineru_source_formats() -> None:
    assert detect_oled_mineru_source_format([{"type": "text", "text": "OLED"}]) == OledMineruSourceFormat.CONTENT_LIST
    assert detect_oled_mineru_source_format([[{"type": "text", "content": {"paragraph_content": ["OLED"]}}]]) == (
        OledMineruSourceFormat.CONTENT_LIST_V2
    )
    assert detect_oled_mineru_source_format({"content_list": [{"type": "text", "text": "OLED"}]}) == (
        OledMineruSourceFormat.MINERU_LIKE
    )
    assert detect_oled_mineru_source_format({"unexpected": "shape"}) == OledMineruSourceFormat.UNKNOWN
