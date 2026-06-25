from __future__ import annotations

import json
from pathlib import Path

import pytest

from ai4s_agent.mineru_output_normalizer import (
    discover_mineru_output_bundle,
    normalize_mineru_output_bundle,
)
from document_parse_test_helpers import fixture_mineru_output_dir, write_synthetic_pdf


def test_mineru_output_normalizer_discovers_and_normalizes_fixture(tmp_path: Path) -> None:
    pdf = write_synthetic_pdf(tmp_path / "paper.pdf")
    bundle_dir = tmp_path / "bundle"
    bundle_dir.mkdir()
    for path in fixture_mineru_output_dir().iterdir():
        (bundle_dir / path.name).write_bytes(path.read_bytes())

    bundle = discover_mineru_output_bundle(bundle_dir)
    normalized = normalize_mineru_output_bundle(
        input_pdf=pdf,
        bundle=bundle,
        parser_backend="mineru_api:hybrid-engine",
    )

    assert bundle.markdown_path.endswith("synthetic.md")
    assert bundle.content_list_json_path.endswith("synthetic_content_list.json")
    assert bundle.middle_json_path.endswith("synthetic_middle.json")
    assert normalized.parsed_document.metadata["mineru_backend"] == "hybrid-engine"
    assert normalized.parsed_document.metadata["mineru_version"] == "mineru-fixture-1"
    assert normalized.parsed_document.pages == [{"page": 1, "width": 1000.0, "height": 1400.0}]
    assert normalized.parsed_document.elements[0].page == 1
    assert normalized.parsed_document.elements[0].type == "title"
    assert normalized.parsed_document.tables[0].caption == "Table 1 OLED measurements"
    assert normalized.parsed_document.tables[0].footnotes == ["Synthetic fixture values."]
    assert normalized.parsed_document.tables[0].headers == ["SMILES", "PLQY", "lambda_em"]
    assert normalized.parsed_document.tables[0].rows == [{"SMILES": "CCO", "PLQY": "0.65", "lambda_em": "520"}]
    assert "table_p1_0003" in normalized.parsed_document.metadata["table_html_by_id"]
    by_type = {element.type: element for element in normalized.parsed_document.elements}
    assert by_type["image"].metadata["image_path"] == "images/figure_1.png"
    assert by_type["code"].text == "print('oled')"
    assert by_type["list"].text == "Validate PLQY\nPreserve provenance"
    assert normalized.warnings == []


def test_mineru_output_normalizer_falls_back_to_markdown_with_warning(tmp_path: Path) -> None:
    pdf = write_synthetic_pdf(tmp_path / "paper.pdf")
    bundle_dir = tmp_path / "bundle"
    bundle_dir.mkdir()
    (bundle_dir / "synthetic.md").write_text("# Markdown Only\n\nFallback body", encoding="utf-8")

    bundle = discover_mineru_output_bundle(bundle_dir)
    normalized = normalize_mineru_output_bundle(
        input_pdf=pdf,
        bundle=bundle,
        parser_backend="mineru_api:hybrid-engine",
    )

    assert normalized.parsed_document.elements[0].type == "markdown"
    assert normalized.warnings == ["structured content_list JSON missing; falling back to Markdown-only normalization"]


def test_mineru_output_normalizer_rejects_invalid_structured_output(tmp_path: Path) -> None:
    pdf = write_synthetic_pdf(tmp_path / "paper.pdf")
    bundle_dir = tmp_path / "bundle"
    bundle_dir.mkdir()
    (bundle_dir / "synthetic.md").write_text("# Broken Bundle", encoding="utf-8")
    (bundle_dir / "synthetic_content_list.json").write_text(json.dumps({"content_list": ["bad"]}), encoding="utf-8")

    bundle = discover_mineru_output_bundle(bundle_dir)

    with pytest.raises(ValueError, match="does not contain valid structured content items"):
        normalize_mineru_output_bundle(
            input_pdf=pdf,
            bundle=bundle,
            parser_backend="mineru_api:hybrid-engine",
        )


def test_mineru_output_normalizer_supports_content_list_v2_nested_pages(tmp_path: Path) -> None:
    pdf = write_synthetic_pdf(tmp_path / "paper.pdf")
    bundle_dir = tmp_path / "bundle"
    bundle_dir.mkdir()
    (bundle_dir / "synthetic.md").write_text("V2 fixture body without heading", encoding="utf-8")
    (bundle_dir / "synthetic_content_list_v2.json").write_text(
        json.dumps(
            [
                [
                    {
                        "type": "title",
                        "content": {
                            "title_content": [
                                {"type": "text", "text": "Official "},
                                {"type": "text", "text": "V2 Title"},
                            ]
                        },
                        "bbox": [0, 0, 100, 20],
                    },
                    {
                        "type": "text",
                        "content": {
                            "paragraph_content": [
                                {"type": "text", "text": "Nested "},
                                {
                                    "type": "hyperlink",
                                    "url": "https://example.test/paper",
                                    "content": [{"type": "text", "text": "linked"}],
                                    "children": [{"type": "text", "text": "linked"}],
                                },
                                {"type": "text", "text": " page text"},
                            ]
                        },
                        "bbox": [1, 2, 3, 4],
                    },
                    {
                        "type": "list",
                        "content": {
                            "list_items": [
                                [{"type": "text", "text": "first bullet"}],
                                [{"type": "text", "text": "second bullet"}],
                            ]
                        },
                    },
                    {
                        "type": "code",
                        "content": {"code_content": [{"type": "text", "text": "print('v2')"}]},
                    },
                    {
                        "type": "algorithm",
                        "content": {"algorithm_content": [{"type": "text", "text": "for i in range(n)"}]},
                    },
                    {
                        "type": "equation_interline",
                        "content": {"math_content": [{"type": "text", "text": "E = h nu"}]},
                    },
                    {
                        "type": "image",
                        "content": {
                            "img_path": "images/v2-figure.png",
                            "image_caption": [{"type": "text", "text": "Official V2 figure"}],
                        },
                    },
                ],
                [
                    {
                        "type": "table",
                        "content": {
                            "table_caption": [
                                {"type": "text", "text": "Second "},
                                {"type": "text", "text": "page table"},
                            ],
                            "table_footnote": [
                                {"type": "text", "text": "Official "},
                                {"type": "text", "text": "V2 footnote"},
                            ],
                            "table_body": "<table><tr><th>A</th></tr><tr><td>B</td></tr></table>",
                        },
                    }
                ],
            ]
        ),
        encoding="utf-8",
    )
    (bundle_dir / "synthetic_middle.json").write_text(
        json.dumps(
            {
                "pdf_info": [
                    {"page_idx": 0, "page_size": [1000, 1400]},
                    {"page_idx": 1, "page_size": [1000, 1400]},
                ]
            }
        ),
        encoding="utf-8",
    )

    bundle = discover_mineru_output_bundle(bundle_dir)
    normalized = normalize_mineru_output_bundle(
        input_pdf=pdf,
        bundle=bundle,
        parser_backend="mineru_api:hybrid-engine",
    )

    assert [page["page"] for page in normalized.parsed_document.pages] == [1, 2]
    by_type = {element.type: element for element in normalized.parsed_document.elements}
    assert normalized.parsed_document.metadata["title"] == "Official V2 Title"
    assert by_type["title"].text == "Official V2 Title"
    assert by_type["paragraph"].text == "Nested linked page text"
    assert by_type["list"].text == "first bullet\nsecond bullet"
    element_texts = {element.text for element in normalized.parsed_document.elements}
    assert "print('v2')" in element_texts
    assert "for i in range(n)" in element_texts
    assert by_type["equation"].text == "E = h nu"
    assert by_type["image"].metadata["image_path"] == "images/v2-figure.png"
    assert normalized.parsed_document.tables[0].page == 2
    assert normalized.parsed_document.tables[0].caption == "Second page table"
    assert normalized.parsed_document.tables[0].footnotes == ["Official V2 footnote"]
