from __future__ import annotations

import json
from pathlib import Path

import pytest

from ai4s_agent.custom_corpus_manifest import (
    CustomCorpusManifestError,
    load_custom_corpus_manifest,
    safe_document_label,
    sha256_file,
    validate_custom_corpus_manifest,
)


def test_valid_manifest_loads_and_normalizes_sha256(tmp_path: Path) -> None:
    pdf = tmp_path / "paper.pdf"
    pdf.write_bytes(b"%PDF-1.4 synthetic\n")
    digest = sha256_file(pdf).removeprefix("sha256:")
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(_manifest(pdf_path=str(pdf), pdf_sha256=digest)),
        encoding="utf-8",
    )

    manifest = load_custom_corpus_manifest(manifest_path)

    assert manifest.schema_version == "custom_corpus_manifest.v1"
    assert manifest.documents[0].pdf_sha256 == f"sha256:{digest}"
    assert safe_document_label(manifest.documents[0]) == "doc-001"


def test_duplicate_document_id_fails(tmp_path: Path) -> None:
    payload = _manifest(pdf_path=str(tmp_path / "a.pdf"))
    payload["documents"].append(dict(payload["documents"][0]))

    with pytest.raises(CustomCorpusManifestError, match="duplicate document_id"):
        validate_custom_corpus_manifest(payload)


def test_invalid_schema_version_fails(tmp_path: Path) -> None:
    payload = _manifest(pdf_path=str(tmp_path / "a.pdf"))
    payload["schema_version"] = "custom_corpus_manifest.v0"

    with pytest.raises(CustomCorpusManifestError, match="schema_version"):
        validate_custom_corpus_manifest(payload)


def test_unsafe_corpus_and_document_ids_fail(tmp_path: Path) -> None:
    payload = _manifest(pdf_path=str(tmp_path / "a.pdf"))
    payload["corpus_id"] = "../private"

    with pytest.raises(CustomCorpusManifestError, match="corpus_id"):
        validate_custom_corpus_manifest(payload)

    payload = _manifest(pdf_path=str(tmp_path / "a.pdf"))
    payload["documents"][0]["document_id"] = "doc/001"
    with pytest.raises(CustomCorpusManifestError, match="document_id"):
        validate_custom_corpus_manifest(payload)


def test_private_and_unknown_corpora_force_commit_flags_false(tmp_path: Path) -> None:
    private_payload = _manifest(pdf_path=str(tmp_path / "a.pdf"), corpus_class="private_literature")
    private_payload["documents"][0]["allow_raw_pdf_commit"] = True
    with pytest.raises(CustomCorpusManifestError, match="commit flags"):
        validate_custom_corpus_manifest(private_payload)

    unknown_payload = _manifest(pdf_path=str(tmp_path / "b.pdf"), corpus_class="unknown_or_mixed")
    unknown_payload["documents"][0]["allow_mineru_bundle_commit"] = True
    with pytest.raises(CustomCorpusManifestError, match="commit flags"):
        validate_custom_corpus_manifest(unknown_payload)


def test_token_like_source_url_fails_without_leaking_secret(tmp_path: Path) -> None:
    payload = _manifest(pdf_path=str(tmp_path / "a.pdf"))
    payload["documents"][0]["source_url"] = "https://example.org/paper?token=abc123"

    with pytest.raises(CustomCorpusManifestError) as excinfo:
        validate_custom_corpus_manifest(payload)

    message = str(excinfo.value).lower()
    assert "source_url" in message
    assert "abc123" not in message
    assert "token=abc123" not in message


@pytest.mark.parametrize(
    "source_url",
    [
        "https://user@example.org/paper",
        "https://example.org/paper?download=1",
        "https://example.org/paper#section",
    ],
)
def test_source_url_with_userinfo_query_or_fragment_fails(tmp_path: Path, source_url: str) -> None:
    payload = _manifest(pdf_path=str(tmp_path / "a.pdf"))
    payload["documents"][0]["source_url"] = source_url

    with pytest.raises(CustomCorpusManifestError, match="source_url"):
        validate_custom_corpus_manifest(payload)


def test_placeholder_example_manifest_loads_as_contract_example() -> None:
    manifest = load_custom_corpus_manifest(
        Path(__file__).parents[1] / "docs" / "examples" / "custom-corpus-manifest.example.json"
    )

    assert manifest.corpus_id == "example-public-corpus"
    assert manifest.documents[0].document_id == "doc-example-001"


def _manifest(
    *,
    pdf_path: str,
    pdf_sha256: str = "",
    corpus_class: str = "public_literature",
) -> dict[str, object]:
    return {
        "schema_version": "custom_corpus_manifest.v1",
        "corpus_id": "test-corpus",
        "corpus_class": corpus_class,
        "created_at": "2026-06-28T00:00:00Z",
        "created_by": "tester",
        "description": "Synthetic manifest for tests.",
        "source_policy": "test-only",
        "default_redaction_policy": {
            "commit_raw_pdfs": False,
            "commit_parsed_documents": False,
            "commit_mineru_bundles": False,
            "commit_full_reports": False,
        },
        "documents": [
            {
                "document_id": "doc-001",
                "pdf_path": pdf_path,
                "pdf_sha256": pdf_sha256,
                "title": "Synthetic Paper",
                "doi": "",
                "source_url": "https://example.org/paper",
                "license_or_access": "synthetic-test",
                "provenance_note": "Synthetic local test fixture.",
                "allow_raw_pdf_commit": False,
                "allow_parsed_document_commit": False,
                "allow_mineru_bundle_commit": False,
                "redaction_required": True,
                "expected_document_type": "scientific_paper",
                "notes": "",
            }
        ],
    }
