from __future__ import annotations

import hashlib
import json
import os
import shlex
import shutil
import struct
import zlib
from copy import deepcopy
from io import StringIO
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from pydantic import ValidationError

from ai4s_agent._utils import write_json
from ai4s_agent.domains.oled_supplementary_scoped_candidate_request import (
    OledSupplementaryScopedCandidateRequestArtifact,
)
from ai4s_agent.domains.oled_supplementary_scoped_candidate_response import (
    OledSupplementaryScopedCandidateResponseArtifact,
    OledSupplementaryScopedCandidateResponseManifest,
)
from ai4s_agent.domains.oled_supplementary_semantic_review import (
    OledSupplementarySemanticAdjudicationArtifact,
    OledSupplementarySemanticDecisionManifest,
    OledSupplementarySemanticReviewPacket,
)
from ai4s_agent.domains import (
    oled_supplementary_source_transcription_review as transcription_domain,
)
from ai4s_agent.domains.oled_supplementary_source_transcription_review import (
    OledSupplementarySourceTranscriptionAdjudicationArtifact,
    OledSupplementarySourceTranscriptionDecisionManifest,
    OledSupplementarySourceTranscriptionReviewItem,
    OledSupplementarySourceTranscriptionReviewPacket,
    build_oled_supplementary_source_page_asset,
    build_oled_supplementary_source_pdf_evidence,
    build_oled_supplementary_source_transcription_adjudication_artifact,
    build_oled_supplementary_source_transcription_review_packet,
    render_oled_supplementary_source_transcription_review_markdown,
)
from ai4s_agent.oled_supplementary_scoped_candidate_response import (
    build_oled_supplementary_scoped_candidate_response_from_files,
)
from ai4s_agent.oled_supplementary_semantic_review import (
    build_oled_supplementary_semantic_adjudication_from_files,
    build_oled_supplementary_semantic_review_packet_from_files,
)
from ai4s_agent import (
    oled_supplementary_source_transcription_review as transcription_runner,
)
from ai4s_agent.oled_supplementary_source_transcription_review import (
    build_oled_supplementary_source_transcription_adjudication_from_files,
    build_oled_supplementary_source_transcription_review_packet_from_files,
    main,
    render_oled_supplementary_source_transcription_review_packet_from_files,
)
from tests.test_oled_supplementary_scoped_candidate_response import (
    _request_payload,
    _response_payload,
    _scope_id,
    _sha256_file,
    _stable_hash,
)
from tests.test_oled_supplementary_semantic_review import _decision_payload


_RESPONSE_GENERATED_AT = "2026-07-13T22:00:00+08:00"
_SEMANTIC_PACKET_GENERATED_AT = "2026-07-13T22:10:00+08:00"
_SEMANTIC_ADJUDICATED_AT = "2026-07-13T22:30:00+08:00"
_TRANSCRIPTION_PACKET_GENERATED_AT = "2026-07-13T22:40:00+08:00"
_TRANSCRIPTION_REVIEWED_AT = "2026-07-13T22:50:00+08:00"
_TRANSCRIPTION_ADJUDICATED_AT = "2026-07-13T23:00:00+08:00"

_COMPONENT_FIELDS = (
    "page_anchor_check",
    "caption_check",
    "headers_check",
    "row_structure_check",
    "cell_literals_check",
    "footnotes_check",
    "table_extent_check",
)

_PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


def _png_chunk(chunk_type: bytes, data: bytes) -> bytes:
    checksum = zlib.crc32(chunk_type + data) & 0xFFFFFFFF
    return (
        struct.pack(">I", len(data))
        + chunk_type
        + data
        + struct.pack(">I", checksum)
    )


def _png_bytes(
    *,
    width: int = 1700,
    height: int = 2200,
    color_type: int = 2,
    bit_depth: int = 8,
) -> bytes:
    """Build a complete, CRC-valid, non-interlaced RGB/RGBA PNG fixture."""

    if color_type == 2:
        pixel = b"\x21\x43\x65" if bit_depth == 8 else b"\x00\x21\x00\x43\x00\x65"
    elif color_type == 6:
        pixel = (
            b"\x21\x43\x65\xff"
            if bit_depth == 8
            else b"\x00\x21\x00\x43\x00\x65\xff\xff"
        )
    else:
        raise ValueError("unsupported test PNG color type")
    scanline = b"\x00" + pixel * width
    ihdr = struct.pack(">IIBBBBB", width, height, bit_depth, color_type, 0, 0, 0)
    return b"".join(
        (
            _PNG_SIGNATURE,
            _png_chunk(b"IHDR", ihdr),
            _png_chunk(b"IDAT", zlib.compress(scanline * height)),
            _png_chunk(b"IEND", b""),
        )
    )


_FAKE_PNG_BYTES = _png_bytes()


def _png_with_declared_dimensions(*, width: int, height: int) -> bytes:
    """Build a tiny envelope used to test pre-decode total-pixel rejection."""

    ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    return b"".join(
        (
            _PNG_SIGNATURE,
            _png_chunk(b"IHDR", ihdr),
            _png_chunk(b"IDAT", zlib.compress(b"\x00")),
            _png_chunk(b"IEND", b""),
        )
    )


def _real_poppler_bin_dir() -> Path:
    configured = os.environ.get("MOLLY_TEST_POPPLER_BIN_DIR")
    pdfinfo = shutil.which("pdfinfo")
    pdftoppm = shutil.which("pdftoppm")
    candidates: list[Path] = []
    if configured:
        candidates.append(Path(configured).expanduser())
    resolved_tools = [
        Path(value).resolve()
        for value in (pdfinfo, pdftoppm)
        if value is not None
    ]
    if len(resolved_tools) == 2 and resolved_tools[0].parent == resolved_tools[1].parent:
        candidates.append(resolved_tools[0].parent)
    for tool in resolved_tools:
        for ancestor in tool.parents:
            candidates.extend(
                (
                    ancestor / "native" / "poppler" / "poppler" / "bin",
                    ancestor / "poppler" / "bin",
                )
            )
    candidates.append(Path("/usr/bin"))

    seen: set[Path] = set()
    native_magics = transcription_runner._NATIVE_EXECUTABLE_MAGICS
    for candidate in candidates:
        try:
            candidate = candidate.resolve(strict=True)
            if candidate in seen:
                continue
            seen.add(candidate)
            headers = [
                (candidate / name).read_bytes()[:4]
                for name in ("pdfinfo", "pdftoppm")
            ]
        except OSError:
            continue
        if all(header in native_magics for header in headers):
            return candidate
    pytest.skip("one directory containing native pdfinfo and pdftoppm is unavailable")


def _path_shim_directory(tmp_path: Path) -> tuple[Path, Path]:
    shim_dir = tmp_path / "hostile-path"
    shim_dir.mkdir()
    marker = tmp_path / "path-shim-executed"
    for executable_name in ("pdfinfo", "pdftoppm"):
        shim = shim_dir / executable_name
        shim.write_text(
            "#!/bin/sh\n"
            f"printf '%s' {shlex.quote(executable_name)} > "
            f"{shlex.quote(str(marker))}\n"
            "exit 99\n",
            encoding="utf-8",
        )
        os.chmod(shim, 0o700)
    return shim_dir, marker


def _minimal_pdf_bytes(*, page_count: int = 39) -> bytes:
    """Return a small valid PDF with enough pages for the paper016 page-38 fixture."""

    content_object = page_count + 3
    objects: list[bytes] = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        (
            b"<< /Type /Pages /Count "
            + str(page_count).encode("ascii")
            + b" /Kids ["
            + b" ".join(
                f"{page_object} 0 R".encode("ascii")
                for page_object in range(3, page_count + 3)
            )
            + b"] >>"
        ),
    ]
    objects.extend(
        (
            b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
            + f"/Contents {content_object} 0 R".encode("ascii")
            + b" /Resources << >> >>"
        )
        for _ in range(page_count)
    )
    objects.append(b"<< /Length 0 >>\nstream\n\nendstream")

    output = bytearray(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n")
    offsets = [0]
    for object_number, payload in enumerate(objects, start=1):
        offsets.append(len(output))
        output.extend(f"{object_number} 0 obj\n".encode("ascii"))
        output.extend(payload)
        output.extend(b"\nendobj\n")
    xref_offset = len(output)
    output.extend(f"xref\n0 {len(objects) + 1}\n".encode("ascii"))
    output.extend(b"0000000000 65535 f \n")
    for offset in offsets[1:]:
        output.extend(f"{offset:010d} 00000 n \n".encode("ascii"))
    output.extend(
        (
            f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\n"
            f"startxref\n{xref_offset}\n%%EOF\n"
        ).encode("ascii")
    )
    return bytes(output)


def _recompute_request(request: dict[str, Any]) -> None:
    for scope in request["scopes"]:
        table = scope["matched_table"]
        table["row_count"] = len(table["rows"])
        table["column_count"] = len(table["headers"])
        table["table_content_digest"] = _stable_hash(
            {key: value for key, value in table.items() if key != "table_content_digest"}
        )
        scope["scope_id"] = _scope_id(scope)
    request["scopes"].sort(key=lambda scope: scope["scope_id"])
    request["scope_count"] = len(request["scopes"])
    request["semantic_review_required_count"] = sum(
        bool(scope["semantic_review_required"]) for scope in request["scopes"]
    )
    request["request_digest"] = _stable_hash(
        {key: value for key, value in request.items() if key != "request_digest"}
    )
    OledSupplementaryScopedCandidateRequestArtifact.model_validate(request)


def _response_for_request(
    request_path: Path,
    request: dict[str, Any],
) -> dict[str, Any]:
    first = deepcopy(request)
    first["scopes"] = [deepcopy(request["scopes"][0])]
    payload = _response_payload(request_path, first)
    first_scope_result = payload["scope_results"][0]
    scope_results = []
    for scope in request["scopes"]:
        table = scope["matched_table"]
        result = deepcopy(first_scope_result)
        result.update(
            {
                "scope_id": scope["scope_id"],
                "source_review_item_digest": scope["source_review_item_digest"],
                "source_pdf_sha256": scope["source_pdf_sha256"],
                "parsed_document_sha256": scope["parsed_document_sha256"],
                "table_id": table["table_id"],
                "table_content_digest": table["table_content_digest"],
                "semantic_note": scope["semantic_note"],
                "semantic_note_status": (
                    "unresolved" if scope["semantic_review_required"] else "not_applicable"
                ),
                "subject_column_name": table["headers"][0],
            }
        )
        for disposition in result["cell_dispositions"]:
            row_index = disposition["row_index"]
            column_index = disposition["column_index"]
            column_name = table["headers"][column_index]
            row = table["rows"][row_index]
            disposition.update(
                {
                    "scope_id": scope["scope_id"],
                    "table_id": table["table_id"],
                    "table_content_digest": table["table_content_digest"],
                    "column_name": column_name,
                    "cell_value": row[column_name],
                    "reported_value_text": row[column_name],
                    "subject_column_name": table["headers"][0],
                    "reported_subject_text": row[table["headers"][0]],
                }
            )
            if "property_label" in disposition:
                disposition["property_label"] = column_name
        scope_results.append(result)
    payload.update(
        {
            "request_artifact_sha256": _sha256_file(request_path),
            "request_digest": request["request_digest"],
            "scope_results": scope_results,
        }
    )
    return payload


def _build_semantic_chain(
    tmp_path: Path,
    *,
    scope_count: int = 1,
    unsafe_source_markup: bool = False,
) -> dict[str, Any]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    source_pdf_path = tmp_path / "paper016-si.pdf"
    source_pdf_path.write_bytes(_minimal_pdf_bytes())
    request = _request_payload()
    base_scope = request["scopes"][0]
    if unsafe_source_markup:
        base_scope["matched_table"]["caption"] = (
            "<script>alert(1)</script>|caption\ncontinued\u202e"
        )
        base_scope["matched_table"]["footnotes"] = [
            "<b>footnote</b>|line one\nline two\u202e"
        ]
    scopes = []
    for index in range(scope_count):
        scope = deepcopy(base_scope)
        scope["review_item_id"] = (
            f"supplementary-locator-review:supplementary-recovery:item-{index + 1:03d}"
        )
        scope["recovery_item_id"] = f"supplementary-recovery:item-{index + 1:03d}"
        scope["source_review_item_digest"] = "sha256:" + f"{index + 1:x}" * 64
        scope["source_review_item_digest"] = scope["source_review_item_digest"][:71]
        scope["source_pdf_sha256"] = _sha256_file(source_pdf_path)
        scope["matched_table"]["table_id"] = f"table_p{38 - index}_{index + 1:04d}"
        scope["matched_table"]["page"] = 38 - index
        scope["matched_table"]["caption"] = (
            f"Supplementary Table S{index + 1}. TD-DFT properties"
            if not unsafe_source_markup
            else (
                f"<script>alert({index + 1})</script>|caption\n"
                "continued\u202e"
            )
        )
        scope["target_locator"] = f"S{index + 1}"
        scope["canonical_locator"] = f"S{index + 1}"
        scopes.append(scope)
    request["scopes"] = scopes
    _recompute_request(request)
    request_path = tmp_path / "scoped-candidate-request.json"
    write_json(request_path, request)

    response = _response_for_request(request_path, request)
    response_path = tmp_path / "scoped-candidate-response-manifest.json"
    write_json(response_path, response)
    response_artifact_path = tmp_path / "scoped-candidate-response.json"
    build_oled_supplementary_scoped_candidate_response_from_files(
        request_artifact_json=request_path,
        response_manifest_json=response_path,
        output_json=response_artifact_path,
        generated_at=_RESPONSE_GENERATED_AT,
    )

    semantic_packet_path = tmp_path / "semantic-review-packet.json"
    semantic_packet = build_oled_supplementary_semantic_review_packet_from_files(
        request_artifact_json=request_path,
        response_manifest_json=response_path,
        response_artifact_json=response_artifact_path,
        output_json=semantic_packet_path,
        generated_at=_SEMANTIC_PACKET_GENERATED_AT,
    )
    semantic_decision_path = tmp_path / "semantic-decisions.json"
    semantic_decisions = _decision_payload(semantic_packet, semantic_packet_path)
    write_json(semantic_decision_path, semantic_decisions)
    semantic_adjudication_path = tmp_path / "semantic-adjudication.json"
    semantic_adjudication = build_oled_supplementary_semantic_adjudication_from_files(
        request_artifact_json=request_path,
        response_manifest_json=response_path,
        response_artifact_json=response_artifact_path,
        review_packet_json=semantic_packet_path,
        decision_manifest_json=semantic_decision_path,
        output_json=semantic_adjudication_path,
        generated_at=_SEMANTIC_ADJUDICATED_AT,
    )
    return {
        "source_pdf_path": source_pdf_path,
        "request_path": request_path,
        "response_path": response_path,
        "response_artifact_path": response_artifact_path,
        "semantic_packet_path": semantic_packet_path,
        "semantic_decision_path": semantic_decision_path,
        "semantic_adjudication_path": semantic_adjudication_path,
        "semantic_adjudication": semantic_adjudication,
        "request": request,
    }


def _load_semantic_chain_models(chain: dict[str, Any]) -> dict[str, Any]:
    return {
        "request": OledSupplementaryScopedCandidateRequestArtifact.model_validate_json(
            chain["request_path"].read_text(encoding="utf-8")
        ),
        "response_manifest": OledSupplementaryScopedCandidateResponseManifest.model_validate_json(
            chain["response_path"].read_text(encoding="utf-8")
        ),
        "response_artifact": OledSupplementaryScopedCandidateResponseArtifact.model_validate_json(
            chain["response_artifact_path"].read_text(encoding="utf-8")
        ),
        "semantic_packet": OledSupplementarySemanticReviewPacket.model_validate_json(
            chain["semantic_packet_path"].read_text(encoding="utf-8")
        ),
        "semantic_decisions": OledSupplementarySemanticDecisionManifest.model_validate_json(
            chain["semantic_decision_path"].read_text(encoding="utf-8")
        ),
        "semantic_adjudication": OledSupplementarySemanticAdjudicationArtifact.model_validate_json(
            chain["semantic_adjudication_path"].read_text(encoding="utf-8")
        ),
    }


def _build_domain_packet(
    tmp_path: Path,
    *,
    scope_count: int = 1,
    generated_at: str = _TRANSCRIPTION_PACKET_GENERATED_AT,
    unsafe_source_markup: bool = False,
) -> tuple[
    dict[str, Any],
    OledSupplementarySourceTranscriptionReviewPacket,
    Path,
    Path,
]:
    chain = _build_semantic_chain(
        tmp_path,
        scope_count=scope_count,
        unsafe_source_markup=unsafe_source_markup,
    )
    models = _load_semantic_chain_models(chain)
    request = models["request"]
    source_pdf_path = chain["source_pdf_path"]
    page_assets = []
    asset_dir = tmp_path / "assets"
    asset_dir.mkdir()
    for page in sorted({scope.matched_table.page for scope in request.scopes}):
        asset = build_oled_supplementary_source_page_asset(
            source_id=request.scopes[0].source_id,
            source_pdf_sha256=_sha256_file(source_pdf_path),
            pdf_page_number_one_based=page,
            renderer_id="poppler-pdftoppm",
            renderer_version="26.05.0",
            render_profile="png-200dpi-rgb-full-page-v1",
            rendered_asset_sha256=(
                "sha256:" + hashlib.sha256(_FAKE_PNG_BYTES).hexdigest()
            ),
            rendered_asset_byte_size=len(_FAKE_PNG_BYTES),
            pixel_width=1700,
            pixel_height=2200,
        )
        (asset_dir / asset.asset_filename).write_bytes(_FAKE_PNG_BYTES)
        page_assets.append(asset)
    evidence = build_oled_supplementary_source_pdf_evidence(
        source_id=request.scopes[0].source_id,
        source_pdf_sha256=_sha256_file(source_pdf_path),
        source_pdf_byte_size=source_pdf_path.stat().st_size,
        source_pdf_page_count=39,
        page_counter_version="26.05.0",
        page_counter_executable_sha256="sha256:" + "c" * 64,
        renderer_executable_sha256="sha256:" + "d" * 64,
        page_assets=page_assets,
    )
    packet = build_oled_supplementary_source_transcription_review_packet(
        request_artifact=models["request"],
        request_artifact_sha256=_sha256_file(chain["request_path"]),
        response_manifest=models["response_manifest"],
        response_manifest_sha256=_sha256_file(chain["response_path"]),
        response_artifact=models["response_artifact"],
        response_artifact_sha256=_sha256_file(chain["response_artifact_path"]),
        semantic_review_packet=models["semantic_packet"],
        semantic_review_packet_sha256=_sha256_file(chain["semantic_packet_path"]),
        semantic_decision_manifest=models["semantic_decisions"],
        semantic_decision_manifest_sha256=_sha256_file(
            chain["semantic_decision_path"]
        ),
        semantic_adjudication_artifact=models["semantic_adjudication"],
        semantic_adjudication_artifact_sha256=_sha256_file(
            chain["semantic_adjudication_path"]
        ),
        source_pdf_evidence=evidence,
        generated_at=generated_at,
    )
    packet_path = tmp_path / "source-transcription-review-packet.json"
    write_json(packet_path, packet.model_dump(mode="json"))
    chain.update({"models": models, "source_pdf_evidence": evidence})
    return chain, packet, packet_path, asset_dir


def _component_results(
    value: str = "verified_equivalent",
    **overrides: str,
) -> dict[str, str]:
    payload = {field_name: value for field_name in _COMPONENT_FIELDS}
    payload.update(overrides)
    return payload


def _transcription_decision_payload(
    packet: OledSupplementarySourceTranscriptionReviewPacket,
    packet_path: Path,
    *,
    decisions_by_scope: dict[str, tuple[str, dict[str, str], str]] | None = None,
) -> dict[str, Any]:
    decisions = []
    for item in packet.review_items:
        decision, results, note = (
            decisions_by_scope[item.scope_id]
            if decisions_by_scope is not None
            else (
                "accept_bounded_source_transcription",
                _component_results(),
                "",
            )
        )
        decisions.append(
            {
                "review_item_id": item.review_item_id,
                "review_item_digest": item.review_item_digest,
                "item_kind": item.item_kind.value,
                "decision": decision,
                "component_results": results,
                "review_note": note,
            }
        )
    return {
        "schema_version": (
            "oled_supplementary_source_transcription_decision_manifest.v1"
        ),
        "run_id": packet.run_id,
        "paper_id": packet.paper_id,
        "review_packet_sha256": _sha256_file(packet_path),
        "review_packet_digest": packet.review_packet_digest,
        "source_pdf_evidence_digest": packet.source_pdf_evidence_digest,
        "reviewed_by": "Benton",
        "reviewed_at": _TRANSCRIPTION_REVIEWED_AT,
        "adjudication_confirmed": True,
        "decisions": decisions,
    }


def _adjudicate_domain(
    chain: dict[str, Any],
    packet: OledSupplementarySourceTranscriptionReviewPacket,
    packet_path: Path,
    decision_payload: dict[str, Any],
) -> OledSupplementarySourceTranscriptionAdjudicationArtifact:
    models = chain["models"]
    decisions = OledSupplementarySourceTranscriptionDecisionManifest.model_validate(
        decision_payload
    )
    decision_bytes = (
        json.dumps(decision_payload, ensure_ascii=False, indent=2) + "\n"
    ).encode("utf-8")
    return build_oled_supplementary_source_transcription_adjudication_artifact(
        request_artifact=models["request"],
        request_artifact_sha256=_sha256_file(chain["request_path"]),
        response_manifest=models["response_manifest"],
        response_manifest_sha256=_sha256_file(chain["response_path"]),
        response_artifact=models["response_artifact"],
        response_artifact_sha256=_sha256_file(chain["response_artifact_path"]),
        semantic_review_packet=models["semantic_packet"],
        semantic_review_packet_sha256=_sha256_file(chain["semantic_packet_path"]),
        semantic_decision_manifest=models["semantic_decisions"],
        semantic_decision_manifest_sha256=_sha256_file(
            chain["semantic_decision_path"]
        ),
        semantic_adjudication_artifact=models["semantic_adjudication"],
        semantic_adjudication_artifact_sha256=_sha256_file(
            chain["semantic_adjudication_path"]
        ),
        source_pdf_evidence=chain["source_pdf_evidence"],
        review_packet=packet,
        review_packet_sha256=_sha256_file(packet_path),
        decision_manifest=decisions,
        decision_manifest_sha256=(
            "sha256:" + hashlib.sha256(decision_bytes).hexdigest()
        ),
        generated_at=_TRANSCRIPTION_ADJUDICATED_AT,
    )


def _install_fake_renderer(
    monkeypatch: pytest.MonkeyPatch,
    chain: dict[str, Any],
) -> None:
    request = _load_semantic_chain_models(chain)["request"]
    rendered_assets: dict[str, bytes] = {}
    page_assets = []
    for page in sorted({scope.matched_table.page for scope in request.scopes}):
        payload = _FAKE_PNG_BYTES
        asset = build_oled_supplementary_source_page_asset(
            source_id=request.scopes[0].source_id,
            source_pdf_sha256=_sha256_file(chain["source_pdf_path"]),
            pdf_page_number_one_based=page,
            renderer_id="poppler-pdftoppm",
            renderer_version="test-26.05.0",
            render_profile="png-200dpi-rgb-full-page-v1",
            rendered_asset_sha256=(
                "sha256:" + hashlib.sha256(payload).hexdigest()
            ),
            rendered_asset_byte_size=len(payload),
            pixel_width=1700,
            pixel_height=2200,
        )
        page_assets.append(asset)
        rendered_assets[asset.asset_filename] = payload
    evidence = build_oled_supplementary_source_pdf_evidence(
        source_id=request.scopes[0].source_id,
        source_pdf_sha256=_sha256_file(chain["source_pdf_path"]),
        source_pdf_byte_size=chain["source_pdf_path"].stat().st_size,
        source_pdf_page_count=39,
        page_counter_version="test-26.05.0",
        page_counter_executable_sha256="sha256:" + "c" * 64,
        renderer_executable_sha256="sha256:" + "d" * 64,
        page_assets=page_assets,
    )

    def fake_render_bound_source_pdf(**_: Any) -> tuple[Any, dict[str, bytes]]:
        return evidence, rendered_assets

    monkeypatch.setattr(
        transcription_runner,
        "_render_bound_source_pdf",
        fake_render_bound_source_pdf,
    )


def _packet_file_kwargs(
    chain: dict[str, Any],
    *,
    asset_dir: Path,
    output_path: Path,
    poppler_bin_dir: Path | None = None,
) -> dict[str, Any]:
    return {
        "request_artifact_json": chain["request_path"],
        "response_manifest_json": chain["response_path"],
        "response_artifact_json": chain["response_artifact_path"],
        "semantic_review_packet_json": chain["semantic_packet_path"],
        "semantic_decision_manifest_json": chain["semantic_decision_path"],
        "semantic_adjudication_json": chain["semantic_adjudication_path"],
        "source_pdf_path": chain["source_pdf_path"],
        "poppler_bin_dir": poppler_bin_dir,
        "asset_dir": asset_dir,
        "output_json": output_path,
        "generated_at": _TRANSCRIPTION_PACKET_GENERATED_AT,
    }


def _build_file_packet(
    tmp_path: Path,
    *,
    monkeypatch: pytest.MonkeyPatch | None = None,
    use_real_renderer: bool = False,
    poppler_bin_dir: Path | None = None,
) -> tuple[
    dict[str, Any],
    OledSupplementarySourceTranscriptionReviewPacket,
    Path,
    Path,
]:
    chain = _build_semantic_chain(tmp_path)
    if not use_real_renderer:
        assert monkeypatch is not None
        _install_fake_renderer(monkeypatch, chain)
    else:
        poppler_bin_dir = poppler_bin_dir or _real_poppler_bin_dir()
        chain["poppler_bin_dir"] = poppler_bin_dir
    packet_path = tmp_path / "source-transcription-review-packet.json"
    asset_dir = tmp_path / "assets"
    packet = build_oled_supplementary_source_transcription_review_packet_from_files(
        **_packet_file_kwargs(
            chain,
            asset_dir=asset_dir,
            output_path=packet_path,
            poppler_bin_dir=poppler_bin_dir,
        )
    )
    return chain, packet, packet_path, asset_dir


def _write_transcription_decisions(
    tmp_path: Path,
    packet: OledSupplementarySourceTranscriptionReviewPacket,
    packet_path: Path,
) -> tuple[Path, dict[str, Any]]:
    payload = _transcription_decision_payload(packet, packet_path)
    decision_path = tmp_path / "source-transcription-decisions.json"
    write_json(decision_path, payload)
    return decision_path, payload


def _adjudicate_from_files(
    chain: dict[str, Any],
    *,
    packet_path: Path,
    decision_path: Path,
    asset_dir: Path,
    output_path: Path,
) -> OledSupplementarySourceTranscriptionAdjudicationArtifact:
    return build_oled_supplementary_source_transcription_adjudication_from_files(
        request_artifact_json=chain["request_path"],
        response_manifest_json=chain["response_path"],
        response_artifact_json=chain["response_artifact_path"],
        semantic_review_packet_json=chain["semantic_packet_path"],
        semantic_decision_manifest_json=chain["semantic_decision_path"],
        semantic_adjudication_json=chain["semantic_adjudication_path"],
        source_pdf_path=chain["source_pdf_path"],
        poppler_bin_dir=chain.get("poppler_bin_dir"),
        transcription_review_packet_json=packet_path,
        transcription_decision_manifest_json=decision_path,
        asset_dir=asset_dir,
        output_json=output_path,
        generated_at=_TRANSCRIPTION_ADJUDICATED_AT,
    )


def test_request_bound_renderer_delegates_exact_source_hash_and_pages(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    expected_sha256 = "sha256:" + "a" * 64
    source_path = Path("/operator/source.pdf")
    poppler_path = Path("/operator/poppler/bin")
    evidence = SimpleNamespace(marker="evidence")
    rendered_assets = {"source-page.png": b"rendered"}
    observed: dict[str, Any] = {}

    def fake_render_exact_bound_source_pdf_pages(**kwargs: Any) -> tuple[Any, Any]:
        observed.update(kwargs)
        return evidence, rendered_assets

    monkeypatch.setattr(
        transcription_runner,
        "_render_exact_bound_source_pdf_pages",
        fake_render_exact_bound_source_pdf_pages,
    )
    request = SimpleNamespace(
        scopes=[
            SimpleNamespace(
                source_id="paper016",
                source_pdf_sha256=expected_sha256,
                matched_table=SimpleNamespace(page=38),
            ),
            SimpleNamespace(
                source_id="paper016",
                source_pdf_sha256=expected_sha256,
                matched_table=SimpleNamespace(page=12),
            ),
            SimpleNamespace(
                source_id="paper016",
                source_pdf_sha256=expected_sha256,
                matched_table=SimpleNamespace(page=38),
            ),
        ]
    )

    actual_evidence, actual_assets = transcription_runner._render_bound_source_pdf(
        source_pdf_path=source_path,
        request_artifact=request,
        poppler_bin_dir=poppler_path,
    )

    assert actual_evidence is evidence
    assert actual_assets is rendered_assets
    assert observed == {
        "source_pdf_path": source_path,
        "source_id": "paper016",
        "expected_sha256": expected_sha256,
        "pages": [12, 38],
        "poppler_bin_dir": poppler_path,
    }


@pytest.mark.parametrize(
    "pages",
    (
        [],
        [0],
        [-1],
        [True],
        [1.5],
        [1, 1],
        [2, 1],
    ),
    ids=(
        "empty",
        "zero",
        "negative",
        "boolean",
        "non-integer",
        "duplicate",
        "unsorted",
    ),
)
def test_exact_bound_source_pdf_renderer_rejects_invalid_page_roster(
    pages: list[Any],
) -> None:
    with pytest.raises(ValueError, match="page roster|sorted unique positive"):
        transcription_runner._render_exact_bound_source_pdf_pages(
            source_pdf_path=Path("/must-not-be-opened.pdf"),
            source_id="paper016",
            expected_sha256="sha256:" + "a" * 64,
            pages=pages,
        )


@pytest.mark.parametrize(
    ("source_id", "expected_sha256", "message"),
    (
        ("../paper016", "sha256:" + "a" * 64, "safe path segment"),
        (" paper016 ", "sha256:" + "a" * 64, "source_id must be exact"),
        ("paper016", "sha256:bad", "SHA-256 digest"),
    ),
)
def test_exact_bound_source_pdf_renderer_rejects_invalid_source_binding(
    source_id: str,
    expected_sha256: str,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        transcription_runner._render_exact_bound_source_pdf_pages(
            source_pdf_path=Path("/must-not-be-opened.pdf"),
            source_id=source_id,
            expected_sha256=expected_sha256,
            pages=[1],
        )


def test_paper016_packet_binds_one_complete_table_and_replays_pr_g_h_i(
    tmp_path: Path,
) -> None:
    chain, packet, packet_path, asset_dir = _build_domain_packet(tmp_path)

    assert packet.status.value == "ready_for_human_source_transcription_review"
    assert packet.scope_count == packet.review_item_count == 1
    assert packet.page_asset_count == 1
    assert packet.full_table_cell_count == 56
    assert packet.numeric_source_cell_count == 49
    assert packet.upstream_later_eligible_cell_count == 35
    assert packet.upstream_ontology_review_pending_cell_count == 14
    assert packet.upstream_source_check_pending_cell_count == 0
    assert packet.upstream_rejected_cell_count == 0
    assert packet.request_artifact_sha256 == _sha256_file(chain["request_path"])
    assert packet.response_manifest_sha256 == _sha256_file(chain["response_path"])
    assert packet.response_artifact_sha256 == _sha256_file(
        chain["response_artifact_path"]
    )
    assert packet.semantic_review_packet_sha256 == _sha256_file(
        chain["semantic_packet_path"]
    )
    assert packet.semantic_decision_manifest_sha256 == _sha256_file(
        chain["semantic_decision_path"]
    )
    assert packet.semantic_adjudication_artifact_sha256 == _sha256_file(
        chain["semantic_adjudication_path"]
    )
    assert packet.source_pdf_evidence.source_pdf_sha256 == _sha256_file(
        chain["source_pdf_path"]
    )

    item = packet.review_items[0]
    table = item.matched_table
    assert item.item_kind.value == "table_transcription_scope"
    assert len(table.headers) == table.column_count == 8
    assert len(table.rows) == table.row_count == 7
    assert item.full_table_cell_count == 56
    assert item.numeric_source_cell_count == len(item.source_cell_digests) == 49
    assert item.upstream_later_eligible_cell_count == 35
    assert item.upstream_ontology_review_pending_cell_count == 14
    assert len(item.header_review_bindings) == len(table.headers) == 8
    first_header_binding = item.header_review_bindings[0]
    assert first_header_binding.column_index == 0
    assert first_header_binding.parser_key == "column_1"
    assert first_header_binding.source_visible_header_candidate == ""
    assert (
        first_header_binding.binding_kind.value
        == "parser_placeholder_candidate_for_blank_header"
    )
    assert all(
        binding.binding_kind.value == "reported_literal"
        and binding.source_visible_header_candidate == binding.parser_key
        for binding in item.header_review_bindings[1:]
    )
    assert len(table.footnotes) == 1
    assert {
        row["column_1"]: row
        for row in table.rows
    }["TDBA-Si"] == {
        "column_1": "TDBA-Si",
        "HOMO (eV)": "-1.70",
        "LUMO (eV)": "-5.50",
        "$\\Delta E_{\\text{HOMO} \\rightarrow \\text{LUMO}}$ (eV)": "3.80",
        "$S_1$ (eV)": "3.30",
        "$T_1$ (eV)": "2.78",
        "$\\Delta E_{ST}^a$ (eV)": "0.52",
        "$f(S_0-S_1)^b$": "0.1280",
    }
    assert set(item.component_digests.model_dump()) == {
        name.removesuffix("_check") for name in _COMPONENT_FIELDS
    }
    asset = packet.source_pdf_evidence.page_assets[0]
    assert asset.pdf_page_number_one_based == 38
    assert asset.render_profile == "png-200dpi-rgb-full-page-v1"
    assert asset.full_page_rendered is True
    assert asset.source_bbox_crop_applied is False
    assert (asset_dir / asset.asset_filename).is_file()
    assert OledSupplementarySourceTranscriptionReviewPacket.model_validate_json(
        packet_path.read_text(encoding="utf-8")
    ) == packet
    assert packet.upstream_chain_replayed is True
    assert packet.strict_scope_partition_validated is True
    assert packet.strict_cell_partition_validated is True
    assert packet.all_reviewed_scopes_transcription_validated is False
    assert packet.schema_candidates_created is False
    assert packet.gold_records_created is False
    assert packet.dataset_written is False


@pytest.mark.parametrize(
    ("field_name", "invalid_value", "message"),
    [
        (
            "renderer_id",
            "imagemagick-convert",
            "requires the bound Poppler renderer",
        ),
        (
            "render_profile",
            "png-rgb-200dpi-full-page",
            "requires the bound 200 dpi RGB full-page render profile",
        ),
    ],
)
def test_page_asset_rejects_unbound_renderer_or_render_profile(
    field_name: str,
    invalid_value: str,
    message: str,
) -> None:
    kwargs = {
        "source_id": "paper016",
        "source_pdf_sha256": "sha256:" + "a" * 64,
        "pdf_page_number_one_based": 38,
        "renderer_id": "poppler-pdftoppm",
        "renderer_version": "26.05.0",
        "render_profile": "png-200dpi-rgb-full-page-v1",
        "rendered_asset_sha256": "sha256:" + "b" * 64,
        "rendered_asset_byte_size": len(_FAKE_PNG_BYTES),
        "pixel_width": 1700,
        "pixel_height": 2200,
    }
    kwargs[field_name] = invalid_value

    with pytest.raises(ValidationError, match=message):
        build_oled_supplementary_source_page_asset(**kwargs)


def test_source_pdf_evidence_rejects_two_assets_for_one_page() -> None:
    common = {
        "source_id": "paper016",
        "source_pdf_sha256": "sha256:" + "a" * 64,
        "pdf_page_number_one_based": 38,
        "renderer_id": "poppler-pdftoppm",
        "renderer_version": "26.05.0",
        "render_profile": "png-200dpi-rgb-full-page-v1",
        "rendered_asset_byte_size": len(_FAKE_PNG_BYTES),
        "pixel_width": 1700,
        "pixel_height": 2200,
    }
    first = build_oled_supplementary_source_page_asset(
        **common,
        rendered_asset_sha256="sha256:" + "b" * 64,
    )
    second = build_oled_supplementary_source_page_asset(
        **common,
        rendered_asset_sha256="sha256:" + "e" * 64,
    )

    with pytest.raises(ValidationError, match="one asset per page"):
        build_oled_supplementary_source_pdf_evidence(
            source_id="paper016",
            source_pdf_sha256="sha256:" + "a" * 64,
            source_pdf_byte_size=123,
            source_pdf_page_count=54,
            page_counter_version="26.05.0",
            page_counter_executable_sha256="sha256:" + "c" * 64,
            renderer_executable_sha256="sha256:" + "d" * 64,
            page_assets=[first, second],
        )


def test_rgb_png_fixture_is_complete_and_decodable() -> None:
    assert _FAKE_PNG_BYTES.startswith(_PNG_SIGNATURE)
    assert _FAKE_PNG_BYTES.endswith(_png_chunk(b"IEND", b""))
    assert transcription_runner._png_dimensions(_FAKE_PNG_BYTES) == (1700, 2200)


def _png_with_corrupt_ihdr_crc() -> bytes:
    payload = bytearray(_FAKE_PNG_BYTES)
    payload[32] ^= 0x01
    return bytes(payload)


@pytest.mark.parametrize(
    "payload",
    [
        _PNG_SIGNATURE + b"\x00\x00\x00\rIHDR" + struct.pack(">II", 1700, 2200),
        _png_with_corrupt_ihdr_crc(),
        _FAKE_PNG_BYTES[:-20],
        _png_bytes(color_type=6),
        _png_bytes(width=1, height=1, bit_depth=16),
        _png_bytes(width=20_001, height=1),
        _png_with_declared_dimensions(width=10_000, height=10_000),
    ],
    ids=(
        "24-byte-stub",
        "bad-crc",
        "truncated",
        "rgba",
        "16-bit-rgb",
        "oversized-dimension",
        "oversized-total-pixels",
    ),
)
def test_png_validator_rejects_malformed_or_unbound_png(payload: bytes) -> None:
    with pytest.raises(ValueError, match="PNG|asset"):
        transcription_runner._png_dimensions(payload)


def test_render_pdf_page_rejects_wrong_full_page_dimensions(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_pdf = tmp_path / "source.pdf"
    source_pdf.write_bytes(_minimal_pdf_bytes(page_count=1))

    def fake_run_renderer(
        _toolchain: Any,
        _executable: Any,
        arguments: Any,
        **_: Any,
    ) -> None:
        Path(arguments[-1]).with_suffix(".png").write_bytes(
            _png_bytes(width=1700, height=2190)
        )

    monkeypatch.setattr(transcription_runner, "_run_renderer", fake_run_renderer)

    with pytest.raises(ValueError, match="full-page|dimension|size"):
        transcription_runner._render_pdf_page(
            source_pdf,
            pdf_descriptor=-1,
            page=1,
            temp_dir=tmp_path,
            toolchain=SimpleNamespace(pdftoppm=object()),
            allowed_pixel_widths={1699, 1700, 1701},
            allowed_pixel_heights={2199, 2200, 2201},
        )


def test_render_pdf_page_rejects_symlink_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_pdf = tmp_path / "source.pdf"
    source_pdf.write_bytes(_minimal_pdf_bytes(page_count=1))
    external_png = tmp_path / "external.png"
    external_payload = _FAKE_PNG_BYTES
    external_png.write_bytes(external_payload)

    def fake_run_renderer(
        _toolchain: Any,
        _executable: Any,
        arguments: Any,
        **_: Any,
    ) -> None:
        Path(arguments[-1]).with_suffix(".png").symlink_to(external_png)

    monkeypatch.setattr(transcription_runner, "_run_renderer", fake_run_renderer)

    with pytest.raises(ValueError, match="PNG|regular|symlink|unavailable"):
        transcription_runner._render_pdf_page(
            source_pdf,
            pdf_descriptor=-1,
            page=1,
            temp_dir=tmp_path,
            toolchain=SimpleNamespace(pdftoppm=object()),
            allowed_pixel_widths={1699, 1700, 1701},
            allowed_pixel_heights={2199, 2200, 2201},
        )

    assert external_png.read_bytes() == external_payload


def test_default_poppler_resolution_never_executes_an_inherited_path_shim(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    shim_dir, marker = _path_shim_directory(tmp_path)
    monkeypatch.setenv("PATH", str(shim_dir))

    try:
        with transcription_runner._pinned_poppler_toolchain(
            None,
            execution_root=tmp_path,
        ) as toolchain:
            transcription_runner._poppler_version(
                toolchain,
                toolchain.pdfinfo,
                command_name="pdfinfo",
                working_dir=tmp_path,
            )
    except ValueError:
        # A platform without Poppler in the fixed /usr/bin trust root fails closed.
        pass

    assert not marker.exists()


def test_explicit_shell_wrapper_toolchain_is_rejected_without_execution(
    tmp_path: Path,
) -> None:
    shim_dir, marker = _path_shim_directory(tmp_path)

    with pytest.raises(ValueError, match="native binary"):
        with transcription_runner._pinned_poppler_toolchain(
            shim_dir,
            execution_root=tmp_path,
        ):
            pytest.fail("shell-wrapper toolchain unexpectedly became trusted")

    assert not marker.exists()


def test_explicit_poppler_directory_ignores_a_hostile_inherited_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    poppler_bin_dir = _real_poppler_bin_dir()
    shim_dir, marker = _path_shim_directory(tmp_path)
    monkeypatch.setenv("PATH", str(shim_dir))

    _, packet, packet_path, asset_dir = _build_file_packet(
        tmp_path / "explicit-toolchain",
        use_real_renderer=True,
        poppler_bin_dir=poppler_bin_dir,
    )

    assert packet_path.is_file()
    assert asset_dir.is_dir()
    assert packet.source_pdf_evidence.page_counter_executable_sha256.startswith(
        "sha256:"
    )
    assert packet.source_pdf_evidence.renderer_executable_sha256.startswith(
        "sha256:"
    )
    assert not marker.exists()


def test_poppler_execution_paths_are_private_copies_of_pinned_bytes(
    tmp_path: Path,
) -> None:
    poppler_bin_dir = _real_poppler_bin_dir()

    with transcription_runner._pinned_poppler_toolchain(
        poppler_bin_dir,
        execution_root=tmp_path,
    ) as toolchain:
        for executable in (toolchain.pdfinfo, toolchain.pdftoppm):
            assert executable.path.parent == tmp_path / "pinned-poppler-bin"
            assert executable.path.parent != poppler_bin_dir
            assert _sha256_file(executable.path) == executable.sha256
            assert _sha256_file(poppler_bin_dir / executable.path.name) == (
                executable.sha256
            )


def test_bound_pdf_is_unlinked_and_consumed_through_its_pinned_descriptor(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.pdf"
    source_bytes = _minimal_pdf_bytes(page_count=2)
    source.write_bytes(source_bytes)
    destination = tmp_path / "bound-source.pdf"

    digest, byte_size, descriptor = transcription_runner._copy_and_hash_pdf(
        source,
        destination,
    )
    try:
        assert not destination.exists()
        with pytest.raises(OSError):
            os.write(descriptor, b"forbidden")
        destination.write_bytes(_minimal_pdf_bytes(page_count=3))
        os.lseek(descriptor, 0, os.SEEK_SET)
        pinned_bytes = os.read(descriptor, byte_size + 1)
        assert pinned_bytes == source_bytes
        assert "sha256:" + hashlib.sha256(pinned_bytes).hexdigest() == digest
        assert Path(transcription_runner._descriptor_reference(descriptor)).name == (
            str(descriptor)
        )
    finally:
        os.close(descriptor)


def test_renderer_log_limit_fails_closed_without_buffering_unbounded_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeProcess:
        pid = 41001

        def wait(self, timeout: int | None = None) -> int:
            assert timeout == transcription_runner._RENDER_TIMEOUT_SECONDS
            return 0

    def fake_popen(_command: Any, **kwargs: Any) -> FakeProcess:
        kwargs["stdout"].write(
            b"x" * (transcription_runner._MAX_RENDER_LOG_BYTES + 1)
        )
        kwargs["stdout"].flush()
        return FakeProcess()

    monkeypatch.setattr(
        transcription_runner,
        "_validate_pinned_toolchain",
        lambda _toolchain: None,
    )
    monkeypatch.setattr(transcription_runner.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(
        transcription_runner,
        "_kill_renderer_process_group",
        lambda _process_id: None,
    )

    with pytest.raises(ValueError, match="log exceeded the size limit"):
        transcription_runner._run_renderer(
            SimpleNamespace(bin_dir=tmp_path),
            SimpleNamespace(path=Path("/trusted/pdftoppm")),
            ["-v"],
            working_dir=tmp_path,
        )


def test_renderer_timeout_kills_the_process_group(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    wait_timeouts: list[int | None] = []
    killed_process_groups: list[int] = []

    class FakeProcess:
        pid = 41002

        def wait(self, timeout: int | None = None) -> int:
            wait_timeouts.append(timeout)
            if timeout is not None:
                raise transcription_runner.subprocess.TimeoutExpired(
                    cmd="pdftoppm",
                    timeout=timeout,
                )
            return -9

    monkeypatch.setattr(
        transcription_runner,
        "_validate_pinned_toolchain",
        lambda _toolchain: None,
    )
    monkeypatch.setattr(
        transcription_runner.subprocess,
        "Popen",
        lambda _command, **_kwargs: FakeProcess(),
    )
    monkeypatch.setattr(
        transcription_runner,
        "_kill_renderer_process_group",
        killed_process_groups.append,
    )

    with pytest.raises(ValueError, match="renderer timed out"):
        transcription_runner._run_renderer(
            SimpleNamespace(bin_dir=tmp_path),
            SimpleNamespace(path=Path("/trusted/pdftoppm")),
            ["-f", "1"],
            working_dir=tmp_path,
        )

    assert wait_timeouts == [transcription_runner._RENDER_TIMEOUT_SECONDS, None]
    assert killed_process_groups
    assert set(killed_process_groups) == {FakeProcess.pid}


@pytest.mark.parametrize(
    "component_name",
    [name.removesuffix("_check") for name in _COMPONENT_FIELDS],
)
def test_packet_model_rejects_each_component_digest_tamper(
    tmp_path: Path,
    component_name: str,
) -> None:
    _, packet, _, _ = _build_domain_packet(tmp_path)
    payload = packet.model_dump(mode="json")
    payload["review_items"][0]["component_digests"][component_name] = (
        "sha256:" + "f" * 64
    )
    payload["review_packet_digest"] = transcription_domain._stable_hash(
        {
            key: value
            for key, value in payload.items()
            if key != "review_packet_digest"
        }
    )

    with pytest.raises(ValidationError, match="component digest mismatch"):
        OledSupplementarySourceTranscriptionReviewPacket.model_validate(payload)


@pytest.mark.parametrize(
    "tamper",
    ("coverage", "index", "kind", "candidate", "parser-key"),
)
def test_packet_rejects_header_review_binding_tamper(
    tmp_path: Path,
    tamper: str,
) -> None:
    _, packet, _, _ = _build_domain_packet(tmp_path)
    payload = packet.model_dump(mode="json")
    bindings = payload["review_items"][0]["header_review_bindings"]
    if tamper == "coverage":
        bindings.pop()
    elif tamper == "index":
        bindings[1]["column_index"] = 7
    elif tamper == "kind":
        bindings[1]["binding_kind"] = (
            "parser_placeholder_candidate_for_blank_header"
        )
    elif tamper == "candidate":
        bindings[0]["source_visible_header_candidate"] = "column_1"
    else:
        bindings[1]["parser_key"] = "Changed header"
        bindings[1]["source_visible_header_candidate"] = "Changed header"
    payload["review_packet_digest"] = transcription_domain._stable_hash(
        {
            key: value
            for key, value in payload.items()
            if key != "review_packet_digest"
        }
    )

    with pytest.raises(
        ValidationError,
        match="header|placeholder|reported-literal|component",
    ):
        OledSupplementarySourceTranscriptionReviewPacket.model_validate(payload)


def test_packet_cannot_predate_pr_i_adjudication(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="predates PR-I adjudication"):
        _build_domain_packet(
            tmp_path,
            generated_at="2026-07-13T22:29:59+08:00",
        )


def test_markdown_shows_one_full_table_one_decision_and_all_bindings(
    tmp_path: Path,
) -> None:
    _, packet, packet_path, _ = _build_domain_packet(tmp_path)
    markdown = render_oled_supplementary_source_transcription_review_markdown(
        packet,
        review_packet_sha256=_sha256_file(packet_path),
    )

    assert markdown.count("## T01:") == 1
    assert markdown.count("### Parsed selected table") == 1
    assert markdown.count("Allowed decisions:") == 1
    assert markdown.count("### Exact bound source page") == 1
    assert markdown.count("### Decision checklist and component bindings") == 1
    assert "56 total / 49 numeric" in markdown
    assert "PR-I later-eligible cells: 35" in markdown
    assert "PR-I ontology-pending cells: 14" in markdown
    assert "0.1280" in markdown
    assert "-1.70" in markdown
    assert "-5.50" in markdown
    assert (
        "source-visible candidate <em>[blank candidate]</em>; parser key "
        "<code>column_1</code>"
    ) in markdown
    assert "|  | <code>HOMO (eV)</code>" in markdown
    assert "| <code>column_1</code> |" not in markdown
    assert packet.review_items[0].review_item_id in markdown
    assert packet.review_items[0].review_item_digest in markdown
    assert _sha256_file(packet_path) in markdown
    for component_name in (
        name.removesuffix("_check") for name in _COMPONENT_FIELDS
    ):
        assert f"`{component_name}_check` (`{component_name}`):" in markdown
    assert "assets/source-page-" in markdown


def test_markdown_escapes_untrusted_source_markup_and_display_controls(
    tmp_path: Path,
) -> None:
    _, packet, packet_path, _ = _build_domain_packet(
        tmp_path,
        unsafe_source_markup=True,
    )
    markdown = render_oled_supplementary_source_transcription_review_markdown(
        packet,
        review_packet_sha256=_sha256_file(packet_path),
    )

    assert "<script>" not in markdown
    assert "&lt;script&gt;" in markdown
    assert "<b>" not in markdown
    assert "&lt;b&gt;" in markdown
    assert "&#124;" in markdown
    assert " ⏎ " in markdown
    assert "\\u202E" in markdown


@pytest.mark.parametrize(
    ("decision", "component_results", "review_note"),
    [
        (
            "accept_bounded_source_transcription",
            _component_results(),
            "",
        ),
        (
            "needs_reparse",
            _component_results(caption_check="mismatch"),
            "The parsed caption differs from the readable source page.",
        ),
        (
            "needs_source_check",
            _component_results(footnotes_check="not_checked"),
            "The source footnote cannot be read from the available page.",
        ),
        (
            "reject_scope",
            _component_results(
                "not_checked",
                page_anchor_check="mismatch",
            ),
            "The rendered page contains the wrong table.",
        ),
    ],
)
def test_four_table_decisions_have_valid_component_result_shapes(
    tmp_path: Path,
    decision: str,
    component_results: dict[str, str],
    review_note: str,
) -> None:
    _, packet, packet_path, _ = _build_domain_packet(tmp_path)
    payload = _transcription_decision_payload(packet, packet_path)
    payload["decisions"][0].update(
        {
            "decision": decision,
            "component_results": component_results,
            "review_note": review_note,
        }
    )

    manifest = OledSupplementarySourceTranscriptionDecisionManifest.model_validate(
        payload
    )

    assert manifest.decisions[0].decision.value == decision
    assert set(manifest.decisions[0].component_results.model_dump()) == set(
        _COMPONENT_FIELDS
    )


@pytest.mark.parametrize(
    ("decision", "component_results", "review_note", "message"),
    [
        (
            "accept_bounded_source_transcription",
            _component_results(caption_check="mismatch"),
            "",
            "every component verified",
        ),
        (
            "accept_bounded_source_transcription",
            _component_results(footnotes_check="not_checked"),
            "",
            "every component verified",
        ),
        (
            "needs_reparse",
            _component_results(),
            "A reparse was requested without a mismatch.",
            "needs_reparse requires",
        ),
        (
            "needs_reparse",
            _component_results(
                caption_check="mismatch",
                footnotes_check="not_checked",
            ),
            "The result mixes mismatch and unchecked states.",
            "no unchecked component",
        ),
        (
            "needs_reparse",
            _component_results(page_anchor_check="mismatch"),
            "The wrong page cannot be treated as a parser-only mismatch.",
            "needs_reparse requires",
        ),
        (
            "needs_source_check",
            _component_results(),
            "No component is actually unchecked.",
            "requires an unchecked component",
        ),
        (
            "needs_source_check",
            _component_results(caption_check="mismatch"),
            "A known mismatch requires a reparse.",
            "no mismatch",
        ),
        (
            "reject_scope",
            _component_results(),
            "",
            "requires review_note",
        ),
    ],
)
def test_incompatible_component_result_and_decision_combinations_fail_closed(
    tmp_path: Path,
    decision: str,
    component_results: dict[str, str],
    review_note: str,
    message: str,
) -> None:
    _, packet, packet_path, _ = _build_domain_packet(tmp_path)
    payload = _transcription_decision_payload(packet, packet_path)
    payload["decisions"][0].update(
        {
            "decision": decision,
            "component_results": component_results,
            "review_note": review_note,
        }
    )

    with pytest.raises(ValidationError, match=message):
        OledSupplementarySourceTranscriptionDecisionManifest.model_validate(payload)


def test_accept_adjudication_preserves_35_eligible_and_14_ontology_cells(
    tmp_path: Path,
) -> None:
    chain, packet, packet_path, _ = _build_domain_packet(tmp_path)
    decisions = _transcription_decision_payload(packet, packet_path)

    artifact = _adjudicate_domain(chain, packet, packet_path, decisions)

    assert artifact.status.value == "ready_for_later_identity_review"
    assert artifact.review_item_count == artifact.accepted_scope_count == 1
    assert artifact.reparse_required_scope_count == 0
    assert artifact.source_check_pending_scope_count == 0
    assert artifact.rejected_scope_count == 0
    assert artifact.unresolved_review_item_count == 0
    assert artifact.bounded_transcription_validated_cell_count == 49
    assert artifact.later_identity_review_eligible_cell_count == 35
    assert artifact.upstream_ontology_review_pending_cell_count == 14
    assert artifact.all_reviewed_scopes_transcription_validated is True
    assert artifact.adjudicated_tables[0].table_transcription_validated is True
    assert artifact.adjudicated_tables[0].bounded_selected_table_extent_validated is True
    assert artifact.adjudicated_tables[0].later_identity_review_eligible_cell_count == 35
    for field_name in (
        "table_exhaustiveness_validated",
        "scientific_content_validated",
        "physical_semantics_validated",
        "material_identity_resolved",
        "source_values_corrected",
        "ontology_extensions_applied",
        "schema_candidates_created",
        "automatic_candidate_merge",
        "reviewed_evidence_staging",
        "direct_admission_eligible",
        "device_only_admitted",
        "gold_records_created",
        "dataset_written",
        "network_accessed",
        "external_service_called",
        "llm_called",
        "mineru_called",
    ):
        assert getattr(artifact, field_name) is False


def test_ready_status_requires_nonempty_identity_eligible_intersection() -> None:
    status = transcription_domain._source_transcription_adjudication_status(
        accepted_count=1,
        unresolved_count=0,
        eligible_cell_count=0,
    )

    assert status.value == "review_complete_no_eligible_scopes"


def test_mixed_accept_reparse_source_check_and_reject_are_scope_local(
    tmp_path: Path,
) -> None:
    chain, packet, packet_path, _ = _build_domain_packet(tmp_path, scope_count=4)
    items = sorted(packet.review_items, key=lambda item: item.review_item_id)
    choices = {
        items[0].scope_id: (
            "accept_bounded_source_transcription",
            _component_results(),
            "",
        ),
        items[1].scope_id: (
            "needs_reparse",
            _component_results(cell_literals_check="mismatch"),
            "One or more parsed cell literals differ from the readable source.",
        ),
        items[2].scope_id: (
            "needs_source_check",
            _component_results(footnotes_check="not_checked"),
            "The source footnote cannot be checked from the available page.",
        ),
        items[3].scope_id: (
            "reject_scope",
            _component_results("not_checked", page_anchor_check="mismatch"),
            "The selected page contains the wrong table.",
        ),
    }
    decisions = _transcription_decision_payload(
        packet,
        packet_path,
        decisions_by_scope=choices,
    )

    artifact = _adjudicate_domain(chain, packet, packet_path, decisions)

    assert artifact.status.value == "review_complete_with_unresolved_items"
    assert artifact.review_item_count == 4
    assert artifact.accepted_scope_count == 1
    assert artifact.reparse_required_scope_count == 1
    assert artifact.source_check_pending_scope_count == 1
    assert artifact.rejected_scope_count == 1
    assert artifact.unresolved_review_item_count == 2
    assert artifact.bounded_transcription_validated_cell_count == 49
    assert artifact.later_identity_review_eligible_cell_count == 35
    assert artifact.upstream_ontology_review_pending_cell_count == 56
    assert artifact.all_reviewed_scopes_transcription_validated is False
    assert sum(
        item.table_transcription_validated for item in artifact.adjudicated_tables
    ) == 1
    assert sum(item.reparse_required for item in artifact.adjudicated_tables) == 1
    assert sum(item.source_check_pending for item in artifact.adjudicated_tables) == 1
    assert sum(item.rejected for item in artifact.adjudicated_tables) == 1


@pytest.mark.parametrize(
    "coverage_error",
    ["missing", "duplicate", "unknown", "item_digest", "item_kind"],
)
def test_decisions_must_exactly_cover_and_bind_review_items(
    tmp_path: Path,
    coverage_error: str,
) -> None:
    chain, packet, packet_path, _ = _build_domain_packet(tmp_path)
    decisions = _transcription_decision_payload(packet, packet_path)
    entry = decisions["decisions"][0]
    if coverage_error == "missing":
        decisions["decisions"] = []
    elif coverage_error == "duplicate":
        decisions["decisions"].append(deepcopy(entry))
    elif coverage_error == "unknown":
        entry["review_item_id"] = "supplementary-source-transcription:unknown"
    elif coverage_error == "item_digest":
        entry["review_item_digest"] = "sha256:" + "f" * 64
    else:
        entry["item_kind"] = "scope_semantic_note"

    with pytest.raises((ValidationError, ValueError)):
        _adjudicate_domain(chain, packet, packet_path, decisions)


def test_decision_order_does_not_change_mixed_scope_results(tmp_path: Path) -> None:
    chain, packet, packet_path, _ = _build_domain_packet(tmp_path, scope_count=2)
    decisions = _transcription_decision_payload(packet, packet_path)
    decisions["decisions"] = list(reversed(decisions["decisions"]))

    artifact = _adjudicate_domain(chain, packet, packet_path, decisions)

    assert artifact.accepted_scope_count == 2
    assert artifact.later_identity_review_eligible_cell_count == 70


@pytest.mark.parametrize(
    "forgery",
    ["component_result", "eligibility", "downstream_flag"],
)
def test_adjudication_model_rejects_self_consistent_forgery(
    tmp_path: Path,
    forgery: str,
) -> None:
    chain, packet, packet_path, _ = _build_domain_packet(tmp_path)
    decisions = _transcription_decision_payload(packet, packet_path)
    payload = _adjudicate_domain(
        chain,
        packet,
        packet_path,
        decisions,
    ).model_dump(mode="json")
    if forgery == "component_result":
        payload["adjudicated_tables"][0]["component_results"]["caption_check"] = (
            "mismatch"
        )
    elif forgery == "eligibility":
        payload["adjudicated_tables"][0][
            "later_identity_review_eligible_cell_count"
        ] += 1
        payload["later_identity_review_eligible_cell_count"] += 1
    else:
        payload["scientific_content_validated"] = True
    payload["adjudication_artifact_digest"] = transcription_domain._stable_hash(
        {
            key: value
            for key, value in payload.items()
            if key != "adjudication_artifact_digest"
        }
    )

    with pytest.raises(ValidationError):
        OledSupplementarySourceTranscriptionAdjudicationArtifact.model_validate(
            payload
        )


def test_real_pdf_poppler_packet_render_and_adjudication_smoke(
    tmp_path: Path,
) -> None:
    chain, packet, packet_path, asset_dir = _build_file_packet(
        tmp_path,
        use_real_renderer=True,
    )
    asset = packet.source_pdf_evidence.page_assets[0]
    asset_path = asset_dir / asset.asset_filename
    assert asset_path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    assert asset.pixel_width > 1
    assert asset.pixel_height > 1
    assert asset.render_profile == "png-200dpi-rgb-full-page-v1"

    markdown_path = tmp_path / "source-transcription-review.md"
    rendered = render_oled_supplementary_source_transcription_review_packet_from_files(
        review_packet_json=packet_path,
        asset_dir=asset_dir,
        output_markdown=markdown_path,
    )
    assert rendered == packet
    markdown = markdown_path.read_text(encoding="utf-8")
    assert markdown.count("## T01:") == 1
    assert f"assets/{asset.asset_filename}" in markdown

    decision_path, _ = _write_transcription_decisions(
        tmp_path,
        packet,
        packet_path,
    )
    output_path = tmp_path / "source-transcription-adjudication.json"
    artifact = _adjudicate_from_files(
        chain,
        packet_path=packet_path,
        decision_path=decision_path,
        asset_dir=asset_dir,
        output_path=output_path,
    )
    assert artifact.accepted_scope_count == 1
    assert artifact.bounded_transcription_validated_cell_count == 49
    assert artifact.later_identity_review_eligible_cell_count == 35
    assert output_path.is_file()


def test_packet_file_entry_rejects_wrong_source_pdf_hash(tmp_path: Path) -> None:
    chain = _build_semantic_chain(tmp_path)
    wrong_pdf = tmp_path / "wrong-source.pdf"
    wrong_pdf.write_bytes(_minimal_pdf_bytes(page_count=40))
    chain["source_pdf_path"] = wrong_pdf
    asset_dir = tmp_path / "assets"
    output_path = tmp_path / "packet.json"

    with pytest.raises(ValueError, match="does not match the bound hash"):
        build_oled_supplementary_source_transcription_review_packet_from_files(
            **_packet_file_kwargs(
                chain,
                asset_dir=asset_dir,
                output_path=output_path,
                poppler_bin_dir=_real_poppler_bin_dir(),
            )
        )

    assert not output_path.exists()
    assert not asset_dir.exists()


def test_packet_file_entry_rejects_pdf_symlink(tmp_path: Path) -> None:
    chain = _build_semantic_chain(tmp_path)
    symlink_path = tmp_path / "linked-source.pdf"
    symlink_path.symlink_to(chain["source_pdf_path"])
    chain["source_pdf_path"] = symlink_path
    asset_dir = tmp_path / "assets"
    output_path = tmp_path / "packet.json"

    with pytest.raises(ValueError, match="PDF is unavailable"):
        build_oled_supplementary_source_transcription_review_packet_from_files(
            **_packet_file_kwargs(
                chain,
                asset_dir=asset_dir,
                output_path=output_path,
                poppler_bin_dir=_real_poppler_bin_dir(),
            )
        )

    assert not output_path.exists()
    assert not asset_dir.exists()


def test_packet_file_entry_requires_a_fresh_asset_directory(
    tmp_path: Path,
) -> None:
    chain = _build_semantic_chain(tmp_path)
    asset_dir = tmp_path / "assets"
    asset_dir.mkdir()
    output_path = tmp_path / "packet.json"

    with pytest.raises(ValueError, match="asset directory must be fresh"):
        build_oled_supplementary_source_transcription_review_packet_from_files(
            **_packet_file_kwargs(
                chain,
                asset_dir=asset_dir,
                output_path=output_path,
            )
        )

    assert not output_path.exists()


def test_packet_output_cannot_overwrite_the_source_pdf(tmp_path: Path) -> None:
    chain = _build_semantic_chain(tmp_path)
    original = chain["source_pdf_path"].read_bytes()

    with pytest.raises(ValueError):
        build_oled_supplementary_source_transcription_review_packet_from_files(
            **_packet_file_kwargs(
                chain,
                asset_dir=tmp_path / "assets",
                output_path=chain["source_pdf_path"],
            )
        )

    assert chain["source_pdf_path"].read_bytes() == original


@pytest.mark.parametrize("replacement_timing", ["before-write", "after-write"])
def test_asset_path_replacement_fails_binding_without_deleting_replacement(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    replacement_timing: str,
) -> None:
    chain = _build_semantic_chain(tmp_path)
    _install_fake_renderer(monkeypatch, chain)
    asset_dir = tmp_path / "assets"
    moved_asset_dir = tmp_path / "fixed-assets-moved"
    replacement_sentinel = asset_dir / "replacement-sentinel.txt"
    output_path = tmp_path / "packet.json"
    original_write_asset_bundle = transcription_runner._write_asset_bundle

    def replacing_write_asset_bundle(
        asset_descriptor: int,
        packet: OledSupplementarySourceTranscriptionReviewPacket,
        rendered_assets: dict[str, bytes],
        *,
        created_asset_stats: dict[str, os.stat_result],
    ) -> None:
        if replacement_timing == "before-write":
            asset_dir.rename(moved_asset_dir)
            asset_dir.mkdir()
            replacement_sentinel.write_text("preserve", encoding="utf-8")
        original_write_asset_bundle(
            asset_descriptor,
            packet,
            rendered_assets,
            created_asset_stats=created_asset_stats,
        )
        if replacement_timing == "after-write":
            asset_dir.rename(moved_asset_dir)
            asset_dir.mkdir()
            replacement_sentinel.write_text("preserve", encoding="utf-8")

    monkeypatch.setattr(
        transcription_runner,
        "_write_asset_bundle",
        replacing_write_asset_bundle,
    )

    with pytest.raises(ValueError, match="asset directory (?:binding is unavailable|changed)"):
        build_oled_supplementary_source_transcription_review_packet_from_files(
            **_packet_file_kwargs(
                chain,
                asset_dir=asset_dir,
                output_path=output_path,
            )
        )

    assert replacement_sentinel.read_text(encoding="utf-8") == "preserve"
    assert {path.name for path in asset_dir.iterdir()} == {
        replacement_sentinel.name
    }
    assert any(path.suffix == ".png" for path in moved_asset_dir.iterdir())
    assert not output_path.exists()


def test_parent_path_replacement_never_redirects_asset_writes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    chain = _build_semantic_chain(tmp_path)
    _install_fake_renderer(monkeypatch, chain)
    visible_parent = tmp_path / "publish"
    visible_parent.mkdir()
    held_parent = tmp_path / "fixed-parent-moved"
    asset_dir = visible_parent / "assets"
    replacement_sentinel = visible_parent / "replacement-sentinel.txt"
    output_path = tmp_path / "packet.json"
    original_write_asset_bundle = transcription_runner._write_asset_bundle

    def replacing_parent_before_write(
        asset_descriptor: int,
        packet: OledSupplementarySourceTranscriptionReviewPacket,
        rendered_assets: dict[str, bytes],
        *,
        created_asset_stats: dict[str, os.stat_result],
    ) -> None:
        visible_parent.rename(held_parent)
        visible_parent.mkdir()
        replacement_sentinel.write_text("preserve", encoding="utf-8")
        original_write_asset_bundle(
            asset_descriptor,
            packet,
            rendered_assets,
            created_asset_stats=created_asset_stats,
        )

    monkeypatch.setattr(
        transcription_runner,
        "_write_asset_bundle",
        replacing_parent_before_write,
    )

    with pytest.raises(ValueError, match="asset directory (?:binding is unavailable|changed)"):
        build_oled_supplementary_source_transcription_review_packet_from_files(
            **_packet_file_kwargs(
                chain,
                asset_dir=asset_dir,
                output_path=output_path,
            )
        )

    assert replacement_sentinel.read_text(encoding="utf-8") == "preserve"
    assert {path.name for path in visible_parent.iterdir()} == {
        replacement_sentinel.name
    }
    assert not (held_parent / "assets").exists()
    assert not output_path.exists()


def test_keyboard_interrupt_during_asset_write_cleans_only_known_assets(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    chain = _build_semantic_chain(tmp_path)
    _install_fake_renderer(monkeypatch, chain)
    asset_dir = tmp_path / "assets"
    output_path = tmp_path / "packet.json"
    sentinel_name = "operator-sentinel.txt"
    original_write_asset_bundle = transcription_runner._write_asset_bundle

    def interrupting_asset_write(
        asset_descriptor: int,
        packet: OledSupplementarySourceTranscriptionReviewPacket,
        rendered_assets: dict[str, bytes],
        *,
        created_asset_stats: dict[str, os.stat_result],
    ) -> None:
        original_write_asset_bundle(
            asset_descriptor,
            packet,
            rendered_assets,
            created_asset_stats=created_asset_stats,
        )
        transcription_runner._write_fresh_bytes_at(
            asset_descriptor,
            sentinel_name,
            b"preserve",
        )
        raise KeyboardInterrupt

    monkeypatch.setattr(
        transcription_runner,
        "_write_asset_bundle",
        interrupting_asset_write,
    )

    with pytest.raises(KeyboardInterrupt):
        build_oled_supplementary_source_transcription_review_packet_from_files(
            **_packet_file_kwargs(
                chain,
                asset_dir=asset_dir,
                output_path=output_path,
            )
        )

    assert (asset_dir / sentinel_name).read_bytes() == b"preserve"
    assert {path.name for path in asset_dir.iterdir()} == {sentinel_name}
    assert not output_path.exists()


def test_keyboard_interrupt_during_packet_publish_never_deletes_replacement(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    chain = _build_semantic_chain(tmp_path)
    _install_fake_renderer(monkeypatch, chain)
    asset_dir = tmp_path / "assets"
    moved_asset_dir = tmp_path / "fixed-assets-moved"
    replacement_sentinel = asset_dir / "replacement-sentinel.txt"
    output_path = tmp_path / "packet.json"

    def interrupting_packet_publish(
        _path: Path,
        _payload: str,
        *,
        post_publish_validator: Any,
    ) -> None:
        del post_publish_validator
        asset_dir.rename(moved_asset_dir)
        asset_dir.mkdir()
        replacement_sentinel.write_text("preserve", encoding="utf-8")
        raise KeyboardInterrupt

    monkeypatch.setattr(
        transcription_runner,
        "_publish_packet_text",
        interrupting_packet_publish,
    )

    with pytest.raises(KeyboardInterrupt):
        build_oled_supplementary_source_transcription_review_packet_from_files(
            **_packet_file_kwargs(
                chain,
                asset_dir=asset_dir,
                output_path=output_path,
            )
        )

    assert replacement_sentinel.read_text(encoding="utf-8") == "preserve"
    assert {path.name for path in asset_dir.iterdir()} == {
        replacement_sentinel.name
    }
    assert any(path.suffix == ".png" for path in moved_asset_dir.iterdir())
    assert not output_path.exists()


def test_keyboard_interrupt_during_packet_publish_removes_only_the_known_bundle(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    chain = _build_semantic_chain(tmp_path)
    _install_fake_renderer(monkeypatch, chain)
    asset_dir = tmp_path / "assets"
    output_path = tmp_path / "packet.json"

    def interrupting_packet_publish(
        _path: Path,
        _payload: str,
        *,
        post_publish_validator: Any,
    ) -> None:
        del post_publish_validator
        raise KeyboardInterrupt

    monkeypatch.setattr(
        transcription_runner,
        "_publish_packet_text",
        interrupting_packet_publish,
    )

    with pytest.raises(KeyboardInterrupt):
        build_oled_supplementary_source_transcription_review_packet_from_files(
            **_packet_file_kwargs(
                chain,
                asset_dir=asset_dir,
                output_path=output_path,
            )
        )

    assert not asset_dir.exists()
    assert not output_path.exists()


def test_packet_publication_transaction_rolls_back_interrupt_after_link(
    tmp_path: Path,
) -> None:
    output_path = tmp_path / "packet.json"

    def interrupt_after_link() -> None:
        assert output_path.is_file()
        raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        transcription_runner._publish_packet_text(
            output_path,
            '{"status":"test"}\n',
            post_publish_validator=interrupt_after_link,
        )

    assert not output_path.exists()
    assert not list(tmp_path.glob(".packet.json.*.tmp"))


def test_packet_publish_revalidates_asset_path_and_revokes_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    chain = _build_semantic_chain(tmp_path)
    _install_fake_renderer(monkeypatch, chain)
    asset_dir = tmp_path / "assets"
    moved_asset_dir = tmp_path / "moved-assets"
    replacement_sentinel = asset_dir / "replacement-sentinel.txt"
    output_path = tmp_path / "packet.json"
    real_publish = transcription_runner._publish_packet_text

    def publish_then_replace_assets(
        path: Path,
        payload: str,
        *,
        post_publish_validator: Any,
    ) -> None:
        replacement_done = False

        def validator_with_replacement() -> None:
            nonlocal replacement_done
            post_publish_validator()
            if not replacement_done:
                asset_dir.rename(moved_asset_dir)
                asset_dir.mkdir()
                replacement_sentinel.write_text("preserve", encoding="utf-8")
                replacement_done = True

        real_publish(
            path,
            payload,
            post_publish_validator=validator_with_replacement,
        )

    monkeypatch.setattr(
        transcription_runner,
        "_publish_packet_text",
        publish_then_replace_assets,
    )

    with pytest.raises(ValueError, match="asset directory"):
        build_oled_supplementary_source_transcription_review_packet_from_files(
            **_packet_file_kwargs(
                chain,
                asset_dir=asset_dir,
                output_path=output_path,
            )
        )

    assert not output_path.exists()
    assert replacement_sentinel.read_text(encoding="utf-8") == "preserve"
    assert any(path.suffix == ".png" for path in moved_asset_dir.iterdir())


def test_packet_publish_rejects_output_parent_switched_into_assets(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    chain = _build_semantic_chain(tmp_path)
    _install_fake_renderer(monkeypatch, chain)
    asset_dir = tmp_path / "assets"
    safe_output_parent = tmp_path / "safe-output"
    safe_output_parent.mkdir()
    output_parent_link = tmp_path / "output-link"
    output_parent_link.symlink_to(safe_output_parent, target_is_directory=True)
    output_path = output_parent_link / "packet.json"
    real_publish = transcription_runner._publish_packet_text

    def switch_parent_then_publish(
        path: Path,
        payload: str,
        *,
        post_publish_validator: Any,
    ) -> None:
        output_parent_link.unlink()
        output_parent_link.symlink_to(asset_dir, target_is_directory=True)
        real_publish(
            path,
            payload,
            post_publish_validator=post_publish_validator,
        )

    monkeypatch.setattr(
        transcription_runner,
        "_publish_packet_text",
        switch_parent_then_publish,
    )

    with pytest.raises(
        ValueError,
        match=r"published (?:inside assets|asset coverage changed)",
    ):
        build_oled_supplementary_source_transcription_review_packet_from_files(
            **_packet_file_kwargs(
                chain,
                asset_dir=asset_dir,
                output_path=output_path,
            )
        )

    assert not output_path.exists()
    assert not asset_dir.exists()


@pytest.mark.parametrize("tamper", ["bytes", "extra", "symlink"])
def test_render_rejects_tampered_or_substituted_asset_bundle(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    tamper: str,
) -> None:
    _, packet, packet_path, asset_dir = _build_file_packet(
        tmp_path,
        monkeypatch=monkeypatch,
    )
    asset_path = asset_dir / packet.source_pdf_evidence.page_assets[0].asset_filename
    if tamper == "bytes":
        asset_path.write_bytes(asset_path.read_bytes() + b"tamper")
    elif tamper == "extra":
        (asset_dir / "unbound-extra.png").write_bytes(_FAKE_PNG_BYTES)
    else:
        replacement = tmp_path / "replacement.png"
        replacement.write_bytes(asset_path.read_bytes())
        asset_path.unlink()
        asset_path.symlink_to(replacement)
    markdown_path = tmp_path / "review.md"

    with pytest.raises(ValueError):
        render_oled_supplementary_source_transcription_review_packet_from_files(
            review_packet_json=packet_path,
            asset_dir=asset_dir,
            output_markdown=markdown_path,
        )

    assert not markdown_path.exists()


def test_render_rejects_packet_component_tamper(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, _, packet_path, asset_dir = _build_file_packet(
        tmp_path,
        monkeypatch=monkeypatch,
    )
    payload = json.loads(packet_path.read_text(encoding="utf-8"))
    payload["review_items"][0]["component_digests"]["caption"] = (
        "sha256:" + "f" * 64
    )
    payload["review_packet_digest"] = transcription_domain._stable_hash(
        {
            key: value
            for key, value in payload.items()
            if key != "review_packet_digest"
        }
    )
    write_json(packet_path, payload)
    markdown_path = tmp_path / "review.md"

    with pytest.raises(ValidationError, match="component digest mismatch"):
        render_oled_supplementary_source_transcription_review_packet_from_files(
            review_packet_json=packet_path,
            asset_dir=asset_dir,
            output_markdown=markdown_path,
        )

    assert not markdown_path.exists()


def test_adjudication_rejects_duplicate_json_keys(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    chain, packet, packet_path, asset_dir = _build_file_packet(
        tmp_path,
        monkeypatch=monkeypatch,
    )
    decision_path = tmp_path / "duplicate-decisions.json"
    decision_path.write_text(
        '{"schema_version":"oled_supplementary_source_transcription_decision_manifest.v1",'
        '"schema_version":"oled_supplementary_source_transcription_decision_manifest.v1"}',
        encoding="utf-8",
    )
    output_path = tmp_path / "adjudication.json"

    with pytest.raises(ValueError, match="duplicate keys"):
        _adjudicate_from_files(
            chain,
            packet_path=packet_path,
            decision_path=decision_path,
            asset_dir=asset_dir,
            output_path=output_path,
        )

    assert packet.review_item_count == 1
    assert not output_path.exists()


def test_adjudication_output_cannot_be_inside_asset_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    chain, packet, packet_path, asset_dir = _build_file_packet(
        tmp_path,
        monkeypatch=monkeypatch,
    )
    decision_path, _ = _write_transcription_decisions(
        tmp_path,
        packet,
        packet_path,
    )
    output_path = asset_dir / "adjudication.json"

    with pytest.raises(ValueError, match="outside the asset directory"):
        _adjudicate_from_files(
            chain,
            packet_path=packet_path,
            decision_path=decision_path,
            asset_dir=asset_dir,
            output_path=output_path,
        )

    assert not output_path.exists()


@pytest.mark.parametrize(
    "upstream_key",
    [
        "request_path",
        "response_path",
        "response_artifact_path",
        "semantic_packet_path",
        "semantic_decision_path",
        "semantic_adjudication_path",
    ],
)
def test_adjudication_replays_and_rejects_changed_upstream_bytes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    upstream_key: str,
) -> None:
    chain, packet, packet_path, asset_dir = _build_file_packet(
        tmp_path,
        monkeypatch=monkeypatch,
    )
    decision_path, _ = _write_transcription_decisions(
        tmp_path,
        packet,
        packet_path,
    )
    upstream_path = chain[upstream_key]
    upstream_payload = json.loads(upstream_path.read_text(encoding="utf-8"))
    upstream_path.write_text(
        json.dumps(upstream_payload, ensure_ascii=False, indent=4) + "\n",
        encoding="utf-8",
    )
    output_path = tmp_path / "adjudication.json"

    with pytest.raises((ValidationError, ValueError)):
        _adjudicate_from_files(
            chain,
            packet_path=packet_path,
            decision_path=decision_path,
            asset_dir=asset_dir,
            output_path=output_path,
        )

    assert not output_path.exists()


def test_cli_success_and_failure_outputs_are_redacted(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    chain = _build_semantic_chain(tmp_path)
    _install_fake_renderer(monkeypatch, chain)
    packet_path = tmp_path / "packet.json"
    asset_dir = tmp_path / "assets"
    stdout = StringIO()
    packet_args = [
        "packet",
        "--request-artifact",
        str(chain["request_path"]),
        "--response-manifest",
        str(chain["response_path"]),
        "--response-artifact",
        str(chain["response_artifact_path"]),
        "--semantic-review-packet",
        str(chain["semantic_packet_path"]),
        "--semantic-decision-manifest",
        str(chain["semantic_decision_path"]),
        "--semantic-adjudication",
        str(chain["semantic_adjudication_path"]),
        "--source-pdf",
        str(chain["source_pdf_path"]),
        "--asset-dir",
        str(asset_dir),
        "--output",
        str(packet_path),
    ]

    assert main(packet_args, stdout=stdout) == 0
    output = stdout.getvalue()
    assert '"review_item_count": 1' in output
    assert '"full_table_cell_count": 56' in output
    assert '"numeric_source_cell_count": 49' in output
    assert str(tmp_path) not in output
    assert "paper016" not in output
    assert "0.1280" not in output

    packet = OledSupplementarySourceTranscriptionReviewPacket.model_validate_json(
        packet_path.read_text(encoding="utf-8")
    )
    unsafe = _transcription_decision_payload(packet, packet_path)
    unsafe["decisions"][0]["review_note"] = "token=secret-value"
    decision_path = tmp_path / "unsafe-decisions.json"
    write_json(decision_path, unsafe)
    failed_output = tmp_path / "failed-adjudication.json"
    stdout = StringIO()
    adjudicate_args = [
        "adjudicate",
        "--request-artifact",
        str(chain["request_path"]),
        "--response-manifest",
        str(chain["response_path"]),
        "--response-artifact",
        str(chain["response_artifact_path"]),
        "--semantic-review-packet",
        str(chain["semantic_packet_path"]),
        "--semantic-decision-manifest",
        str(chain["semantic_decision_path"]),
        "--semantic-adjudication",
        str(chain["semantic_adjudication_path"]),
        "--source-pdf",
        str(chain["source_pdf_path"]),
        "--review-packet",
        str(packet_path),
        "--decision-manifest",
        str(decision_path),
        "--asset-dir",
        str(asset_dir),
        "--output",
        str(failed_output),
    ]

    assert main(adjudicate_args, stdout=stdout) == 2
    failure = stdout.getvalue()
    assert "supplementary_source_transcription_review_failed" in failure
    assert "secret-value" not in failure
    assert str(tmp_path) not in failure
    assert not failed_output.exists()
