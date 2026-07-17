from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import pytest
from PIL import Image, ImageDraw

import ai4s_agent.ocsr_crop_preprocessing as crop_module
from ai4s_agent.ocsr_candidate_execution import OcsrCandidateRequest
from ai4s_agent.ocsr_crop_preprocessing import (
    OcsrCropPreprocessingArtifact,
    OcsrCropPreprocessingRequest,
    build_ocsr_crop_preprocessing_from_files,
)


def _sha(path: Path) -> str:
    return f"sha256:{hashlib.sha256(path.read_bytes()).hexdigest()}"


def _make_source(path: Path, *, touches_edge: bool = False) -> None:
    image = Image.new("RGB", (420, 260), "white")
    draw = ImageDraw.Draw(image)
    start = 20 if touches_edge else 55
    # Three substantial graph fragments model the way atom labels split bonds.
    draw.line(
        (start, 120, 110, 75, 170, 120, 110, 165, start, 120), fill="black", width=5
    )
    draw.line((170, 120, 225, 75, 280, 120, 225, 165, 170, 120), fill="black", width=5)
    draw.line((280, 120, 330, 80, 375, 120, 330, 160, 280, 120), fill="black", width=5)
    draw.text((174, 112), "N", fill="black")
    draw.text((276, 112), "O", fill="black")
    draw.rectangle((150, 205, 270, 230), fill="black")  # reported alias
    image.save(path)


def _request_payload(source: Path, *, touches_edge: bool = False) -> dict[str, object]:
    left = 20 if touches_edge else 25
    return {
        "schema_version": "ocsr_crop_preprocessing_request.v1",
        "run_id": "crop-test",
        "items": [
            {
                "candidate_id": "candidate-001",
                "reported_alias": "TEST-1",
                "source_document_id": "paper-test",
                "source_locator": "page=1;figure=1;row=1",
                "source_image_file": source.name,
                "source_image_sha256": _sha(source),
                "crop_bbox": {"left": left, "top": 35, "right": 400, "bottom": 240},
                "exclusions": [
                    {
                        "box": {"left": 145, "top": 200, "right": 280, "bottom": 235},
                        "reason": "reported_alias",
                    }
                ],
            }
        ],
    }


def _write_request(path: Path, payload: dict[str, object]) -> None:
    path.write_text(
        json.dumps(payload, sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )


def test_deterministic_crop_publishes_exact_ocsr_request(tmp_path: Path) -> None:
    source = tmp_path / "page.png"
    _make_source(source)
    request = tmp_path / "request.json"
    _write_request(request, _request_payload(source))

    first, first_dir = build_ocsr_crop_preprocessing_from_files(
        request_json=request,
        output_dir=tmp_path / "first",
        generated_at="2026-07-17T00:00:00Z",
    )
    second, second_dir = build_ocsr_crop_preprocessing_from_files(
        request_json=request,
        output_dir=tmp_path / "second",
        generated_at="2026-07-17T00:00:00Z",
    )

    assert first == second
    assert first.batch_ready is True
    assert first.ready_count == 1
    assert first.results[0].quality.exclusion_pixel_count == 4_725
    image_name = first.results[0].output_image_file
    assert (first_dir / image_name).read_bytes() == (
        second_dir / image_name
    ).read_bytes()
    request_bytes = (first_dir / "ocsr_request.json").read_bytes()
    assert (
        f"sha256:{hashlib.sha256(request_bytes).hexdigest()}"
        == first.ocsr_request_sha256
    )
    downstream = OcsrCandidateRequest.model_validate_json(request_bytes)
    assert downstream.items[0].image_sha256 == first.results[0].output_image_sha256
    persisted = OcsrCropPreprocessingArtifact.model_validate_json(
        (first_dir / "crop_artifact.json").read_bytes()
    )
    assert persisted == first


def test_edge_touching_structure_fails_closed_without_ocsr_request(
    tmp_path: Path,
) -> None:
    source = tmp_path / "page.png"
    _make_source(source, touches_edge=True)
    request = tmp_path / "request.json"
    _write_request(request, _request_payload(source, touches_edge=True))

    artifact, output = build_ocsr_crop_preprocessing_from_files(
        request_json=request,
        output_dir=tmp_path / "rejected",
    )

    assert artifact.batch_ready is False
    assert artifact.results[0].status == "crop_rejected"
    assert "structure_touches_crop_edge" in artifact.results[0].quality.rejection_codes
    assert not (output / "ocsr_request.json").exists()
    assert (output / artifact.results[0].output_image_file).is_file()


def test_source_sha_mismatch_and_symlink_fail_closed(tmp_path: Path) -> None:
    source = tmp_path / "page.png"
    _make_source(source)
    request = tmp_path / "request.json"
    payload = _request_payload(source)
    payload["items"][0]["source_image_sha256"] = "sha256:" + "0" * 64  # type: ignore[index]
    _write_request(request, payload)
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        build_ocsr_crop_preprocessing_from_files(
            request_json=request, output_dir=tmp_path / "bad-sha"
        )

    symlink = tmp_path / "linked.png"
    symlink.symlink_to(source)
    payload = _request_payload(source)
    payload["items"][0]["source_image_file"] = symlink.name  # type: ignore[index]
    _write_request(request, payload)
    with pytest.raises(ValueError, match="unavailable or symbolic"):
        build_ocsr_crop_preprocessing_from_files(
            request_json=request, output_dir=tmp_path / "bad-link"
        )


def test_duplicate_json_keys_and_duplicate_evidence_are_rejected(
    tmp_path: Path,
) -> None:
    source = tmp_path / "page.png"
    _make_source(source)
    request = tmp_path / "request.json"
    request.write_text('{"run_id":"a","run_id":"b","items":[]}', encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate JSON keys"):
        build_ocsr_crop_preprocessing_from_files(
            request_json=request, output_dir=tmp_path / "dup-key"
        )

    payload = _request_payload(source)
    duplicate = dict(payload["items"][0])  # type: ignore[index]
    duplicate["candidate_id"] = "candidate-002"
    payload["items"].append(duplicate)  # type: ignore[union-attr]
    with pytest.raises(ValueError, match="exact evidence binding"):
        OcsrCropPreprocessingRequest.model_validate(payload)

    payload = _request_payload(source)
    payload["items"][0]["candidate_id"] = "unsafe:filename"  # type: ignore[index]
    with pytest.raises(ValueError, match="safe for a crop filename"):
        OcsrCropPreprocessingRequest.model_validate(payload)

    payload = _request_payload(source)
    payload["items"][0]["exclusions"].append(  # type: ignore[index,union-attr]
        {
            "box": {"left": 160, "top": 210, "right": 290, "bottom": 239},
            "reason": "other_non_structure_annotation",
        }
    )
    with pytest.raises(ValueError, match="must not overlap"):
        OcsrCropPreprocessingRequest.model_validate(payload)


def test_existing_and_concurrently_created_targets_are_preserved(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "page.png"
    _make_source(source)
    request = tmp_path / "request.json"
    _write_request(request, _request_payload(source))
    existing = tmp_path / "existing"
    existing.mkdir()
    marker = existing / "marker"
    marker.write_text("owned elsewhere", encoding="utf-8")
    with pytest.raises(ValueError, match="must be fresh"):
        build_ocsr_crop_preprocessing_from_files(
            request_json=request, output_dir=existing
        )
    assert marker.read_text(encoding="utf-8") == "owned elsewhere"

    original = crop_module._atomic_rename_owned_directory_noreplace
    target = tmp_path / "concurrent"

    def race(**kwargs: object) -> None:
        target.mkdir()
        (target / "marker").write_text("concurrent owner", encoding="utf-8")
        original(**kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(crop_module, "_atomic_rename_owned_directory_noreplace", race)
    with pytest.raises(ValueError, match="must be fresh"):
        build_ocsr_crop_preprocessing_from_files(
            request_json=request, output_dir=target
        )
    assert (target / "marker").read_text(encoding="utf-8") == "concurrent owner"


def test_parent_replacement_and_post_publish_tamper_fail_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "page.png"
    _make_source(source)
    request = tmp_path / "request.json"
    _write_request(request, _request_payload(source))
    parent = tmp_path / "output-parent"
    parent.mkdir()
    original = crop_module._atomic_rename_owned_directory_noreplace

    def replace_parent(**kwargs: object) -> None:
        parent.rename(tmp_path / "moved-parent")
        parent.mkdir()
        original(**kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(
        crop_module, "_atomic_rename_owned_directory_noreplace", replace_parent
    )
    with pytest.raises(ValueError, match="parent changed"):
        build_ocsr_crop_preprocessing_from_files(
            request_json=request, output_dir=parent / "bundle"
        )
    assert not (parent / "bundle").exists()

    monkeypatch.setattr(
        crop_module, "_atomic_rename_owned_directory_noreplace", original
    )
    tamper_parent = tmp_path / "tamper-parent"
    tamper_parent.mkdir()

    def tamper(**kwargs: object) -> None:
        original(**kwargs)  # type: ignore[arg-type]
        descriptor = os.open(
            str(tamper_parent / "bundle" / "crop-candidate-001.png"),
            os.O_WRONLY | os.O_TRUNC,
        )
        try:
            os.write(descriptor, b"tampered")
        finally:
            os.close(descriptor)

    monkeypatch.setattr(crop_module, "_atomic_rename_owned_directory_noreplace", tamper)
    with pytest.raises(ValueError, match="content mismatch"):
        build_ocsr_crop_preprocessing_from_files(
            request_json=request, output_dir=tamper_parent / "bundle"
        )
    assert not (tamper_parent / "bundle").exists()
