from __future__ import annotations

import hashlib
from pathlib import Path

import pytest
from PIL import Image

from ai4s_agent import ocsr_candidate_execution as ocsr_execution
from ai4s_agent.ocsr_candidate_execution import (
    OcsrCandidateRequest,
    OcsrModelProvenance,
    _load_json_without_duplicate_keys,
    execute_ocsr_candidate_request,
    execute_ocsr_candidate_request_from_files,
)


def _image(path: Path) -> str:
    Image.new("RGB", (32, 24), "white").save(path)
    return f"sha256:{hashlib.sha256(path.read_bytes()).hexdigest()}"


def _request(tmp_path: Path) -> OcsrCandidateRequest:
    first = tmp_path / "first.png"
    second = tmp_path / "second.png"
    return OcsrCandidateRequest.model_validate(
        {
            "run_id": "ocsr-test-001",
            "items": [
                {
                    "candidate_id": "candidate-001",
                    "reported_alias": "Molecule A",
                    "image_file": first.name,
                    "image_sha256": _image(first),
                },
                {
                    "candidate_id": "candidate-002",
                    "reported_alias": "Molecule B",
                    "image_file": second.name,
                    "image_sha256": _image(second),
                },
            ],
        }
    )


def _model() -> OcsrModelProvenance:
    return OcsrModelProvenance(
        engine_version="1.1.1",
        checkpoint_sha256="sha256:" + "1" * 64,
        device="cuda",
    )


def test_execute_ocsr_candidate_request_records_ready_and_rejected_candidates(
    tmp_path: Path,
) -> None:
    request = _request(tmp_path)
    calls = 0

    def predictor(_: str) -> dict[str, object]:
        nonlocal calls
        calls += 1
        if calls == 1:
            return {"smiles": "C(C)O", "confidence": 0.75}
        return {"smiles": "not-smiles", "confidence": 0.2}

    artifact = execute_ocsr_candidate_request(
        request,
        request_sha256="sha256:" + "2" * 64,
        image_base_dir=tmp_path,
        predictor=predictor,
        model=_model(),
        generated_at="2026-07-17T00:00:00Z",
    )

    assert artifact.candidate_count == 1
    assert artifact.rejected_count == 1
    assert artifact.results[0].canonical_isomeric_smiles == "CCO"
    assert artifact.results[0].inchikey == "LFQSCWFLJHTTHZ-UHFFFAOYSA-N"
    assert artifact.results[0].candidate_only
    assert not artifact.results[0].identity_resolved
    assert artifact.results[1].status == "candidate_rejected"
    assert not artifact.registry_mutated
    assert not artifact.gold_written
    assert not artifact.dataset_written


def test_execute_ocsr_candidate_request_rejects_changed_image_bytes(
    tmp_path: Path,
) -> None:
    request = _request(tmp_path)
    (tmp_path / "first.png").write_bytes(b"changed")

    with pytest.raises(ValueError, match="image SHA-256 mismatch"):
        execute_ocsr_candidate_request(
            request,
            request_sha256="sha256:" + "2" * 64,
            image_base_dir=tmp_path,
            predictor=lambda _: {"smiles": "CC"},
            model=_model(),
        )


def test_execute_ocsr_candidate_request_rejects_symlinked_image(
    tmp_path: Path,
) -> None:
    request = _request(tmp_path)
    first = tmp_path / "first.png"
    target = tmp_path / "target.png"
    first.rename(target)
    first.symlink_to(target)

    with pytest.raises(ValueError, match="must not be a symlink"):
        execute_ocsr_candidate_request(
            request,
            request_sha256="sha256:" + "2" * 64,
            image_base_dir=tmp_path,
            predictor=lambda _: {"smiles": "CC"},
            model=_model(),
        )


def test_ocsr_request_requires_sorted_unique_candidate_ids(tmp_path: Path) -> None:
    request = _request(tmp_path).model_dump(mode="json")
    request["items"].reverse()

    with pytest.raises(ValueError, match="sorted and unique"):
        OcsrCandidateRequest.model_validate(request)


def test_ocsr_request_rejects_duplicate_json_keys() -> None:
    with pytest.raises(ValueError, match="duplicate JSON keys"):
        _load_json_without_duplicate_keys(b'{"run_id":"first","run_id":"second"}')


def test_ocsr_output_never_replaces_existing_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request = _request(tmp_path)
    request_path = tmp_path / "request.json"
    request_path.write_text(request.model_dump_json(), encoding="utf-8")
    checkpoint = tmp_path / "checkpoint.pth"
    checkpoint.write_bytes(b"checkpoint")
    output = tmp_path / "candidates.json"
    output.write_text("keep-me", encoding="utf-8")
    monkeypatch.setattr(
        ocsr_execution,
        "_load_molscribe_predictor",
        lambda *_args, **_kwargs: (
            lambda _: {"smiles": "CC", "confidence": 0.9},
            _model(),
        ),
    )

    with pytest.raises(ValueError, match="output already exists"):
        execute_ocsr_candidate_request_from_files(
            request_json=request_path,
            checkpoint_path=checkpoint,
            output_json=output,
            device="cpu",
        )

    assert output.read_text(encoding="utf-8") == "keep-me"
