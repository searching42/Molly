from __future__ import annotations

import hashlib
import os
import sys
import types
from pathlib import Path

import pytest
from PIL import Image

from ai4s_agent import ocsr_candidate_execution as ocsr_execution
from ai4s_agent.ocsr_candidate_execution import (
    OcsrCandidateRequest,
    OcsrCandidateArtifact,
    OcsrModelProvenance,
    _load_json_without_duplicate_keys,
    _load_molscribe_predictor,
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

    with pytest.raises(ValueError, match="symbolic"):
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


def _patch_fake_loader(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        ocsr_execution,
        "_load_molscribe_predictor",
        lambda *_args, **_kwargs: (
            lambda _: {"smiles": "CC", "confidence": 0.9},
            _model(),
        ),
    )


def _file_request_paths(tmp_path: Path) -> tuple[Path, Path, Path]:
    request = _request(tmp_path)
    request_path = tmp_path / "request.json"
    request_path.write_text(request.model_dump_json(), encoding="utf-8")
    checkpoint = tmp_path / "checkpoint.pth"
    checkpoint.write_bytes(b"checkpoint")
    return request_path, checkpoint, tmp_path / "output" / "candidates.json"


def test_checkpoint_swap_cannot_change_owned_bytes_loaded_by_model(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkpoint = tmp_path / "checkpoint.pth"
    checkpoint.write_bytes(b"checkpoint-A")
    replacement = tmp_path / "checkpoint-B.pth"
    replacement.write_bytes(b"checkpoint-B")
    observed: dict[str, object] = {}

    class FakeMolScribe:
        def __init__(self, model_path: str, *, device: object) -> None:
            original = tmp_path / "checkpoint-A-original.pth"
            checkpoint.rename(original)
            replacement.rename(checkpoint)
            observed["model_path"] = model_path
            observed["loaded_bytes"] = Path(model_path).read_bytes()
            checkpoint.rename(replacement)
            original.rename(checkpoint)
            observed["device"] = device

        def predict_image_file(
            self,
            *_args: object,
            **_kwargs: object,
        ) -> dict[str, object]:
            return {"smiles": "CC"}

    torch_module = types.ModuleType("torch")
    torch_module.device = lambda value: value  # type: ignore[attr-defined]
    molscribe_module = types.ModuleType("molscribe")
    molscribe_module.MolScribe = FakeMolScribe  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "torch", torch_module)
    monkeypatch.setitem(sys.modules, "molscribe", molscribe_module)

    _, provenance = _load_molscribe_predictor(checkpoint, device="cpu")

    assert observed["loaded_bytes"] == b"checkpoint-A"
    assert Path(str(observed["model_path"])) != checkpoint
    assert checkpoint.read_bytes() == b"checkpoint-A"
    assert replacement.read_bytes() == b"checkpoint-B"
    assert provenance.checkpoint_sha256 == (
        "sha256:" + hashlib.sha256(b"checkpoint-A").hexdigest()
    )


def test_checkpoint_rejects_symlink_path_component(tmp_path: Path) -> None:
    real_parent = tmp_path / "real"
    real_parent.mkdir()
    (real_parent / "checkpoint.pth").write_bytes(b"checkpoint")
    linked_parent = tmp_path / "linked"
    linked_parent.symlink_to(real_parent, target_is_directory=True)

    with pytest.raises(ValueError, match="symbolic or unsafe"):
        _load_molscribe_predictor(linked_parent / "checkpoint.pth", device="cpu")


def test_output_short_writes_are_retried_until_complete(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request_path, checkpoint, output = _file_request_paths(tmp_path)
    _patch_fake_loader(monkeypatch)
    original_write = ocsr_execution.os.write
    write_sizes: list[int] = []

    def short_write(descriptor: int, payload: object) -> int:
        view = memoryview(payload)  # type: ignore[arg-type]
        limited = view[: min(7, len(view))]
        written = original_write(descriptor, limited)
        write_sizes.append(written)
        return written

    monkeypatch.setattr(ocsr_execution.os, "write", short_write)

    artifact = execute_ocsr_candidate_request_from_files(
        request_json=request_path,
        checkpoint_path=checkpoint,
        output_json=output,
        device="cpu",
    )

    published = OcsrCandidateArtifact.model_validate_json(output.read_bytes())
    assert published.model_dump(mode="json") == artifact.model_dump(mode="json")
    assert len(write_sizes) > 1
    assert all(0 < size <= 7 for size in write_sizes)


def test_output_parent_symlink_redirect_fails_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request_path, checkpoint, output = _file_request_paths(tmp_path)
    _patch_fake_loader(monkeypatch)
    original_validate = ocsr_execution._validate_published_ocsr_output
    displaced_parent = tmp_path / "displaced-output"
    redirect_target = tmp_path / "redirect-output"

    def perform_redirect(**kwargs: object) -> OcsrCandidateArtifact:
        output.parent.rename(displaced_parent)
        redirect_target.mkdir()
        (redirect_target / "marker").write_text("keep-me", encoding="utf-8")
        output.parent.symlink_to(redirect_target, target_is_directory=True)
        return original_validate(**kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(
        ocsr_execution,
        "_validate_published_ocsr_output",
        perform_redirect,
    )

    with pytest.raises(ValueError, match="parent changed|symbolic or unsafe"):
        execute_ocsr_candidate_request_from_files(
            request_json=request_path,
            checkpoint_path=checkpoint,
            output_json=output,
            device="cpu",
        )

    assert (redirect_target / "marker").read_text(encoding="utf-8") == "keep-me"
    assert not (redirect_target / output.name).exists()
    assert not (displaced_parent / output.name).exists()


def test_output_rejects_existing_symlink_parent_component(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request_path, checkpoint, _ = _file_request_paths(tmp_path)
    _patch_fake_loader(monkeypatch)
    actual_parent = tmp_path / "actual-output"
    actual_parent.mkdir()
    linked_parent = tmp_path / "linked-output"
    linked_parent.symlink_to(actual_parent, target_is_directory=True)

    with pytest.raises(ValueError, match="symbolic or unsafe"):
        execute_ocsr_candidate_request_from_files(
            request_json=request_path,
            checkpoint_path=checkpoint,
            output_json=linked_parent / "candidates.json",
            device="cpu",
        )

    assert list(actual_parent.iterdir()) == []


def test_post_write_byte_tamper_is_detected_and_owned_inode_is_cleaned(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request_path, checkpoint, output = _file_request_paths(tmp_path)
    _patch_fake_loader(monkeypatch)
    original_validate = ocsr_execution._validate_published_ocsr_output

    def tamper(**kwargs: object) -> OcsrCandidateArtifact:
        expected = bytes(kwargs["expected_bytes"])  # type: ignore[arg-type]
        with output.open("r+b") as handle:
            handle.write(b"X" + expected[1:])
            handle.flush()
            os.fsync(handle.fileno())
        return original_validate(**kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(ocsr_execution, "_validate_published_ocsr_output", tamper)

    with pytest.raises(ValueError, match="output bytes changed"):
        execute_ocsr_candidate_request_from_files(
            request_json=request_path,
            checkpoint_path=checkpoint,
            output_json=output,
            device="cpu",
        )

    assert not output.exists()


def test_post_write_path_replacement_is_preserved_and_fails_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request_path, checkpoint, output = _file_request_paths(tmp_path)
    _patch_fake_loader(monkeypatch)
    original_validate = ocsr_execution._validate_published_ocsr_output
    replacement = b"concurrent-writer-owned\n"

    def replace_output(**kwargs: object) -> OcsrCandidateArtifact:
        output.unlink()
        output.write_bytes(replacement)
        return original_validate(**kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(
        ocsr_execution,
        "_validate_published_ocsr_output",
        replace_output,
    )

    with pytest.raises(ValueError, match="output publication changed"):
        execute_ocsr_candidate_request_from_files(
            request_json=request_path,
            checkpoint_path=checkpoint,
            output_json=output,
            device="cpu",
        )

    assert output.read_bytes() == replacement
