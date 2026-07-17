from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from urllib.parse import quote

import pytest

import ai4s_agent.contextual_alias_resolution as runner
from ai4s_agent.contextual_alias_resolution import (
    ContextualAliasResolutionArtifact,
    ContextualAliasResolutionRequest,
    OpsinExchange,
    build_contextual_alias_resolution_artifact,
    build_contextual_alias_resolution_from_files,
)
from ai4s_agent.ocsr_candidate_execution import _sha256_bytes


_GENERATED_AT = "2026-07-17T00:00:00Z"
_BENZENE = {
    "status": "SUCCESS",
    "smiles": "c1ccccc1",
    "stdinchi": "InChI=1S/C6H6/c1-2-4-6-5-3-1/h1-6H",
    "stdinchikey": "UHOVQNZJYSORNB-UHFFFAOYSA-N",
}


def _resolver(name: str) -> OpsinExchange:
    assert name in {"benzene", "2-phenylbenzen-1-amine"}
    return OpsinExchange(
        endpoint_url=f"https://www.ebi.ac.uk/opsin/ws/{quote(name, safe='')}.json",
        http_status=200,
        response_bytes=json.dumps(_BENZENE, sort_keys=True).encode(),
    )


def _request(text: str, *, alias: str = "Bz") -> ContextualAliasResolutionRequest:
    return ContextualAliasResolutionRequest(
        run_id="run-an-test",
        paper_id="paper-test",
        source_document_id="paper-test-si",
        parsed_text_file="parsed.txt",
        parsed_text_sha256=_sha256_bytes(text.encode()),
        items=[{"candidate_id": "candidate-001", "reported_alias": alias}],
    )


def _file_inputs(tmp_path: Path, text: str = "=== PAGE 2 ===\nbenzene (Bz): 90%\n") -> tuple[Path, Path]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    request = _request(text)
    text_path = tmp_path / request.parsed_text_file
    request_path = tmp_path / "request.json"
    text_path.write_text(text, encoding="utf-8")
    request_path.write_text(json.dumps(request.model_dump(mode="json")), encoding="utf-8")
    return request_path, tmp_path / "result.json"


def test_wrapped_heading_resolves_and_persists_candidate_only_artifact(tmp_path: Path) -> None:
    text = "=== PAGE 7 ===\n2-phenylben-\nzen-1-amine (Bz): 81%\n"
    request = _request(text)
    artifact = build_contextual_alias_resolution_artifact(
        request,
        request_sha256=_sha256_bytes(b"request"),
        parsed_text=text,
        resolver=_resolver,
        generated_at=_GENERATED_AT,
    )

    result = artifact.results[0]
    assert result.status == "candidate_ready"
    assert result.systematic_name == "2-phenylbenzen-1-amine"
    assert result.source_locator == "page=7;lines=2-3;alias=Bz"
    assert result.inchikey == _BENZENE["stdinchikey"]
    assert artifact.candidate_only is True
    assert artifact.source_match_validated is False
    assert artifact.identity_resolved is False
    assert artifact.registry_mutated is False
    assert artifact.gold_written is False
    assert artifact.dataset_written is False
    ContextualAliasResolutionArtifact.model_validate(artifact.model_dump(mode="json"))


@pytest.mark.parametrize(
    ("text", "status"),
    [
        ("=== PAGE 1 ===\nno matching heading\n", "alias_not_found"),
        ("benzene (Bz):\nbenzene (Bz):\n", "alias_ambiguous"),
    ],
)
def test_missing_or_ambiguous_alias_never_calls_resolver(text: str, status: str) -> None:
    def forbidden(_: str) -> OpsinExchange:
        raise AssertionError("resolver must not run")

    artifact = build_contextual_alias_resolution_artifact(
        _request(text),
        request_sha256=_sha256_bytes(b"request"),
        parsed_text=text,
        resolver=forbidden,
        generated_at=_GENERATED_AT,
    )
    assert artifact.results[0].status == status
    assert artifact.results[0].systematic_name == ""


def test_heading_join_never_crosses_a_page_marker() -> None:
    text = "previous-page-name\n=== PAGE 2 ===\n(Bz): no heading on this page\n"

    def forbidden(_: str) -> OpsinExchange:
        raise AssertionError("resolver must not run")

    artifact = build_contextual_alias_resolution_artifact(
        _request(text),
        request_sha256=_sha256_bytes(b"request"),
        parsed_text=text,
        resolver=forbidden,
        generated_at=_GENERATED_AT,
    )
    assert artifact.results[0].status == "alias_not_found"


def test_opsin_rejection_remains_a_rejected_candidate() -> None:
    text = "benzene (Bz):\n"

    def rejected(_: str) -> OpsinExchange:
        return OpsinExchange(
            endpoint_url="https://www.ebi.ac.uk/opsin/ws/benzene.json",
            http_status=404,
            response_bytes=b'{"status":"FAILURE","message":"not parsed"}',
        )

    artifact = build_contextual_alias_resolution_artifact(
        _request(text),
        request_sha256=_sha256_bytes(b"request"),
        parsed_text=text,
        resolver=rejected,
        generated_at=_GENERATED_AT,
    )
    assert artifact.ready_count == 0
    assert artifact.results[0].status == "name_resolution_rejected"
    assert artifact.results[0].rejection_reason == "not parsed"


def test_opsin_and_rdkit_identifier_disagreement_fails_closed() -> None:
    text = "benzene (Bz):\n"

    def dishonest(_: str) -> OpsinExchange:
        payload = {**_BENZENE, "stdinchikey": "AAAAAAAAAAAAAA-BBBBBBBBBB-C"}
        return OpsinExchange(
            endpoint_url="https://www.ebi.ac.uk/opsin/ws/benzene.json",
            http_status=200,
            response_bytes=json.dumps(payload).encode(),
        )

    with pytest.raises(ValueError, match="OPSIN and RDKit"):
        build_contextual_alias_resolution_artifact(
            _request(text),
            request_sha256=_sha256_bytes(b"request"),
            parsed_text=text,
            resolver=dishonest,
            generated_at=_GENERATED_AT,
        )


def test_resigned_embedded_response_and_request_tampering_fail_validation() -> None:
    text = "benzene (Bz):\n"
    artifact = build_contextual_alias_resolution_artifact(
        _request(text),
        request_sha256=_sha256_bytes(b"request"),
        parsed_text=text,
        resolver=_resolver,
        generated_at=_GENERATED_AT,
    )
    payload = artifact.model_dump(mode="json")
    result = payload["results"][0]
    result["resolver_response_sha256"] = "sha256:" + "a" * 64
    result["result_digest"] = runner._stable_hash(
        {key: value for key, value in result.items() if key != "result_digest"}
    )
    payload["artifact_digest"] = runner._stable_hash(
        {key: value for key, value in payload.items() if key != "artifact_digest"}
    )
    with pytest.raises(ValueError, match="resolver provenance mismatch"):
        ContextualAliasResolutionArtifact.model_validate(payload)

    payload = artifact.model_dump(mode="json")
    payload["request"]["paper_id"] = "paper-substituted"
    payload["request_digest"] = runner._stable_hash(payload["request"])
    payload["artifact_digest"] = runner._stable_hash(
        {key: value for key, value in payload.items() if key != "artifact_digest"}
    )
    with pytest.raises(ValueError, match="request binding mismatch"):
        ContextualAliasResolutionArtifact.model_validate(payload)

def test_file_runner_rechecks_inputs_and_writes_exact_validated_artifact(tmp_path: Path) -> None:
    request_path, output_path = _file_inputs(tmp_path)
    artifact = build_contextual_alias_resolution_from_files(
        request_json=request_path,
        output_json=output_path,
        generated_at=_GENERATED_AT,
        resolver=_resolver,
    )
    assert output_path.exists()
    assert ContextualAliasResolutionArtifact.model_validate_json(output_path.read_bytes()) == artifact


def test_duplicate_request_keys_and_symlinked_text_are_rejected(tmp_path: Path) -> None:
    request_path, output_path = _file_inputs(tmp_path)
    request_path.write_text('{"run_id":"a","run_id":"b"}', encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate JSON keys"):
        build_contextual_alias_resolution_from_files(
            request_json=request_path, output_json=output_path, resolver=_resolver
        )


def test_parsed_text_digest_mismatch_is_rejected(tmp_path: Path) -> None:
    request_path, output_path = _file_inputs(tmp_path)
    (tmp_path / "parsed.txt").write_text("changed before execution", encoding="utf-8")
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        build_contextual_alias_resolution_from_files(
            request_json=request_path, output_json=output_path, resolver=_resolver
        )
    assert not output_path.exists()


def test_symlinked_parsed_text_is_rejected(tmp_path: Path) -> None:
    request_path, output_path = _file_inputs(tmp_path / "second")
    text_path = request_path.parent / "parsed.txt"
    real_text = request_path.parent / "real.txt"
    text_path.rename(real_text)
    text_path.symlink_to(real_text)
    with pytest.raises(ValueError, match="symbolic|unsafe"):
        build_contextual_alias_resolution_from_files(
            request_json=request_path, output_json=output_path, resolver=_resolver
        )


def test_existing_output_is_preserved(tmp_path: Path) -> None:
    request_path, output_path = _file_inputs(tmp_path)
    output_path.write_text("operator-owned\n", encoding="utf-8")
    with pytest.raises(ValueError, match="already exists"):
        build_contextual_alias_resolution_from_files(
            request_json=request_path, output_json=output_path, resolver=_resolver
        )
    assert output_path.read_text(encoding="utf-8") == "operator-owned\n"


@pytest.mark.parametrize("replacement_kind", ("directory", "symlink"))
def test_output_parent_replacement_fails_without_redirect(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, replacement_kind: str
) -> None:
    input_dir = tmp_path / "inputs"
    input_dir.mkdir()
    request_path, _ = _file_inputs(input_dir)
    output_parent = tmp_path / "output"
    output_parent.mkdir()
    output_path = output_parent / "result.json"
    displaced = tmp_path / "displaced"
    redirected = tmp_path / "redirected"
    redirected.mkdir()
    original = runner.build_contextual_alias_resolution_artifact

    def replace_after_build(*args: Any, **kwargs: Any) -> ContextualAliasResolutionArtifact:
        artifact = original(*args, **kwargs)
        output_parent.rename(displaced)
        if replacement_kind == "symlink":
            output_parent.symlink_to(redirected, target_is_directory=True)
        else:
            output_parent.mkdir()
        return artifact

    monkeypatch.setattr(runner, "build_contextual_alias_resolution_artifact", replace_after_build)
    with pytest.raises(ValueError, match="output parent changed|symbolic"):
        build_contextual_alias_resolution_from_files(
            request_json=request_path, output_json=output_path, resolver=_resolver
        )
    assert not output_path.exists()
    assert not (redirected / output_path.name).exists()
    assert not (displaced / output_path.name).exists()


def test_short_writes_are_completed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    request_path, output_path = _file_inputs(tmp_path)
    real_write = runner.os.write

    def short_write(descriptor: int, data: bytes | memoryview) -> int:
        return real_write(descriptor, data[: max(1, len(data) // 3)])

    monkeypatch.setattr(runner.os, "write", short_write)
    artifact = build_contextual_alias_resolution_from_files(
        request_json=request_path, output_json=output_path, resolver=_resolver
    )
    assert ContextualAliasResolutionArtifact.model_validate_json(output_path.read_bytes()) == artifact


def test_input_pair_swap_after_build_fails_before_publication(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    request_path, output_path = _file_inputs(tmp_path)
    original = runner.build_contextual_alias_resolution_artifact

    def swap_after_build(*args: Any, **kwargs: Any) -> ContextualAliasResolutionArtifact:
        artifact = original(*args, **kwargs)
        (tmp_path / "parsed.txt").write_text("replacement", encoding="utf-8")
        return artifact

    monkeypatch.setattr(runner, "build_contextual_alias_resolution_artifact", swap_after_build)
    with pytest.raises(ValueError, match="inputs changed"):
        build_contextual_alias_resolution_from_files(
            request_json=request_path, output_json=output_path, resolver=_resolver
        )
    assert not output_path.exists()
