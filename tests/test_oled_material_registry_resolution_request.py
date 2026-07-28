from __future__ import annotations

import json
from copy import deepcopy
from io import StringIO
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from ai4s_agent._utils import write_json
from ai4s_agent import oled_material_registry_resolution_request as registry_runner
from ai4s_agent.domains.oled_material_registry_resolution_request import (
    OledMaterialRegistryEntry,
    OledMaterialRegistryMatchStatus,
    OledMaterialRegistryResolutionRequestArtifact,
    OledMaterialRegistrySnapshot,
    _classify_match,
    build_oled_material_registry_entry,
    build_oled_material_registry_snapshot,
)
from ai4s_agent.oled_material_registry_resolution_request import (
    build_oled_material_registry_resolution_request_from_files,
    main,
    render_oled_material_registry_resolution_request_from_files,
)
from ai4s_agent.oled_supplementary_material_identity_review import (
    build_oled_supplementary_material_identity_adjudication_from_files,
)
from tests.test_oled_supplementary_material_identity_review import (
    _ADJUDICATED_AT,
    _adjudication_kwargs,
    _build_packet,
    _write_decisions,
)
from tests.test_oled_supplementary_scoped_candidate_response import _sha256_file


_SNAPSHOT_AT = "2026-07-14T00:05:00+08:00"
_REQUEST_AT = "2026-07-14T00:10:00+08:00"


def _accepted_source(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[Any, Path]:
    _, context = _build_packet(tmp_path, monkeypatch)
    decision_path, _ = _write_decisions(context)
    output_path = context["review_dir"] / "material-identity-adjudication.json"
    artifact = build_oled_supplementary_material_identity_adjudication_from_files(
        **_adjudication_kwargs(
            context,
            decision_path=decision_path,
            output_path=output_path,
        )
    )
    assert artifact.generated_at == _ADJUDICATED_AT
    return artifact, output_path


def _accepted_candidate(source: Any) -> tuple[Any, Any]:
    group = next(
        group
        for group in source.adjudicated_groups
        if group.eligible_for_later_registry_review
    )
    result = group.review_item.validated_result
    return group, result.response_result.structure_candidate


def _snapshot(
    entries: list[OledMaterialRegistryEntry],
) -> OledMaterialRegistrySnapshot:
    return build_oled_material_registry_snapshot(
        registry_id="molly-material-registry",
        registry_version="snapshot-0001",
        generated_at=_SNAPSHOT_AT,
        entries=entries,
    )


def _write_snapshot(tmp_path: Path, snapshot: OledMaterialRegistrySnapshot) -> Path:
    path = tmp_path / "material-registry-snapshot.json"
    write_json(path, snapshot.model_dump(mode="json"))
    return path


def _build_request(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    entries: list[OledMaterialRegistryEntry],
) -> tuple[Any, Any, Path, Path, Path]:
    source, source_path = _accepted_source(tmp_path, monkeypatch)
    snapshot = _snapshot(entries)
    snapshot_path = _write_snapshot(tmp_path, snapshot)
    output_path = tmp_path / "registry-resolution-request.json"
    request = build_oled_material_registry_resolution_request_from_files(
        source_adjudication_json=source_path,
        registry_snapshot_json=snapshot_path,
        output_json=output_path,
        generated_at=_REQUEST_AT,
    )
    return request, source, source_path, snapshot_path, output_path


def test_empty_registry_preserves_one_pr_m_candidate_as_unresolved_request(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request, source, source_path, snapshot_path, output_path = _build_request(
        tmp_path,
        monkeypatch,
        entries=[],
    )

    assert request.resolution_item_count == 1
    assert request.registry_eligible_group_count == 1
    assert request.registry_eligible_cell_count == 5
    assert request.source_review_item_count == 7
    assert request.no_exact_structural_candidate_count == 1
    assert request.device_only_cell_count == 0
    assert request.resolution_items[0].match_status == (
        OledMaterialRegistryMatchStatus.NO_EXACT_STRUCTURAL_CANDIDATE
    )
    assert not request.material_identity_resolved
    assert not request.canonical_material_id_assigned
    assert not request.registry_written
    assert not request.pr_m_upstream_chain_revalidated
    assert not request.source_pdf_read
    assert request.source_adjudication_sha256 == _sha256_file(source_path)
    assert request.registry_snapshot_sha256 == _sha256_file(snapshot_path)
    assert OledMaterialRegistryResolutionRequestArtifact.model_validate_json(
        output_path.read_text(encoding="utf-8")
    ) == request
    assert source.later_registry_review_eligible_group_count == 1


def test_consistent_smiles_and_inchikey_hit_is_only_a_review_candidate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source, source_path = _accepted_source(tmp_path, monkeypatch)
    group, candidate = _accepted_candidate(source)
    reported = group.review_item.validated_result.bound_identity_group.reported_subject_text
    entry = build_oled_material_registry_entry(
        material_id="material-0001",
        canonical_name=reported,
        aliases=["fixture-alias"],
        canonical_isomeric_smiles=candidate.canonical_isomeric_smiles_candidate,
    )
    snapshot_path = _write_snapshot(tmp_path, _snapshot([entry]))
    output_path = tmp_path / "exact-request.json"

    request = build_oled_material_registry_resolution_request_from_files(
        source_adjudication_json=source_path,
        registry_snapshot_json=snapshot_path,
        output_json=output_path,
        generated_at=_REQUEST_AT,
    )
    item = request.resolution_items[0]

    assert item.match_status == (
        OledMaterialRegistryMatchStatus.ONE_CONSISTENT_EXACT_STRUCTURAL_CANDIDATE
    )
    assert item.canonical_smiles_candidate_material_ids == ["material-0001"]
    assert item.inchikey_candidate_material_ids == ["material-0001"]
    assert item.consistent_exact_candidate_material_id == "material-0001"
    assert len(item.exact_alias_literal_hits) == 1
    assert item.exact_alias_literal_hits[0].matched_field.value == "canonical_name"
    assert not item.exact_alias_literal_hits[0].identity_evidence
    assert item.registry_resolution_required
    assert not item.material_identity_resolved
    assert not item.canonical_material_id_assigned
    assert request.consistent_exact_structural_candidate_count == 1
    assert not request.cross_paper_identity_merge


def test_duplicate_structural_keys_and_names_are_explicit_without_merge(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source, source_path = _accepted_source(tmp_path, monkeypatch)
    group, candidate = _accepted_candidate(source)
    reported = group.review_item.validated_result.bound_identity_group.reported_subject_text
    entries = [
        build_oled_material_registry_entry(
            material_id=material_id,
            canonical_name=reported if index == 0 else f"fixture-{index}",
            aliases=[] if index == 0 else [reported],
            canonical_isomeric_smiles=candidate.canonical_isomeric_smiles_candidate,
        )
        for index, material_id in enumerate(("material-0001", "material-0002"))
    ]
    snapshot_path = _write_snapshot(tmp_path, _snapshot(entries))
    output_path = tmp_path / "collision-request.json"

    request = build_oled_material_registry_resolution_request_from_files(
        source_adjudication_json=source_path,
        registry_snapshot_json=snapshot_path,
        output_json=output_path,
        generated_at=_REQUEST_AT,
    )
    item = request.resolution_items[0]

    assert item.match_status == (
        OledMaterialRegistryMatchStatus.AMBIGUOUS_DUPLICATE_STRUCTURAL_KEY
    )
    assert item.canonical_smiles_candidate_material_ids == [
        "material-0001",
        "material-0002",
    ]
    assert item.inchikey_candidate_material_ids == [
        "material-0001",
        "material-0002",
    ]
    assert item.consistent_exact_candidate_material_id == ""
    assert len(item.exact_alias_literal_hits) == 2
    assert request.registry_conflict_finding_count == 3
    assert {finding.finding_kind.value for finding in request.registry_conflict_findings} == {
        "duplicate_canonical_smiles",
        "duplicate_inchikey",
        "duplicate_reported_name_literal",
    }
    assert all(
        not finding.automatic_merge_performed
        for finding in request.registry_conflict_findings
    )
    assert not request.automatic_candidate_merge
    assert request.status.value == "registry_conflicts_require_human_review"


def test_unrelated_registry_collisions_do_not_expand_the_bounded_review_packet(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source, source_path = _accepted_source(tmp_path, monkeypatch)
    del source
    entries = [
        build_oled_material_registry_entry(
            material_id=material_id,
            canonical_name=f"unrelated-{index}",
            canonical_isomeric_smiles="c1ccccc1",
        )
        for index, material_id in enumerate(("material-0101", "material-0102"))
    ]
    snapshot_path = _write_snapshot(tmp_path, _snapshot(entries))

    request = build_oled_material_registry_resolution_request_from_files(
        source_adjudication_json=source_path,
        registry_snapshot_json=snapshot_path,
        output_json=tmp_path / "bounded-request.json",
        generated_at=_REQUEST_AT,
    )

    assert request.registry_conflict_finding_count == 0
    assert request.no_exact_structural_candidate_count == 1
    assert request.status.value == "ready_for_human_registry_resolution_review"


@pytest.mark.parametrize(
    ("smiles_ids", "inchikey_ids", "expected"),
    (
        ([], [], "no_exact_structural_candidate"),
        (["m1"], [], "partial_structural_key_match"),
        ([], ["m1"], "partial_structural_key_match"),
        (["m1"], ["m1"], "one_consistent_exact_structural_candidate"),
        (["m1", "m2"], ["m1", "m2"], "ambiguous_duplicate_structural_key"),
        (["m1"], ["m2"], "conflicting_structural_key_matches"),
    ),
)
def test_match_classification_is_fail_closed(
    smiles_ids: list[str],
    inchikey_ids: list[str],
    expected: str,
) -> None:
    status, consistent = _classify_match(smiles_ids, inchikey_ids)
    assert status.value == expected
    assert consistent == ("m1" if expected.startswith("one_consistent") else "")


def test_registry_entry_rejects_inconsistent_chemical_identifiers() -> None:
    entry = build_oled_material_registry_entry(
        material_id="material-0001",
        canonical_name="benzene",
        canonical_isomeric_smiles="c1ccccc1",
    )
    payload = entry.model_dump(mode="json")
    payload["inchikey"] = "AAAAAAAAAAAAAA-BBBBBBBBBB-C"

    with pytest.raises(ValidationError, match="inconsistent|digest"):
        OledMaterialRegistryEntry.model_validate(payload)


@pytest.mark.parametrize(
    "unsafe_name",
    (
        "<script>alert(1)</script>",
        "token=do-not-store",
        "https://example.test/material",
        "/Users/operator/private/material",
    ),
)
def test_registry_names_reject_active_or_sensitive_text(unsafe_name: str) -> None:
    with pytest.raises(ValidationError):
        build_oled_material_registry_entry(
            material_id="material-0001",
            canonical_name=unsafe_name,
            canonical_isomeric_smiles="c1ccccc1",
        )


def test_snapshot_digest_or_runtime_tamper_is_rejected() -> None:
    snapshot = _snapshot([])
    digest_tamper = snapshot.model_dump(mode="json")
    digest_tamper["registry_version"] = "snapshot-0002"
    with pytest.raises(ValidationError, match="digest"):
        OledMaterialRegistrySnapshot.model_validate(digest_tamper)

    runtime_tamper = snapshot.model_dump(mode="json")
    runtime_tamper["toolkit_version"] = "0.0.0"
    with pytest.raises(ValidationError, match="runtime"):
        OledMaterialRegistrySnapshot.model_validate(runtime_tamper)


def test_request_omits_every_nonaccepted_group_and_all_ontology_device_cells(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request, source, _, _, _ = _build_request(
        tmp_path,
        monkeypatch,
        entries=[],
    )

    source_ids = {
        group.review_item.validated_result.bound_identity_group.identity_group_id
        for group in source.adjudicated_groups
        if group.eligible_for_later_registry_review
    }
    request_ids = {
        item.adjudicated_group.review_item.validated_result.bound_identity_group
        .identity_group_id
        for item in request.resolution_items
    }
    assert request_ids == source_ids
    assert len(request_ids) == 1
    assert request.registry_eligible_cell_count == 5
    assert source.upstream_ontology_review_pending_cell_count == 14
    assert request.device_only_cell_count == 0


def test_markdown_is_evidence_first_and_does_not_claim_resolution(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source, source_path = _accepted_source(tmp_path, monkeypatch)
    group, candidate = _accepted_candidate(source)
    reported = group.review_item.validated_result.bound_identity_group.reported_subject_text
    entry = build_oled_material_registry_entry(
        material_id="material-0001",
        canonical_name=reported,
        canonical_isomeric_smiles=candidate.canonical_isomeric_smiles_candidate,
    )
    snapshot_path = _write_snapshot(tmp_path, _snapshot([entry]))
    request_path = tmp_path / "request.json"
    build_oled_material_registry_resolution_request_from_files(
        source_adjudication_json=source_path,
        registry_snapshot_json=snapshot_path,
        output_json=request_path,
        generated_at=_REQUEST_AT,
    )
    markdown_path = tmp_path / "request.md"

    render_oled_material_registry_resolution_request_from_files(
        request_artifact_json=request_path,
        output_markdown=markdown_path,
    )
    markdown = markdown_path.read_text(encoding="utf-8")

    assert markdown.index("PR-M accepted paper-local evidence") < markdown.index(
        "Deterministic Registry lookup"
    )
    assert "alias hits are exact literal hints only" in markdown
    assert "material identity resolved: `false`" in markdown
    assert "No canonical material ID has been assigned" in markdown
    assert "Registry candidate entry projections (snapshot data, not source evidence)" in (
        markdown
    )
    assert "- material ID: `material-0001`" in markdown
    assert "- Registry entry count: `1`" in markdown
    assert "Human decision required in the next stage" in markdown
    assert "device-only cells admitted: `0`" in markdown
    assert "<script" not in markdown.lower()


@pytest.mark.parametrize("protected_key", ("source", "snapshot"))
def test_request_output_cannot_overwrite_either_input(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    protected_key: str,
) -> None:
    source, source_path = _accepted_source(tmp_path, monkeypatch)
    del source
    snapshot_path = _write_snapshot(tmp_path, _snapshot([]))
    protected = {"source": source_path, "snapshot": snapshot_path}[protected_key]
    before = protected.read_bytes()

    with pytest.raises(ValueError, match="overwrite"):
        build_oled_material_registry_resolution_request_from_files(
            source_adjudication_json=source_path,
            registry_snapshot_json=snapshot_path,
            output_json=protected,
            generated_at=_REQUEST_AT,
        )

    assert protected.read_bytes() == before


def test_symlinked_snapshot_and_output_parent_are_rejected_without_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, source_path = _accepted_source(tmp_path, monkeypatch)
    snapshot_path = _write_snapshot(tmp_path, _snapshot([]))
    snapshot_alias = tmp_path / "snapshot-alias.json"
    snapshot_alias.symlink_to(snapshot_path)
    output_path = tmp_path / "must-not-exist.json"

    with pytest.raises(ValueError):
        build_oled_material_registry_resolution_request_from_files(
            source_adjudication_json=source_path,
            registry_snapshot_json=snapshot_alias,
            output_json=output_path,
            generated_at=_REQUEST_AT,
        )
    assert not output_path.exists()

    output_parent = tmp_path / "output-alias"
    real_parent = tmp_path / "real-output"
    real_parent.mkdir()
    output_parent.symlink_to(real_parent, target_is_directory=True)
    with pytest.raises(ValueError):
        build_oled_material_registry_resolution_request_from_files(
            source_adjudication_json=source_path,
            registry_snapshot_json=snapshot_path,
            output_json=output_parent / "must-not-exist.json",
            generated_at=_REQUEST_AT,
        )
    assert not (real_parent / "must-not-exist.json").exists()


@pytest.mark.parametrize("replacement_kind", ("symlink", "directory"))
def test_output_parent_replacement_during_build_fails_without_publication(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    replacement_kind: str,
) -> None:
    _, source_path = _accepted_source(tmp_path, monkeypatch)
    snapshot_path = _write_snapshot(tmp_path, _snapshot([]))
    output_parent = tmp_path / "pinned-output"
    output_parent.mkdir()
    displaced = tmp_path / "pinned-output-displaced"
    redirected = tmp_path / "pinned-output-redirected"
    redirected.mkdir()
    output_path = output_parent / "must-not-exist.json"
    original_builder = (
        registry_runner.build_oled_material_registry_resolution_request_artifact
    )

    def replace_parent_after_build(**kwargs: Any) -> Any:
        artifact = original_builder(**kwargs)
        output_parent.rename(displaced)
        if replacement_kind == "symlink":
            output_parent.symlink_to(redirected, target_is_directory=True)
        else:
            output_parent.mkdir()
        return artifact

    monkeypatch.setattr(
        registry_runner,
        "build_oled_material_registry_resolution_request_artifact",
        replace_parent_after_build,
    )

    with pytest.raises(ValueError, match="parent changed"):
        build_oled_material_registry_resolution_request_from_files(
            source_adjudication_json=source_path,
            registry_snapshot_json=snapshot_path,
            output_json=output_path,
            generated_at=_REQUEST_AT,
        )

    assert not (displaced / output_path.name).exists()
    assert not (redirected / output_path.name).exists()
    assert not (output_parent / output_path.name).exists()


def test_output_parent_replacement_during_render_fails_without_publication(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request_workspace = tmp_path / "request-workspace"
    request_workspace.mkdir()
    _, _, _, _, request_path = _build_request(
        request_workspace,
        monkeypatch,
        entries=[],
    )
    output_parent = tmp_path / "render-output"
    output_parent.mkdir()
    displaced = tmp_path / "render-output-displaced"
    redirected = tmp_path / "render-output-redirected"
    redirected.mkdir()
    output_path = output_parent / "must-not-exist.md"
    original_renderer = (
        registry_runner.render_oled_material_registry_resolution_request_markdown
    )

    def replace_parent_after_render(*args: Any, **kwargs: Any) -> str:
        markdown = original_renderer(*args, **kwargs)
        output_parent.rename(displaced)
        output_parent.symlink_to(redirected, target_is_directory=True)
        return markdown

    monkeypatch.setattr(
        registry_runner,
        "render_oled_material_registry_resolution_request_markdown",
        replace_parent_after_render,
    )

    with pytest.raises(ValueError, match="parent changed"):
        render_oled_material_registry_resolution_request_from_files(
            request_artifact_json=request_path,
            output_markdown=output_path,
        )

    assert not (displaced / output_path.name).exists()
    assert not (redirected / output_path.name).exists()


def test_embedded_snapshot_or_item_tamper_breaks_request_validation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request, _, _, _, _ = _build_request(tmp_path, monkeypatch, entries=[])
    payload = request.model_dump(mode="json")
    payload["resolution_items"][0]["match_status"] = (
        "one_consistent_exact_structural_candidate"
    )
    with pytest.raises(ValidationError, match="classification|derivation|digest"):
        OledMaterialRegistryResolutionRequestArtifact.model_validate(payload)

    payload = request.model_dump(mode="json")
    payload["registry_snapshot"]["registry_version"] = "changed"
    with pytest.raises(ValidationError, match="digest"):
        OledMaterialRegistryResolutionRequestArtifact.model_validate(payload)


def test_cli_failure_is_redacted_and_does_not_publish_output(tmp_path: Path) -> None:
    sensitive = tmp_path / "token=do-not-disclose.json"
    output = tmp_path / "must-not-exist.json"
    stream = StringIO()

    status = main(
        [
            "build",
            "--source-adjudication",
            str(sensitive),
            "--registry-snapshot",
            str(tmp_path / "missing-snapshot.json"),
            "--output",
            str(output),
        ],
        stdout=stream,
    )

    assert status == 2
    assert json.loads(stream.getvalue()) == {
        "error_code": "material_registry_resolution_request_failed",
        "error_type": "ValueError",
        "status": "error",
    }
    assert str(tmp_path) not in stream.getvalue()
    assert "do-not-disclose" not in stream.getvalue()
    assert not output.exists()


def test_request_generated_at_cannot_predate_either_input(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, source_path = _accepted_source(tmp_path, monkeypatch)
    snapshot_path = _write_snapshot(tmp_path, _snapshot([]))
    output_path = tmp_path / "must-not-exist.json"

    with pytest.raises(ValidationError, match="predates"):
        build_oled_material_registry_resolution_request_from_files(
            source_adjudication_json=source_path,
            registry_snapshot_json=snapshot_path,
            output_json=output_path,
            generated_at="2026-07-13T23:59:00+08:00",
        )
    assert not output_path.exists()


def test_registry_snapshot_input_whitespace_changes_exact_byte_binding(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, source_path = _accepted_source(tmp_path, monkeypatch)
    snapshot = _snapshot([])
    compact_path = tmp_path / "compact-snapshot.json"
    pretty_path = tmp_path / "pretty-snapshot.json"
    compact_path.write_text(
        json.dumps(snapshot.model_dump(mode="json"), separators=(",", ":")),
        encoding="utf-8",
    )
    pretty_path.write_text(
        json.dumps(snapshot.model_dump(mode="json"), indent=2) + "\n",
        encoding="utf-8",
    )

    compact_request = build_oled_material_registry_resolution_request_from_files(
        source_adjudication_json=source_path,
        registry_snapshot_json=compact_path,
        output_json=tmp_path / "compact-request.json",
        generated_at=_REQUEST_AT,
    )
    pretty_request = build_oled_material_registry_resolution_request_from_files(
        source_adjudication_json=source_path,
        registry_snapshot_json=pretty_path,
        output_json=tmp_path / "pretty-request.json",
        generated_at=_REQUEST_AT,
    )

    assert compact_request.registry_snapshot_digest == (
        pretty_request.registry_snapshot_digest
    )
    assert compact_request.registry_snapshot_sha256 != (
        pretty_request.registry_snapshot_sha256
    )


def test_source_adjudication_payload_is_not_mutated_by_request_builder(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source, source_path = _accepted_source(tmp_path, monkeypatch)
    before = deepcopy(source.model_dump(mode="json"))
    snapshot_path = _write_snapshot(tmp_path, _snapshot([]))

    request = build_oled_material_registry_resolution_request_from_files(
        source_adjudication_json=source_path,
        registry_snapshot_json=snapshot_path,
        output_json=tmp_path / "request.json",
        generated_at=_REQUEST_AT,
    )

    assert source.model_dump(mode="json") == before
    assert request.source_adjudication.model_dump(mode="json") == before
