from __future__ import annotations

import hashlib
import json
import re
from copy import deepcopy
from io import StringIO
from pathlib import Path
from typing import Any, Callable

import pytest

from ai4s_agent._utils import write_json
from ai4s_agent.domains.oled_supplementary_material_identity_evidence_response import (
    oled_supplementary_material_identity_evidence_anchor_digest,
)
from ai4s_agent.domains.oled_supplementary_material_identity_review import (
    OledSupplementaryMaterialIdentityReviewPacket,
    build_oled_supplementary_material_identity_candidate_depiction_asset,
)
from ai4s_agent.domains.oled_supplementary_source_transcription_review import (
    build_oled_supplementary_source_page_asset,
    build_oled_supplementary_source_pdf_evidence,
)
from ai4s_agent import (
    oled_supplementary_material_identity_review as identity_review_runner,
)
from ai4s_agent.oled_supplementary_material_identity_review import (
    build_oled_supplementary_material_identity_adjudication_from_files,
    build_oled_supplementary_material_identity_review_packet_from_files,
    main,
    render_oled_supplementary_material_identity_review_packet_from_files,
)
from tests.test_oled_supplementary_material_identity_evidence_response import (
    _anchor_only_result,
    _build_from_files,
    _candidate_result,
    _group_binding,
    _source_reported_inchi_result,
    _source_check_result,
    _source_reported_stereo_smiles_result,
)
from tests.test_oled_supplementary_scoped_candidate_response import _sha256_file
from tests.test_oled_supplementary_source_transcription_review import (
    _FAKE_PNG_BYTES,
)


_PACKET_GENERATED_AT = "2026-07-13T23:40:00+08:00"
_REVIEWED_AT = "2026-07-13T23:50:00+08:00"
_ADJUDICATED_AT = "2026-07-14T00:00:00+08:00"
_FIXTURE_PDF_PAGE_COUNT = 39


def test_cli_failure_is_stable_and_does_not_disclose_local_paths(
    tmp_path: Path,
) -> None:
    sensitive_path = tmp_path / "token=do-not-disclose.json"
    stream = StringIO()

    status = main(
        [
            "packet",
            "--request-artifact",
            str(sensitive_path),
            "--transcription-review-packet",
            str(tmp_path / "missing-j.json"),
            "--response-manifest",
            str(tmp_path / "missing-manifest.json"),
            "--response-artifact",
            str(tmp_path / "missing-l.json"),
            "--source-pdf",
            str(tmp_path / "missing.pdf"),
            "--asset-dir",
            str(tmp_path / "assets"),
            "--output",
            str(tmp_path / "packet.json"),
        ],
        stdout=stream,
    )

    assert status == 2
    payload = json.loads(stream.getvalue())
    assert payload == {
        "error_code": "supplementary_material_identity_review_failed",
        "error_type": "ValueError",
        "status": "error",
    }
    assert str(tmp_path) not in stream.getvalue()
    assert "do-not-disclose" not in stream.getvalue()


def _stable_hash(value: Any) -> str:
    canonical = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return f"sha256:{hashlib.sha256(canonical.encode('utf-8')).hexdigest()}"


def _ambiguous_result(group: Any) -> dict[str, Any]:
    return {
        **_group_binding(group),
        "disposition": "ambiguous_identity",
        "ambiguity_reason": "multiple_structures_associated",
        "review_note": "The bounded source does not support one unique assignment.",
    }


def _exclusion_result(group: Any) -> dict[str, Any]:
    return {
        **_group_binding(group),
        "disposition": "exclude_identity_group",
        "exclusion_reason": "not_a_material_entity",
        "review_note": "The bounded row is proposed as not being a material entity.",
    }


def _mixed_result(group: Any) -> dict[str, Any]:
    factories: dict[int, Callable[[Any], dict[str, Any]]] = {
        0: _candidate_result,
        1: _anchor_only_result,
        2: _source_check_result,
        3: _ambiguous_result,
        4: _exclusion_result,
        5: _anchor_only_result,
        6: _anchor_only_result,
    }
    return factories[group.row_index](group)


def _two_candidate_result(group: Any) -> dict[str, Any]:
    if group.row_index == 0:
        return _candidate_result(group)
    if group.row_index == 1:
        return _source_reported_stereo_smiles_result(group)
    return _source_check_result(group)


def _collision_result(group: Any) -> dict[str, Any]:
    if group.row_index in {0, 1}:
        return _candidate_result(group)
    return _mixed_result(group)


def _inchi_candidate_result(group: Any) -> dict[str, Any]:
    if group.row_index == 0:
        return _source_reported_inchi_result(group)
    return _mixed_result(group)


def _install_fake_page_renderer(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_render_exact_bound_source_pdf_pages(
        *,
        source_pdf_path: Path,
        source_id: str,
        expected_sha256: str,
        pages: list[int],
        poppler_bin_dir: str | Path | None = None,
        reject_symlink_components: bool = False,
    ) -> tuple[Any, dict[str, bytes]]:
        del poppler_bin_dir
        source_path = Path(source_pdf_path)
        if reject_symlink_components:
            current = Path(source_path.anchor)
            for part in source_path.parts[1:]:
                current /= part
                if current.is_symlink():
                    raise ValueError("fake renderer rejects symlink path components")
        if _sha256_file(source_path) != expected_sha256:
            raise ValueError("fake renderer PDF does not match the bound hash")
        if not pages or pages != sorted(set(pages)) or any(page < 1 for page in pages):
            raise ValueError("fake renderer requires sorted unique positive pages")
        assets = []
        rendered: dict[str, bytes] = {}
        for page in pages:
            asset = build_oled_supplementary_source_page_asset(
                source_id=source_id,
                source_pdf_sha256=expected_sha256,
                pdf_page_number_one_based=page,
                renderer_id="poppler-pdftoppm",
                renderer_version="test-26.05.0",
                render_profile="png-200dpi-rgb-full-page-v1",
                rendered_asset_sha256=(
                    "sha256:" + hashlib.sha256(_FAKE_PNG_BYTES).hexdigest()
                ),
                rendered_asset_byte_size=len(_FAKE_PNG_BYTES),
                pixel_width=1700,
                pixel_height=2200,
            )
            assets.append(asset)
            rendered[asset.asset_filename] = _FAKE_PNG_BYTES
        evidence = build_oled_supplementary_source_pdf_evidence(
            source_id=source_id,
            source_pdf_sha256=expected_sha256,
            source_pdf_byte_size=source_path.stat().st_size,
            source_pdf_page_count=_FIXTURE_PDF_PAGE_COUNT,
            page_counter_version="test-26.05.0",
            page_counter_executable_sha256="sha256:" + "c" * 64,
            renderer_executable_sha256="sha256:" + "d" * 64,
            page_assets=assets,
        )
        return evidence, rendered

    monkeypatch.setattr(
        identity_review_runner,
        "_render_exact_bound_source_pdf_pages",
        fake_render_exact_bound_source_pdf_pages,
    )


def _packet_kwargs(
    context: dict[str, Any],
    *,
    source_pdf_path: Path | None = None,
    asset_dir: Path | None = None,
    output_path: Path | None = None,
) -> dict[str, Any]:
    return {
        "request_artifact_json": context["request_path"],
        "transcription_review_packet_json": context[
            "transcription_review_packet_path"
        ],
        "response_manifest_json": context["response_manifest_path"],
        "response_artifact_json": context["response_artifact_path"],
        "source_pdf_path": source_pdf_path or context["source_pdf_path"],
        "asset_dir": asset_dir or context["asset_dir"],
        "output_json": output_path or context["packet_path"],
        "generated_at": _PACKET_GENERATED_AT,
    }


def _adjudication_kwargs(
    context: dict[str, Any],
    *,
    decision_path: Path,
    output_path: Path,
    source_pdf_path: Path | None = None,
) -> dict[str, Any]:
    return {
        "request_artifact_json": context["request_path"],
        "transcription_review_packet_json": context[
            "transcription_review_packet_path"
        ],
        "response_manifest_json": context["response_manifest_path"],
        "response_artifact_json": context["response_artifact_path"],
        "source_pdf_path": source_pdf_path or context["source_pdf_path"],
        "review_packet_json": context["packet_path"],
        "decision_manifest_json": decision_path,
        "asset_dir": context["asset_dir"],
        "output_json": output_path,
        "generated_at": _ADJUDICATED_AT,
    }


def _build_packet(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    result_factory: Callable[[Any], dict[str, Any]] = _mixed_result,
) -> tuple[Any, dict[str, Any]]:
    (
        response_artifact,
        chain,
        request,
        request_path,
        response_manifest_path,
        response_artifact_path,
    ) = _build_from_files(tmp_path, result_factory=result_factory)
    del request
    _install_fake_page_renderer(monkeypatch)
    review_dir = tmp_path / "material-identity-review"
    review_dir.mkdir()
    context = {
        "chain": chain,
        "response_artifact": response_artifact,
        "request_path": request_path,
        "transcription_review_packet_path": chain["transcription_packet_path"],
        "response_manifest_path": response_manifest_path,
        "response_artifact_path": response_artifact_path,
        "source_pdf_path": chain["source_pdf_path"],
        "review_dir": review_dir,
        "asset_dir": review_dir / "assets",
        "packet_path": review_dir / "material-identity-review-packet.json",
    }
    packet = build_oled_supplementary_material_identity_review_packet_from_files(
        **_packet_kwargs(context)
    )
    context["packet"] = packet
    return packet, context


def _decision_payload(packet: Any, packet_path: Path) -> dict[str, Any]:
    decision_by_row = {
        0: "accept_structure_candidate",
        1: "accept_structure_anchor_only",
        2: "confirm_source_check",
        3: "confirm_ambiguous_identity",
        4: "accept_identity_exclusion",
        5: "needs_source_check",
        6: "reject_response_evidence",
    }
    entries = []
    for item in packet.review_items:
        response = item.validated_result.response_result
        row_index = item.validated_result.bound_identity_group.row_index
        anchor_assessments = []
        for anchor in getattr(response, "evidence_anchors", []):
            assessment = "supports_claim"
            review_note = ""
            if row_index == 5:
                assessment = "not_checked"
                review_note = "The source page needs a closer manual check."
            elif row_index == 6:
                assessment = "does_not_support_claim"
                review_note = "The asserted panel does not support this row assignment."
            anchor_assessments.append(
                {
                    "evidence_anchor_digest": (
                        oled_supplementary_material_identity_evidence_anchor_digest(
                            anchor
                        )
                    ),
                    "assessment": assessment,
                    "review_note": review_note,
                }
            )
        anchor_assessments.sort(key=lambda entry: entry["evidence_anchor_digest"])
        review_note = ""
        if row_index == 5:
            review_note = "More source inspection is required before identity review."
        elif row_index == 6:
            review_note = "Reject only this response evidence; keep the row unresolved."
        entries.append(
            {
                "review_item_id": item.review_item_id,
                "review_item_digest": item.review_item_digest,
                "item_kind": item.item_kind.value,
                "decision": decision_by_row[row_index],
                "anchor_assessments": anchor_assessments,
                "candidate_source_match": (
                    "matches_source" if row_index == 0 else "not_applicable"
                ),
                "reviewed_collision_finding_digests": [
                    finding.finding_digest
                    for finding in item.related_collision_findings
                ],
                "review_note": review_note,
            }
        )
    return {
        "schema_version": "oled_supplementary_material_identity_decision_manifest.v1",
        "run_id": packet.run_id,
        "paper_id": packet.paper_id,
        "review_packet_sha256": _sha256_file(packet_path),
        "review_packet_digest": packet.review_packet_digest,
        "review_source_pdf_evidence_digest": (
            packet.review_source_pdf_evidence_digest
        ),
        "reviewed_by": "human_reviewer",
        "reviewed_at": _REVIEWED_AT,
        "adjudication_confirmed": True,
        "decisions": entries,
    }


def _write_decisions(
    context: dict[str, Any],
    *,
    mutate: Callable[[dict[str, Any]], None] | None = None,
    filename: str = "material-identity-decisions.json",
) -> tuple[Path, dict[str, Any]]:
    payload = _decision_payload(context["packet"], context["packet_path"])
    if mutate is not None:
        mutate(payload)
    decision_path = context["review_dir"] / filename
    write_json(decision_path, payload)
    return decision_path, payload


def _write_collision_decisions(
    context: dict[str, Any],
    *,
    mutate: Callable[[dict[str, Any]], None] | None = None,
    filename: str = "material-identity-collision-decisions.json",
) -> tuple[Path, dict[str, Any]]:
    payload = _decision_payload(context["packet"], context["packet_path"])
    row_one_item = next(
        item
        for item in context["packet"].review_items
        if item.validated_result.bound_identity_group.row_index == 1
    )
    row_one_decision = next(
        entry
        for entry in payload["decisions"]
        if entry["review_item_id"] == row_one_item.review_item_id
    )
    row_one_decision["decision"] = "accept_structure_candidate"
    row_one_decision["candidate_source_match"] = "matches_source"
    if mutate is not None:
        mutate(payload)
    decision_path = context["review_dir"] / filename
    write_json(decision_path, payload)
    return decision_path, payload


def test_paper016_packet_renders_exact_pages_and_binds_candidate_depiction(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    packet, context = _build_packet(tmp_path, monkeypatch)

    assert packet.identity_group_count == packet.review_item_count == 7
    assert packet.identity_dependent_cell_count == 35
    assert packet.bounded_transcription_validated_cell_count == 49
    assert packet.upstream_ontology_review_pending_cell_count == 14
    assert packet.device_only_cell_count == 0
    assert packet.structure_candidate_count == 1
    assert packet.structure_anchor_only_count == 3
    assert packet.source_check_count == 1
    assert packet.ambiguous_identity_count == 1
    assert packet.exclusion_proposal_count == 1
    assert packet.cited_source_page_count == 2
    assert {
        asset.pdf_page_number_one_based
        for asset in packet.review_source_pdf_evidence.page_assets
    } == {27, 38}
    assert all(
        item.row_context_page.pdf_page_number_one_based == 38
        for item in packet.review_items
    )
    assert {
        reference.pdf_page_number_one_based
        for item in packet.review_items
        for reference in item.identity_evidence_pages
    } == {27}
    candidate_item = next(
        item
        for item in packet.review_items
        if item.candidate_depiction_asset is not None
    )
    depiction = candidate_item.candidate_depiction_asset
    chemistry = candidate_item.validated_result.chemistry_validation
    assert depiction is not None and chemistry is not None
    assert depiction.validated_result_id == candidate_item.validated_result.validated_result_id
    assert depiction.candidate_digest == chemistry.candidate_digest
    depiction_path = context["asset_dir"] / depiction.asset_filename
    assert depiction_path.is_file()
    assert _sha256_file(depiction_path) == depiction.rendered_asset_sha256
    assert OledSupplementaryMaterialIdentityReviewPacket.model_validate_json(
        context["packet_path"].read_text(encoding="utf-8")
    ) == packet


def test_source_reported_inchi_candidate_has_exact_bound_depiction(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    packet, context = _build_packet(
        tmp_path,
        monkeypatch,
        result_factory=_inchi_candidate_result,
    )
    candidate_item = next(
        item
        for item in packet.review_items
        if item.candidate_depiction_asset is not None
    )
    candidate = candidate_item.validated_result.response_result.structure_candidate
    depiction = candidate_item.candidate_depiction_asset
    assert candidate.structure_encoding_kind.value == "inchi"
    assert candidate.structure_candidate_text.startswith("InChI=1S/")
    assert depiction is not None
    assert depiction.candidate_digest == (
        candidate_item.validated_result.chemistry_validation.candidate_digest
    )
    depiction_path = context["asset_dir"] / depiction.asset_filename
    assert _sha256_file(depiction_path) == depiction.rendered_asset_sha256


def test_packet_render_decision_adjudication_chain_preserves_decision_truths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    packet, context = _build_packet(tmp_path, monkeypatch)
    markdown_path = context["review_dir"] / "material-identity-review.md"
    render_oled_supplementary_material_identity_review_packet_from_files(
        review_packet_json=context["packet_path"],
        asset_dir=context["asset_dir"],
        output_markdown=markdown_path,
    )
    markdown = markdown_path.read_text(encoding="utf-8")
    first_context = markdown.index("### Row context (not identity evidence)")
    first_evidence = markdown.index("### Identity evidence", first_context)
    first_candidate = markdown.index(
        "### Untrusted candidate depiction - not source evidence",
        first_evidence,
    )
    assert first_context < first_evidence < first_candidate
    assert "non-authoritative review projections" in markdown
    assert "bound PDF remains authoritative" in markdown
    assert (
        f"- Source ID: `{packet.review_source_pdf_evidence.source_id}`" in markdown
    )
    assert "| Allowed outcomes |" in markdown
    assert "  - source page: [PDF page 27](#pdf-page-27)" in markdown
    assert "  - anchor kind: `figure`" in markdown
    assert "<script" not in markdown.lower()
    image_urls = re.findall(r"!\[[^\]]*\]\(([^)]+)\)", markdown)
    assert image_urls
    assert all(
        value.startswith("assets/")
        and "://" not in value
        and ".." not in value
        for value in image_urls
    )

    decision_path, _ = _write_decisions(context)
    output_path = context["review_dir"] / "material-identity-adjudication.json"
    artifact = build_oled_supplementary_material_identity_adjudication_from_files(
        **_adjudication_kwargs(
            context,
            decision_path=decision_path,
            output_path=output_path,
        )
    )

    assert artifact.review_item_count == 7
    assert artifact.accepted_structure_candidate_count == 1
    assert artifact.confirmed_structure_anchor_only_count == 1
    assert artifact.source_check_pending_group_count == 2
    assert artifact.ambiguous_identity_pending_group_count == 1
    assert artifact.identity_exclusion_confirmed_group_count == 1
    assert artifact.response_evidence_rejected_group_count == 1
    assert artifact.unresolved_review_item_count == 5
    assert artifact.later_registry_review_eligible_group_count == 1
    assert artifact.later_registry_review_eligible_cell_count == 5
    assert artifact.upstream_ontology_review_pending_cell_count == 14
    assert artifact.device_only_cell_count == 0
    assert output_path.is_file()
    by_row = {
        group.review_item.validated_result.bound_identity_group.row_index: group
        for group in artifact.adjudicated_groups
    }
    assert by_row[0].paper_local_structure_candidate_accepted
    assert by_row[0].source_to_candidate_match_human_validated
    assert by_row[1].structure_anchor_only_confirmed
    assert by_row[2].source_check_pending
    assert by_row[3].ambiguous_identity_pending
    assert by_row[4].identity_exclusion_confirmed
    assert by_row[5].source_check_pending
    assert by_row[6].response_evidence_rejected
    assert not by_row[6].identity_exclusion_confirmed
    assert not artifact.material_identity_resolved
    assert not artifact.canonical_smiles_assigned
    assert not artifact.inchikey_assigned
    assert not artifact.automatic_candidate_merge
    assert not artifact.registry_written
    assert not artifact.gold_records_created
    assert not artifact.dataset_written


def test_collision_findings_are_acknowledged_without_automatic_merge(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    packet, context = _build_packet(
        tmp_path,
        monkeypatch,
        result_factory=_collision_result,
    )
    assert packet.collision_finding_count == 2
    assert all(not finding.automatic_merge_performed for finding in packet.collision_findings)
    collision_items = [
        item for item in packet.review_items if item.related_collision_findings
    ]
    assert len(collision_items) == 2
    assert all(len(item.related_collision_findings) == 2 for item in collision_items)
    decision_path, _ = _write_collision_decisions(context)
    output_path = context["review_dir"] / "collision-adjudication.json"

    artifact = build_oled_supplementary_material_identity_adjudication_from_files(
        **_adjudication_kwargs(
            context,
            decision_path=decision_path,
            output_path=output_path,
        )
    )

    assert artifact.accepted_structure_candidate_count == 2
    assert artifact.later_registry_review_eligible_group_count == 2
    assert not artifact.automatic_candidate_merge
    assert not artifact.cross_paper_identity_merge
    accepted_group_ids = {
        group.review_item.validated_result.bound_identity_group.identity_group_id
        for group in artifact.adjudicated_groups
        if group.paper_local_structure_candidate_accepted
    }
    assert len(accepted_group_ids) == 2


@pytest.mark.parametrize("tamper_kind", ("missing", "unknown"))
def test_collision_review_coverage_tamper_fails_without_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    tamper_kind: str,
) -> None:
    _, context = _build_packet(
        tmp_path,
        monkeypatch,
        result_factory=_collision_result,
    )

    def mutate(payload: dict[str, Any]) -> None:
        entry = next(
            item
            for item in payload["decisions"]
            if item["reviewed_collision_finding_digests"]
        )
        if tamper_kind == "missing":
            entry["reviewed_collision_finding_digests"].pop()
        else:
            entry["reviewed_collision_finding_digests"] = sorted(
                [
                    *entry["reviewed_collision_finding_digests"],
                    "sha256:" + "f" * 64,
                ]
            )

    decision_path, _ = _write_collision_decisions(
        context,
        mutate=mutate,
        filename=f"collision-{tamper_kind}-decisions.json",
    )
    output_path = context["review_dir"] / f"collision-{tamper_kind}-must-not-exist.json"

    with pytest.raises(ValueError, match="collision-review coverage"):
        build_oled_supplementary_material_identity_adjudication_from_files(
            **_adjudication_kwargs(
                context,
                decision_path=decision_path,
                output_path=output_path,
            )
        )
    assert not output_path.exists()


def test_incompatible_decision_fails_without_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, context = _build_packet(tmp_path, monkeypatch)

    def mutate(payload: dict[str, Any]) -> None:
        anchor_only = next(
            entry
            for entry in payload["decisions"]
            if entry["decision"] == "accept_structure_anchor_only"
        )
        anchor_only["decision"] = "accept_structure_candidate"

    decision_path, _ = _write_decisions(context, mutate=mutate)
    output_path = context["review_dir"] / "must-not-exist.json"
    with pytest.raises(ValueError, match="incompatible"):
        build_oled_supplementary_material_identity_adjudication_from_files(
            **_adjudication_kwargs(
                context,
                decision_path=decision_path,
                output_path=output_path,
            )
        )
    assert not output_path.exists()


@pytest.mark.parametrize(
    "tamper_kind",
    (
        "candidate_anchor_mismatch",
        "candidate_graph_mismatch",
        "anchor_only_mismatch",
        "needs_source_check_without_note",
        "reject_without_mismatch",
    ),
)
def test_decision_evidence_compatibility_tamper_fails_without_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    tamper_kind: str,
) -> None:
    _, context = _build_packet(tmp_path, monkeypatch)

    def mutate(payload: dict[str, Any]) -> None:
        decision_name = {
            "candidate_anchor_mismatch": "accept_structure_candidate",
            "candidate_graph_mismatch": "accept_structure_candidate",
            "anchor_only_mismatch": "accept_structure_anchor_only",
            "needs_source_check_without_note": "needs_source_check",
            "reject_without_mismatch": "reject_response_evidence",
        }[tamper_kind]
        entry = next(
            item for item in payload["decisions"] if item["decision"] == decision_name
        )
        if tamper_kind in {
            "candidate_anchor_mismatch",
            "anchor_only_mismatch",
        }:
            entry["anchor_assessments"][0]["assessment"] = (
                "does_not_support_claim"
            )
            entry["anchor_assessments"][0]["review_note"] = (
                "The readable source does not support this asserted anchor."
            )
        elif tamper_kind == "candidate_graph_mismatch":
            entry["candidate_source_match"] = "does_not_match_source"
        elif tamper_kind == "needs_source_check_without_note":
            entry["review_note"] = ""
        else:
            entry["anchor_assessments"][0]["assessment"] = "supports_claim"

    decision_path, _ = _write_decisions(
        context,
        mutate=mutate,
        filename=f"{tamper_kind}-decisions.json",
    )
    output_path = context["review_dir"] / f"{tamper_kind}-must-not-exist.json"

    with pytest.raises(ValueError, match="requires|rejection"):
        build_oled_supplementary_material_identity_adjudication_from_files(
            **_adjudication_kwargs(
                context,
                decision_path=decision_path,
                output_path=output_path,
            )
        )
    assert not output_path.exists()


@pytest.mark.parametrize("failure", ("missing", "wrong_digest"))
def test_decision_coverage_or_item_digest_failure_does_not_publish(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure: str,
) -> None:
    _, context = _build_packet(tmp_path, monkeypatch)

    def mutate(payload: dict[str, Any]) -> None:
        if failure == "missing":
            payload["decisions"].pop()
        else:
            payload["decisions"][0]["review_item_digest"] = "sha256:" + "f" * 64

    decision_path, _ = _write_decisions(
        context,
        mutate=mutate,
        filename=f"{failure}-decisions.json",
    )
    output_path = context["review_dir"] / f"{failure}-must-not-exist.json"
    with pytest.raises(ValueError, match="coverage|binding"):
        build_oled_supplementary_material_identity_adjudication_from_files(
            **_adjudication_kwargs(
                context,
                decision_path=decision_path,
                output_path=output_path,
            )
        )
    assert not output_path.exists()


def test_wrong_pdf_hash_fails_before_packet_publication(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, context = _build_packet(tmp_path, monkeypatch)
    wrong_pdf = context["review_dir"] / "wrong.pdf"
    wrong_pdf.write_bytes(context["source_pdf_path"].read_bytes() + b"\n")
    second_dir = tmp_path / "wrong-pdf-review"
    second_dir.mkdir()
    output_path = second_dir / "packet.json"
    asset_dir = second_dir / "assets"

    with pytest.raises(ValueError, match="bound hash"):
        build_oled_supplementary_material_identity_review_packet_from_files(
            **_packet_kwargs(
                context,
                source_pdf_path=wrong_pdf,
                asset_dir=asset_dir,
                output_path=output_path,
            )
        )
    assert not output_path.exists()
    assert not asset_dir.exists()


@pytest.mark.parametrize("tamper_kind", ("bytes", "content"))
def test_changed_pr_l_artifact_fails_exact_replay_without_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    tamper_kind: str,
) -> None:
    _, context = _build_packet(tmp_path, monkeypatch)
    decision_path, _ = _write_decisions(context)
    response_path = context["response_artifact_path"]
    if tamper_kind == "bytes":
        response_path.write_text(
            response_path.read_text(encoding="utf-8") + "\n",
            encoding="utf-8",
        )
    else:
        payload = json.loads(response_path.read_text(encoding="utf-8"))
        payload["structure_anchor_only_count"] += 1
        write_json(response_path, payload)
    output_path = context["review_dir"] / f"{tamper_kind}-must-not-exist.json"

    with pytest.raises(ValueError, match="replay|bind|mismatch"):
        build_oled_supplementary_material_identity_adjudication_from_files(
            **_adjudication_kwargs(
                context,
                decision_path=decision_path,
                output_path=output_path,
            )
        )
    assert not output_path.exists()


@pytest.mark.parametrize(
    "context_key",
    (
        "request_path",
        "transcription_review_packet_path",
        "response_manifest_path",
    ),
)
def test_upstream_byte_change_fails_joint_replay_without_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    context_key: str,
) -> None:
    _, context = _build_packet(tmp_path, monkeypatch)
    decision_path, _ = _write_decisions(context)
    upstream_path = context[context_key]
    upstream_path.write_text(
        upstream_path.read_text(encoding="utf-8") + "\n",
        encoding="utf-8",
    )
    output_path = context["review_dir"] / f"{context_key}-must-not-exist.json"

    with pytest.raises(ValueError, match="replay|bind|mismatch"):
        build_oled_supplementary_material_identity_adjudication_from_files(
            **_adjudication_kwargs(
                context,
                decision_path=decision_path,
                output_path=output_path,
            )
        )
    assert not output_path.exists()


@pytest.mark.parametrize("tamper_kind", ("bytes", "missing", "extra", "symlink"))
def test_asset_bundle_tampering_blocks_markdown_and_publishes_nothing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    tamper_kind: str,
) -> None:
    packet, context = _build_packet(tmp_path, monkeypatch)
    first_asset = packet.review_source_pdf_evidence.page_assets[0]
    asset_path = context["asset_dir"] / first_asset.asset_filename
    if tamper_kind == "bytes":
        asset_path.write_bytes(asset_path.read_bytes() + b"tamper")
    elif tamper_kind == "missing":
        asset_path.unlink()
    elif tamper_kind == "extra":
        (context["asset_dir"] / "unexpected.png").write_bytes(_FAKE_PNG_BYTES)
    else:
        replacement = context["review_dir"] / "replacement.png"
        replacement.write_bytes(asset_path.read_bytes())
        asset_path.unlink()
        asset_path.symlink_to(replacement)
    markdown_path = context["review_dir"] / f"{tamper_kind}-must-not-exist.md"

    with pytest.raises(ValueError, match="asset"):
        render_oled_supplementary_material_identity_review_packet_from_files(
            review_packet_json=context["packet_path"],
            asset_dir=context["asset_dir"],
            output_markdown=markdown_path,
        )
    assert not markdown_path.exists()


def test_second_post_publish_asset_validation_failure_revokes_markdown(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, context = _build_packet(tmp_path, monkeypatch)
    markdown_path = context["review_dir"] / "must-be-revoked.md"
    original_validator = identity_review_runner._validate_identity_asset_bundle
    validation_count = 0

    def fail_second_post_publish_validation(*args: Any, **kwargs: Any) -> None:
        nonlocal validation_count
        validation_count += 1
        original_validator(*args, **kwargs)
        if validation_count == 3:
            raise ValueError("simulated second post-publish asset validation failure")

    monkeypatch.setattr(
        identity_review_runner,
        "_validate_identity_asset_bundle",
        fail_second_post_publish_validation,
    )

    with pytest.raises(ValueError, match="second post-publish"):
        render_oled_supplementary_material_identity_review_packet_from_files(
            review_packet_json=context["packet_path"],
            asset_dir=context["asset_dir"],
            output_markdown=markdown_path,
        )

    assert validation_count == 3
    assert not markdown_path.exists()


def _replace_directory_with_redirecting_symlink(
    directory: Path,
    *,
    displaced: Path,
    redirected: Path,
) -> None:
    redirected.mkdir()
    directory.rename(displaced)
    directory.symlink_to(redirected, target_is_directory=True)


def test_packet_parent_replacement_during_render_fails_without_redirected_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, context = _build_packet(tmp_path, monkeypatch)
    review_dir = tmp_path / "packet-race-review"
    displaced = tmp_path / "packet-race-displaced"
    redirected = tmp_path / "packet-race-redirected"
    review_dir.mkdir()
    output_path = review_dir / "packet.json"
    asset_dir = review_dir / "assets"
    original_renderer = identity_review_runner._render_review_source_pages

    def replace_parent_after_render(**kwargs: Any) -> tuple[Any, Any]:
        rendered = original_renderer(**kwargs)
        _replace_directory_with_redirecting_symlink(
            review_dir,
            displaced=displaced,
            redirected=redirected,
        )
        return rendered

    monkeypatch.setattr(
        identity_review_runner,
        "_render_review_source_pages",
        replace_parent_after_render,
    )

    with pytest.raises(ValueError, match="parent changed|parent is unavailable"):
        build_oled_supplementary_material_identity_review_packet_from_files(
            **_packet_kwargs(
                context,
                asset_dir=asset_dir,
                output_path=output_path,
            )
        )

    assert not (redirected / output_path.name).exists()
    assert not (redirected / asset_dir.name).exists()
    assert not (displaced / output_path.name).exists()
    assert not (displaced / asset_dir.name).exists()


def test_packet_distinct_asset_parent_symlink_to_pinned_inode_fails_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, context = _build_packet(tmp_path, monkeypatch)
    output_parent = tmp_path / "packet-distinct-output"
    asset_parent = tmp_path / "packet-distinct-assets"
    displaced_asset_parent = tmp_path / "packet-distinct-assets-displaced"
    output_parent.mkdir()
    asset_parent.mkdir()
    output_path = output_parent / "packet.json"
    asset_dir = asset_parent / "assets"
    original_writer = identity_review_runner._write_identity_asset_bundle

    def replace_asset_parent_after_write(**kwargs: Any) -> None:
        original_writer(**kwargs)
        asset_parent.rename(displaced_asset_parent)
        asset_parent.symlink_to(
            displaced_asset_parent,
            target_is_directory=True,
        )

    monkeypatch.setattr(
        identity_review_runner,
        "_write_identity_asset_bundle",
        replace_asset_parent_after_write,
    )

    with pytest.raises(ValueError, match="asset directory changed|parent changed"):
        build_oled_supplementary_material_identity_review_packet_from_files(
            **_packet_kwargs(
                context,
                asset_dir=asset_dir,
                output_path=output_path,
            )
        )

    assert not output_path.exists()
    assert not (displaced_asset_parent / asset_dir.name).exists()


def test_interrupt_after_packet_publish_never_leaves_packet_without_assets(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, context = _build_packet(tmp_path, monkeypatch)
    review_dir = tmp_path / "packet-interrupt-review"
    review_dir.mkdir()
    output_path = review_dir / "packet.json"
    asset_dir = review_dir / "assets"
    original_publisher = identity_review_runner._publish_packet_text

    def interrupt_after_publish(*args: Any, **kwargs: Any) -> None:
        original_publisher(*args, **kwargs)
        raise KeyboardInterrupt("simulated post-publication interrupt")

    monkeypatch.setattr(
        identity_review_runner,
        "_publish_packet_text",
        interrupt_after_publish,
    )

    with pytest.raises(KeyboardInterrupt, match="post-publication"):
        build_oled_supplementary_material_identity_review_packet_from_files(
            **_packet_kwargs(
                context,
                asset_dir=asset_dir,
                output_path=output_path,
            )
        )

    packet = OledSupplementaryMaterialIdentityReviewPacket.model_validate_json(
        output_path.read_text(encoding="utf-8")
    )
    expected_filenames = {
        asset.asset_filename
        for asset in packet.review_source_pdf_evidence.page_assets
    } | {
        item.candidate_depiction_asset.asset_filename
        for item in packet.review_items
        if item.candidate_depiction_asset is not None
    }
    assert {path.name for path in asset_dir.iterdir()} == expected_filenames


def test_markdown_parent_replacement_during_render_fails_without_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, context = _build_packet(tmp_path, monkeypatch)
    review_dir = context["review_dir"]
    displaced = tmp_path / "markdown-race-displaced"
    redirected = tmp_path / "markdown-race-redirected"
    markdown_path = review_dir / "must-not-exist.md"
    original_renderer = (
        identity_review_runner.render_oled_supplementary_material_identity_review_markdown
    )

    def replace_parent_after_render(*args: Any, **kwargs: Any) -> str:
        markdown = original_renderer(*args, **kwargs)
        _replace_directory_with_redirecting_symlink(
            review_dir,
            displaced=displaced,
            redirected=redirected,
        )
        return markdown

    monkeypatch.setattr(
        identity_review_runner,
        "render_oled_supplementary_material_identity_review_markdown",
        replace_parent_after_render,
    )

    with pytest.raises(ValueError, match="parent changed|parent is unavailable"):
        render_oled_supplementary_material_identity_review_packet_from_files(
            review_packet_json=context["packet_path"],
            asset_dir=context["asset_dir"],
            output_markdown=markdown_path,
        )

    assert not (redirected / markdown_path.name).exists()
    assert not (displaced / markdown_path.name).exists()


def test_markdown_ancestor_symlink_to_pinned_tree_fails_without_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = tmp_path / "markdown-ancestor-workspace"
    displaced_workspace = tmp_path / "markdown-ancestor-displaced"
    workspace.mkdir()
    _, context = _build_packet(workspace, monkeypatch)
    markdown_path = context["review_dir"] / "must-not-exist.md"
    original_renderer = (
        identity_review_runner.render_oled_supplementary_material_identity_review_markdown
    )

    def replace_ancestor_after_render(*args: Any, **kwargs: Any) -> str:
        markdown = original_renderer(*args, **kwargs)
        workspace.rename(displaced_workspace)
        workspace.symlink_to(displaced_workspace, target_is_directory=True)
        return markdown

    monkeypatch.setattr(
        identity_review_runner,
        "render_oled_supplementary_material_identity_review_markdown",
        replace_ancestor_after_render,
    )

    with pytest.raises(ValueError, match="parent changed"):
        render_oled_supplementary_material_identity_review_packet_from_files(
            review_packet_json=context["packet_path"],
            asset_dir=context["asset_dir"],
            output_markdown=markdown_path,
        )

    displaced_output = (
        displaced_workspace / context["review_dir"].name / markdown_path.name
    )
    assert not displaced_output.exists()


def test_adjudication_parent_replacement_during_render_fails_without_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, context = _build_packet(tmp_path, monkeypatch)
    decision_path, _ = _write_decisions(context)
    review_dir = context["review_dir"]
    displaced = tmp_path / "adjudication-race-displaced"
    redirected = tmp_path / "adjudication-race-redirected"
    output_path = review_dir / "must-not-exist.json"
    original_renderer = identity_review_runner._render_review_source_pages

    def replace_parent_after_render(**kwargs: Any) -> tuple[Any, Any]:
        rendered = original_renderer(**kwargs)
        _replace_directory_with_redirecting_symlink(
            review_dir,
            displaced=displaced,
            redirected=redirected,
        )
        return rendered

    monkeypatch.setattr(
        identity_review_runner,
        "_render_review_source_pages",
        replace_parent_after_render,
    )

    with pytest.raises(ValueError, match="asset directory is invalid|parent changed"):
        build_oled_supplementary_material_identity_adjudication_from_files(
            **_adjudication_kwargs(
                context,
                decision_path=decision_path,
                output_path=output_path,
            )
        )

    assert not (redirected / output_path.name).exists()
    assert not (displaced / output_path.name).exists()


def test_adjudication_ancestor_symlink_to_pinned_tree_fails_without_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = tmp_path / "adjudication-ancestor-workspace"
    displaced_workspace = tmp_path / "adjudication-ancestor-displaced"
    workspace.mkdir()
    _, context = _build_packet(workspace, monkeypatch)
    decision_path, _ = _write_decisions(context)
    output_path = context["review_dir"] / "must-not-exist.json"
    original_renderer = identity_review_runner._render_review_source_pages

    def replace_ancestor_after_render(**kwargs: Any) -> tuple[Any, Any]:
        rendered = original_renderer(**kwargs)
        workspace.rename(displaced_workspace)
        workspace.symlink_to(displaced_workspace, target_is_directory=True)
        return rendered

    monkeypatch.setattr(
        identity_review_runner,
        "_render_review_source_pages",
        replace_ancestor_after_render,
    )

    with pytest.raises(ValueError, match="asset parent changed|parent changed"):
        build_oled_supplementary_material_identity_adjudication_from_files(
            **_adjudication_kwargs(
                context,
                decision_path=decision_path,
                output_path=output_path,
            )
        )

    displaced_output = (
        displaced_workspace / context["review_dir"].name / output_path.name
    )
    assert not displaced_output.exists()


def test_candidate_depiction_bytes_are_exactly_bound_at_adjudication(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    packet, context = _build_packet(tmp_path, monkeypatch)
    decision_path, _ = _write_decisions(context)
    depiction = next(
        item.candidate_depiction_asset
        for item in packet.review_items
        if item.candidate_depiction_asset is not None
    )
    assert depiction is not None
    depiction_path = context["asset_dir"] / depiction.asset_filename
    depiction_path.write_bytes(_FAKE_PNG_BYTES)
    output_path = context["review_dir"] / "depiction-must-not-exist.json"

    with pytest.raises(ValueError, match="asset|depiction"):
        build_oled_supplementary_material_identity_adjudication_from_files(
            **_adjudication_kwargs(
                context,
                decision_path=decision_path,
                output_path=output_path,
            )
        )
    assert not output_path.exists()


def test_source_pdf_symlinked_ancestor_is_rejected_without_publication(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, context = _build_packet(tmp_path, monkeypatch)
    alias_parent = context["review_dir"] / "source-parent-alias"
    alias_parent.symlink_to(
        context["source_pdf_path"].parent,
        target_is_directory=True,
    )
    aliased_pdf = alias_parent / context["source_pdf_path"].name
    second_dir = tmp_path / "symlinked-pdf-review"
    second_dir.mkdir()
    output_path = second_dir / "packet.json"
    asset_dir = second_dir / "assets"

    with pytest.raises((ValueError, OSError)):
        build_oled_supplementary_material_identity_review_packet_from_files(
            **_packet_kwargs(
                context,
                source_pdf_path=aliased_pdf,
                asset_dir=asset_dir,
                output_path=output_path,
            )
        )
    assert not output_path.exists()
    assert not asset_dir.exists()


def test_packet_output_cannot_overwrite_any_json_or_source_pdf(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, context = _build_packet(tmp_path, monkeypatch)
    protected = [
        context["request_path"],
        context["transcription_review_packet_path"],
        context["response_manifest_path"],
        context["response_artifact_path"],
        context["source_pdf_path"],
    ]
    original_bytes = {path: path.read_bytes() for path in protected}

    for index, output_path in enumerate(protected):
        attempt_dir = tmp_path / f"collision-attempt-{index}"
        attempt_dir.mkdir()
        asset_dir = attempt_dir / "assets"
        with pytest.raises(ValueError):
            build_oled_supplementary_material_identity_review_packet_from_files(
                **_packet_kwargs(
                    context,
                    asset_dir=asset_dir,
                    output_path=output_path,
                )
            )
        assert output_path.read_bytes() == original_bytes[output_path]
        assert not asset_dir.exists()


def test_adjudication_output_inside_asset_directory_is_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, context = _build_packet(tmp_path, monkeypatch)
    decision_path, _ = _write_decisions(context)
    output_path = context["asset_dir"] / "must-not-exist.json"

    with pytest.raises(ValueError, match="outside the asset directory"):
        build_oled_supplementary_material_identity_adjudication_from_files(
            **_adjudication_kwargs(
                context,
                decision_path=decision_path,
                output_path=output_path,
            )
        )
    assert not output_path.exists()


@pytest.mark.parametrize(
    ("target", "unsafe_text"),
    (
        ("reviewed_by", "token=abc123"),
        ("review_note", "secret=abc123"),
        ("review_note", "&lt;script&gt;alert(1)&lt;/script&gt;"),
        ("reviewed_by", "reviewer\u202Ehidden"),
    ),
    ids=("token", "secret", "encoded-html", "display-control"),
)
def test_unsafe_reviewer_text_is_rejected_without_adjudication_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    target: str,
    unsafe_text: str,
) -> None:
    _, context = _build_packet(tmp_path, monkeypatch)

    def mutate(payload: dict[str, Any]) -> None:
        if target == "reviewed_by":
            payload["reviewed_by"] = unsafe_text
        else:
            payload["decisions"][0]["review_note"] = unsafe_text

    decision_path, _ = _write_decisions(
        context,
        mutate=mutate,
        filename=f"unsafe-{target}.json",
    )
    output_path = context["review_dir"] / "unsafe-must-not-exist.json"

    with pytest.raises(ValueError):
        build_oled_supplementary_material_identity_adjudication_from_files(
            **_adjudication_kwargs(
                context,
                decision_path=decision_path,
                output_path=output_path,
            )
        )
    assert not output_path.exists()


def test_packet_whitespace_change_invalidates_preexisting_decision_binding(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, context = _build_packet(tmp_path, monkeypatch)
    decision_path, _ = _write_decisions(context)
    context["packet_path"].write_text(
        context["packet_path"].read_text(encoding="utf-8") + "\n",
        encoding="utf-8",
    )
    output_path = context["review_dir"] / "whitespace-must-not-exist.json"

    with pytest.raises(ValueError, match="packet binding"):
        build_oled_supplementary_material_identity_adjudication_from_files(
            **_adjudication_kwargs(
                context,
                decision_path=decision_path,
                output_path=output_path,
            )
        )
    assert not output_path.exists()


def test_self_consistent_swapped_candidate_depictions_fail_exact_packet_replay(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    packet, context = _build_packet(
        tmp_path,
        monkeypatch,
        result_factory=_two_candidate_result,
    )
    payload = packet.model_dump(mode="json")
    candidate_indexes = [
        index
        for index, item in enumerate(payload["review_items"])
        if item["candidate_depiction_asset"] is not None
    ]
    assert len(candidate_indexes) == 2
    first_index, second_index = candidate_indexes
    first_item = payload["review_items"][first_index]
    second_item = payload["review_items"][second_index]
    first_depiction = first_item["candidate_depiction_asset"]
    second_depiction = second_item["candidate_depiction_asset"]
    assert first_depiction["rendered_asset_sha256"] != second_depiction[
        "rendered_asset_sha256"
    ]

    first_forged = build_oled_supplementary_material_identity_candidate_depiction_asset(
        validated_result_id=first_depiction["validated_result_id"],
        candidate_digest=first_depiction["candidate_digest"],
        toolkit_version=first_depiction["toolkit_version"],
        rendered_asset_sha256=second_depiction["rendered_asset_sha256"],
        rendered_asset_byte_size=second_depiction["rendered_asset_byte_size"],
        pixel_width=second_depiction["pixel_width"],
        pixel_height=second_depiction["pixel_height"],
    )
    second_forged = build_oled_supplementary_material_identity_candidate_depiction_asset(
        validated_result_id=second_depiction["validated_result_id"],
        candidate_digest=second_depiction["candidate_digest"],
        toolkit_version=second_depiction["toolkit_version"],
        rendered_asset_sha256=first_depiction["rendered_asset_sha256"],
        rendered_asset_byte_size=first_depiction["rendered_asset_byte_size"],
        pixel_width=first_depiction["pixel_width"],
        pixel_height=first_depiction["pixel_height"],
    )
    first_item["candidate_depiction_asset"] = first_forged.model_dump(mode="json")
    second_item["candidate_depiction_asset"] = second_forged.model_dump(mode="json")
    for item in (first_item, second_item):
        item["review_item_digest"] = _stable_hash(
            {
                key: value
                for key, value in item.items()
                if key != "review_item_digest"
            }
        )
    payload["review_packet_digest"] = _stable_hash(
        {
            key: value
            for key, value in payload.items()
            if key != "review_packet_digest"
        }
    )
    forged_packet = OledSupplementaryMaterialIdentityReviewPacket.model_validate(
        payload
    )
    forged_path = context["review_dir"] / "forged-swapped-packet.json"
    write_json(forged_path, forged_packet.model_dump(mode="json"))
    forged_context = {
        **context,
        "packet": forged_packet,
        "packet_path": forged_path,
    }
    decision_path, _ = _write_decisions(
        forged_context,
        filename="forged-swapped-decisions.json",
    )
    output_path = context["review_dir"] / "forged-must-not-exist.json"

    with pytest.raises(ValueError, match="depictions changed|exact packet replay"):
        build_oled_supplementary_material_identity_adjudication_from_files(
            **_adjudication_kwargs(
                forged_context,
                decision_path=decision_path,
                output_path=output_path,
            )
        )
    assert not output_path.exists()
