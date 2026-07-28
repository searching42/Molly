from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import stat
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Sequence, TextIO

from rdkit import Chem, rdBase
from rdkit.Chem import rdDepictor
from rdkit.Chem import rdinchi as rd_inchi
from rdkit.Chem.Draw import rdMolDraw2D

from ai4s_agent._utils import now_iso
from ai4s_agent.domains.oled_supplementary_material_identity_candidate_request import (
    OledSupplementaryMaterialIdentityCandidateRequestArtifact,
)
from ai4s_agent.domains.oled_supplementary_material_identity_evidence_response import (
    OledSupplementaryMaterialIdentityEvidenceResponseArtifact,
    OledSupplementaryMaterialIdentityEvidenceResponseManifest,
    OledSupplementaryMaterialIdentityProposeStructureCandidate,
    OledSupplementaryMaterialIdentityStructureEncodingKind,
)
from ai4s_agent.domains.oled_supplementary_material_identity_review import (
    SUPPLEMENTARY_MATERIAL_IDENTITY_DEPICTION_PROFILE,
    OledSupplementaryMaterialIdentityAdjudicationArtifact,
    OledSupplementaryMaterialIdentityCandidateDepictionAsset,
    OledSupplementaryMaterialIdentityDecisionManifest,
    OledSupplementaryMaterialIdentityReviewPacket,
    build_oled_supplementary_material_identity_adjudication_artifact,
    build_oled_supplementary_material_identity_candidate_depiction_asset,
    build_oled_supplementary_material_identity_review_packet,
    render_oled_supplementary_material_identity_review_markdown,
)
from ai4s_agent.domains.oled_supplementary_source_transcription_review import (
    OledSupplementarySourcePageAsset,
    OledSupplementarySourceTranscriptionReviewPacket,
)
from ai4s_agent.oled_supplementary_scoped_candidate_response import (
    _absolute_local_path,
    _open_or_create_output_parent_without_symlinks,
    _read_bound_json,
    _validate_fresh_output,
)
from ai4s_agent.oled_supplementary_source_transcription_review import (
    _create_private_asset_directory,
    _path_is_within,
    _png_dimensions,
    _publish_packet_text,
    _read_bound_binary_at,
    _remove_private_asset_directory,
    _render_exact_bound_source_pdf_pages,
    _validate_open_asset_directory_binding,
    _validate_packet_outputs,
    _validate_pinned_directory_path_without_symlinks,
    _validate_published_packet_paths,
    _validate_review_asset_location,
    _write_fresh_bytes_at,
)


_MAX_REQUEST_BYTES = 100 * 1024 * 1024
_MAX_TRANSCRIPTION_PACKET_BYTES = 200 * 1024 * 1024
_MAX_RESPONSE_MANIFEST_BYTES = 100 * 1024 * 1024
_MAX_RESPONSE_ARTIFACT_BYTES = 250 * 1024 * 1024
_MAX_REVIEW_PACKET_BYTES = 300 * 1024 * 1024
_MAX_DECISION_MANIFEST_BYTES = 50 * 1024 * 1024
_MAX_ASSET_BYTES = 100 * 1024 * 1024
_DEPICTION_DIMENSIONS_RE = re.compile(r"_([1-9][0-9]*)x([1-9][0-9]*)_rgb_png\.")


def build_oled_supplementary_material_identity_review_packet_from_files(
    *,
    request_artifact_json: str | Path,
    transcription_review_packet_json: str | Path,
    response_manifest_json: str | Path,
    response_artifact_json: str | Path,
    source_pdf_path: str | Path,
    asset_dir: str | Path,
    output_json: str | Path,
    poppler_bin_dir: str | Path | None = None,
    generated_at: str | None = None,
) -> OledSupplementaryMaterialIdentityReviewPacket:
    paths = _upstream_paths(
        request_artifact_json=request_artifact_json,
        transcription_review_packet_json=transcription_review_packet_json,
        response_manifest_json=response_manifest_json,
        response_artifact_json=response_artifact_json,
    )
    upstream = _load_upstream(paths)
    source_path = _absolute_local_path(source_pdf_path)
    assets_path = _absolute_local_path(asset_dir)
    output_path = _absolute_local_path(output_json)
    protected_paths = {*paths.values(), source_path}
    with _pinned_output_parents_without_symlink_components(
        output_path.parent,
        assets_path.parent,
    ) as pinned:
        _validate_packet_outputs(
            output_path=output_path,
            asset_dir=assets_path,
            protected_paths=protected_paths,
        )

        source_evidence, page_bytes = _render_review_source_pages(
            upstream=upstream,
            source_pdf_path=source_path,
            poppler_bin_dir=poppler_bin_dir,
        )
        depiction_assets, depiction_bytes = _render_candidate_depictions(
            upstream["response_artifact"]
        )
        packet = build_oled_supplementary_material_identity_review_packet(
            request_artifact=upstream["request"],
            request_artifact_sha256=upstream["request_sha256"],
            transcription_review_packet=upstream["transcription_packet"],
            transcription_review_packet_sha256=upstream[
                "transcription_packet_sha256"
            ],
            response_manifest=upstream["response_manifest"],
            response_manifest_sha256=upstream["response_manifest_sha256"],
            response_artifact=upstream["response_artifact"],
            response_artifact_sha256=upstream["response_artifact_sha256"],
            review_source_pdf_evidence=source_evidence,
            candidate_depiction_assets=depiction_assets,
            generated_at=generated_at or now_iso(),
        )
        rendered_assets = _merge_rendered_assets(page_bytes, depiction_bytes)
        _publish_packet_and_assets(
            packet=packet,
            rendered_assets=rendered_assets,
            output_path=output_path,
            asset_dir=assets_path,
            protected_paths=protected_paths,
            output_parent_descriptor=pinned[output_path.parent],
            asset_parent_descriptor=pinned[assets_path.parent],
        )
        return packet


def render_oled_supplementary_material_identity_review_packet_from_files(
    *,
    review_packet_json: str | Path,
    asset_dir: str | Path,
    output_markdown: str | Path,
) -> OledSupplementaryMaterialIdentityReviewPacket:
    packet_path = _absolute_local_path(review_packet_json)
    assets_path = _absolute_local_path(asset_dir)
    output_path = _absolute_local_path(output_markdown)
    payload, packet_sha256 = _read_bound_json(
        packet_path,
        "supplementary material identity review packet",
        max_bytes=_MAX_REVIEW_PACKET_BYTES,
        reject_symlink_components=True,
    )
    packet = OledSupplementaryMaterialIdentityReviewPacket.model_validate(payload)
    with _pinned_output_parents_without_symlink_components(
        output_path.parent,
        assets_path.parent,
    ) as pinned:
        output_parent_descriptor = pinned[output_path.parent]
        asset_parent_descriptor = pinned[assets_path.parent]
        _validate_review_asset_location(
            asset_dir=assets_path,
            output_markdown=output_path,
        )
        _validate_identity_asset_bundle(
            assets_path,
            packet,
            pinned_parent_descriptor=asset_parent_descriptor,
        )
        protected_paths = {
            packet_path,
            *(
                _absolute_local_path(assets_path / filename)
                for filename in _expected_asset_metadata(packet)
            ),
        }
        _validate_fresh_output(output_path, protected_paths=protected_paths)
        if _path_is_within(output_path, assets_path):
            raise ValueError(
                "material identity Markdown must be outside the asset directory"
            )
        markdown = render_oled_supplementary_material_identity_review_markdown(
            packet,
            review_packet_sha256=packet_sha256,
        )

        def validate_markdown_publication() -> None:
            _validate_identity_asset_bundle(
                assets_path,
                packet,
                pinned_parent_descriptor=asset_parent_descriptor,
            )
            _validate_published_packet_paths(
                output_path=output_path,
                asset_dir=assets_path,
                protected_paths=protected_paths,
            )

        _publish_packet_text(
            output_path,
            markdown,
            post_publish_validator=validate_markdown_publication,
            pinned_parent_descriptor=output_parent_descriptor,
        )
        return packet


def build_oled_supplementary_material_identity_adjudication_from_files(
    *,
    request_artifact_json: str | Path,
    transcription_review_packet_json: str | Path,
    response_manifest_json: str | Path,
    response_artifact_json: str | Path,
    source_pdf_path: str | Path,
    review_packet_json: str | Path,
    decision_manifest_json: str | Path,
    asset_dir: str | Path,
    output_json: str | Path,
    poppler_bin_dir: str | Path | None = None,
    generated_at: str | None = None,
) -> OledSupplementaryMaterialIdentityAdjudicationArtifact:
    paths = _upstream_paths(
        request_artifact_json=request_artifact_json,
        transcription_review_packet_json=transcription_review_packet_json,
        response_manifest_json=response_manifest_json,
        response_artifact_json=response_artifact_json,
    )
    upstream = _load_upstream(paths)
    source_path = _absolute_local_path(source_pdf_path)
    packet_path = _absolute_local_path(review_packet_json)
    decision_path = _absolute_local_path(decision_manifest_json)
    assets_path = _absolute_local_path(asset_dir)
    output_path = _absolute_local_path(output_json)
    packet_payload, packet_sha256 = _read_bound_json(
        packet_path,
        "supplementary material identity review packet",
        max_bytes=_MAX_REVIEW_PACKET_BYTES,
        reject_symlink_components=True,
    )
    decision_payload, decision_sha256 = _read_bound_json(
        decision_path,
        "supplementary material identity decision manifest",
        max_bytes=_MAX_DECISION_MANIFEST_BYTES,
        reject_symlink_components=True,
    )
    packet = OledSupplementaryMaterialIdentityReviewPacket.model_validate(
        packet_payload
    )
    decisions = OledSupplementaryMaterialIdentityDecisionManifest.model_validate(
        decision_payload
    )
    protected_paths = {
        *paths.values(),
        source_path,
        packet_path,
        decision_path,
        *(_absolute_local_path(assets_path / filename)
          for filename in _expected_asset_metadata(packet)),
    }
    with _pinned_output_parents_without_symlink_components(
        output_path.parent,
        assets_path.parent,
    ) as pinned:
        output_parent_descriptor = pinned[output_path.parent]
        asset_parent_descriptor = pinned[assets_path.parent]
        _validate_fresh_output(output_path, protected_paths=protected_paths)
        if _path_is_within(output_path, assets_path):
            raise ValueError(
                "material identity adjudication must be outside the asset directory"
            )

        source_evidence, page_bytes = _render_review_source_pages(
            upstream=upstream,
            source_pdf_path=source_path,
            poppler_bin_dir=poppler_bin_dir,
        )
        depiction_assets, depiction_bytes = _render_candidate_depictions(
            upstream["response_artifact"]
        )
        rendered_assets = _merge_rendered_assets(page_bytes, depiction_bytes)
        if source_evidence.model_dump(mode="json") != (
            packet.review_source_pdf_evidence.model_dump(mode="json")
        ):
            raise ValueError("material identity source PDF evidence changed")
        expected_depictions = _packet_depiction_assets(packet)
        if [item.model_dump(mode="json") for item in depiction_assets] != [
            item.model_dump(mode="json") for item in expected_depictions
        ]:
            raise ValueError("material identity candidate depictions changed")
        _validate_identity_asset_bundle(
            assets_path,
            packet,
            expected_bytes=rendered_assets,
            pinned_parent_descriptor=asset_parent_descriptor,
        )
        artifact = build_oled_supplementary_material_identity_adjudication_artifact(
            request_artifact=upstream["request"],
            request_artifact_sha256=upstream["request_sha256"],
            transcription_review_packet=upstream["transcription_packet"],
            transcription_review_packet_sha256=upstream[
                "transcription_packet_sha256"
            ],
            response_manifest=upstream["response_manifest"],
            response_manifest_sha256=upstream["response_manifest_sha256"],
            response_artifact=upstream["response_artifact"],
            response_artifact_sha256=upstream["response_artifact_sha256"],
            review_source_pdf_evidence=source_evidence,
            candidate_depiction_assets=depiction_assets,
            review_packet=packet,
            review_packet_sha256=packet_sha256,
            decision_manifest=decisions,
            decision_manifest_sha256=decision_sha256,
            generated_at=generated_at or now_iso(),
        )

        def validate_adjudication_publication() -> None:
            _validate_identity_asset_bundle(
                assets_path,
                packet,
                expected_bytes=rendered_assets,
                pinned_parent_descriptor=asset_parent_descriptor,
            )
            _validate_published_packet_paths(
                output_path=output_path,
                asset_dir=assets_path,
                protected_paths=protected_paths,
            )

        _publish_packet_text(
            output_path,
            json.dumps(
                artifact.model_dump(mode="json"),
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            post_publish_validator=validate_adjudication_publication,
            pinned_parent_descriptor=output_parent_descriptor,
        )
        return artifact


def _upstream_paths(
    *,
    request_artifact_json: str | Path,
    transcription_review_packet_json: str | Path,
    response_manifest_json: str | Path,
    response_artifact_json: str | Path,
) -> dict[str, Path]:
    paths = {
        "request": _absolute_local_path(request_artifact_json),
        "transcription_packet": _absolute_local_path(
            transcription_review_packet_json
        ),
        "response_manifest": _absolute_local_path(response_manifest_json),
        "response_artifact": _absolute_local_path(response_artifact_json),
    }
    if len(set(paths.values())) != len(paths):
        raise ValueError("material identity review inputs must be distinct files")
    return paths


@contextmanager
def _pinned_output_parents_without_symlink_components(
    *parents: Path,
) -> Iterator[dict[Path, int]]:
    no_follow = getattr(os, "O_NOFOLLOW", None)
    directory_flag = getattr(os, "O_DIRECTORY", None)
    if no_follow is None or directory_flag is None:
        raise ValueError("material identity review requires safe dirfd support")
    pinned: dict[Path, int] = {}
    try:
        for parent in dict.fromkeys(parents):
            descriptor = _open_or_create_output_parent_without_symlinks(
                parent,
                no_follow=no_follow,
                directory_flag=directory_flag,
            )
            _validate_pinned_directory_path_without_symlinks(
                parent,
                descriptor,
                error_message="material identity output parent changed",
            )
            parent_stat = os.fstat(descriptor)
            current_stat = os.stat(parent, follow_symlinks=False)
            if (
                not stat.S_ISDIR(parent_stat.st_mode)
                or not stat.S_ISDIR(current_stat.st_mode)
                or current_stat.st_dev != parent_stat.st_dev
                or current_stat.st_ino != parent_stat.st_ino
            ):
                os.close(descriptor)
                raise ValueError("material identity output parent changed")
            pinned[parent] = descriptor
        yield pinned
    except ValueError:
        raise
    except OSError as exc:
        raise ValueError("material identity output parent is unavailable") from exc
    finally:
        for descriptor in pinned.values():
            os.close(descriptor)


def _load_upstream(paths: dict[str, Path]) -> dict[str, Any]:
    request_payload, request_sha256 = _read_bound_json(
        paths["request"],
        "supplementary material identity candidate request",
        max_bytes=_MAX_REQUEST_BYTES,
        reject_symlink_components=True,
    )
    transcription_payload, transcription_sha256 = _read_bound_json(
        paths["transcription_packet"],
        "supplementary source transcription review packet",
        max_bytes=_MAX_TRANSCRIPTION_PACKET_BYTES,
        reject_symlink_components=True,
    )
    manifest_payload, manifest_sha256 = _read_bound_json(
        paths["response_manifest"],
        "supplementary material identity evidence response manifest",
        max_bytes=_MAX_RESPONSE_MANIFEST_BYTES,
        reject_symlink_components=True,
    )
    response_payload, response_sha256 = _read_bound_json(
        paths["response_artifact"],
        "supplementary material identity evidence response artifact",
        max_bytes=_MAX_RESPONSE_ARTIFACT_BYTES,
        reject_symlink_components=True,
    )
    return {
        "request": OledSupplementaryMaterialIdentityCandidateRequestArtifact.model_validate(
            request_payload
        ),
        "request_sha256": request_sha256,
        "transcription_packet": OledSupplementarySourceTranscriptionReviewPacket.model_validate(
            transcription_payload
        ),
        "transcription_packet_sha256": transcription_sha256,
        "response_manifest": (
            OledSupplementaryMaterialIdentityEvidenceResponseManifest.model_validate(
                manifest_payload
            )
        ),
        "response_manifest_sha256": manifest_sha256,
        "response_artifact": (
            OledSupplementaryMaterialIdentityEvidenceResponseArtifact.model_validate(
                response_payload
            )
        ),
        "response_artifact_sha256": response_sha256,
    }


def _review_page_numbers(
    response: OledSupplementaryMaterialIdentityEvidenceResponseArtifact,
) -> list[int]:
    pages = {
        result.bound_identity_group.pdf_page_number_one_based
        for result in response.validated_results
    }
    for result in response.validated_results:
        pages.update(
            anchor.pdf_page_number_one_based
            for anchor in getattr(result.response_result, "evidence_anchors", [])
        )
    if not pages:
        raise ValueError("material identity review requires at least one source page")
    return sorted(pages)


def _render_review_source_pages(
    *,
    upstream: dict[str, Any],
    source_pdf_path: Path,
    poppler_bin_dir: str | Path | None,
) -> tuple[Any, dict[str, bytes]]:
    transcription = upstream["transcription_packet"]
    response = upstream["response_artifact"]
    source = transcription.source_pdf_evidence
    evidence, rendered = _render_exact_bound_source_pdf_pages(
        source_pdf_path=source_pdf_path,
        source_id=source.source_id,
        expected_sha256=source.source_pdf_sha256,
        pages=_review_page_numbers(response),
        poppler_bin_dir=poppler_bin_dir,
        reject_symlink_components=True,
    )
    if (
        evidence.source_pdf_byte_size != source.source_pdf_byte_size
        or evidence.source_pdf_page_count != source.source_pdf_page_count
    ):
        raise ValueError("material identity review PDF size or page count changed")
    return evidence, rendered


def _depiction_dimensions() -> tuple[int, int]:
    match = _DEPICTION_DIMENSIONS_RE.search(
        SUPPLEMENTARY_MATERIAL_IDENTITY_DEPICTION_PROFILE
    )
    if match is None:
        raise ValueError("unsupported material identity depiction profile")
    width, height = (int(match.group(1)), int(match.group(2)))
    if width > 20_000 or height > 20_000 or width * height > 80_000_000:
        raise ValueError("material identity depiction dimensions are unsafe")
    return width, height


def _candidate_molecule(candidate: Any) -> Any:
    parameters = Chem.SmilesParserParams()
    parameters.sanitize = True
    parameters.removeHs = True
    parameters.parseName = False
    parameters.allowCXSMILES = False
    parameters.strictCXSMILES = True
    if candidate.structure_encoding_kind == (
        OledSupplementaryMaterialIdentityStructureEncodingKind.SMILES
    ):
        molecule = Chem.MolFromSmiles(
            candidate.structure_candidate_text,
            parameters,
        )
    elif candidate.structure_encoding_kind == (
        OledSupplementaryMaterialIdentityStructureEncodingKind.INCHI
    ):
        if not candidate.structure_candidate_text.startswith("InChI=1S/"):
            raise ValueError("candidate depiction requires a standard InChI")
        molecule, return_code, _, _ = rd_inchi.InchiToMol(
            candidate.structure_candidate_text,
            True,
            True,
        )
        if return_code not in {0, 1}:
            raise ValueError("RDKit rejected the exact candidate InChI")
    else:
        raise ValueError("unsupported material identity candidate encoding")
    if molecule is None or molecule.GetNumAtoms() <= 0:
        raise ValueError("RDKit could not parse the exact material identity candidate")
    Chem.SanitizeMol(molecule)
    Chem.AssignStereochemistry(
        molecule,
        cleanIt=True,
        force=True,
        flagPossibleStereoCenters=True,
    )
    return molecule


def _render_candidate_png(candidate: Any) -> bytes:
    width, height = _depiction_dimensions()
    with rdBase.BlockLogs():
        molecule = _candidate_molecule(candidate)
        rdDepictor.Compute2DCoords(
            molecule,
            canonOrient=True,
            clearConfs=True,
        )
        drawer = rdMolDraw2D.MolDraw2DCairo(width, height)
        drawer.drawOptions().addStereoAnnotation = True
        rdMolDraw2D.PrepareAndDrawMolecule(drawer, molecule)
        drawer.FinishDrawing()
        payload = bytes(drawer.GetDrawingText())
    if _png_dimensions(payload) != (width, height):
        raise ValueError("RDKit candidate depiction has unexpected dimensions")
    return payload


def _render_candidate_depictions(
    response: OledSupplementaryMaterialIdentityEvidenceResponseArtifact,
) -> tuple[
    list[OledSupplementaryMaterialIdentityCandidateDepictionAsset],
    dict[str, bytes],
]:
    runtime_version = str(rdBase.rdkitVersion)
    if runtime_version != response.rdkit_version:
        raise ValueError("material identity depiction RDKit runtime changed")
    assets: list[OledSupplementaryMaterialIdentityCandidateDepictionAsset] = []
    rendered: dict[str, bytes] = {}
    for result in response.validated_results:
        result_response = result.response_result
        if not isinstance(
            result_response,
            OledSupplementaryMaterialIdentityProposeStructureCandidate,
        ):
            continue
        if result.chemistry_validation is None:
            raise ValueError("material identity candidate lacks chemistry validation")
        payload = _render_candidate_png(result_response.structure_candidate)
        width, height = _png_dimensions(payload)
        asset = build_oled_supplementary_material_identity_candidate_depiction_asset(
            validated_result_id=result.validated_result_id,
            candidate_digest=result.chemistry_validation.candidate_digest,
            toolkit_version=runtime_version,
            rendered_asset_sha256=(
                f"sha256:{hashlib.sha256(payload).hexdigest()}"
            ),
            rendered_asset_byte_size=len(payload),
            pixel_width=width,
            pixel_height=height,
        )
        if asset.asset_filename in rendered:
            raise ValueError("material identity candidate depiction filename repeated")
        assets.append(asset)
        rendered[asset.asset_filename] = payload
    assets.sort(key=lambda item: item.validated_result_id)
    return assets, rendered


def _merge_rendered_assets(
    page_bytes: dict[str, bytes],
    depiction_bytes: dict[str, bytes],
) -> dict[str, bytes]:
    overlap = set(page_bytes).intersection(depiction_bytes)
    if overlap:
        raise ValueError("source-page and candidate depiction filenames collide")
    return {**page_bytes, **depiction_bytes}


def _packet_depiction_assets(
    packet: OledSupplementaryMaterialIdentityReviewPacket,
) -> list[OledSupplementaryMaterialIdentityCandidateDepictionAsset]:
    return sorted(
        (
            item.candidate_depiction_asset
            for item in packet.review_items
            if item.candidate_depiction_asset is not None
        ),
        key=lambda item: item.validated_result_id,
    )


def _expected_asset_metadata(
    packet: OledSupplementaryMaterialIdentityReviewPacket,
) -> dict[
    str,
    OledSupplementarySourcePageAsset
    | OledSupplementaryMaterialIdentityCandidateDepictionAsset,
]:
    expected: dict[
        str,
        OledSupplementarySourcePageAsset
        | OledSupplementaryMaterialIdentityCandidateDepictionAsset,
    ] = {}
    all_assets: list[
        OledSupplementarySourcePageAsset
        | OledSupplementaryMaterialIdentityCandidateDepictionAsset
    ] = [
        *packet.review_source_pdf_evidence.page_assets,
        *_packet_depiction_assets(packet),
    ]
    for asset in all_assets:
        if asset.asset_filename in expected:
            raise ValueError("material identity review asset filename repeated")
        expected[asset.asset_filename] = asset
    return expected


def _validate_asset_payload(
    asset: OledSupplementarySourcePageAsset
    | OledSupplementaryMaterialIdentityCandidateDepictionAsset,
    payload: bytes,
) -> None:
    if (
        len(payload) != asset.rendered_asset_byte_size
        or f"sha256:{hashlib.sha256(payload).hexdigest()}"
        != asset.rendered_asset_sha256
        or _png_dimensions(payload) != (asset.pixel_width, asset.pixel_height)
    ):
        raise ValueError("material identity review asset bytes do not match metadata")
    if isinstance(
        asset,
        OledSupplementaryMaterialIdentityCandidateDepictionAsset,
    ) and (asset.pixel_width, asset.pixel_height) != _depiction_dimensions():
        raise ValueError("candidate depiction dimensions do not match its profile")


def _publish_packet_and_assets(
    *,
    packet: OledSupplementaryMaterialIdentityReviewPacket,
    rendered_assets: dict[str, bytes],
    output_path: Path,
    asset_dir: Path,
    protected_paths: set[Path],
    output_parent_descriptor: int,
    asset_parent_descriptor: int,
) -> None:
    parent_descriptor = -1
    asset_descriptor = -1
    asset_directory_stat: os.stat_result | None = None
    created_asset_stats: dict[str, os.stat_result] = {}
    keep_assets = False
    publication_attempted = False
    packet_text = (
        json.dumps(packet.model_dump(mode="json"), ensure_ascii=False, indent=2)
        + "\n"
    )
    try:
        (
            parent_descriptor,
            asset_descriptor,
            asset_directory_stat,
        ) = _create_private_asset_directory(
            asset_dir,
            pinned_parent_descriptor=asset_parent_descriptor,
        )
        _write_identity_asset_bundle(
            asset_descriptor=asset_descriptor,
            packet=packet,
            rendered_assets=rendered_assets,
            created_asset_stats=created_asset_stats,
        )
        _validate_open_asset_directory_binding(
            asset_dir=asset_dir,
            parent_descriptor=parent_descriptor,
            asset_descriptor=asset_descriptor,
            asset_directory_stat=asset_directory_stat,
            reject_symlink_components=True,
        )

        def validate_joint_publication() -> None:
            _validate_open_asset_directory_binding(
                asset_dir=asset_dir,
                parent_descriptor=parent_descriptor,
                asset_descriptor=asset_descriptor,
                asset_directory_stat=asset_directory_stat,
                reject_symlink_components=True,
            )
            _validate_open_created_identity_asset_bundle(
                asset_descriptor=asset_descriptor,
                packet=packet,
                rendered_assets=rendered_assets,
                created_asset_stats=created_asset_stats,
            )
            _validate_published_packet_paths(
                output_path=output_path,
                asset_dir=asset_dir,
                protected_paths=protected_paths,
            )

        publication_attempted = True
        _publish_packet_text(
            output_path,
            packet_text,
            post_publish_validator=validate_joint_publication,
            pinned_parent_descriptor=output_parent_descriptor,
        )
        keep_assets = True
    finally:
        if (
            not keep_assets
            and publication_attempted
            and asset_directory_stat is not None
            and _joint_identity_publication_is_complete(
                packet=packet,
                rendered_assets=rendered_assets,
                packet_text=packet_text,
                output_path=output_path,
                asset_dir=asset_dir,
                protected_paths=protected_paths,
                output_parent_descriptor=output_parent_descriptor,
                asset_parent_descriptor=parent_descriptor,
                asset_descriptor=asset_descriptor,
                asset_directory_stat=asset_directory_stat,
                created_asset_stats=created_asset_stats,
            )
        ):
            keep_assets = True
        if not keep_assets and asset_directory_stat is not None:
            _remove_private_asset_directory(
                asset_dir=asset_dir,
                parent_descriptor=parent_descriptor,
                asset_descriptor=asset_descriptor,
                asset_directory_stat=asset_directory_stat,
                created_asset_stats=created_asset_stats,
            )
        if asset_descriptor != -1:
            os.close(asset_descriptor)
        if parent_descriptor != -1:
            os.close(parent_descriptor)


def _joint_identity_publication_is_complete(
    *,
    packet: OledSupplementaryMaterialIdentityReviewPacket,
    rendered_assets: dict[str, bytes],
    packet_text: str,
    output_path: Path,
    asset_dir: Path,
    protected_paths: set[Path],
    output_parent_descriptor: int,
    asset_parent_descriptor: int,
    asset_descriptor: int,
    asset_directory_stat: os.stat_result,
    created_asset_stats: dict[str, os.stat_result],
) -> bool:
    try:
        _validate_pinned_directory_path_without_symlinks(
            output_path.parent,
            output_parent_descriptor,
            error_message="material identity packet parent changed",
        )
        if _read_bound_binary_at(
            output_parent_descriptor,
            output_path.name,
            max_bytes=_MAX_REVIEW_PACKET_BYTES,
        ) != packet_text.encode("utf-8"):
            return False
        _validate_open_asset_directory_binding(
            asset_dir=asset_dir,
            parent_descriptor=asset_parent_descriptor,
            asset_descriptor=asset_descriptor,
            asset_directory_stat=asset_directory_stat,
            reject_symlink_components=True,
        )
        _validate_open_created_identity_asset_bundle(
            asset_descriptor=asset_descriptor,
            packet=packet,
            rendered_assets=rendered_assets,
            created_asset_stats=created_asset_stats,
        )
        _validate_published_packet_paths(
            output_path=output_path,
            asset_dir=asset_dir,
            protected_paths=protected_paths,
        )
    except (ValueError, OSError):
        return False
    return True


def _write_identity_asset_bundle(
    *,
    asset_descriptor: int,
    packet: OledSupplementaryMaterialIdentityReviewPacket,
    rendered_assets: dict[str, bytes],
    created_asset_stats: dict[str, os.stat_result],
) -> None:
    expected = _expected_asset_metadata(packet)
    if set(expected) != set(rendered_assets):
        raise ValueError("material identity rendered asset coverage mismatch")
    for filename in sorted(expected):
        payload = rendered_assets[filename]
        _validate_asset_payload(expected[filename], payload)
        created_asset_stats[filename] = _write_fresh_bytes_at(
            asset_descriptor,
            filename,
            payload,
        )
    os.fsync(asset_descriptor)


def _validate_open_created_identity_asset_bundle(
    *,
    asset_descriptor: int,
    packet: OledSupplementaryMaterialIdentityReviewPacket,
    rendered_assets: dict[str, bytes],
    created_asset_stats: dict[str, os.stat_result],
) -> None:
    expected = _expected_asset_metadata(packet)
    if (
        set(expected) != set(rendered_assets)
        or set(expected) != set(created_asset_stats)
        or set(os.listdir(asset_descriptor)) != set(expected)
    ):
        raise ValueError("material identity published asset coverage changed")
    initial_directory_stat = os.fstat(asset_descriptor)
    for filename, asset in expected.items():
        current_stat = os.stat(
            filename,
            dir_fd=asset_descriptor,
            follow_symlinks=False,
        )
        created_stat = created_asset_stats[filename]
        if (
            not stat.S_ISREG(current_stat.st_mode)
            or current_stat.st_dev != created_stat.st_dev
            or current_stat.st_ino != created_stat.st_ino
        ):
            raise ValueError("material identity published asset inode changed")
        payload = _read_bound_binary_at(
            asset_descriptor,
            filename,
            max_bytes=_MAX_ASSET_BYTES,
        )
        if payload != rendered_assets[filename]:
            raise ValueError("material identity published asset bytes changed")
        _validate_asset_payload(asset, payload)
    final_directory_stat = os.fstat(asset_descriptor)
    if (
        final_directory_stat.st_dev != initial_directory_stat.st_dev
        or final_directory_stat.st_ino != initial_directory_stat.st_ino
        or final_directory_stat.st_mtime_ns != initial_directory_stat.st_mtime_ns
        or final_directory_stat.st_ctime_ns != initial_directory_stat.st_ctime_ns
    ):
        raise ValueError("material identity published asset directory changed")


def _validate_identity_asset_bundle(
    asset_dir: Path,
    packet: OledSupplementaryMaterialIdentityReviewPacket,
    *,
    expected_bytes: dict[str, bytes] | None = None,
    pinned_parent_descriptor: int | None = None,
) -> None:
    no_follow = getattr(os, "O_NOFOLLOW", None)
    directory_flag = getattr(os, "O_DIRECTORY", None)
    if no_follow is None or directory_flag is None:
        raise ValueError("material identity review requires safe dirfd support")
    expected = _expected_asset_metadata(packet)
    if expected_bytes is not None and set(expected_bytes) != set(expected):
        raise ValueError("material identity rerendered asset coverage mismatch")
    descriptor = -1
    try:
        if pinned_parent_descriptor is None:
            descriptor = os.open(
                asset_dir,
                os.O_RDONLY | directory_flag | no_follow,
            )
            parent_stat = os.stat(asset_dir.parent, follow_symlinks=False)
        else:
            _validate_pinned_directory_path_without_symlinks(
                asset_dir.parent,
                pinned_parent_descriptor,
                error_message="material identity asset parent changed",
            )
            descriptor = os.open(
                asset_dir.name,
                os.O_RDONLY | directory_flag | no_follow,
                dir_fd=pinned_parent_descriptor,
            )
            parent_stat = os.fstat(pinned_parent_descriptor)
        if pinned_parent_descriptor is None:
            current_parent_stat = os.stat(
                asset_dir.parent,
                follow_symlinks=False,
            )
        else:
            current_parent_stat = os.fstat(pinned_parent_descriptor)
        initial_stat = os.fstat(descriptor)
        if (
            not stat.S_ISDIR(parent_stat.st_mode)
            or not stat.S_ISDIR(current_parent_stat.st_mode)
            or current_parent_stat.st_dev != parent_stat.st_dev
            or current_parent_stat.st_ino != parent_stat.st_ino
            or not stat.S_ISDIR(initial_stat.st_mode)
        ):
            raise ValueError("material identity asset directory is invalid")
        if set(os.listdir(descriptor)) != set(expected):
            raise ValueError("material identity asset coverage mismatch")
        for filename, asset in expected.items():
            payload = _read_bound_binary_at(
                descriptor,
                filename,
                max_bytes=_MAX_ASSET_BYTES,
            )
            _validate_asset_payload(asset, payload)
            if expected_bytes is not None and payload != expected_bytes[filename]:
                raise ValueError("material identity rerendered asset changed")
        final_stat = os.fstat(descriptor)
        current_stat = os.stat(asset_dir, follow_symlinks=False)
        if pinned_parent_descriptor is None:
            final_parent_stat = os.stat(
                asset_dir.parent,
                follow_symlinks=False,
            )
        else:
            _validate_pinned_directory_path_without_symlinks(
                asset_dir.parent,
                pinned_parent_descriptor,
                error_message="material identity asset parent changed",
            )
            final_parent_stat = os.fstat(pinned_parent_descriptor)
        if (
            not stat.S_ISDIR(current_stat.st_mode)
            or final_stat.st_dev != initial_stat.st_dev
            or final_stat.st_ino != initial_stat.st_ino
            or final_stat.st_mtime_ns != initial_stat.st_mtime_ns
            or final_stat.st_ctime_ns != initial_stat.st_ctime_ns
            or current_stat.st_dev != initial_stat.st_dev
            or current_stat.st_ino != initial_stat.st_ino
            or not stat.S_ISDIR(final_parent_stat.st_mode)
            or final_parent_stat.st_dev != parent_stat.st_dev
            or final_parent_stat.st_ino != parent_stat.st_ino
        ):
            raise ValueError("material identity asset directory changed while read")
    except ValueError:
        raise
    except OSError as exc:
        raise ValueError("material identity asset directory is unavailable") from exc
    finally:
        if descriptor != -1:
            os.close(descriptor)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build, render, and adjudicate exact-bound supplementary "
            "material-identity review packets without Registry or dataset writes."
        )
    )
    commands = parser.add_subparsers(dest="command", required=True)
    packet = commands.add_parser(
        "packet",
        help="render exact PDF pages and candidate depictions for human review",
    )
    _add_upstream_arguments(packet)
    packet.add_argument("--source-pdf", required=True)
    _add_poppler_argument(packet)
    packet.add_argument("--asset-dir", required=True)
    packet.add_argument("--output", required=True)

    render = commands.add_parser(
        "render",
        help="render a validated reviewer-facing Markdown packet",
    )
    render.add_argument("--review-packet", required=True)
    render.add_argument("--asset-dir", required=True)
    render.add_argument("--output-markdown", required=True)

    adjudicate = commands.add_parser(
        "adjudicate",
        help="apply one complete exact-bound human decision per identity group",
    )
    _add_upstream_arguments(adjudicate)
    adjudicate.add_argument("--source-pdf", required=True)
    _add_poppler_argument(adjudicate)
    adjudicate.add_argument("--review-packet", required=True)
    adjudicate.add_argument("--decision-manifest", required=True)
    adjudicate.add_argument("--asset-dir", required=True)
    adjudicate.add_argument("--output", required=True)
    return parser


def _add_upstream_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--request-artifact", required=True)
    parser.add_argument("--transcription-review-packet", required=True)
    parser.add_argument("--response-manifest", required=True)
    parser.add_argument("--response-artifact", required=True)


def _add_poppler_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--poppler-bin-dir",
        help=(
            "absolute operator-trusted directory containing pdfinfo and pdftoppm; "
            "defaults to /usr/bin and never searches inherited PATH"
        ),
    )


def main(argv: Sequence[str] | None = None, *, stdout: TextIO | None = None) -> int:
    stream = stdout or sys.stdout
    args = build_parser().parse_args(argv)
    try:
        if args.command == "packet":
            packet = build_oled_supplementary_material_identity_review_packet_from_files(
                request_artifact_json=args.request_artifact,
                transcription_review_packet_json=args.transcription_review_packet,
                response_manifest_json=args.response_manifest,
                response_artifact_json=args.response_artifact,
                source_pdf_path=args.source_pdf,
                poppler_bin_dir=args.poppler_bin_dir,
                asset_dir=args.asset_dir,
                output_json=args.output,
            )
            result = {
                "status": packet.status.value,
                "review_item_count": packet.review_item_count,
                "source_page_asset_count": packet.cited_source_page_count,
                "candidate_depiction_asset_count": (
                    packet.candidate_depiction_asset_count
                ),
            }
        elif args.command == "render":
            packet = render_oled_supplementary_material_identity_review_packet_from_files(
                review_packet_json=args.review_packet,
                asset_dir=args.asset_dir,
                output_markdown=args.output_markdown,
            )
            result = {
                "status": "rendered",
                "review_item_count": packet.review_item_count,
                "asset_count": (
                    packet.cited_source_page_count
                    + packet.candidate_depiction_asset_count
                ),
            }
        else:
            artifact = build_oled_supplementary_material_identity_adjudication_from_files(
                request_artifact_json=args.request_artifact,
                transcription_review_packet_json=args.transcription_review_packet,
                response_manifest_json=args.response_manifest,
                response_artifact_json=args.response_artifact,
                source_pdf_path=args.source_pdf,
                poppler_bin_dir=args.poppler_bin_dir,
                review_packet_json=args.review_packet,
                decision_manifest_json=args.decision_manifest,
                asset_dir=args.asset_dir,
                output_json=args.output,
            )
            result = {
                "status": artifact.status.value,
                "review_item_count": artifact.review_item_count,
                "later_registry_review_eligible_group_count": (
                    artifact.later_registry_review_eligible_group_count
                ),
                "unresolved_review_item_count": (
                    artifact.unresolved_review_item_count
                ),
            }
    except Exception as exc:
        print(
            json.dumps(
                {
                    "status": "error",
                    "error_code": "supplementary_material_identity_review_failed",
                    "error_type": type(exc).__name__,
                },
                sort_keys=True,
            ),
            file=stream,
        )
        return 2
    print(json.dumps(result, sort_keys=True), file=stream)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "build_oled_supplementary_material_identity_adjudication_from_files",
    "build_oled_supplementary_material_identity_review_packet_from_files",
    "main",
    "render_oled_supplementary_material_identity_review_packet_from_files",
]
