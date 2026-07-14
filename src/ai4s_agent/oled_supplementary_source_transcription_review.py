from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import os
import re
import resource
import secrets
import signal
import stat
import struct
import subprocess
import sys
import tempfile
import warnings
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterator, Sequence, TextIO

from PIL import Image, UnidentifiedImageError

from ai4s_agent._utils import now_iso
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
    _normalize_sha256,
    _validate_path_segment,
)
from ai4s_agent.domains.oled_supplementary_source_transcription_review import (
    SUPPLEMENTARY_SOURCE_TRANSCRIPTION_RENDERER_ID,
    SUPPLEMENTARY_SOURCE_TRANSCRIPTION_RENDER_PROFILE,
    OledSupplementarySourcePageAsset,
    OledSupplementarySourcePdfEvidence,
    OledSupplementarySourceTranscriptionAdjudicationArtifact,
    OledSupplementarySourceTranscriptionDecisionManifest,
    OledSupplementarySourceTranscriptionReviewPacket,
    build_oled_supplementary_source_page_asset,
    build_oled_supplementary_source_pdf_evidence,
    build_oled_supplementary_source_transcription_adjudication_artifact,
    build_oled_supplementary_source_transcription_review_packet,
    render_oled_supplementary_source_transcription_review_markdown,
)
from ai4s_agent.oled_supplementary_scoped_candidate_response import (
    _absolute_local_path,
    _open_regular_file_without_symlink_components,
    _read_bound_json,
    _validate_fresh_output,
    _write_fresh_text,
)


_MAX_REQUEST_BYTES = 100 * 1024 * 1024
_MAX_RESPONSE_MANIFEST_BYTES = 50 * 1024 * 1024
_MAX_RESPONSE_ARTIFACT_BYTES = 100 * 1024 * 1024
_MAX_SEMANTIC_REVIEW_PACKET_BYTES = 150 * 1024 * 1024
_MAX_SEMANTIC_DECISION_MANIFEST_BYTES = 20 * 1024 * 1024
_MAX_SEMANTIC_ADJUDICATION_BYTES = 200 * 1024 * 1024
_MAX_TRANSCRIPTION_REVIEW_PACKET_BYTES = 200 * 1024 * 1024
_MAX_TRANSCRIPTION_DECISION_MANIFEST_BYTES = 20 * 1024 * 1024
_MAX_SOURCE_PDF_BYTES = 200 * 1024 * 1024
_MAX_RENDERED_PNG_BYTES = 100 * 1024 * 1024
_MAX_RENDERED_PIXEL_DIMENSION = 20_000
_MAX_RENDERED_PIXEL_COUNT = 80_000_000
_MAX_RENDER_LOG_BYTES = 1024 * 1024
_MAX_RENDER_ADDRESS_SPACE_BYTES = 4 * 1024 * 1024 * 1024
_MAX_RENDER_CPU_SECONDS = 55
_PDF_TRAILER_SCAN_BYTES = 8192
_RENDER_TIMEOUT_SECONDS = 60
_PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
_SYSTEM_POPPLER_BIN_DIR = Path("/usr/bin")
_NATIVE_EXECUTABLE_MAGICS = {
    b"\x7fELF",
    b"\xce\xfa\xed\xfe",
    b"\xcf\xfa\xed\xfe",
    b"\xca\xfe\xba\xbe",
    b"\xbe\xba\xfe\xca",
    b"\xfe\xed\xfa\xce",
    b"\xfe\xed\xfa\xcf",
}


@dataclass(frozen=True)
class _PinnedExecutable:
    path: Path
    descriptor: int
    sha256: str
    initial_stat: os.stat_result


@dataclass(frozen=True)
class _PinnedPopplerToolchain:
    bin_dir: Path
    directory_descriptor: int
    directory_stat: os.stat_result
    pdfinfo: _PinnedExecutable
    pdftoppm: _PinnedExecutable


def build_oled_supplementary_source_transcription_review_packet_from_files(
    *,
    request_artifact_json: str | Path,
    response_manifest_json: str | Path,
    response_artifact_json: str | Path,
    semantic_review_packet_json: str | Path,
    semantic_decision_manifest_json: str | Path,
    semantic_adjudication_json: str | Path,
    source_pdf_path: str | Path,
    poppler_bin_dir: str | Path | None = None,
    asset_dir: str | Path,
    output_json: str | Path,
    generated_at: str | None = None,
) -> OledSupplementarySourceTranscriptionReviewPacket:
    paths = _input_paths(
        request_artifact_json=request_artifact_json,
        response_manifest_json=response_manifest_json,
        response_artifact_json=response_artifact_json,
        semantic_review_packet_json=semantic_review_packet_json,
        semantic_decision_manifest_json=semantic_decision_manifest_json,
        semantic_adjudication_json=semantic_adjudication_json,
    )
    source_path = _absolute_local_path(source_pdf_path)
    output_path = _absolute_local_path(output_json)
    assets_path = _absolute_local_path(asset_dir)
    upstream = _load_upstream(paths)
    request = upstream["request"]
    assert isinstance(request, OledSupplementaryScopedCandidateRequestArtifact)
    protected_paths = {*paths.values(), source_path}
    _validate_packet_outputs(
        output_path=output_path,
        asset_dir=assets_path,
        protected_paths=protected_paths,
    )
    asset_parent_descriptor = -1
    asset_descriptor = -1
    asset_directory_stat: os.stat_result | None = None
    created_asset_stats: dict[str, os.stat_result] = {}
    keep_assets = False
    try:
        evidence, rendered_assets = _render_bound_source_pdf(
            source_pdf_path=source_path,
            request_artifact=request,
            poppler_bin_dir=poppler_bin_dir,
        )
        packet = build_oled_supplementary_source_transcription_review_packet(
            request_artifact=request,
            request_artifact_sha256=upstream["request_sha256"],
            response_manifest=upstream["response_manifest"],
            response_manifest_sha256=upstream["response_manifest_sha256"],
            response_artifact=upstream["response_artifact"],
            response_artifact_sha256=upstream["response_artifact_sha256"],
            semantic_review_packet=upstream["semantic_review_packet"],
            semantic_review_packet_sha256=upstream["semantic_review_packet_sha256"],
            semantic_decision_manifest=upstream["semantic_decision_manifest"],
            semantic_decision_manifest_sha256=upstream[
                "semantic_decision_manifest_sha256"
            ],
            semantic_adjudication_artifact=upstream["semantic_adjudication"],
            semantic_adjudication_artifact_sha256=upstream[
                "semantic_adjudication_sha256"
            ],
            source_pdf_evidence=evidence,
            generated_at=generated_at or now_iso(),
        )
        (
            asset_parent_descriptor,
            asset_descriptor,
            asset_directory_stat,
        ) = _create_private_asset_directory(assets_path)
        _write_asset_bundle(
            asset_descriptor,
            packet,
            rendered_assets,
            created_asset_stats=created_asset_stats,
        )
        _validate_open_asset_directory_binding(
            asset_dir=assets_path,
            parent_descriptor=asset_parent_descriptor,
            asset_descriptor=asset_descriptor,
            asset_directory_stat=asset_directory_stat,
        )
        packet_text = (
            json.dumps(packet.model_dump(mode="json"), ensure_ascii=False, indent=2)
            + "\n"
        )
        def validate_joint_publication() -> None:
            _validate_open_asset_directory_binding(
                asset_dir=assets_path,
                parent_descriptor=asset_parent_descriptor,
                asset_descriptor=asset_descriptor,
                asset_directory_stat=asset_directory_stat,
            )
            _validate_open_created_asset_bundle(
                asset_descriptor=asset_descriptor,
                packet=packet,
                rendered_assets=rendered_assets,
                created_asset_stats=created_asset_stats,
            )
            _validate_published_packet_paths(
                output_path=output_path,
                asset_dir=assets_path,
                protected_paths=protected_paths,
            )

        _publish_packet_text(
            output_path,
            packet_text,
            post_publish_validator=validate_joint_publication,
        )
        keep_assets = True
        return packet
    finally:
        if not keep_assets and asset_directory_stat is not None:
            _remove_private_asset_directory(
                asset_dir=assets_path,
                parent_descriptor=asset_parent_descriptor,
                asset_descriptor=asset_descriptor,
                asset_directory_stat=asset_directory_stat,
                created_asset_stats=created_asset_stats,
            )
        if asset_descriptor != -1:
            os.close(asset_descriptor)
        if asset_parent_descriptor != -1:
            os.close(asset_parent_descriptor)


def render_oled_supplementary_source_transcription_review_packet_from_files(
    *,
    review_packet_json: str | Path,
    asset_dir: str | Path,
    output_markdown: str | Path,
) -> OledSupplementarySourceTranscriptionReviewPacket:
    packet_path = _absolute_local_path(review_packet_json)
    assets_path = _absolute_local_path(asset_dir)
    output_path = _absolute_local_path(output_markdown)
    packet_payload, packet_sha256 = _read_bound_json(
        packet_path,
        "supplementary source transcription review packet",
        max_bytes=_MAX_TRANSCRIPTION_REVIEW_PACKET_BYTES,
    )
    packet = OledSupplementarySourceTranscriptionReviewPacket.model_validate(
        packet_payload
    )
    _validate_review_asset_location(
        asset_dir=assets_path,
        output_markdown=output_path,
    )
    _validate_asset_bundle(assets_path, packet)
    _validate_fresh_output(
        output_path,
        protected_paths={
            packet_path,
            *(_absolute_local_path(assets_path / asset.asset_filename)
              for asset in packet.source_pdf_evidence.page_assets),
        },
    )
    _write_fresh_text(
        output_path,
        render_oled_supplementary_source_transcription_review_markdown(
            packet,
            review_packet_sha256=packet_sha256,
        ),
    )
    return packet


def build_oled_supplementary_source_transcription_adjudication_from_files(
    *,
    request_artifact_json: str | Path,
    response_manifest_json: str | Path,
    response_artifact_json: str | Path,
    semantic_review_packet_json: str | Path,
    semantic_decision_manifest_json: str | Path,
    semantic_adjudication_json: str | Path,
    source_pdf_path: str | Path,
    poppler_bin_dir: str | Path | None = None,
    transcription_review_packet_json: str | Path,
    transcription_decision_manifest_json: str | Path,
    asset_dir: str | Path,
    output_json: str | Path,
    generated_at: str | None = None,
) -> OledSupplementarySourceTranscriptionAdjudicationArtifact:
    paths = _input_paths(
        request_artifact_json=request_artifact_json,
        response_manifest_json=response_manifest_json,
        response_artifact_json=response_artifact_json,
        semantic_review_packet_json=semantic_review_packet_json,
        semantic_decision_manifest_json=semantic_decision_manifest_json,
        semantic_adjudication_json=semantic_adjudication_json,
    )
    source_path = _absolute_local_path(source_pdf_path)
    review_packet_path = _absolute_local_path(transcription_review_packet_json)
    decision_path = _absolute_local_path(transcription_decision_manifest_json)
    assets_path = _absolute_local_path(asset_dir)
    output_path = _absolute_local_path(output_json)
    upstream = _load_upstream(paths)
    packet_payload, packet_sha256 = _read_bound_json(
        review_packet_path,
        "supplementary source transcription review packet",
        max_bytes=_MAX_TRANSCRIPTION_REVIEW_PACKET_BYTES,
    )
    decision_payload, decision_sha256 = _read_bound_json(
        decision_path,
        "supplementary source transcription decision manifest",
        max_bytes=_MAX_TRANSCRIPTION_DECISION_MANIFEST_BYTES,
    )
    packet = OledSupplementarySourceTranscriptionReviewPacket.model_validate(
        packet_payload
    )
    decisions = OledSupplementarySourceTranscriptionDecisionManifest.model_validate(
        decision_payload
    )
    protected_paths = {
        *paths.values(),
        source_path,
        review_packet_path,
        decision_path,
        *(_absolute_local_path(assets_path / asset.asset_filename)
          for asset in packet.source_pdf_evidence.page_assets),
    }
    _validate_fresh_output(output_path, protected_paths=protected_paths)
    if _path_is_within(output_path, assets_path):
        raise ValueError(
            "supplementary source transcription output must be outside the asset directory"
        )
    request = upstream["request"]
    assert isinstance(request, OledSupplementaryScopedCandidateRequestArtifact)
    evidence, rendered_assets = _render_bound_source_pdf(
        source_pdf_path=source_path,
        request_artifact=request,
        poppler_bin_dir=poppler_bin_dir,
    )
    if evidence.model_dump(mode="json") != packet.source_pdf_evidence.model_dump(
        mode="json"
    ):
        raise ValueError("supplementary source transcription PDF evidence changed")
    _validate_asset_bundle(assets_path, packet, expected_bytes=rendered_assets)
    artifact = build_oled_supplementary_source_transcription_adjudication_artifact(
        request_artifact=request,
        request_artifact_sha256=upstream["request_sha256"],
        response_manifest=upstream["response_manifest"],
        response_manifest_sha256=upstream["response_manifest_sha256"],
        response_artifact=upstream["response_artifact"],
        response_artifact_sha256=upstream["response_artifact_sha256"],
        semantic_review_packet=upstream["semantic_review_packet"],
        semantic_review_packet_sha256=upstream["semantic_review_packet_sha256"],
        semantic_decision_manifest=upstream["semantic_decision_manifest"],
        semantic_decision_manifest_sha256=upstream[
            "semantic_decision_manifest_sha256"
        ],
        semantic_adjudication_artifact=upstream["semantic_adjudication"],
        semantic_adjudication_artifact_sha256=upstream[
            "semantic_adjudication_sha256"
        ],
        source_pdf_evidence=evidence,
        review_packet=packet,
        review_packet_sha256=packet_sha256,
        decision_manifest=decisions,
        decision_manifest_sha256=decision_sha256,
        generated_at=generated_at or now_iso(),
    )
    _write_fresh_text(
        output_path,
        json.dumps(artifact.model_dump(mode="json"), ensure_ascii=False, indent=2)
        + "\n",
    )
    return artifact


def _input_paths(
    *,
    request_artifact_json: str | Path,
    response_manifest_json: str | Path,
    response_artifact_json: str | Path,
    semantic_review_packet_json: str | Path,
    semantic_decision_manifest_json: str | Path,
    semantic_adjudication_json: str | Path,
) -> dict[str, Path]:
    return {
        "request": _absolute_local_path(request_artifact_json),
        "response_manifest": _absolute_local_path(response_manifest_json),
        "response_artifact": _absolute_local_path(response_artifact_json),
        "semantic_review_packet": _absolute_local_path(semantic_review_packet_json),
        "semantic_decision_manifest": _absolute_local_path(
            semantic_decision_manifest_json
        ),
        "semantic_adjudication": _absolute_local_path(semantic_adjudication_json),
    }


def _load_upstream(paths: dict[str, Path]) -> dict[str, Any]:
    request_payload, request_sha256 = _read_bound_json(
        paths["request"],
        "supplementary source transcription request artifact",
        max_bytes=_MAX_REQUEST_BYTES,
    )
    manifest_payload, manifest_sha256 = _read_bound_json(
        paths["response_manifest"],
        "supplementary source transcription response manifest",
        max_bytes=_MAX_RESPONSE_MANIFEST_BYTES,
    )
    response_payload, response_sha256 = _read_bound_json(
        paths["response_artifact"],
        "supplementary source transcription response artifact",
        max_bytes=_MAX_RESPONSE_ARTIFACT_BYTES,
    )
    semantic_packet_payload, semantic_packet_sha256 = _read_bound_json(
        paths["semantic_review_packet"],
        "supplementary source transcription semantic packet",
        max_bytes=_MAX_SEMANTIC_REVIEW_PACKET_BYTES,
    )
    semantic_decision_payload, semantic_decision_sha256 = _read_bound_json(
        paths["semantic_decision_manifest"],
        "supplementary source transcription semantic decisions",
        max_bytes=_MAX_SEMANTIC_DECISION_MANIFEST_BYTES,
    )
    semantic_adjudication_payload, semantic_adjudication_sha256 = _read_bound_json(
        paths["semantic_adjudication"],
        "supplementary source transcription semantic adjudication",
        max_bytes=_MAX_SEMANTIC_ADJUDICATION_BYTES,
    )
    return {
        "request": OledSupplementaryScopedCandidateRequestArtifact.model_validate(
            request_payload
        ),
        "request_sha256": request_sha256,
        "response_manifest": OledSupplementaryScopedCandidateResponseManifest.model_validate(
            manifest_payload
        ),
        "response_manifest_sha256": manifest_sha256,
        "response_artifact": OledSupplementaryScopedCandidateResponseArtifact.model_validate(
            response_payload
        ),
        "response_artifact_sha256": response_sha256,
        "semantic_review_packet": OledSupplementarySemanticReviewPacket.model_validate(
            semantic_packet_payload
        ),
        "semantic_review_packet_sha256": semantic_packet_sha256,
        "semantic_decision_manifest": OledSupplementarySemanticDecisionManifest.model_validate(
            semantic_decision_payload
        ),
        "semantic_decision_manifest_sha256": semantic_decision_sha256,
        "semantic_adjudication": OledSupplementarySemanticAdjudicationArtifact.model_validate(
            semantic_adjudication_payload
        ),
        "semantic_adjudication_sha256": semantic_adjudication_sha256,
    }


@contextmanager
def _pinned_poppler_toolchain(
    poppler_bin_dir: str | Path | None,
    *,
    execution_root: Path,
) -> Iterator[_PinnedPopplerToolchain]:
    candidate = (
        _SYSTEM_POPPLER_BIN_DIR
        if poppler_bin_dir is None
        else Path(poppler_bin_dir).expanduser()
    )
    if not candidate.is_absolute():
        raise ValueError(
            "supplementary source transcription Poppler bin directory must be absolute"
        )
    try:
        bin_dir = candidate.resolve(strict=True)
    except OSError as exc:
        raise ValueError(
            "supplementary source transcription trusted Poppler toolchain is unavailable"
        ) from exc
    no_follow = getattr(os, "O_NOFOLLOW", None)
    directory_flag = getattr(os, "O_DIRECTORY", None)
    if no_follow is None or directory_flag is None:
        raise ValueError(
            "supplementary source transcription requires safe dirfd support"
        )
    directory_descriptor = -1
    pinned: list[_PinnedExecutable] = []
    try:
        execution_dir = execution_root / "pinned-poppler-bin"
        execution_dir.mkdir(mode=0o700)
        directory_descriptor = os.open(
            bin_dir,
            os.O_RDONLY | directory_flag | no_follow,
        )
        directory_stat = os.fstat(directory_descriptor)
        if not stat.S_ISDIR(directory_stat.st_mode):
            raise ValueError(
                "supplementary source transcription Poppler bin path is not a directory"
            )
        current_stat = os.stat(bin_dir, follow_symlinks=False)
        if (
            not stat.S_ISDIR(current_stat.st_mode)
            or current_stat.st_dev != directory_stat.st_dev
            or current_stat.st_ino != directory_stat.st_ino
        ):
            raise ValueError(
                "supplementary source transcription Poppler bin directory changed"
            )
        pdfinfo = _pin_executable(
            directory_descriptor=directory_descriptor,
            filename="pdfinfo",
            execution_dir=execution_dir,
        )
        pinned.append(pdfinfo)
        pdftoppm = _pin_executable(
            directory_descriptor=directory_descriptor,
            filename="pdftoppm",
            execution_dir=execution_dir,
        )
        pinned.append(pdftoppm)
        toolchain = _PinnedPopplerToolchain(
            bin_dir=bin_dir,
            directory_descriptor=directory_descriptor,
            directory_stat=directory_stat,
            pdfinfo=pdfinfo,
            pdftoppm=pdftoppm,
        )
        _validate_pinned_toolchain(toolchain)
        yield toolchain
        _validate_pinned_toolchain(toolchain)
    except ValueError:
        raise
    except OSError as exc:
        raise ValueError(
            "supplementary source transcription trusted Poppler toolchain is unavailable"
        ) from exc
    finally:
        for executable in reversed(pinned):
            os.close(executable.descriptor)
        if directory_descriptor != -1:
            os.close(directory_descriptor)


def _pin_executable(
    *,
    directory_descriptor: int,
    filename: str,
    execution_dir: Path,
) -> _PinnedExecutable:
    no_follow = getattr(os, "O_NOFOLLOW", None)
    if no_follow is None:
        raise ValueError("supplementary source transcription requires O_NOFOLLOW support")
    descriptor = -1
    try:
        descriptor = os.open(
            filename,
            os.O_RDONLY | no_follow | getattr(os, "O_NONBLOCK", 0),
            dir_fd=directory_descriptor,
        )
        initial_stat = os.fstat(descriptor)
        if (
            not stat.S_ISREG(initial_stat.st_mode)
            or not initial_stat.st_mode & 0o111
            or initial_stat.st_size <= 0
            or initial_stat.st_size > _MAX_SOURCE_PDF_BYTES
        ):
            raise ValueError(
                "supplementary source transcription Poppler executable is invalid"
            )
        executable_header = os.read(descriptor, 4)
        os.lseek(descriptor, 0, os.SEEK_SET)
        if executable_header not in _NATIVE_EXECUTABLE_MAGICS:
            raise ValueError(
                "supplementary source transcription Poppler executable must be a native binary"
            )
        named_stat = os.stat(
            filename,
            dir_fd=directory_descriptor,
            follow_symlinks=False,
        )
        if (
            not stat.S_ISREG(named_stat.st_mode)
            or named_stat.st_dev != initial_stat.st_dev
            or named_stat.st_ino != initial_stat.st_ino
        ):
            raise ValueError(
                "supplementary source transcription Poppler executable changed"
            )
        digest = _hash_open_descriptor(descriptor, max_bytes=_MAX_SOURCE_PDF_BYTES)
        execution_path = _materialize_pinned_executable(
            source_descriptor=descriptor,
            destination=execution_dir / filename,
            expected_sha256=digest,
        )
        return _PinnedExecutable(
            path=execution_path,
            descriptor=descriptor,
            sha256=digest,
            initial_stat=initial_stat,
        )
    except BaseException:
        if descriptor != -1:
            os.close(descriptor)
        raise


def _materialize_pinned_executable(
    *,
    source_descriptor: int,
    destination: Path,
    expected_sha256: str,
) -> Path:
    no_follow = getattr(os, "O_NOFOLLOW", None)
    if no_follow is None:
        raise ValueError("supplementary source transcription requires O_NOFOLLOW support")
    destination_descriptor = -1
    try:
        destination_descriptor = os.open(
            destination,
            os.O_RDWR | os.O_CREAT | os.O_EXCL | no_follow,
            0o500,
        )
        os.lseek(source_descriptor, 0, os.SEEK_SET)
        while True:
            chunk = os.read(source_descriptor, 1024 * 1024)
            if not chunk:
                break
            _write_all(destination_descriptor, chunk)
        os.lseek(source_descriptor, 0, os.SEEK_SET)
        os.fchmod(destination_descriptor, 0o500)
        os.fsync(destination_descriptor)
        if (
            _hash_open_descriptor(
                destination_descriptor,
                max_bytes=_MAX_SOURCE_PDF_BYTES,
            )
            != expected_sha256
        ):
            raise ValueError(
                "supplementary source transcription private Poppler copy changed"
            )
        return destination
    except ValueError:
        raise
    except OSError as exc:
        raise ValueError(
            "supplementary source transcription cannot pin Poppler executable bytes"
        ) from exc
    finally:
        if destination_descriptor != -1:
            os.close(destination_descriptor)


def _hash_open_descriptor(descriptor: int, *, max_bytes: int) -> str:
    os.lseek(descriptor, 0, os.SEEK_SET)
    digest = hashlib.sha256()
    bytes_read = 0
    while True:
        chunk = os.read(descriptor, min(1024 * 1024, max_bytes + 1 - bytes_read))
        if not chunk:
            break
        digest.update(chunk)
        bytes_read += len(chunk)
        if bytes_read > max_bytes:
            raise ValueError(
                "supplementary source transcription Poppler executable is too large"
            )
    os.lseek(descriptor, 0, os.SEEK_SET)
    return f"sha256:{digest.hexdigest()}"


def _validate_pinned_toolchain(toolchain: _PinnedPopplerToolchain) -> None:
    try:
        current_directory_stat = os.stat(toolchain.bin_dir, follow_symlinks=False)
        open_directory_stat = os.fstat(toolchain.directory_descriptor)
    except OSError as exc:
        raise ValueError(
            "supplementary source transcription Poppler toolchain changed"
        ) from exc
    if (
        not stat.S_ISDIR(current_directory_stat.st_mode)
        or current_directory_stat.st_dev != toolchain.directory_stat.st_dev
        or current_directory_stat.st_ino != toolchain.directory_stat.st_ino
        or open_directory_stat.st_dev != toolchain.directory_stat.st_dev
        or open_directory_stat.st_ino != toolchain.directory_stat.st_ino
    ):
        raise ValueError(
            "supplementary source transcription Poppler toolchain changed"
        )
    for executable in (toolchain.pdfinfo, toolchain.pdftoppm):
        try:
            current = os.stat(
                executable.path.name,
                dir_fd=toolchain.directory_descriptor,
                follow_symlinks=False,
            )
            opened = os.fstat(executable.descriptor)
        except OSError as exc:
            raise ValueError(
                "supplementary source transcription Poppler executable changed"
            ) from exc
        if (
            not stat.S_ISREG(current.st_mode)
            or current.st_dev != executable.initial_stat.st_dev
            or current.st_ino != executable.initial_stat.st_ino
            or opened.st_dev != executable.initial_stat.st_dev
            or opened.st_ino != executable.initial_stat.st_ino
            or opened.st_size != executable.initial_stat.st_size
            or opened.st_mtime_ns != executable.initial_stat.st_mtime_ns
            or opened.st_ctime_ns != executable.initial_stat.st_ctime_ns
            or _hash_open_descriptor(
                executable.descriptor,
                max_bytes=_MAX_SOURCE_PDF_BYTES,
            )
            != executable.sha256
        ):
            raise ValueError(
                "supplementary source transcription Poppler executable changed"
            )
        if (
            f"sha256:{hashlib.sha256(_read_bound_binary_path(executable.path, max_bytes=_MAX_SOURCE_PDF_BYTES)).hexdigest()}"
            != executable.sha256
        ):
            raise ValueError(
                "supplementary source transcription private Poppler copy changed"
            )


def _render_bound_source_pdf(
    *,
    source_pdf_path: Path,
    request_artifact: OledSupplementaryScopedCandidateRequestArtifact,
    poppler_bin_dir: str | Path | None = None,
) -> tuple[OledSupplementarySourcePdfEvidence, dict[str, bytes]]:
    source_pairs = {
        (scope.source_id, scope.source_pdf_sha256) for scope in request_artifact.scopes
    }
    if len(source_pairs) != 1:
        raise ValueError(
            "supplementary source transcription v1 requires one bound source PDF"
        )
    source_id, expected_sha256 = next(iter(source_pairs))
    pages = sorted({scope.matched_table.page for scope in request_artifact.scopes})
    return _render_exact_bound_source_pdf_pages(
        source_pdf_path=source_pdf_path,
        source_id=source_id,
        expected_sha256=expected_sha256,
        pages=pages,
        poppler_bin_dir=poppler_bin_dir,
    )


def _render_exact_bound_source_pdf_pages(
    *,
    source_pdf_path: Path,
    source_id: str,
    expected_sha256: str,
    pages: Sequence[int],
    poppler_bin_dir: str | Path | None = None,
    reject_symlink_components: bool = False,
) -> tuple[OledSupplementarySourcePdfEvidence, dict[str, bytes]]:
    """Render exact full pages from one source/hash-bound PDF."""

    if not isinstance(source_id, str):
        raise ValueError("supplementary source transcription source_id must be text")
    source_id_clean = _validate_path_segment(source_id, field_name="source_id")
    if source_id_clean != source_id:
        raise ValueError("supplementary source transcription source_id must be exact")
    expected_sha256_clean = _normalize_sha256(
        expected_sha256,
        field_name="expected_sha256",
    )
    page_numbers = list(pages)
    if not page_numbers:
        raise ValueError(
            "supplementary source transcription requires a non-empty page roster"
        )
    if (
        any(
            not isinstance(page, int) or isinstance(page, bool) or page < 1
            for page in page_numbers
        )
        or page_numbers != sorted(page_numbers)
        or len(page_numbers) != len(set(page_numbers))
    ):
        raise ValueError(
            "supplementary source transcription pages must be sorted unique "
            "positive integers"
        )
    with tempfile.TemporaryDirectory(prefix="molly-transcription-pdf-") as temp_dir, _pinned_poppler_toolchain(
        poppler_bin_dir,
        execution_root=Path(temp_dir),
    ) as toolchain:
        temp_path = Path(temp_dir)
        pinned_pdf = temp_path / "bound-source.pdf"
        pdf_sha256, pdf_byte_size, pdf_descriptor = _copy_and_hash_pdf(
            source_pdf_path,
            pinned_pdf,
            reject_symlink_components=reject_symlink_components,
        )
        try:
            if pdf_sha256 != expected_sha256_clean:
                raise ValueError(
                    "supplementary source transcription PDF does not match the bound hash"
                )
            pinned_pdf_reference = Path(_descriptor_reference(pdf_descriptor))
            page_counter_version = _poppler_version(
                toolchain=toolchain,
                executable=toolchain.pdfinfo,
                command_name="pdfinfo",
                working_dir=temp_path,
            )
            renderer_version = _poppler_version(
                toolchain=toolchain,
                executable=toolchain.pdftoppm,
                command_name="pdftoppm",
                working_dir=temp_path,
            )
            if page_counter_version != renderer_version:
                raise ValueError(
                    "supplementary source transcription Poppler tool versions differ"
                )
            page_count = _pdf_page_count(
                pinned_pdf_reference,
                pdf_descriptor=pdf_descriptor,
                toolchain=toolchain,
                working_dir=temp_path,
            )
            if any(page > page_count for page in page_numbers):
                raise ValueError(
                    "supplementary source transcription page is out of range"
                )
            page_assets: list[OledSupplementarySourcePageAsset] = []
            rendered_assets: dict[str, bytes] = {}
            for page in page_numbers:
                allowed_widths, allowed_heights = _pdf_page_pixel_dimensions(
                    pinned_pdf_reference,
                    pdf_descriptor=pdf_descriptor,
                    page=page,
                    toolchain=toolchain,
                    working_dir=temp_path,
                )
                rendered = _render_pdf_page(
                    pinned_pdf_reference,
                    pdf_descriptor=pdf_descriptor,
                    page=page,
                    temp_dir=temp_path,
                    toolchain=toolchain,
                    allowed_pixel_widths=allowed_widths,
                    allowed_pixel_heights=allowed_heights,
                )
                width, height = _png_dimensions(rendered)
                rendered_sha256 = f"sha256:{hashlib.sha256(rendered).hexdigest()}"
                asset = build_oled_supplementary_source_page_asset(
                    source_id=source_id,
                    source_pdf_sha256=pdf_sha256,
                    pdf_page_number_one_based=page,
                    renderer_id=SUPPLEMENTARY_SOURCE_TRANSCRIPTION_RENDERER_ID,
                    renderer_version=renderer_version,
                    render_profile=SUPPLEMENTARY_SOURCE_TRANSCRIPTION_RENDER_PROFILE,
                    rendered_asset_sha256=rendered_sha256,
                    rendered_asset_byte_size=len(rendered),
                    pixel_width=width,
                    pixel_height=height,
                )
                page_assets.append(asset)
                rendered_assets[asset.asset_filename] = rendered
            evidence = build_oled_supplementary_source_pdf_evidence(
                source_id=source_id,
                source_pdf_sha256=pdf_sha256,
                source_pdf_byte_size=pdf_byte_size,
                source_pdf_page_count=page_count,
                page_counter_version=page_counter_version,
                page_counter_executable_sha256=toolchain.pdfinfo.sha256,
                renderer_executable_sha256=toolchain.pdftoppm.sha256,
                page_assets=page_assets,
            )
            return evidence, rendered_assets
        finally:
            final_pdf_stat = os.fstat(pdf_descriptor)
            final_pdf_sha256 = _hash_open_descriptor(
                pdf_descriptor,
                max_bytes=_MAX_SOURCE_PDF_BYTES,
            )
            os.close(pdf_descriptor)
            if (
                not stat.S_ISREG(final_pdf_stat.st_mode)
                or final_pdf_stat.st_size != pdf_byte_size
                or final_pdf_sha256 != pdf_sha256
            ):
                raise ValueError(
                    "supplementary source transcription pinned PDF changed during rendering"
                )


def _descriptor_reference(descriptor: int) -> str:
    prefix = "/proc/self/fd" if sys.platform.startswith("linux") else "/dev/fd"
    return f"{prefix}/{descriptor}"


def _copy_and_hash_pdf(
    source: Path,
    destination: Path,
    *,
    reject_symlink_components: bool = False,
) -> tuple[str, int, int]:
    if source.suffix.lower() != ".pdf":
        raise ValueError("supplementary source transcription input must be a PDF")
    no_follow = getattr(os, "O_NOFOLLOW", None)
    directory_flag = getattr(os, "O_DIRECTORY", None)
    if no_follow is None or (reject_symlink_components and directory_flag is None):
        raise ValueError("supplementary source transcription requires O_NOFOLLOW support")
    source_descriptor = -1
    destination_descriptor = -1
    read_only_descriptor = -1
    try:
        if reject_symlink_components:
            assert directory_flag is not None
            source_descriptor = _open_regular_file_without_symlink_components(
                source,
                no_follow=no_follow,
                directory_flag=directory_flag,
            )
        else:
            source_descriptor = os.open(
                source,
                os.O_RDONLY | no_follow | getattr(os, "O_NONBLOCK", 0),
            )
        destination_descriptor = os.open(
            destination,
            os.O_RDWR | os.O_CREAT | os.O_EXCL | no_follow,
            0o600,
        )
        source_stat = os.fstat(source_descriptor)
        if not stat.S_ISREG(source_stat.st_mode):
            raise ValueError(
                "supplementary source transcription PDF must be a regular file"
            )
        if source_stat.st_size <= 0 or source_stat.st_size > _MAX_SOURCE_PDF_BYTES:
            raise ValueError(
                "supplementary source transcription PDF has an unsupported byte size"
            )
        digest = hashlib.sha256()
        header = b""
        trailer = b""
        bytes_read = 0
        while True:
            chunk = os.read(source_descriptor, 1024 * 1024)
            if not chunk:
                break
            if not header:
                header = chunk[:8]
            trailer = (trailer + chunk)[-_PDF_TRAILER_SCAN_BYTES:]
            digest.update(chunk)
            bytes_read += len(chunk)
            if bytes_read > _MAX_SOURCE_PDF_BYTES:
                raise ValueError(
                    "supplementary source transcription PDF exceeds the size limit"
                )
            _write_all(destination_descriptor, chunk)
        os.fsync(destination_descriptor)
        final_stat = os.fstat(source_descriptor)
        if (
            bytes_read != source_stat.st_size
            or final_stat.st_size != source_stat.st_size
            or final_stat.st_mtime_ns != source_stat.st_mtime_ns
            or final_stat.st_ctime_ns != source_stat.st_ctime_ns
        ):
            raise ValueError(
                "supplementary source transcription PDF changed while being read"
            )
        if not header.startswith(b"%PDF-") or b"%%EOF" not in trailer:
            raise ValueError(
                "supplementary source transcription PDF envelope is invalid"
            )
        os.fchmod(destination_descriptor, 0o400)
        os.fsync(destination_descriptor)
        destination_stat = os.fstat(destination_descriptor)
        read_only_descriptor = os.open(
            destination,
            os.O_RDONLY | no_follow | getattr(os, "O_NONBLOCK", 0),
        )
        read_only_stat = os.fstat(read_only_descriptor)
        if (
            not stat.S_ISREG(read_only_stat.st_mode)
            or read_only_stat.st_dev != destination_stat.st_dev
            or read_only_stat.st_ino != destination_stat.st_ino
            or read_only_stat.st_size != bytes_read
            or _hash_open_descriptor(
                read_only_descriptor,
                max_bytes=_MAX_SOURCE_PDF_BYTES,
            )
            != f"sha256:{digest.hexdigest()}"
        ):
            raise ValueError(
                "supplementary source transcription pinned PDF copy changed"
            )
        os.close(destination_descriptor)
        destination_descriptor = -1
        _unlink_path_if_same_inode(
            destination,
            expected_stat=read_only_stat,
            mismatch_message=(
                "supplementary source transcription pinned PDF path changed"
            ),
        )
        pinned_descriptor = read_only_descriptor
        read_only_descriptor = -1
        return f"sha256:{digest.hexdigest()}", bytes_read, pinned_descriptor
    except ValueError:
        raise
    except OSError as exc:
        raise ValueError(
            "supplementary source transcription PDF is unavailable"
        ) from exc
    finally:
        if source_descriptor != -1:
            os.close(source_descriptor)
        if destination_descriptor != -1:
            os.close(destination_descriptor)
        if read_only_descriptor != -1:
            os.close(read_only_descriptor)


def _unlink_path_if_same_inode(
    path: Path,
    *,
    expected_stat: os.stat_result,
    mismatch_message: str,
) -> None:
    no_follow = getattr(os, "O_NOFOLLOW", None)
    directory_flag = getattr(os, "O_DIRECTORY", None)
    if no_follow is None or directory_flag is None:
        raise ValueError(
            "supplementary source transcription requires safe dirfd support"
        )
    parent_descriptor = -1
    try:
        parent_descriptor = os.open(
            path.parent.resolve(strict=True),
            os.O_RDONLY | directory_flag | no_follow,
        )
        named_stat = os.stat(
            path.name,
            dir_fd=parent_descriptor,
            follow_symlinks=False,
        )
        if (
            not stat.S_ISREG(named_stat.st_mode)
            or named_stat.st_dev != expected_stat.st_dev
            or named_stat.st_ino != expected_stat.st_ino
        ):
            raise ValueError(mismatch_message)
        os.unlink(path.name, dir_fd=parent_descriptor)
        os.fsync(parent_descriptor)
    except ValueError:
        raise
    except OSError as exc:
        raise ValueError(mismatch_message) from exc
    finally:
        if parent_descriptor != -1:
            os.close(parent_descriptor)


def _write_all(descriptor: int, payload: bytes) -> None:
    view = memoryview(payload)
    while view:
        written = os.write(descriptor, view)
        if written <= 0:
            raise OSError("short write")
        view = view[written:]


def _pdf_page_count(
    source_pdf: Path,
    *,
    pdf_descriptor: int,
    toolchain: _PinnedPopplerToolchain,
    working_dir: Path,
) -> int:
    result = _run_renderer(
        toolchain,
        toolchain.pdfinfo,
        [str(source_pdf)],
        working_dir=working_dir,
        pass_descriptors=(pdf_descriptor,),
    )
    match = re.search(r"(?m)^Pages:\s+(\d+)\s*$", result.stdout)
    if match is None:
        raise ValueError("supplementary source transcription cannot read PDF page count")
    page_count = int(match.group(1))
    if page_count < 1:
        raise ValueError("supplementary source transcription PDF has no pages")
    return page_count


def _poppler_version(
    toolchain: _PinnedPopplerToolchain,
    executable: _PinnedExecutable,
    *,
    command_name: str,
    working_dir: Path,
) -> str:
    result = _run_renderer(
        toolchain,
        executable,
        ["-v"],
        working_dir=working_dir,
    )
    combined = "\n".join((result.stdout, result.stderr))
    match = re.search(
        rf"{re.escape(command_name)} version\s+([^\s]+)",
        combined,
        re.IGNORECASE,
    )
    if match is None:
        raise ValueError("supplementary source transcription cannot identify renderer")
    version = match.group(1).strip()
    if not re.fullmatch(r"[A-Za-z0-9._-]+", version):
        raise ValueError("supplementary source transcription renderer version is unsafe")
    return version


def _pdf_page_pixel_dimensions(
    source_pdf: Path,
    *,
    pdf_descriptor: int,
    page: int,
    toolchain: _PinnedPopplerToolchain,
    working_dir: Path,
) -> tuple[set[int], set[int]]:
    result = _run_renderer(
        toolchain,
        toolchain.pdfinfo,
        ["-f", str(page), "-l", str(page), "-box", str(source_pdf)],
        working_dir=working_dir,
        pass_descriptors=(pdf_descriptor,),
    )
    media_box_match = re.search(
        rf"(?m)^Page\s+{page}\s+MediaBox:\s+"
        r"(-?(?:\d+(?:\.\d*)?|\.\d+))\s+"
        r"(-?(?:\d+(?:\.\d*)?|\.\d+))\s+"
        r"(-?(?:\d+(?:\.\d*)?|\.\d+))\s+"
        r"(-?(?:\d+(?:\.\d*)?|\.\d+))\s*$",
        result.stdout,
    )
    rotation_match = re.search(
        rf"(?m)^Page\s+{page}\s+rot:\s+(-?\d+)\s*$",
        result.stdout,
    )
    if media_box_match is None or rotation_match is None:
        raise ValueError(
            "supplementary source transcription cannot read PDF page geometry"
        )
    x1, y1, x2, y2 = (float(value) for value in media_box_match.groups())
    width_points = abs(x2 - x1)
    height_points = abs(y2 - y1)
    rotation = int(rotation_match.group(1)) % 360
    if (
        not math.isfinite(width_points)
        or not math.isfinite(height_points)
        or width_points <= 0
        or height_points <= 0
        or rotation not in {0, 90, 180, 270}
    ):
        raise ValueError(
            "supplementary source transcription PDF page geometry is unsafe"
        )
    if rotation in {90, 270}:
        width_points, height_points = height_points, width_points
    allowed_widths = _allowed_200dpi_pixels(width_points)
    allowed_heights = _allowed_200dpi_pixels(height_points)
    if (
        max(allowed_widths) > _MAX_RENDERED_PIXEL_DIMENSION
        or max(allowed_heights) > _MAX_RENDERED_PIXEL_DIMENSION
        or max(allowed_widths) * max(allowed_heights) > _MAX_RENDERED_PIXEL_COUNT
    ):
        raise ValueError(
            "supplementary source transcription PDF page is too large to render"
        )
    return allowed_widths, allowed_heights


def _allowed_200dpi_pixels(points: float) -> set[int]:
    exact = points * 200.0 / 72.0
    rounded = {math.floor(exact), round(exact), math.ceil(exact)}
    return {value for value in rounded if value >= 1}


def _render_pdf_page(
    source_pdf: Path,
    *,
    pdf_descriptor: int,
    page: int,
    temp_dir: Path,
    toolchain: _PinnedPopplerToolchain,
    allowed_pixel_widths: set[int],
    allowed_pixel_heights: set[int],
) -> bytes:
    output_prefix = temp_dir / f"page-{page:06d}"
    _run_renderer(
        toolchain,
        toolchain.pdftoppm,
        [
            "-f",
            str(page),
            "-l",
            str(page),
            "-r",
            "200",
            "-png",
            "-singlefile",
            str(source_pdf),
            str(output_prefix),
        ],
        working_dir=temp_dir,
        pass_descriptors=(pdf_descriptor,),
    )
    output_path = output_prefix.with_suffix(".png")
    try:
        rendered = _read_bound_binary_path(
            output_path,
            max_bytes=_MAX_RENDERED_PNG_BYTES,
        )
    except FileNotFoundError as exc:
        raise ValueError(
            "supplementary source transcription renderer produced no PNG"
        ) from exc
    width, height = _png_dimensions(rendered)
    if width not in allowed_pixel_widths or height not in allowed_pixel_heights:
        raise ValueError(
            "supplementary source transcription renderer did not produce a full-page 200 dpi image"
        )
    return rendered


def _read_bound_binary_path(path: Path, *, max_bytes: int) -> bytes:
    no_follow = getattr(os, "O_NOFOLLOW", None)
    if no_follow is None:
        raise ValueError("supplementary source transcription requires O_NOFOLLOW support")
    descriptor = -1
    try:
        descriptor = os.open(
            path,
            os.O_RDONLY | no_follow | getattr(os, "O_NONBLOCK", 0),
        )
        initial_stat = os.fstat(descriptor)
        if not stat.S_ISREG(initial_stat.st_mode):
            raise ValueError(
                "supplementary source transcription renderer output must be a regular file"
            )
        if initial_stat.st_size <= 0 or initial_stat.st_size > max_bytes:
            raise ValueError(
                "supplementary source transcription renderer output has an unsupported size"
            )
        payload = bytearray()
        while len(payload) <= max_bytes:
            chunk = os.read(
                descriptor,
                min(1024 * 1024, max_bytes + 1 - len(payload)),
            )
            if not chunk:
                break
            payload.extend(chunk)
        final_stat = os.fstat(descriptor)
        if (
            len(payload) != initial_stat.st_size
            or final_stat.st_size != initial_stat.st_size
            or final_stat.st_mtime_ns != initial_stat.st_mtime_ns
            or final_stat.st_ctime_ns != initial_stat.st_ctime_ns
        ):
            raise ValueError(
                "supplementary source transcription renderer output changed while being read"
            )
        return bytes(payload)
    except FileNotFoundError:
        raise
    except ValueError:
        raise
    except OSError as exc:
        raise ValueError(
            "supplementary source transcription renderer output is unavailable"
        ) from exc
    finally:
        if descriptor != -1:
            os.close(descriptor)


def _run_renderer(
    toolchain: _PinnedPopplerToolchain,
    executable: _PinnedExecutable,
    arguments: Sequence[str],
    *,
    working_dir: Path,
    pass_descriptors: Sequence[int] = (),
) -> subprocess.CompletedProcess[str]:
    _validate_pinned_toolchain(toolchain)
    command = [str(executable.path), *arguments]
    process: subprocess.Popen[bytes] | None = None
    try:
        with tempfile.TemporaryFile(dir=working_dir) as stdout_file, tempfile.TemporaryFile(
            dir=working_dir
        ) as stderr_file:
            process = subprocess.Popen(
                command,
                stdin=subprocess.DEVNULL,
                stdout=stdout_file,
                stderr=stderr_file,
                cwd=working_dir,
                env=_renderer_environment(toolchain, working_dir),
                pass_fds=tuple(pass_descriptors),
                start_new_session=True,
                preexec_fn=_apply_renderer_resource_limits,
            )
            try:
                return_code = process.wait(timeout=_RENDER_TIMEOUT_SECONDS)
            except subprocess.TimeoutExpired as exc:
                _kill_renderer_process_group(process.pid)
                process.wait()
                raise ValueError(
                    "supplementary source transcription renderer timed out"
                ) from exc
            finally:
                _kill_renderer_process_group(process.pid)
            stdout = _read_limited_renderer_log(stdout_file)
            stderr = _read_limited_renderer_log(stderr_file)
    except ValueError:
        raise
    except OSError as exc:
        raise ValueError(
            "supplementary source transcription renderer is unavailable"
        ) from exc
    finally:
        _validate_pinned_toolchain(toolchain)
    if return_code != 0:
        raise ValueError("supplementary source transcription renderer failed")
    return subprocess.CompletedProcess(
        args=command,
        returncode=return_code,
        stdout=stdout,
        stderr=stderr,
    )


def _renderer_environment(
    toolchain: _PinnedPopplerToolchain,
    working_dir: Path,
) -> dict[str, str]:
    environment = {
        "HOME": str(working_dir),
        "LANG": "C",
        "LC_ALL": "C",
        "PATH": "/usr/bin:/bin",
        "TMPDIR": str(working_dir),
    }
    if sys.platform == "darwin":
        library_dir = (toolchain.bin_dir.parent / "lib").resolve(strict=False)
        if library_dir.is_dir():
            environment["DYLD_FALLBACK_LIBRARY_PATH"] = str(library_dir)
    return environment


def _apply_renderer_resource_limits() -> None:
    _cap_resource(resource.RLIMIT_CPU, _MAX_RENDER_CPU_SECONDS)
    _cap_resource(resource.RLIMIT_FSIZE, _MAX_RENDERED_PNG_BYTES)
    if sys.platform != "darwin" and hasattr(resource, "RLIMIT_AS"):
        _cap_resource(resource.RLIMIT_AS, _MAX_RENDER_ADDRESS_SPACE_BYTES)


def _cap_resource(resource_name: int, requested_limit: int) -> None:
    soft, hard = resource.getrlimit(resource_name)
    effective = requested_limit if hard == resource.RLIM_INFINITY else min(
        requested_limit,
        hard,
    )
    resource.setrlimit(resource_name, (effective, effective))


def _kill_renderer_process_group(process_id: int) -> None:
    try:
        os.killpg(process_id, signal.SIGKILL)
    except ProcessLookupError:
        pass


def _read_limited_renderer_log(handle: Any) -> str:
    file_size = os.fstat(handle.fileno()).st_size
    if file_size > _MAX_RENDER_LOG_BYTES:
        raise ValueError(
            "supplementary source transcription renderer log exceeded the size limit"
        )
    handle.seek(0)
    payload = handle.read(_MAX_RENDER_LOG_BYTES + 1)
    if len(payload) > _MAX_RENDER_LOG_BYTES:
        raise ValueError(
            "supplementary source transcription renderer log exceeded the size limit"
        )
    return payload.decode("utf-8", errors="replace")


def _png_dimensions(payload: bytes) -> tuple[int, int]:
    if len(payload) < 33 or not payload.startswith(_PNG_SIGNATURE):
        raise ValueError("supplementary source transcription asset is not a PNG")
    if payload[8:12] != struct.pack(">I", 13) or payload[12:16] != b"IHDR":
        raise ValueError("supplementary source transcription PNG lacks IHDR")
    width, height = struct.unpack(">II", payload[16:24])
    if (
        width < 1
        or height < 1
        or width > _MAX_RENDERED_PIXEL_DIMENSION
        or height > _MAX_RENDERED_PIXEL_DIMENSION
        or width * height > _MAX_RENDERED_PIXEL_COUNT
    ):
        raise ValueError("supplementary source transcription PNG dimensions are unsafe")
    if payload[24] != 8 or payload[25] != 2:
        raise ValueError(
            "supplementary source transcription PNG must be 8-bit RGB"
        )
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(io.BytesIO(payload)) as image:
                if image.format != "PNG" or image.mode != "RGB":
                    raise ValueError(
                        "supplementary source transcription PNG must decode as RGB"
                    )
                if image.size != (width, height):
                    raise ValueError(
                        "supplementary source transcription PNG dimensions changed"
                    )
                image.verify()
            with Image.open(io.BytesIO(payload)) as image:
                image.load()
                if image.mode != "RGB" or image.size != (width, height):
                    raise ValueError(
                        "supplementary source transcription PNG decoding changed"
                    )
    except ValueError:
        raise
    except (Image.DecompressionBombError, UnidentifiedImageError, OSError) as exc:
        raise ValueError(
            "supplementary source transcription PNG is invalid"
        ) from exc
    return width, height


def _validate_packet_outputs(
    *,
    output_path: Path,
    asset_dir: Path,
    protected_paths: set[Path],
) -> None:
    _validate_fresh_output(output_path, protected_paths=protected_paths)
    if asset_dir.exists() or asset_dir.is_symlink():
        raise ValueError("supplementary source transcription asset directory must be fresh")
    if not asset_dir.parent.exists() or not asset_dir.parent.is_dir():
        raise ValueError(
            "supplementary source transcription asset parent must be a directory"
        )
    if any(_paths_collide(asset_dir, path) for path in protected_paths | {output_path}):
        raise ValueError(
            "supplementary source transcription asset directory collides with an input"
        )
    if _path_is_within(output_path, asset_dir):
        raise ValueError(
            "supplementary source transcription packet must be outside the asset directory"
        )


def _validate_pinned_directory_path_without_symlinks(
    path: Path,
    pinned_descriptor: int,
    *,
    error_message: str,
) -> None:
    """Require every named path component to reach one pinned directory inode."""

    no_follow = getattr(os, "O_NOFOLLOW", None)
    directory_flag = getattr(os, "O_DIRECTORY", None)
    if no_follow is None or directory_flag is None or not path.is_absolute():
        raise ValueError(error_message)
    descriptor = -1
    try:
        descriptor = os.open(
            path.anchor,
            os.O_RDONLY | directory_flag | no_follow,
        )
        for component in path.parts[1:]:
            next_descriptor = os.open(
                component,
                os.O_RDONLY | directory_flag | no_follow,
                dir_fd=descriptor,
            )
            os.close(descriptor)
            descriptor = next_descriptor
        named_stat = os.fstat(descriptor)
        pinned_stat = os.fstat(pinned_descriptor)
        if (
            not stat.S_ISDIR(named_stat.st_mode)
            or not stat.S_ISDIR(pinned_stat.st_mode)
            or named_stat.st_dev != pinned_stat.st_dev
            or named_stat.st_ino != pinned_stat.st_ino
        ):
            raise ValueError(error_message)
    except ValueError:
        raise
    except OSError as exc:
        raise ValueError(error_message) from exc
    finally:
        if descriptor != -1:
            os.close(descriptor)


def _publish_packet_text(
    path: Path,
    content: str,
    *,
    post_publish_validator: Callable[[], None],
    pinned_parent_descriptor: int | None = None,
) -> None:
    no_follow = getattr(os, "O_NOFOLLOW", None)
    directory_flag = getattr(os, "O_DIRECTORY", None)
    if no_follow is None or directory_flag is None:
        raise ValueError(
            "supplementary source transcription requires safe dirfd support"
        )
    parent_descriptor = -1
    temp_descriptor = -1
    temp_name = ""
    output_link_created = False
    keep_output = False
    try:
        if pinned_parent_descriptor is None:
            path.parent.mkdir(parents=True, exist_ok=True)
            resolved_parent = path.parent.resolve(strict=True)
            parent_descriptor = os.open(
                resolved_parent,
                os.O_RDONLY | directory_flag | no_follow,
            )
        else:
            parent_descriptor = os.dup(pinned_parent_descriptor)
        parent_stat = os.fstat(parent_descriptor)
        if not stat.S_ISDIR(parent_stat.st_mode):
            raise ValueError(
                "supplementary source transcription packet parent changed"
            )
        if pinned_parent_descriptor is not None:
            _validate_pinned_directory_path_without_symlinks(
                path.parent,
                parent_descriptor,
                error_message=(
                    "supplementary source transcription packet parent changed"
                ),
            )
        try:
            os.stat(path.name, dir_fd=parent_descriptor, follow_symlinks=False)
        except FileNotFoundError:
            pass
        else:
            raise ValueError(
                "supplementary source transcription packet output must be fresh"
            )
        for _ in range(32):
            temp_name = f".{path.name}.{secrets.token_hex(12)}.tmp"
            try:
                temp_descriptor = os.open(
                    temp_name,
                    os.O_RDWR | os.O_CREAT | os.O_EXCL | no_follow,
                    0o600,
                    dir_fd=parent_descriptor,
                )
                break
            except FileExistsError:
                continue
        if temp_descriptor == -1:
            raise ValueError(
                "supplementary source transcription cannot allocate packet output"
            )
        encoded = content.encode("utf-8")
        _write_all(temp_descriptor, encoded)
        os.fsync(temp_descriptor)
        output_stat = os.fstat(temp_descriptor)
        if (
            not stat.S_ISREG(output_stat.st_mode)
            or output_stat.st_size != len(encoded)
            or output_stat.st_size <= 0
            or output_stat.st_size > _MAX_TRANSCRIPTION_REVIEW_PACKET_BYTES
        ):
            raise ValueError(
                "supplementary source transcription packet output is invalid"
            )
        os.link(
            temp_name,
            path.name,
            src_dir_fd=parent_descriptor,
            dst_dir_fd=parent_descriptor,
            follow_symlinks=False,
        )
        output_link_created = True
        os.fsync(parent_descriptor)
        post_publish_validator()
        current_output_stat = os.stat(
            path.name,
            dir_fd=parent_descriptor,
            follow_symlinks=False,
        )
        if pinned_parent_descriptor is None:
            current_parent_stat = os.stat(path.parent)
        else:
            _validate_pinned_directory_path_without_symlinks(
                path.parent,
                parent_descriptor,
                error_message=(
                    "supplementary source transcription packet parent changed"
                ),
            )
            current_parent_stat = os.fstat(parent_descriptor)
        if (
            not stat.S_ISREG(current_output_stat.st_mode)
            or current_output_stat.st_dev != output_stat.st_dev
            or current_output_stat.st_ino != output_stat.st_ino
            or current_output_stat.st_size != output_stat.st_size
            or not stat.S_ISDIR(current_parent_stat.st_mode)
            or current_parent_stat.st_dev != parent_stat.st_dev
            or current_parent_stat.st_ino != parent_stat.st_ino
        ):
            raise ValueError(
                "supplementary source transcription packet publication changed"
            )
        if (
            _hash_open_descriptor(
                temp_descriptor,
                max_bytes=_MAX_TRANSCRIPTION_REVIEW_PACKET_BYTES,
            )
            != f"sha256:{hashlib.sha256(encoded).hexdigest()}"
        ):
            raise ValueError(
                "supplementary source transcription packet bytes changed"
            )
        post_publish_validator()
        final_output_stat = os.stat(
            path.name,
            dir_fd=parent_descriptor,
            follow_symlinks=False,
        )
        if pinned_parent_descriptor is None:
            final_parent_stat = os.stat(path.parent)
        else:
            _validate_pinned_directory_path_without_symlinks(
                path.parent,
                parent_descriptor,
                error_message=(
                    "supplementary source transcription packet parent changed"
                ),
            )
            final_parent_stat = os.fstat(parent_descriptor)
        if (
            not stat.S_ISREG(final_output_stat.st_mode)
            or final_output_stat.st_dev != output_stat.st_dev
            or final_output_stat.st_ino != output_stat.st_ino
            or final_output_stat.st_size != output_stat.st_size
            or not stat.S_ISDIR(final_parent_stat.st_mode)
            or final_parent_stat.st_dev != parent_stat.st_dev
            or final_parent_stat.st_ino != parent_stat.st_ino
            or _hash_open_descriptor(
                temp_descriptor,
                max_bytes=_MAX_TRANSCRIPTION_REVIEW_PACKET_BYTES,
            )
            != f"sha256:{hashlib.sha256(encoded).hexdigest()}"
        ):
            raise ValueError(
                "supplementary source transcription packet changed at commit"
            )
        keep_output = True
    except FileExistsError as exc:
        raise ValueError(
            "supplementary source transcription packet output must be fresh"
        ) from exc
    except ValueError:
        raise
    except OSError as exc:
        raise ValueError(
            "supplementary source transcription packet cannot be published"
        ) from exc
    finally:
        if parent_descriptor != -1:
            if output_link_created and not keep_output:
                try:
                    current_output_stat = os.stat(
                        path.name,
                        dir_fd=parent_descriptor,
                        follow_symlinks=False,
                    )
                    if (
                        temp_descriptor != -1
                        and current_output_stat.st_dev
                        == os.fstat(temp_descriptor).st_dev
                        and current_output_stat.st_ino
                        == os.fstat(temp_descriptor).st_ino
                    ):
                        os.unlink(path.name, dir_fd=parent_descriptor)
                        os.fsync(parent_descriptor)
                except OSError:
                    pass
            if temp_name:
                try:
                    current_temp_stat = os.stat(
                        temp_name,
                        dir_fd=parent_descriptor,
                        follow_symlinks=False,
                    )
                    if (
                        temp_descriptor != -1
                        and current_temp_stat.st_dev
                        == os.fstat(temp_descriptor).st_dev
                        and current_temp_stat.st_ino
                        == os.fstat(temp_descriptor).st_ino
                    ):
                        os.unlink(temp_name, dir_fd=parent_descriptor)
                        os.fsync(parent_descriptor)
                except OSError:
                    pass
        if temp_descriptor != -1:
            os.close(temp_descriptor)
        if parent_descriptor != -1:
            os.close(parent_descriptor)


def _validate_published_packet_paths(
    *,
    output_path: Path,
    asset_dir: Path,
    protected_paths: set[Path],
) -> None:
    if output_path.is_symlink() or not output_path.is_file():
        raise ValueError(
            "supplementary source transcription published packet is invalid"
        )
    if _path_is_within(output_path, asset_dir):
        raise ValueError(
            "supplementary source transcription packet was published inside assets"
        )
    if any(_paths_collide(output_path, path) for path in protected_paths):
        raise ValueError(
            "supplementary source transcription packet overwrote a protected input"
        )


def _create_private_asset_directory(
    asset_dir: Path,
    *,
    pinned_parent_descriptor: int | None = None,
) -> tuple[int, int, os.stat_result]:
    no_follow = getattr(os, "O_NOFOLLOW", None)
    directory_flag = getattr(os, "O_DIRECTORY", None)
    if no_follow is None or directory_flag is None:
        raise ValueError(
            "supplementary source transcription requires safe dirfd support"
        )
    if asset_dir.name in {"", ".", ".."}:
        raise ValueError(
            "supplementary source transcription asset directory name is invalid"
        )
    parent_descriptor = -1
    asset_descriptor = -1
    asset_directory_stat: os.stat_result | None = None
    try:
        if pinned_parent_descriptor is None:
            resolved_parent = asset_dir.parent.resolve(strict=True)
            parent_descriptor = os.open(
                resolved_parent,
                os.O_RDONLY | directory_flag | no_follow,
            )
        else:
            parent_descriptor = os.dup(pinned_parent_descriptor)
        parent_stat = os.fstat(parent_descriptor)
        if pinned_parent_descriptor is None:
            current_parent_stat = os.stat(asset_dir.parent)
        else:
            _validate_pinned_directory_path_without_symlinks(
                asset_dir.parent,
                parent_descriptor,
                error_message=(
                    "supplementary source transcription asset parent changed"
                ),
            )
            current_parent_stat = os.fstat(parent_descriptor)
        if (
            not stat.S_ISDIR(parent_stat.st_mode)
            or not stat.S_ISDIR(current_parent_stat.st_mode)
            or current_parent_stat.st_dev != parent_stat.st_dev
            or current_parent_stat.st_ino != parent_stat.st_ino
        ):
            raise ValueError(
                "supplementary source transcription asset parent changed"
            )
        try:
            os.stat(
                asset_dir.name,
                dir_fd=parent_descriptor,
                follow_symlinks=False,
            )
        except FileNotFoundError:
            pass
        else:
            raise ValueError(
                "supplementary source transcription asset directory must be fresh"
            )
        os.mkdir(asset_dir.name, mode=0o700, dir_fd=parent_descriptor)
        asset_directory_stat = os.stat(
            asset_dir.name,
            dir_fd=parent_descriptor,
            follow_symlinks=False,
        )
        asset_descriptor = os.open(
            asset_dir.name,
            os.O_RDONLY | directory_flag | no_follow,
            dir_fd=parent_descriptor,
        )
        opened_asset_stat = os.fstat(asset_descriptor)
        if (
            not stat.S_ISDIR(asset_directory_stat.st_mode)
            or not stat.S_ISDIR(opened_asset_stat.st_mode)
            or opened_asset_stat.st_dev != asset_directory_stat.st_dev
            or opened_asset_stat.st_ino != asset_directory_stat.st_ino
            or asset_directory_stat.st_mode & 0o077
        ):
            raise ValueError(
                "supplementary source transcription asset directory is not private"
            )
        os.fsync(parent_descriptor)
        _validate_open_asset_directory_binding(
            asset_dir=asset_dir,
            parent_descriptor=parent_descriptor,
            asset_descriptor=asset_descriptor,
            asset_directory_stat=asset_directory_stat,
            reject_symlink_components=pinned_parent_descriptor is not None,
        )
        return parent_descriptor, asset_descriptor, asset_directory_stat
    except ValueError:
        if asset_directory_stat is not None:
            _remove_private_asset_directory(
                asset_dir=asset_dir,
                parent_descriptor=parent_descriptor,
                asset_descriptor=asset_descriptor,
                asset_directory_stat=asset_directory_stat,
            )
        if asset_descriptor != -1:
            os.close(asset_descriptor)
        if parent_descriptor != -1:
            os.close(parent_descriptor)
        raise
    except OSError as exc:
        if asset_directory_stat is not None:
            _remove_private_asset_directory(
                asset_dir=asset_dir,
                parent_descriptor=parent_descriptor,
                asset_descriptor=asset_descriptor,
                asset_directory_stat=asset_directory_stat,
            )
        if asset_descriptor != -1:
            os.close(asset_descriptor)
        if parent_descriptor != -1:
            os.close(parent_descriptor)
        raise ValueError(
            "supplementary source transcription asset directory cannot be created"
        ) from exc
    except BaseException:
        if asset_directory_stat is not None:
            _remove_private_asset_directory(
                asset_dir=asset_dir,
                parent_descriptor=parent_descriptor,
                asset_descriptor=asset_descriptor,
                asset_directory_stat=asset_directory_stat,
            )
        if asset_descriptor != -1:
            os.close(asset_descriptor)
        if parent_descriptor != -1:
            os.close(parent_descriptor)
        raise


def _validate_open_asset_directory_binding(
    *,
    asset_dir: Path,
    parent_descriptor: int,
    asset_descriptor: int,
    asset_directory_stat: os.stat_result,
    reject_symlink_components: bool = False,
) -> None:
    try:
        open_stat = os.fstat(asset_descriptor)
        named_stat = os.stat(
            asset_dir.name,
            dir_fd=parent_descriptor,
            follow_symlinks=False,
        )
        parent_stat = os.fstat(parent_descriptor)
        if reject_symlink_components:
            _validate_pinned_directory_path_without_symlinks(
                asset_dir.parent,
                parent_descriptor,
                error_message=(
                    "supplementary source transcription asset directory changed"
                ),
            )
            current_parent_stat = os.fstat(parent_descriptor)
        else:
            current_parent_stat = os.stat(asset_dir.parent)
    except OSError as exc:
        raise ValueError(
            "supplementary source transcription asset directory binding is unavailable"
        ) from exc
    if (
        not stat.S_ISDIR(open_stat.st_mode)
        or not stat.S_ISDIR(named_stat.st_mode)
        or open_stat.st_dev != asset_directory_stat.st_dev
        or open_stat.st_ino != asset_directory_stat.st_ino
        or named_stat.st_dev != asset_directory_stat.st_dev
        or named_stat.st_ino != asset_directory_stat.st_ino
        or not stat.S_ISDIR(parent_stat.st_mode)
        or not stat.S_ISDIR(current_parent_stat.st_mode)
        or current_parent_stat.st_dev != parent_stat.st_dev
        or current_parent_stat.st_ino != parent_stat.st_ino
    ):
        raise ValueError(
            "supplementary source transcription asset directory changed"
        )


def _write_asset_bundle(
    asset_descriptor: int,
    packet: OledSupplementarySourceTranscriptionReviewPacket,
    rendered_assets: dict[str, bytes],
    *,
    created_asset_stats: dict[str, os.stat_result],
) -> None:
    expected = {
        asset.asset_filename: asset for asset in packet.source_pdf_evidence.page_assets
    }
    if set(expected) != set(rendered_assets):
        raise ValueError("supplementary source transcription rendered asset coverage mismatch")
    for filename in sorted(expected):
        asset = expected[filename]
        payload = rendered_assets[filename]
        if (
            len(payload) != asset.rendered_asset_byte_size
            or f"sha256:{hashlib.sha256(payload).hexdigest()}"
            != asset.rendered_asset_sha256
            or _png_dimensions(payload) != (asset.pixel_width, asset.pixel_height)
        ):
            raise ValueError("supplementary source transcription rendered asset mismatch")
        created_asset_stats[filename] = _write_fresh_bytes_at(
            asset_descriptor,
            filename,
            payload,
        )
    os.fsync(asset_descriptor)


def _validate_open_created_asset_bundle(
    *,
    asset_descriptor: int,
    packet: OledSupplementarySourceTranscriptionReviewPacket,
    rendered_assets: dict[str, bytes],
    created_asset_stats: dict[str, os.stat_result],
) -> None:
    expected = {
        asset.asset_filename: asset for asset in packet.source_pdf_evidence.page_assets
    }
    if (
        set(expected) != set(rendered_assets)
        or set(expected) != set(created_asset_stats)
    ):
        raise ValueError(
            "supplementary source transcription published asset coverage mismatch"
        )
    initial_directory_stat = os.fstat(asset_descriptor)
    if set(os.listdir(asset_descriptor)) != set(expected):
        raise ValueError(
            "supplementary source transcription published asset coverage changed"
        )
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
            raise ValueError(
                "supplementary source transcription published asset inode changed"
            )
        payload = _read_bound_binary_at(
            asset_descriptor,
            filename,
            max_bytes=_MAX_RENDERED_PNG_BYTES,
        )
        if (
            payload != rendered_assets[filename]
            or len(payload) != asset.rendered_asset_byte_size
            or f"sha256:{hashlib.sha256(payload).hexdigest()}"
            != asset.rendered_asset_sha256
            or _png_dimensions(payload) != (asset.pixel_width, asset.pixel_height)
        ):
            raise ValueError(
                "supplementary source transcription published asset bytes changed"
            )
    final_directory_stat = os.fstat(asset_descriptor)
    if (
        final_directory_stat.st_dev != initial_directory_stat.st_dev
        or final_directory_stat.st_ino != initial_directory_stat.st_ino
        or final_directory_stat.st_mtime_ns != initial_directory_stat.st_mtime_ns
        or final_directory_stat.st_ctime_ns != initial_directory_stat.st_ctime_ns
    ):
        raise ValueError(
            "supplementary source transcription published asset directory changed"
        )


def _write_fresh_bytes_at(
    directory_descriptor: int,
    filename: str,
    payload: bytes,
) -> os.stat_result:
    no_follow = getattr(os, "O_NOFOLLOW", None)
    if no_follow is None:
        raise ValueError("supplementary source transcription requires O_NOFOLLOW support")
    if Path(filename).name != filename or filename in {"", ".", ".."}:
        raise ValueError("supplementary source transcription asset filename is invalid")
    descriptor = -1
    created_stat: os.stat_result | None = None
    keep_file = False
    try:
        descriptor = os.open(
            filename,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | no_follow,
            0o600,
            dir_fd=directory_descriptor,
        )
        created_stat = os.fstat(descriptor)
        _write_all(descriptor, payload)
        os.fsync(descriptor)
        final_stat = os.fstat(descriptor)
        keep_file = True
        return final_stat
    except FileExistsError as exc:
        raise ValueError(
            "supplementary source transcription asset must be fresh"
        ) from exc
    except OSError as exc:
        raise ValueError(
            "supplementary source transcription asset cannot be written"
        ) from exc
    finally:
        if descriptor != -1:
            os.close(descriptor)
        if not keep_file and created_stat is not None:
            try:
                named_stat = os.stat(
                    filename,
                    dir_fd=directory_descriptor,
                    follow_symlinks=False,
                )
                if (
                    stat.S_ISREG(named_stat.st_mode)
                    and named_stat.st_dev == created_stat.st_dev
                    and named_stat.st_ino == created_stat.st_ino
                ):
                    os.unlink(filename, dir_fd=directory_descriptor)
                    os.fsync(directory_descriptor)
            except OSError:
                pass


def _validate_asset_bundle(
    asset_dir: Path,
    packet: OledSupplementarySourceTranscriptionReviewPacket,
    *,
    expected_bytes: dict[str, bytes] | None = None,
) -> None:
    no_follow = getattr(os, "O_NOFOLLOW", None)
    directory_flag = getattr(os, "O_DIRECTORY", None)
    if no_follow is None or directory_flag is None:
        raise ValueError(
            "supplementary source transcription requires safe dirfd support"
        )
    expected = {
        asset.asset_filename: asset for asset in packet.source_pdf_evidence.page_assets
    }
    if expected_bytes is not None and set(expected_bytes) != set(expected):
        raise ValueError(
            "supplementary source transcription rerendered asset coverage mismatch"
        )
    directory_descriptor = -1
    try:
        directory_descriptor = os.open(
            asset_dir,
            os.O_RDONLY | directory_flag | no_follow,
        )
        initial_stat = os.fstat(directory_descriptor)
        if not stat.S_ISDIR(initial_stat.st_mode):
            raise ValueError(
                "supplementary source transcription asset directory is invalid"
            )
        observed_names = set(os.listdir(directory_descriptor))
        if observed_names != set(expected):
            raise ValueError(
                "supplementary source transcription asset coverage mismatch"
            )
        for filename, asset in expected.items():
            payload = _read_bound_binary_at(
                directory_descriptor,
                filename,
                max_bytes=_MAX_RENDERED_PNG_BYTES,
            )
            if (
                len(payload) != asset.rendered_asset_byte_size
                or f"sha256:{hashlib.sha256(payload).hexdigest()}"
                != asset.rendered_asset_sha256
                or _png_dimensions(payload) != (asset.pixel_width, asset.pixel_height)
            ):
                raise ValueError(
                    "supplementary source transcription asset binding mismatch"
                )
            if expected_bytes is not None and payload != expected_bytes[filename]:
                raise ValueError(
                    "supplementary source transcription rendered page changed"
                )
        final_stat = os.fstat(directory_descriptor)
        current_stat = os.stat(asset_dir, follow_symlinks=False)
        if (
            not stat.S_ISDIR(current_stat.st_mode)
            or final_stat.st_dev != initial_stat.st_dev
            or final_stat.st_ino != initial_stat.st_ino
            or final_stat.st_mtime_ns != initial_stat.st_mtime_ns
            or final_stat.st_ctime_ns != initial_stat.st_ctime_ns
            or current_stat.st_dev != initial_stat.st_dev
            or current_stat.st_ino != initial_stat.st_ino
        ):
            raise ValueError(
                "supplementary source transcription asset directory changed while being read"
            )
    except ValueError:
        raise
    except OSError as exc:
        raise ValueError(
            "supplementary source transcription asset directory is unavailable"
        ) from exc
    finally:
        if directory_descriptor != -1:
            os.close(directory_descriptor)


def _read_bound_binary_at(
    directory_descriptor: int,
    filename: str,
    *,
    max_bytes: int,
) -> bytes:
    no_follow = getattr(os, "O_NOFOLLOW", None)
    if no_follow is None:
        raise ValueError("supplementary source transcription requires O_NOFOLLOW support")
    descriptor = -1
    try:
        descriptor = os.open(
            filename,
            os.O_RDONLY | no_follow | getattr(os, "O_NONBLOCK", 0),
            dir_fd=directory_descriptor,
        )
        initial_stat = os.fstat(descriptor)
        if not stat.S_ISREG(initial_stat.st_mode):
            raise ValueError(
                "supplementary source transcription asset must be a regular file"
            )
        if initial_stat.st_size <= 0 or initial_stat.st_size > max_bytes:
            raise ValueError(
                "supplementary source transcription asset has an unsupported size"
            )
        payload = bytearray()
        while len(payload) <= max_bytes:
            chunk = os.read(descriptor, min(1024 * 1024, max_bytes + 1 - len(payload)))
            if not chunk:
                break
            payload.extend(chunk)
        final_stat = os.fstat(descriptor)
        if (
            len(payload) != initial_stat.st_size
            or final_stat.st_size != initial_stat.st_size
            or final_stat.st_mtime_ns != initial_stat.st_mtime_ns
            or final_stat.st_ctime_ns != initial_stat.st_ctime_ns
        ):
            raise ValueError(
                "supplementary source transcription asset changed while being read"
            )
        return bytes(payload)
    except ValueError:
        raise
    except OSError as exc:
        raise ValueError(
            "supplementary source transcription asset is unavailable"
        ) from exc
    finally:
        if descriptor != -1:
            os.close(descriptor)


def _validate_review_asset_location(*, asset_dir: Path, output_markdown: Path) -> None:
    if asset_dir.name != "assets" or asset_dir.parent != output_markdown.parent:
        raise ValueError(
            "supplementary source transcription Markdown requires a sibling assets directory"
        )


def _paths_collide(left: Path, right: Path) -> bool:
    try:
        return left.resolve(strict=left.exists()) == right.resolve(strict=right.exists())
    except OSError as exc:
        raise ValueError(
            "supplementary source transcription path cannot be resolved safely"
        ) from exc


def _path_is_within(path: Path, directory: Path) -> bool:
    try:
        return path.resolve(strict=False).is_relative_to(directory.resolve(strict=False))
    except OSError as exc:
        raise ValueError(
            "supplementary source transcription path cannot be resolved safely"
        ) from exc


def _remove_private_asset_directory(
    *,
    asset_dir: Path,
    parent_descriptor: int,
    asset_descriptor: int,
    asset_directory_stat: os.stat_result,
    created_asset_stats: dict[str, os.stat_result] | None = None,
) -> None:
    try:
        if parent_descriptor == -1:
            return
        named_stat = os.stat(
            asset_dir.name,
            dir_fd=parent_descriptor,
            follow_symlinks=False,
        )
        if (
            not stat.S_ISDIR(named_stat.st_mode)
            or named_stat.st_dev != asset_directory_stat.st_dev
            or named_stat.st_ino != asset_directory_stat.st_ino
        ):
            return
        if asset_descriptor != -1:
            for filename, created_stat in sorted(
                (created_asset_stats or {}).items()
            ):
                try:
                    file_stat = os.stat(
                        filename,
                        dir_fd=asset_descriptor,
                        follow_symlinks=False,
                    )
                    if (
                        stat.S_ISREG(file_stat.st_mode)
                        and file_stat.st_dev == created_stat.st_dev
                        and file_stat.st_ino == created_stat.st_ino
                    ):
                        os.unlink(filename, dir_fd=asset_descriptor)
                except FileNotFoundError:
                    pass
            os.fsync(asset_descriptor)
        named_stat = os.stat(
            asset_dir.name,
            dir_fd=parent_descriptor,
            follow_symlinks=False,
        )
        if (
            stat.S_ISDIR(named_stat.st_mode)
            and named_stat.st_dev == asset_directory_stat.st_dev
            and named_stat.st_ino == asset_directory_stat.st_ino
        ):
            os.rmdir(asset_dir.name, dir_fd=parent_descriptor)
            os.fsync(parent_descriptor)
    except OSError:
        pass


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build, render, and adjudicate exact-bound supplementary source-transcription "
            "review packets without changing scientific data or writing datasets."
        )
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    packet = subparsers.add_parser(
        "packet",
        help="render exact source pages and build one table-level review item per scope",
    )
    _add_upstream_arguments(packet)
    packet.add_argument("--source-pdf", required=True)
    _add_poppler_argument(packet)
    packet.add_argument("--asset-dir", required=True)
    packet.add_argument("--output", required=True)
    render = subparsers.add_parser(
        "render",
        help="render a validated reviewer-facing Markdown packet",
    )
    render.add_argument("--review-packet", required=True)
    render.add_argument("--asset-dir", required=True)
    render.add_argument("--output-markdown", required=True)
    adjudicate = subparsers.add_parser(
        "adjudicate",
        help="apply a complete exact-bound table-transcription decision manifest",
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
    parser.add_argument("--response-manifest", required=True)
    parser.add_argument("--response-artifact", required=True)
    parser.add_argument("--semantic-review-packet", required=True)
    parser.add_argument("--semantic-decision-manifest", required=True)
    parser.add_argument("--semantic-adjudication", required=True)


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
            packet = build_oled_supplementary_source_transcription_review_packet_from_files(
                request_artifact_json=args.request_artifact,
                response_manifest_json=args.response_manifest,
                response_artifact_json=args.response_artifact,
                semantic_review_packet_json=args.semantic_review_packet,
                semantic_decision_manifest_json=args.semantic_decision_manifest,
                semantic_adjudication_json=args.semantic_adjudication,
                source_pdf_path=args.source_pdf,
                poppler_bin_dir=args.poppler_bin_dir,
                asset_dir=args.asset_dir,
                output_json=args.output,
            )
            result = {
                "status": packet.status.value,
                "review_item_count": packet.review_item_count,
                "page_asset_count": packet.page_asset_count,
                "full_table_cell_count": packet.full_table_cell_count,
                "numeric_source_cell_count": packet.numeric_source_cell_count,
            }
        elif args.command == "render":
            packet = render_oled_supplementary_source_transcription_review_packet_from_files(
                review_packet_json=args.review_packet,
                asset_dir=args.asset_dir,
                output_markdown=args.output_markdown,
            )
            result = {
                "status": "rendered",
                "review_item_count": packet.review_item_count,
                "page_asset_count": packet.page_asset_count,
            }
        else:
            artifact = build_oled_supplementary_source_transcription_adjudication_from_files(
                request_artifact_json=args.request_artifact,
                response_manifest_json=args.response_manifest,
                response_artifact_json=args.response_artifact,
                semantic_review_packet_json=args.semantic_review_packet,
                semantic_decision_manifest_json=args.semantic_decision_manifest,
                semantic_adjudication_json=args.semantic_adjudication,
                source_pdf_path=args.source_pdf,
                poppler_bin_dir=args.poppler_bin_dir,
                transcription_review_packet_json=args.review_packet,
                transcription_decision_manifest_json=args.decision_manifest,
                asset_dir=args.asset_dir,
                output_json=args.output,
            )
            result = {
                "status": artifact.status.value,
                "review_item_count": artifact.review_item_count,
                "accepted_scope_count": artifact.accepted_scope_count,
                "later_identity_review_eligible_cell_count": (
                    artifact.later_identity_review_eligible_cell_count
                ),
                "unresolved_review_item_count": artifact.unresolved_review_item_count,
            }
    except Exception as exc:
        print(
            json.dumps(
                {
                    "status": "error",
                    "error_code": "supplementary_source_transcription_review_failed",
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
    "build_oled_supplementary_source_transcription_adjudication_from_files",
    "build_oled_supplementary_source_transcription_review_packet_from_files",
    "main",
    "render_oled_supplementary_source_transcription_review_packet_from_files",
]
