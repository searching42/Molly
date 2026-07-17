from __future__ import annotations

import argparse
import io
import json
import os
import re
import stat
import uuid
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from PIL import Image, ImageDraw, ImageOps, __version__ as pillow_version
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictBool,
    field_validator,
    model_validator,
)

from ai4s_agent.ocsr_candidate_execution import (
    OcsrCandidateImageRequest,
    OcsrCandidateRequest,
    _absolute_path,
    _load_json_without_duplicate_keys,
    _normalize_sha256,
    _pinned_output_parent,
    _read_exact_regular_file,
    _sha256_bytes,
    _stable_hash,
    _validate_directory_path_binding,
    _validate_safe_id,
)
from ai4s_agent.oled_material_registry_successor_writer import (
    _atomic_rename_owned_directory_noreplace,
    _remove_owned_directory_if_still_named,
    _same_inode,
)
from ai4s_agent.oled_supplementary_source_transcription_review import (
    _read_bound_binary_at,
    _write_fresh_bytes_at,
)

OCSR_CROP_REQUEST_VERSION = "ocsr_crop_preprocessing_request.v1"
OCSR_CROP_ARTIFACT_VERSION = "ocsr_crop_preprocessing_artifact.v1"
OCSR_CROP_PROFILE_VERSION = "deterministic_structure_crop.v1"

_SAFE_FILENAME_RE = re.compile(r"^[A-Za-z0-9._-]+$")
_MAX_IMAGE_BYTES = 100 * 1024 * 1024
_MAX_IMAGE_PIXELS = 100_000_000
_MAX_CROP_PIXELS = 4_000_000
_INK_THRESHOLD = 245
_STRONG_COMPONENT_RATIO = 0.15
_MIN_STRONG_COMPONENT_INK = 40
_VERTICAL_CLUSTER_TOLERANCE_RATIO = 0.25
_CONTENT_PADDING_RATIO = 0.04
_MIN_CONTENT_WIDTH = 64
_MIN_CONTENT_HEIGHT = 48
_MIN_EDGE_CLEARANCE = 4
_MIN_SELECTED_INK_COVERAGE = 0.55
_MAX_SELECTED_COMPONENT_COUNT = 8
_MIN_FINAL_INK_FRACTION = 0.01
_MAX_FINAL_INK_FRACTION = 0.50
_FINAL_LONG_EDGE = 768


class OcsrPixelBox(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    left: int = Field(ge=0)
    top: int = Field(ge=0)
    right: int = Field(ge=1)
    bottom: int = Field(ge=1)

    @model_validator(mode="after")
    def validate_extent(self) -> OcsrPixelBox:
        if self.right <= self.left or self.bottom <= self.top:
            raise ValueError("pixel box must have positive area")
        return self


class OcsrCropExclusion(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    box: OcsrPixelBox
    reason: Literal[
        "reported_alias",
        "atom_number_annotation",
        "figure_heading",
        "neighboring_fragment",
        "other_non_structure_annotation",
    ]


class OcsrCropRequestItem(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    candidate_id: str
    reported_alias: str
    source_document_id: str
    source_locator: str
    source_image_file: str
    source_image_sha256: str
    crop_bbox: OcsrPixelBox
    exclusions: list[OcsrCropExclusion] = Field(default_factory=list, max_length=100)

    @field_validator("candidate_id", "source_document_id")
    @classmethod
    def validate_ids(cls, value: str, info: Any) -> str:
        clean = _validate_safe_id(value, field_name=info.field_name)
        if (
            info.field_name == "candidate_id"
            and _SAFE_FILENAME_RE.fullmatch(clean) is None
        ):
            raise ValueError("candidate_id must also be safe for a crop filename")
        return clean

    @field_validator("reported_alias", "source_locator")
    @classmethod
    def validate_text(cls, value: str, info: Any) -> str:
        clean = str(value).strip()
        if not clean or len(clean) > 500 or any(char in clean for char in "\r\n\x00"):
            raise ValueError(f"{info.field_name} is invalid")
        return clean

    @field_validator("source_image_file")
    @classmethod
    def validate_source_image_file(cls, value: str) -> str:
        clean = str(value).strip()
        path = Path(clean)
        if (
            not clean
            or path.is_absolute()
            or len(path.parts) != 1
            or not _SAFE_FILENAME_RE.fullmatch(clean)
        ):
            raise ValueError("source_image_file must be one safe relative filename")
        return clean

    @field_validator("source_image_sha256")
    @classmethod
    def validate_source_image_sha256(cls, value: str) -> str:
        return _normalize_sha256(value, field_name="source_image_sha256")

    @model_validator(mode="after")
    def validate_exclusions(self) -> OcsrCropRequestItem:
        keys = [
            (item.box.top, item.box.left, item.box.bottom, item.box.right, item.reason)
            for item in self.exclusions
        ]
        if keys != sorted(keys) or len(keys) != len(set(keys)):
            raise ValueError("crop exclusions must be sorted and unique")
        for exclusion in self.exclusions:
            box = exclusion.box
            crop = self.crop_bbox
            if (
                box.left < crop.left
                or box.top < crop.top
                or box.right > crop.right
                or box.bottom > crop.bottom
            ):
                raise ValueError("crop exclusion must be contained by crop_bbox")
        for index, left in enumerate(self.exclusions):
            for right in self.exclusions[index + 1 :]:
                if (
                    left.box.left < right.box.right
                    and right.box.left < left.box.right
                    and left.box.top < right.box.bottom
                    and right.box.top < left.box.bottom
                ):
                    raise ValueError("crop exclusions must not overlap")
        return self


class OcsrCropPreprocessingRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    schema_version: Literal[OCSR_CROP_REQUEST_VERSION] = OCSR_CROP_REQUEST_VERSION
    run_id: str
    items: list[OcsrCropRequestItem] = Field(min_length=1, max_length=10_000)

    @field_validator("run_id")
    @classmethod
    def validate_run_id(cls, value: str) -> str:
        return _validate_safe_id(value, field_name="run_id")

    @model_validator(mode="after")
    def validate_roster(self) -> OcsrCropPreprocessingRequest:
        ids = [item.candidate_id for item in self.items]
        if ids != sorted(ids) or len(ids) != len(set(ids)):
            raise ValueError("crop request items must be sorted and unique")
        evidence = [
            (
                item.source_image_sha256,
                item.source_locator,
                item.crop_bbox.left,
                item.crop_bbox.top,
                item.crop_bbox.right,
                item.crop_bbox.bottom,
            )
            for item in self.items
        ]
        if len(evidence) != len(set(evidence)):
            raise ValueError("crop request repeats an exact evidence binding")
        return self


class OcsrCropQuality(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    raw_edge_clearance: dict[Literal["left", "top", "right", "bottom"], int]
    strong_component_count: int = Field(ge=0)
    selected_component_count: int = Field(ge=0)
    strong_ink_pixels: int = Field(ge=0)
    selected_ink_pixels: int = Field(ge=0)
    selected_ink_coverage: float = Field(ge=0.0, le=1.0)
    exclusion_pixel_count: int = Field(ge=0)
    final_ink_fraction: float = Field(ge=0.0, le=1.0)
    rejection_codes: list[str]

    @model_validator(mode="after")
    def validate_codes(self) -> OcsrCropQuality:
        if self.rejection_codes != sorted(set(self.rejection_codes)):
            raise ValueError("crop rejection codes must be sorted and unique")
        return self


class OcsrCropResult(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    candidate_id: str
    reported_alias: str
    source_document_id: str
    source_locator: str
    source_image_sha256: str
    source_width: int = Field(ge=1)
    source_height: int = Field(ge=1)
    crop_bbox: OcsrPixelBox
    exclusions: list[OcsrCropExclusion]
    raw_crop_sha256: str
    selected_content_bbox: OcsrPixelBox
    output_image_file: str
    output_image_sha256: str
    output_width: int = Field(ge=1)
    output_height: int = Field(ge=1)
    quality: OcsrCropQuality
    status: Literal["crop_ready", "crop_rejected"]
    result_digest: str

    @field_validator("candidate_id", "source_document_id")
    @classmethod
    def validate_ids(cls, value: str, info: Any) -> str:
        return _validate_safe_id(value, field_name=info.field_name)

    @field_validator("source_image_sha256", "raw_crop_sha256", "output_image_sha256")
    @classmethod
    def validate_hashes(cls, value: str, info: Any) -> str:
        return _normalize_sha256(value, field_name=info.field_name)

    @field_validator("output_image_file")
    @classmethod
    def validate_output_filename(cls, value: str) -> str:
        if Path(value).name != value or _SAFE_FILENAME_RE.fullmatch(value) is None:
            raise ValueError("output_image_file is invalid")
        return value

    @model_validator(mode="after")
    def validate_result(self) -> OcsrCropResult:
        if (self.status == "crop_ready") == bool(self.quality.rejection_codes):
            raise ValueError("crop status and rejection codes disagree")
        expected = _stable_hash(self.model_dump(mode="json", exclude={"result_digest"}))
        if self.result_digest != expected:
            raise ValueError("crop result digest mismatch")
        return self


class OcsrCropPreprocessingArtifact(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    artifact_version: Literal[OCSR_CROP_ARTIFACT_VERSION] = OCSR_CROP_ARTIFACT_VERSION
    profile_version: Literal[OCSR_CROP_PROFILE_VERSION] = OCSR_CROP_PROFILE_VERSION
    run_id: str
    generated_at: str
    request_sha256: str
    request_digest: str
    request_file: Literal["crop_request.json"] = "crop_request.json"
    pillow_version: str
    parameters: dict[str, int | float]
    crop_count: int = Field(ge=1)
    ready_count: int = Field(ge=0)
    rejected_count: int = Field(ge=0)
    results: list[OcsrCropResult]
    batch_ready: StrictBool
    ocsr_request_file: str
    ocsr_request_sha256: str
    artifact_digest: str
    ocsr_executed: StrictBool = False
    source_match_validated: StrictBool = False
    identity_resolved: StrictBool = False
    registry_mutated: StrictBool = False
    gold_written: StrictBool = False
    dataset_written: StrictBool = False

    @field_validator("request_sha256")
    @classmethod
    def validate_request_sha(cls, value: str) -> str:
        return _normalize_sha256(value, field_name="request_sha256")

    @model_validator(mode="after")
    def validate_artifact(self) -> OcsrCropPreprocessingArtifact:
        ids = [item.candidate_id for item in self.results]
        if ids != sorted(ids) or len(ids) != len(set(ids)):
            raise ValueError("crop artifact results must be sorted and unique")
        ready = sum(item.status == "crop_ready" for item in self.results)
        rejected = len(self.results) - ready
        if (self.crop_count, self.ready_count, self.rejected_count) != (
            len(self.results),
            ready,
            rejected,
        ):
            raise ValueError("crop artifact counts mismatch")
        if self.batch_ready != (ready == len(self.results)):
            raise ValueError("crop artifact batch readiness mismatch")
        if self.batch_ready:
            if (
                self.ocsr_request_file != "ocsr_request.json"
                or not self.ocsr_request_sha256
            ):
                raise ValueError("ready crop artifact lacks downstream request binding")
            _normalize_sha256(
                self.ocsr_request_sha256, field_name="ocsr_request_sha256"
            )
        elif self.ocsr_request_file or self.ocsr_request_sha256:
            raise ValueError(
                "rejected crop artifact must not publish a downstream request"
            )
        if any(
            (
                self.ocsr_executed,
                self.source_match_validated,
                self.identity_resolved,
                self.registry_mutated,
                self.gold_written,
                self.dataset_written,
            )
        ):
            raise ValueError("crop preprocessing crossed its publication boundary")
        expected = _stable_hash(
            self.model_dump(mode="json", exclude={"artifact_digest"})
        )
        if self.artifact_digest != expected:
            raise ValueError("crop artifact digest mismatch")
        return self


def _png_bytes(image: Image.Image) -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG", optimize=False, compress_level=9)
    return buffer.getvalue()


def _components(mask: list[list[bool]]) -> list[tuple[int, int, int, int, int]]:
    height = len(mask)
    width = len(mask[0]) if height else 0
    seen = bytearray(width * height)
    found: list[tuple[int, int, int, int, int]] = []
    for y in range(height):
        for x in range(width):
            offset = y * width + x
            if not mask[y][x] or seen[offset]:
                continue
            seen[offset] = 1
            queue = deque([(x, y)])
            left = right = x
            top = bottom = y
            area = 0
            while queue:
                current_x, current_y = queue.popleft()
                area += 1
                left = min(left, current_x)
                right = max(right, current_x)
                top = min(top, current_y)
                bottom = max(bottom, current_y)
                for next_x, next_y in (
                    (current_x - 1, current_y),
                    (current_x + 1, current_y),
                    (current_x, current_y - 1),
                    (current_x, current_y + 1),
                ):
                    if 0 <= next_x < width and 0 <= next_y < height:
                        next_offset = next_y * width + next_x
                        if mask[next_y][next_x] and not seen[next_offset]:
                            seen[next_offset] = 1
                            queue.append((next_x, next_y))
            found.append((left, top, right + 1, bottom + 1, area))
    return found


def _apply_exclusions(
    grayscale: Image.Image,
    *,
    crop_bbox: OcsrPixelBox,
    exclusions: list[OcsrCropExclusion],
) -> tuple[Image.Image, int]:
    exclusion_mask = Image.new("1", grayscale.size, 0)
    mask_draw = ImageDraw.Draw(exclusion_mask)
    for exclusion in exclusions:
        box = exclusion.box
        left = box.left - crop_bbox.left
        top = box.top - crop_bbox.top
        right = box.right - crop_bbox.left
        bottom = box.bottom - crop_bbox.top
        # OcsrPixelBox is half-open. ImageDraw.rectangle includes its endpoint,
        # so subtract one from the exclusive right and bottom coordinates.
        mask_draw.rectangle((left, top, right - 1, bottom - 1), fill=1)
    exclusion_pixel_count = sum(exclusion_mask.get_flattened_data())
    masked = grayscale.copy()
    masked.paste(255, mask=exclusion_mask)
    return masked, exclusion_pixel_count


def _process_item(
    item: OcsrCropRequestItem, source_root: Path
) -> tuple[OcsrCropResult, bytes]:
    image_bytes = _read_exact_regular_file(source_root / item.source_image_file)
    if _sha256_bytes(image_bytes) != item.source_image_sha256:
        raise ValueError(f"source image SHA-256 mismatch for {item.candidate_id}")
    with Image.open(io.BytesIO(image_bytes)) as opened:
        if getattr(opened, "n_frames", 1) != 1:
            raise ValueError("source image must have exactly one frame")
        opened.verify()
    with Image.open(io.BytesIO(image_bytes)) as opened:
        source_width, source_height = opened.size
        if source_width * source_height > _MAX_IMAGE_PIXELS:
            raise ValueError("source image has too many pixels")
        crop_box = item.crop_bbox
        if crop_box.right > source_width or crop_box.bottom > source_height:
            raise ValueError(f"crop bbox exceeds source image for {item.candidate_id}")
        if (crop_box.right - crop_box.left) * (
            crop_box.bottom - crop_box.top
        ) > _MAX_CROP_PIXELS:
            raise ValueError(f"crop bbox is too large for {item.candidate_id}")
        grayscale = ImageOps.grayscale(opened).crop(
            (crop_box.left, crop_box.top, crop_box.right, crop_box.bottom)
        )

    raw_crop_bytes = _png_bytes(grayscale)
    masked, exclusion_pixel_count = _apply_exclusions(
        grayscale,
        crop_bbox=crop_box,
        exclusions=item.exclusions,
    )

    pixels = list(masked.get_flattened_data())
    width, height = masked.size
    mask = [
        [pixels[y * width + x] < _INK_THRESHOLD for x in range(width)]
        for y in range(height)
    ]
    components = _components(mask)
    largest = max((component[4] for component in components), default=0)
    strong_floor = max(
        _MIN_STRONG_COMPONENT_INK, round(largest * _STRONG_COMPONENT_RATIO)
    )
    strong = [component for component in components if component[4] >= strong_floor]
    rejection_codes: list[str] = []
    if not strong:
        rejection_codes.append("no_strong_structure_components")
        strong = [(0, 0, width, height, 0)]

    tolerance = max(12, round(height * _VERTICAL_CLUSTER_TOLERANCE_RATIO))
    groups: list[list[tuple[int, int, int, int, int]]] = []
    for component in sorted(
        strong, key=lambda value: ((value[1] + value[3]) / 2, value[0])
    ):
        center = (component[1] + component[3]) / 2
        matched = None
        for group in groups:
            weighted_center = sum(
                ((part[1] + part[3]) / 2) * part[4] for part in group
            ) / max(1, sum(part[4] for part in group))
            if abs(center - weighted_center) <= tolerance:
                matched = group
                break
        if matched is None:
            groups.append([component])
        else:
            matched.append(component)
    selected = max(groups, key=lambda group: sum(part[4] for part in group))
    selected_ink = sum(part[4] for part in selected)
    strong_ink = sum(part[4] for part in strong)
    content_left = min(part[0] for part in selected)
    content_top = min(part[1] for part in selected)
    content_right = max(part[2] for part in selected)
    content_bottom = max(part[3] for part in selected)
    clearances = {
        "left": content_left,
        "top": content_top,
        "right": width - content_right,
        "bottom": height - content_bottom,
    }
    if min(clearances.values()) < _MIN_EDGE_CLEARANCE:
        rejection_codes.append("structure_touches_crop_edge")
    content_width = content_right - content_left
    content_height = content_bottom - content_top
    if content_width < _MIN_CONTENT_WIDTH or content_height < _MIN_CONTENT_HEIGHT:
        rejection_codes.append("selected_structure_too_small")
    coverage = selected_ink / strong_ink if strong_ink else 0.0
    if coverage < _MIN_SELECTED_INK_COVERAGE:
        rejection_codes.append("ambiguous_component_cluster")
    if len(selected) > _MAX_SELECTED_COMPONENT_COUNT:
        rejection_codes.append("too_many_selected_components")

    padding = max(8, round(max(content_width, content_height) * _CONTENT_PADDING_RATIO))
    selected_box = OcsrPixelBox(
        left=max(0, content_left - padding),
        top=max(0, content_top - padding),
        right=min(width, content_right + padding),
        bottom=min(height, content_bottom + padding),
    )
    final = masked.crop(
        (selected_box.left, selected_box.top, selected_box.right, selected_box.bottom)
    )
    scale = _FINAL_LONG_EDGE / max(final.size)
    final = final.resize(
        (max(1, round(final.width * scale)), max(1, round(final.height * scale))),
        Image.Resampling.LANCZOS,
    )
    final_pixels = list(final.get_flattened_data())
    ink_fraction = sum(value < _INK_THRESHOLD for value in final_pixels) / len(
        final_pixels
    )
    if not _MIN_FINAL_INK_FRACTION <= ink_fraction <= _MAX_FINAL_INK_FRACTION:
        rejection_codes.append("final_ink_fraction_out_of_range")
    rejection_codes = sorted(set(rejection_codes))
    output_bytes = _png_bytes(final)
    filename = f"crop-{item.candidate_id}.png"
    payload: dict[str, Any] = {
        "candidate_id": item.candidate_id,
        "reported_alias": item.reported_alias,
        "source_document_id": item.source_document_id,
        "source_locator": item.source_locator,
        "source_image_sha256": item.source_image_sha256,
        "source_width": source_width,
        "source_height": source_height,
        "crop_bbox": item.crop_bbox.model_dump(mode="json"),
        "exclusions": [entry.model_dump(mode="json") for entry in item.exclusions],
        "raw_crop_sha256": _sha256_bytes(raw_crop_bytes),
        "selected_content_bbox": selected_box.model_dump(mode="json"),
        "output_image_file": filename,
        "output_image_sha256": _sha256_bytes(output_bytes),
        "output_width": final.width,
        "output_height": final.height,
        "quality": {
            "raw_edge_clearance": clearances,
            "strong_component_count": len(strong),
            "selected_component_count": len(selected),
            "strong_ink_pixels": strong_ink,
            "selected_ink_pixels": selected_ink,
            "selected_ink_coverage": coverage,
            "exclusion_pixel_count": exclusion_pixel_count,
            "final_ink_fraction": ink_fraction,
            "rejection_codes": rejection_codes,
        },
        "status": "crop_rejected" if rejection_codes else "crop_ready",
        "result_digest": "sha256:" + "0" * 64,
    }
    payload["result_digest"] = _stable_hash(
        {key: value for key, value in payload.items() if key != "result_digest"}
    )
    return OcsrCropResult.model_validate(payload), output_bytes


def _artifact_and_payloads(
    request: OcsrCropPreprocessingRequest,
    *,
    request_bytes: bytes,
    source_root: Path,
    generated_at: str | None,
) -> tuple[OcsrCropPreprocessingArtifact, dict[str, bytes]]:
    request_sha256 = _sha256_bytes(request_bytes)
    results: list[OcsrCropResult] = []
    payloads: dict[str, bytes] = {}
    for item in request.items:
        result, image_bytes = _process_item(item, source_root)
        results.append(result)
        payloads[result.output_image_file] = image_bytes
    batch_ready = all(item.status == "crop_ready" for item in results)
    ocsr_request_bytes = b""
    if batch_ready:
        downstream = OcsrCandidateRequest(
            run_id=request.run_id,
            items=[
                OcsrCandidateImageRequest(
                    candidate_id=item.candidate_id,
                    reported_alias=item.reported_alias,
                    image_file=item.output_image_file,
                    image_sha256=item.output_image_sha256,
                )
                for item in results
            ],
        )
        ocsr_request_bytes = (
            json.dumps(
                downstream.model_dump(mode="json"),
                ensure_ascii=False,
                allow_nan=False,
                sort_keys=True,
                indent=2,
            )
            + "\n"
        ).encode("utf-8")
        payloads["ocsr_request.json"] = ocsr_request_bytes
    parameters: dict[str, int | float] = {
        "ink_threshold": _INK_THRESHOLD,
        "max_crop_pixels": _MAX_CROP_PIXELS,
        "strong_component_ratio": _STRONG_COMPONENT_RATIO,
        "min_strong_component_ink": _MIN_STRONG_COMPONENT_INK,
        "vertical_cluster_tolerance_ratio": _VERTICAL_CLUSTER_TOLERANCE_RATIO,
        "content_padding_ratio": _CONTENT_PADDING_RATIO,
        "min_content_width": _MIN_CONTENT_WIDTH,
        "min_content_height": _MIN_CONTENT_HEIGHT,
        "min_edge_clearance": _MIN_EDGE_CLEARANCE,
        "min_selected_ink_coverage": _MIN_SELECTED_INK_COVERAGE,
        "max_selected_component_count": _MAX_SELECTED_COMPONENT_COUNT,
        "min_final_ink_fraction": _MIN_FINAL_INK_FRACTION,
        "max_final_ink_fraction": _MAX_FINAL_INK_FRACTION,
        "final_long_edge": _FINAL_LONG_EDGE,
    }
    timestamp = generated_at or datetime.now(timezone.utc).isoformat().replace(
        "+00:00", "Z"
    )
    artifact_payload: dict[str, Any] = {
        "artifact_version": OCSR_CROP_ARTIFACT_VERSION,
        "profile_version": OCSR_CROP_PROFILE_VERSION,
        "run_id": request.run_id,
        "generated_at": timestamp,
        "request_sha256": request_sha256,
        "request_digest": _stable_hash(request.model_dump(mode="json")),
        "request_file": "crop_request.json",
        "pillow_version": pillow_version,
        "parameters": parameters,
        "crop_count": len(results),
        "ready_count": sum(item.status == "crop_ready" for item in results),
        "rejected_count": sum(item.status == "crop_rejected" for item in results),
        "results": [item.model_dump(mode="json") for item in results],
        "batch_ready": batch_ready,
        "ocsr_request_file": "ocsr_request.json" if batch_ready else "",
        "ocsr_request_sha256": _sha256_bytes(ocsr_request_bytes) if batch_ready else "",
        "artifact_digest": "sha256:" + "0" * 64,
        "ocsr_executed": False,
        "source_match_validated": False,
        "identity_resolved": False,
        "registry_mutated": False,
        "gold_written": False,
        "dataset_written": False,
    }
    artifact_payload["artifact_digest"] = _stable_hash(
        {
            key: value
            for key, value in artifact_payload.items()
            if key != "artifact_digest"
        }
    )
    artifact = OcsrCropPreprocessingArtifact.model_validate(artifact_payload)
    payloads["crop_request.json"] = request_bytes
    payloads["crop_artifact.json"] = (
        json.dumps(
            artifact.model_dump(mode="json"),
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            indent=2,
        )
        + "\n"
    ).encode("utf-8")
    return artifact, payloads


def _require_fresh_output(output_dir: Path, parent_descriptor: int) -> None:
    try:
        os.stat(output_dir.name, dir_fd=parent_descriptor, follow_symlinks=False)
    except FileNotFoundError:
        return
    except OSError as exc:
        raise ValueError("crop output cannot be inspected") from exc
    raise ValueError("crop output directory must be fresh")


def _publish_bundle(
    output_dir: Path, parent_descriptor: int, payloads: dict[str, bytes]
) -> None:
    directory_flag = getattr(os, "O_DIRECTORY", None)
    no_follow = getattr(os, "O_NOFOLLOW", None)
    if directory_flag is None or no_follow is None:
        raise ValueError("crop publisher requires safe dirfd support")
    _validate_directory_path_binding(
        output_dir.parent, parent_descriptor, error_message="crop output parent changed"
    )
    _require_fresh_output(output_dir, parent_descriptor)
    temp_name = f".{output_dir.name}.{uuid.uuid4().hex}.tmp"
    temp_descriptor = -1
    owned_stat: os.stat_result | None = None
    created_files: dict[str, os.stat_result] = {}
    committed = False
    try:
        os.mkdir(temp_name, 0o700, dir_fd=parent_descriptor)
        temp_descriptor = os.open(
            temp_name,
            os.O_RDONLY | directory_flag | no_follow,
            dir_fd=parent_descriptor,
        )
        owned_stat = os.fstat(temp_descriptor)
        for filename, payload in payloads.items():
            created_files[filename] = _write_fresh_bytes_at(
                temp_descriptor, filename, payload
            )
        os.fsync(temp_descriptor)
        _validate_directory_path_binding(
            output_dir.parent,
            parent_descriptor,
            error_message="crop output parent changed",
        )
        _require_fresh_output(output_dir, parent_descriptor)
        _atomic_rename_owned_directory_noreplace(
            parent_descriptor=parent_descriptor,
            temp_name=temp_name,
            output_name=output_dir.name,
            temp_descriptor=temp_descriptor,
            owned_stat=owned_stat,
        )
        os.fsync(parent_descriptor)
        named_stat = os.stat(
            output_dir.name, dir_fd=parent_descriptor, follow_symlinks=False
        )
        if (
            not stat.S_ISDIR(named_stat.st_mode)
            or not _same_inode(named_stat, owned_stat)
            or not _same_inode(os.fstat(temp_descriptor), owned_stat)
        ):
            raise ValueError("crop published directory inode mismatch")
        if set(os.listdir(temp_descriptor)) != set(payloads):
            raise ValueError("crop published bundle file coverage mismatch")
        for filename, expected in payloads.items():
            if (
                _read_bound_binary_at(
                    temp_descriptor, filename, max_bytes=_MAX_IMAGE_BYTES
                )
                != expected
            ):
                raise ValueError("crop published bundle content mismatch")
        artifact_bytes = payloads["crop_artifact.json"]
        OcsrCropPreprocessingArtifact.model_validate(
            _load_json_without_duplicate_keys(artifact_bytes)
        )
        if "ocsr_request.json" in payloads:
            OcsrCandidateRequest.model_validate(
                _load_json_without_duplicate_keys(payloads["ocsr_request.json"])
            )
        request = OcsrCropPreprocessingRequest.model_validate(
            _load_json_without_duplicate_keys(payloads["crop_request.json"])
        )
        artifact = OcsrCropPreprocessingArtifact.model_validate(
            _load_json_without_duplicate_keys(payloads["crop_artifact.json"])
        )
        if (
            _sha256_bytes(payloads["crop_request.json"]) != artifact.request_sha256
            or _stable_hash(request.model_dump(mode="json")) != artifact.request_digest
        ):
            raise ValueError("crop published request binding mismatch")
        _validate_directory_path_binding(
            output_dir.parent,
            parent_descriptor,
            error_message="crop output parent changed",
        )
        committed = True
    except FileExistsError as exc:
        raise ValueError("crop output directory must be fresh") from exc
    except OSError as exc:
        raise ValueError("crop bundle publication failed") from exc
    finally:
        if temp_descriptor != -1:
            os.close(temp_descriptor)
        if not committed and owned_stat is not None:
            for name in (temp_name, output_dir.name):
                _remove_owned_directory_if_still_named(
                    parent_descriptor=parent_descriptor,
                    directory_name=name,
                    owned_stat=owned_stat,
                    created_files=created_files,
                )


def build_ocsr_crop_preprocessing_from_files(
    *, request_json: str | Path, output_dir: str | Path, generated_at: str | None = None
) -> tuple[OcsrCropPreprocessingArtifact, Path]:
    request_path = _absolute_path(Path(request_json))
    output_path = _absolute_path(Path(output_dir))
    request_bytes = _read_exact_regular_file(request_path)
    request = OcsrCropPreprocessingRequest.model_validate(
        _load_json_without_duplicate_keys(request_bytes)
    )
    with _pinned_output_parent(output_path.parent) as (parent_descriptor, _):
        _require_fresh_output(output_path, parent_descriptor)
        artifact, payloads = _artifact_and_payloads(
            request,
            request_bytes=request_bytes,
            source_root=request_path.parent,
            generated_at=generated_at,
        )
        if _read_exact_regular_file(request_path) != request_bytes:
            raise ValueError("crop request changed during execution")
        _publish_bundle(output_path, parent_descriptor, payloads)
    return artifact, output_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Deterministically crop source diagrams for OCSR candidate execution"
    )
    parser.add_argument("--request", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args(argv)
    try:
        artifact, output = build_ocsr_crop_preprocessing_from_files(
            request_json=args.request, output_dir=args.output_dir
        )
    except Exception as exc:
        print(
            json.dumps(
                {
                    "error_code": "ocsr_crop_preprocessing_failed",
                    "exception_type": type(exc).__name__,
                },
                sort_keys=True,
            )
        )
        return 1
    print(
        json.dumps(
            {
                "artifact_digest": artifact.artifact_digest,
                "batch_ready": artifact.batch_ready,
                "ready_count": artifact.ready_count,
                "rejected_count": artifact.rejected_count,
                "output_directory": output.name,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
