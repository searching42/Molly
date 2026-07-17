from __future__ import annotations

import argparse
import json
import os
import re
import stat
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Literal
from urllib.parse import quote

import httpx
from pydantic import BaseModel, ConfigDict, Field, StrictBool, field_validator, model_validator

from ai4s_agent.ocsr_candidate_execution import (
    _absolute_path,
    _ensure_fresh_output_at,
    _normalize_sha256,
    _pinned_output_parent,
    _read_exact_regular_file,
    _read_open_descriptor,
    _sha256_bytes,
    _stable_hash,
    _validate_directory_path_binding,
    _validate_safe_id,
    _write_all,
)

try:
    from rdkit import Chem
    from rdkit.Chem import rdMolDescriptors
    from rdkit.Chem import rdinchi as rd_inchi
except ImportError:  # pragma: no cover - reduced deployments fail at execution
    Chem = None  # type: ignore[assignment]
    rdMolDescriptors = None  # type: ignore[assignment]
    rd_inchi = None  # type: ignore[assignment]


CONTEXTUAL_ALIAS_REQUEST_VERSION = "contextual_alias_resolution_request.v1"
CONTEXTUAL_ALIAS_ARTIFACT_VERSION = "contextual_alias_resolution_artifact.v1"
CONTEXTUAL_ALIAS_PROFILE_VERSION = "supplementary_heading_opsin_rdkit.v1"
OPSIN_RESOLVER_ID = "opsin-ebi-web"
OPSIN_RESOLVER_VERSION = "service-unreported"
OPSIN_BASE_URL = "https://www.ebi.ac.uk/opsin/ws"

_SAFE_FILENAME_RE = re.compile(r"^[A-Za-z0-9._-]+$")
_PAGE_MARKER_RE = re.compile(r"^===\s+PAGE\s+([1-9][0-9]*)\s+===$")
_INCHIKEY_RE = re.compile(r"^[A-Z]{14}-[A-Z]{10}-[A-Z]$")
_MAX_INPUT_BYTES = 100 * 1024 * 1024
_MAX_RESPONSE_BYTES = 5 * 1024 * 1024
_MAX_EMBEDDED_RESPONSE_BYTES = 50 * 1024 * 1024
_MAX_OUTPUT_BYTES = 100 * 1024 * 1024


def _load_json_without_duplicate_keys(value: bytes, *, label: str) -> Any:
    def build_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, item in pairs:
            if key in result:
                raise ValueError(f"{label} contains duplicate JSON keys")
            result[key] = item
        return result

    try:
        return json.loads(value, object_pairs_hook=build_object)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} is not valid JSON") from exc


def _clean_authored_text(value: str, *, field_name: str, maximum: int) -> str:
    clean = str(value).strip()
    if not clean or len(clean) > maximum or any(char in clean for char in "\r\n\x00"):
        raise ValueError(f"{field_name} is invalid")
    return clean


class ContextualAliasRequestItem(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    candidate_id: str
    reported_alias: str

    @field_validator("candidate_id")
    @classmethod
    def validate_candidate_id(cls, value: str) -> str:
        return _validate_safe_id(value, field_name="candidate_id")

    @field_validator("reported_alias")
    @classmethod
    def validate_reported_alias(cls, value: str) -> str:
        return _clean_authored_text(value, field_name="reported_alias", maximum=500)


class ContextualAliasResolutionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    schema_version: Literal[CONTEXTUAL_ALIAS_REQUEST_VERSION] = CONTEXTUAL_ALIAS_REQUEST_VERSION
    run_id: str
    paper_id: str
    source_document_id: str
    parsed_text_file: str
    parsed_text_sha256: str
    items: list[ContextualAliasRequestItem] = Field(min_length=1, max_length=10_000)

    @field_validator("run_id", "paper_id", "source_document_id")
    @classmethod
    def validate_ids(cls, value: str, info: Any) -> str:
        return _validate_safe_id(value, field_name=info.field_name)

    @field_validator("parsed_text_file")
    @classmethod
    def validate_parsed_text_file(cls, value: str) -> str:
        clean = str(value).strip()
        path = Path(clean)
        if not clean or path.is_absolute() or len(path.parts) != 1 or _SAFE_FILENAME_RE.fullmatch(clean) is None:
            raise ValueError("parsed_text_file must be one safe relative filename")
        return clean

    @field_validator("parsed_text_sha256")
    @classmethod
    def validate_parsed_text_sha256(cls, value: str) -> str:
        return _normalize_sha256(value, field_name="parsed_text_sha256")

    @model_validator(mode="after")
    def validate_roster(self) -> ContextualAliasResolutionRequest:
        keys = [(item.candidate_id, item.reported_alias) for item in self.items]
        if keys != sorted(keys) or len(keys) != len(set(keys)):
            raise ValueError("contextual alias request items must be sorted and unique")
        aliases = [item.reported_alias.casefold() for item in self.items]
        if len(aliases) != len(set(aliases)):
            raise ValueError("contextual alias request aliases must be unique")
        return self


class OpsinExchange(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    endpoint_url: str
    http_status: int = Field(ge=100, le=599)
    response_bytes: bytes

    @field_validator("endpoint_url")
    @classmethod
    def validate_endpoint_url(cls, value: str) -> str:
        clean = str(value).strip()
        if not clean.startswith(f"{OPSIN_BASE_URL}/") or not clean.endswith(".json"):
            raise ValueError("OPSIN endpoint URL is outside the fixed service origin")
        return clean

    @field_validator("response_bytes")
    @classmethod
    def validate_response_bytes(cls, value: bytes) -> bytes:
        if not value or len(value) > _MAX_RESPONSE_BYTES:
            raise ValueError("OPSIN response has an unsupported size")
        return value


class ContextualAliasResolutionResult(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    candidate_id: str
    reported_alias: str
    status: Literal[
        "candidate_ready",
        "alias_not_found",
        "alias_ambiguous",
        "name_resolution_rejected",
    ]
    systematic_name: str = ""
    source_locator: str = ""
    source_heading_sha256: str = ""
    resolver_id: str = ""
    resolver_version: str = ""
    resolver_endpoint_url: str = ""
    resolver_http_status: int = Field(default=0, ge=0, le=599)
    resolver_response_sha256: str = ""
    resolver_response_json: str = ""
    resolver_status: str = ""
    resolver_message: str = ""
    resolver_warnings: list[str] = Field(default_factory=list, max_length=100)
    resolver_smiles: str = ""
    resolver_standard_inchi: str = ""
    resolver_standard_inchikey: str = ""
    canonical_isomeric_smiles: str = ""
    standard_inchi: str = ""
    inchikey: str = ""
    molecular_formula: str = ""
    rejection_reason: str = ""
    candidate_only: StrictBool = True
    source_match_validated: StrictBool = False
    identity_resolved: StrictBool = False
    result_digest: str

    @field_validator("candidate_id")
    @classmethod
    def validate_candidate_id(cls, value: str) -> str:
        return _validate_safe_id(value, field_name="candidate_id")

    @field_validator("reported_alias")
    @classmethod
    def validate_alias(cls, value: str) -> str:
        return _clean_authored_text(value, field_name="reported_alias", maximum=500)

    @field_validator("source_heading_sha256", "resolver_response_sha256")
    @classmethod
    def validate_optional_hashes(cls, value: str, info: Any) -> str:
        clean = str(value).strip()
        if not clean:
            return ""
        return _normalize_sha256(clean, field_name=info.field_name)

    @field_validator("resolver_endpoint_url")
    @classmethod
    def validate_optional_endpoint(cls, value: str) -> str:
        clean = str(value).strip()
        if clean and (not clean.startswith(f"{OPSIN_BASE_URL}/") or not clean.endswith(".json")):
            raise ValueError("OPSIN endpoint URL is outside the fixed service origin")
        return clean

    @field_validator("resolver_warnings")
    @classmethod
    def validate_warnings(cls, value: list[str]) -> list[str]:
        clean = [
            _clean_authored_text(item, field_name="resolver_warning", maximum=2_000)
            for item in value
        ]
        if clean != sorted(set(clean)):
            raise ValueError("OPSIN warnings must be sorted and unique")
        return clean

    @model_validator(mode="after")
    def validate_result(self) -> ContextualAliasResolutionResult:
        if not self.candidate_only or self.source_match_validated or self.identity_resolved:
            raise ValueError("contextual alias result crossed the candidate boundary")
        resolved_fields = (
            self.systematic_name,
            self.source_locator,
            self.source_heading_sha256,
            self.resolver_id,
            self.resolver_version,
            self.resolver_endpoint_url,
            self.resolver_response_sha256,
            self.resolver_response_json,
            self.resolver_status,
        )
        chemistry_fields = (
            self.resolver_smiles,
            self.resolver_standard_inchi,
            self.resolver_standard_inchikey,
            self.canonical_isomeric_smiles,
            self.standard_inchi,
            self.inchikey,
            self.molecular_formula,
        )
        if self.status == "candidate_ready":
            if not all(resolved_fields) or not all(chemistry_fields) or self.rejection_reason:
                raise ValueError("ready contextual alias result is incomplete")
            if self.resolver_id != OPSIN_RESOLVER_ID or self.resolver_version != OPSIN_RESOLVER_VERSION:
                raise ValueError("ready contextual alias result has unexpected resolver provenance")
            if self.resolver_status != "SUCCESS" or self.resolver_http_status != 200:
                raise ValueError("ready contextual alias result has unsuccessful resolver status")
            self._validate_resolver_replay()
            observation = _rdkit_observation(self.resolver_smiles)
            if (
                self.canonical_isomeric_smiles != observation["canonical_isomeric_smiles"]
                or self.standard_inchi != observation["standard_inchi"]
                or self.inchikey != observation["inchikey"]
                or self.molecular_formula != observation["molecular_formula"]
                or self.resolver_standard_inchi != self.standard_inchi
                or self.resolver_standard_inchikey != self.inchikey
            ):
                raise ValueError("contextual alias chemistry binding mismatch")
        elif self.status == "name_resolution_rejected":
            if not all(resolved_fields) or any(chemistry_fields) or not self.rejection_reason:
                raise ValueError("rejected contextual alias result has an invalid shape")
            if self.resolver_http_status < 100:
                raise ValueError("rejected contextual alias result lacks an HTTP status")
            self._validate_resolver_replay()
        elif any(resolved_fields) or any(chemistry_fields) or not self.rejection_reason:
            raise ValueError("unlocated contextual alias result has an invalid shape")
        expected = _stable_hash(self.model_dump(mode="json", exclude={"result_digest"}))
        if self.result_digest != expected:
            raise ValueError("contextual alias result digest mismatch")
        return self

    def _validate_resolver_replay(self) -> None:
        expected_endpoint = f"{OPSIN_BASE_URL}/{quote(self.systematic_name, safe='')}.json"
        expected_heading = _sha256_bytes(f"{self.systematic_name} ({self.reported_alias}):".encode("utf-8"))
        try:
            raw_response = self.resolver_response_json.encode("utf-8")
        except UnicodeEncodeError as exc:
            raise ValueError("OPSIN response JSON is not UTF-8") from exc
        if (
            self.resolver_endpoint_url != expected_endpoint
            or self.source_heading_sha256 != expected_heading
            or self.resolver_response_sha256 != _sha256_bytes(raw_response)
        ):
            raise ValueError("contextual alias resolver provenance mismatch")
        response = _load_json_without_duplicate_keys(raw_response, label="embedded OPSIN response")
        if not isinstance(response, dict):
            raise ValueError("embedded OPSIN response must be a JSON object")
        status = str(response.get("status") or "").strip().upper() or "UNKNOWN"
        message = str(response.get("message") or "").strip()[:2_000]
        raw_warnings = response.get("warnings") or []
        if isinstance(raw_warnings, str):
            raw_warnings = [raw_warnings]
        if not isinstance(raw_warnings, list):
            raise ValueError("embedded OPSIN warnings have an invalid shape")
        warnings = sorted(
            set(
                _clean_authored_text(str(value), field_name="resolver_warning", maximum=2_000)
                for value in raw_warnings
                if str(value).strip()
            )
        )
        if (
            self.resolver_status != status
            or self.resolver_message != message
            or self.resolver_warnings != warnings
        ):
            raise ValueError("embedded OPSIN response metadata mismatch")
        if self.status == "candidate_ready" and (
            self.resolver_smiles != str(response.get("smiles") or "").strip()
            or self.resolver_standard_inchi != str(response.get("stdinchi") or "").strip()
            or self.resolver_standard_inchikey != str(response.get("stdinchikey") or "").strip()
        ):
            raise ValueError("embedded OPSIN chemistry mismatch")


class ContextualAliasResolutionArtifact(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    artifact_version: Literal[CONTEXTUAL_ALIAS_ARTIFACT_VERSION] = CONTEXTUAL_ALIAS_ARTIFACT_VERSION
    profile_version: Literal[CONTEXTUAL_ALIAS_PROFILE_VERSION] = CONTEXTUAL_ALIAS_PROFILE_VERSION
    run_id: str
    paper_id: str
    source_document_id: str
    generated_at: str
    request_sha256: str
    request_digest: str
    request: ContextualAliasResolutionRequest
    parsed_text_sha256: str
    resolver_id: Literal[OPSIN_RESOLVER_ID] = OPSIN_RESOLVER_ID
    resolver_version: Literal[OPSIN_RESOLVER_VERSION] = OPSIN_RESOLVER_VERSION
    result_count: int = Field(ge=1)
    ready_count: int = Field(ge=0)
    rejected_count: int = Field(ge=0)
    results: list[ContextualAliasResolutionResult]
    artifact_digest: str
    candidate_only: StrictBool = True
    source_match_validated: StrictBool = False
    identity_resolved: StrictBool = False
    registry_mutated: StrictBool = False
    gold_written: StrictBool = False
    dataset_written: StrictBool = False

    @field_validator("request_sha256", "parsed_text_sha256")
    @classmethod
    def validate_hashes(cls, value: str, info: Any) -> str:
        return _normalize_sha256(value, field_name=info.field_name)

    @model_validator(mode="after")
    def validate_artifact(self) -> ContextualAliasResolutionArtifact:
        keys = [(item.candidate_id, item.reported_alias) for item in self.results]
        if keys != sorted(keys) or len(keys) != len(set(keys)):
            raise ValueError("contextual alias results must be sorted and unique")
        ready = sum(item.status == "candidate_ready" for item in self.results)
        rejected = len(self.results) - ready
        if (self.result_count, self.ready_count, self.rejected_count) != (len(self.results), ready, rejected):
            raise ValueError("contextual alias artifact counts mismatch")
        if (
            self.request_digest != _stable_hash(self.request.model_dump(mode="json"))
            or self.run_id != self.request.run_id
            or self.paper_id != self.request.paper_id
            or self.source_document_id != self.request.source_document_id
            or self.parsed_text_sha256 != self.request.parsed_text_sha256
            or keys
            != [(item.candidate_id, item.reported_alias) for item in self.request.items]
        ):
            raise ValueError("contextual alias request binding mismatch")
        if any(
            (
                not self.candidate_only,
                self.source_match_validated,
                self.identity_resolved,
                self.registry_mutated,
                self.gold_written,
                self.dataset_written,
            )
        ):
            raise ValueError("contextual alias artifact crossed its publication boundary")
        expected = _stable_hash(self.model_dump(mode="json", exclude={"artifact_digest"}))
        if self.artifact_digest != expected:
            raise ValueError("contextual alias artifact digest mismatch")
        return self


def _rdkit_observation(smiles: str) -> dict[str, str]:
    if Chem is None or rd_inchi is None or rdMolDescriptors is None:
        raise ValueError("contextual alias resolution requires RDKit/InChI")
    molecule = Chem.MolFromSmiles(str(smiles).strip())
    if molecule is None:
        raise ValueError("OPSIN returned an invalid molecular graph")
    Chem.SanitizeMol(molecule)
    canonical = Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=True)
    inchi_result = rd_inchi.MolToInchi(molecule)
    standard_inchi = str(inchi_result[0])
    if not standard_inchi.startswith("InChI=1S/"):
        raise ValueError("RDKit could not derive a standard InChI")
    inchikey = str(rd_inchi.InchiToInchiKey(standard_inchi))
    if _INCHIKEY_RE.fullmatch(inchikey) is None:
        raise ValueError("RDKit could not derive a standard InChIKey")
    return {
        "canonical_isomeric_smiles": canonical,
        "standard_inchi": standard_inchi,
        "inchikey": inchikey,
        "molecular_formula": str(rdMolDescriptors.CalcMolFormula(molecule)),
    }


def _join_wrapped_heading(left: str, right: str) -> str:
    left = " ".join(left.split())
    right = " ".join(right.split())
    if left.endswith("-"):
        return left[:-1] + right
    return f"{left} {right}".strip()


def _looks_like_heading_start(value: str) -> bool:
    return bool(value) and (value[0].isupper() or value[0].isdigit() or value[0] == "(")


def _locate_systematic_names(parsed_text: str, alias: str) -> list[tuple[str, str, str]]:
    lines = parsed_text.splitlines()
    marker = re.compile(rf"\(\s*{re.escape(alias)}\s*\)\s*:", re.IGNORECASE)
    page = 0
    pages: list[int] = []
    for raw_line in lines:
        match = _PAGE_MARKER_RE.fullmatch(raw_line.strip())
        if match is not None:
            page = int(match.group(1))
        pages.append(page)
    found: list[tuple[str, str, str]] = []
    for index, raw_line in enumerate(lines):
        match = marker.search(raw_line)
        if match is None:
            continue
        fragment = " ".join(raw_line[: match.start()].split())
        start = index
        while (not fragment or not _looks_like_heading_start(fragment)) and start > 0 and index - start < 4:
            start -= 1
            previous = " ".join(lines[start].split())
            if _PAGE_MARKER_RE.fullmatch(previous):
                break
            if not previous:
                continue
            fragment = _join_wrapped_heading(previous, fragment)
        name = " ".join(fragment.split())
        if not name or len(name) > 2_000 or not any(char.isalpha() for char in name):
            continue
        heading = f"{name} ({alias}):"
        locator = f"page={pages[index] or 'unknown'};lines={start + 1}-{index + 1};alias={alias}"
        found.append((name, locator, _sha256_bytes(heading.encode("utf-8"))))
    unique = sorted(set(found))
    return unique


def _opsin_exchange(systematic_name: str, *, transport: httpx.BaseTransport | None = None) -> OpsinExchange:
    endpoint = f"{OPSIN_BASE_URL}/{quote(systematic_name, safe='')}.json"
    try:
        with httpx.Client(
            transport=transport,
            timeout=httpx.Timeout(30.0),
            follow_redirects=False,
            headers={"Accept": "application/json", "User-Agent": "Molly-PR-AN/1"},
        ) as client:
            with client.stream("GET", endpoint) as response:
                payload = bytearray()
                for chunk in response.iter_bytes():
                    payload.extend(chunk)
                    if len(payload) > _MAX_RESPONSE_BYTES:
                        raise ValueError("OPSIN response has an unsupported size")
                return OpsinExchange(
                    endpoint_url=endpoint,
                    http_status=response.status_code,
                    response_bytes=bytes(payload),
                )
    except ValueError:
        raise
    except httpx.RequestError as exc:
        raise ValueError("OPSIN service request failed") from exc


def _result_payload(item: ContextualAliasRequestItem, parsed_text: str, resolver: Callable[[str], OpsinExchange]) -> dict[str, Any]:
    occurrences = _locate_systematic_names(parsed_text, item.reported_alias)
    base: dict[str, Any] = {
        "candidate_id": item.candidate_id,
        "reported_alias": item.reported_alias,
        "status": "alias_not_found",
        "rejection_reason": "alias has no systematic-name heading in the parsed text",
        "result_digest": "sha256:" + "0" * 64,
    }
    if len(occurrences) != 1:
        if occurrences:
            base["status"] = "alias_ambiguous"
            base["rejection_reason"] = "alias has multiple systematic-name headings in the parsed text"
        return base
    systematic_name, locator, heading_sha = occurrences[0]
    exchange = resolver(systematic_name)
    response_sha = _sha256_bytes(exchange.response_bytes)
    response = _load_json_without_duplicate_keys(exchange.response_bytes, label="OPSIN response")
    if not isinstance(response, dict):
        raise ValueError("OPSIN response must be a JSON object")
    resolver_status = str(response.get("status") or "").strip().upper()
    message = str(response.get("message") or "").strip()[:2_000]
    raw_warnings = response.get("warnings") or []
    if isinstance(raw_warnings, str):
        raw_warnings = [raw_warnings]
    if not isinstance(raw_warnings, list):
        raise ValueError("OPSIN warnings have an invalid shape")
    warnings = sorted(
        set(
            _clean_authored_text(str(value), field_name="resolver_warning", maximum=2_000)
            for value in raw_warnings
            if str(value).strip()
        )
    )
    common = {
        "systematic_name": systematic_name,
        "source_locator": locator,
        "source_heading_sha256": heading_sha,
        "resolver_id": OPSIN_RESOLVER_ID,
        "resolver_version": OPSIN_RESOLVER_VERSION,
        "resolver_endpoint_url": exchange.endpoint_url,
        "resolver_http_status": exchange.http_status,
        "resolver_response_sha256": response_sha,
        "resolver_response_json": exchange.response_bytes.decode("utf-8"),
        "resolver_status": resolver_status or "UNKNOWN",
        "resolver_message": message,
        "resolver_warnings": warnings,
    }
    base.update(common)
    if exchange.http_status != 200 or resolver_status != "SUCCESS":
        base["status"] = "name_resolution_rejected"
        base["rejection_reason"] = message or "OPSIN did not resolve the systematic name"
        return base
    smiles = str(response.get("smiles") or "").strip()
    standard_inchi = str(response.get("stdinchi") or "").strip()
    standard_inchikey = str(response.get("stdinchikey") or "").strip()
    observation = _rdkit_observation(smiles)
    if standard_inchi != observation["standard_inchi"] or standard_inchikey != observation["inchikey"]:
        raise ValueError("OPSIN and RDKit standard identifiers disagree")
    base.update(
        {
            "status": "candidate_ready",
            "resolver_smiles": smiles,
            "resolver_standard_inchi": standard_inchi,
            "resolver_standard_inchikey": standard_inchikey,
            **observation,
            "rejection_reason": "",
        }
    )
    return base


def build_contextual_alias_resolution_artifact(
    request: ContextualAliasResolutionRequest,
    *,
    request_sha256: str,
    parsed_text: str,
    resolver: Callable[[str], OpsinExchange],
    generated_at: str | None = None,
) -> ContextualAliasResolutionArtifact:
    results: list[ContextualAliasResolutionResult] = []
    embedded_response_bytes = 0
    for item in request.items:
        payload = _result_payload(item, parsed_text, resolver)
        embedded_response_bytes += len(str(payload.get("resolver_response_json", "")).encode("utf-8"))
        if embedded_response_bytes > _MAX_EMBEDDED_RESPONSE_BYTES:
            raise ValueError("contextual alias resolver evidence exceeds the artifact limit")
        # Digest the complete normalized shape, including defaulted optional
        # fields and publication-boundary flags.
        draft = ContextualAliasResolutionResult.model_construct(**payload)
        complete_payload = draft.model_dump(mode="json")
        complete_payload["result_digest"] = _stable_hash(
            {key: value for key, value in complete_payload.items() if key != "result_digest"}
        )
        results.append(ContextualAliasResolutionResult.model_validate(complete_payload))
    timestamp = generated_at or datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    artifact_payload: dict[str, Any] = {
        "artifact_version": CONTEXTUAL_ALIAS_ARTIFACT_VERSION,
        "profile_version": CONTEXTUAL_ALIAS_PROFILE_VERSION,
        "run_id": request.run_id,
        "paper_id": request.paper_id,
        "source_document_id": request.source_document_id,
        "generated_at": timestamp,
        "request_sha256": _normalize_sha256(request_sha256, field_name="request_sha256"),
        "request_digest": _stable_hash(request.model_dump(mode="json")),
        "request": request.model_dump(mode="json"),
        "parsed_text_sha256": request.parsed_text_sha256,
        "resolver_id": OPSIN_RESOLVER_ID,
        "resolver_version": OPSIN_RESOLVER_VERSION,
        "result_count": len(results),
        "ready_count": sum(item.status == "candidate_ready" for item in results),
        "rejected_count": sum(item.status != "candidate_ready" for item in results),
        "results": [item.model_dump(mode="json") for item in results],
        "artifact_digest": "sha256:" + "0" * 64,
        "candidate_only": True,
        "source_match_validated": False,
        "identity_resolved": False,
        "registry_mutated": False,
        "gold_written": False,
        "dataset_written": False,
    }
    artifact_payload["artifact_digest"] = _stable_hash(
        {key: value for key, value in artifact_payload.items() if key != "artifact_digest"}
    )
    return ContextualAliasResolutionArtifact.model_validate(artifact_payload)


def _publish_artifact(
    *, output_path: Path, parent_descriptor: int, parent_stat: os.stat_result, artifact: ContextualAliasResolutionArtifact
) -> ContextualAliasResolutionArtifact:
    no_follow = getattr(os, "O_NOFOLLOW", None)
    if no_follow is None:
        raise ValueError("contextual alias publisher requires O_NOFOLLOW support")
    encoded = (
        json.dumps(artifact.model_dump(mode="json"), ensure_ascii=False, allow_nan=False, sort_keys=True, indent=2)
        + "\n"
    ).encode("utf-8")
    if len(encoded) > _MAX_OUTPUT_BYTES:
        raise ValueError("contextual alias output has an unsupported size")
    descriptor = -1
    created_stat: os.stat_result | None = None
    keep = False
    try:
        _validate_directory_path_binding(output_path.parent, parent_descriptor, error_message="contextual alias output parent changed")
        _ensure_fresh_output_at(parent_descriptor, output_path.name)
        descriptor = os.open(
            output_path.name,
            os.O_RDWR | os.O_CREAT | os.O_EXCL | no_follow,
            0o600,
            dir_fd=parent_descriptor,
        )
        created_stat = os.fstat(descriptor)
        _write_all(descriptor, encoded)
        os.fsync(descriptor)
        os.fsync(parent_descriptor)
        named_stat = os.stat(output_path.name, dir_fd=parent_descriptor, follow_symlinks=False)
        final_parent = _validate_directory_path_binding(
            output_path.parent, parent_descriptor, error_message="contextual alias output parent changed"
        )
        if (
            not stat.S_ISREG(named_stat.st_mode)
            or named_stat.st_dev != created_stat.st_dev
            or named_stat.st_ino != created_stat.st_ino
            or named_stat.st_size != len(encoded)
            or final_parent.st_dev != parent_stat.st_dev
            or final_parent.st_ino != parent_stat.st_ino
        ):
            raise ValueError("contextual alias output publication changed")
        actual = _read_open_descriptor(descriptor, max_bytes=_MAX_OUTPUT_BYTES)
        if actual != encoded:
            raise ValueError("contextual alias output bytes changed")
        validated = ContextualAliasResolutionArtifact.model_validate(
            _load_json_without_duplicate_keys(actual, label="contextual alias artifact")
        )
        if validated != artifact:
            raise ValueError("contextual alias output artifact changed")
        final_named = os.stat(output_path.name, dir_fd=parent_descriptor, follow_symlinks=False)
        final_descriptor = os.fstat(descriptor)
        final_parent = _validate_directory_path_binding(
            output_path.parent, parent_descriptor, error_message="contextual alias output parent changed"
        )
        if (
            final_named.st_dev != created_stat.st_dev
            or final_named.st_ino != created_stat.st_ino
            or final_descriptor.st_dev != created_stat.st_dev
            or final_descriptor.st_ino != created_stat.st_ino
            or final_named.st_size != len(encoded)
            or final_descriptor.st_size != len(encoded)
            or final_parent.st_dev != parent_stat.st_dev
            or final_parent.st_ino != parent_stat.st_ino
        ):
            raise ValueError("contextual alias output publication changed")
        keep = True
        return validated
    except FileExistsError as exc:
        raise ValueError("contextual alias output already exists") from exc
    except ValueError:
        raise
    except OSError as exc:
        raise ValueError("contextual alias output cannot be published") from exc
    finally:
        if created_stat is not None and not keep:
            try:
                current = os.stat(output_path.name, dir_fd=parent_descriptor, follow_symlinks=False)
                if current.st_dev == created_stat.st_dev and current.st_ino == created_stat.st_ino:
                    os.unlink(output_path.name, dir_fd=parent_descriptor)
                    os.fsync(parent_descriptor)
            except OSError:
                pass
        if descriptor != -1:
            os.close(descriptor)


def build_contextual_alias_resolution_from_files(
    *,
    request_json: str | Path,
    output_json: str | Path,
    generated_at: str | None = None,
    resolver: Callable[[str], OpsinExchange] | None = None,
) -> ContextualAliasResolutionArtifact:
    request_path = _absolute_path(Path(request_json))
    output_path = _absolute_path(Path(output_json))
    request_bytes = _read_exact_regular_file(request_path)
    request = ContextualAliasResolutionRequest.model_validate(
        _load_json_without_duplicate_keys(request_bytes, label="contextual alias request")
    )
    text_path = request_path.parent / request.parsed_text_file
    text_bytes = _read_exact_regular_file(text_path)
    if len(text_bytes) > _MAX_INPUT_BYTES or _sha256_bytes(text_bytes) != request.parsed_text_sha256:
        raise ValueError("contextual alias parsed text SHA-256 mismatch")
    try:
        parsed_text = text_bytes.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError("contextual alias parsed text must be UTF-8") from exc
    with _pinned_output_parent(output_path.parent) as (parent_descriptor, parent_stat):
        _ensure_fresh_output_at(parent_descriptor, output_path.name)
        artifact = build_contextual_alias_resolution_artifact(
            request,
            request_sha256=_sha256_bytes(request_bytes),
            parsed_text=parsed_text,
            resolver=resolver or _opsin_exchange,
            generated_at=generated_at,
        )
        if _read_exact_regular_file(request_path) != request_bytes or _read_exact_regular_file(text_path) != text_bytes:
            raise ValueError("contextual alias inputs changed during execution")
        return _publish_artifact(
            output_path=output_path,
            parent_descriptor=parent_descriptor,
            parent_stat=parent_stat,
            artifact=artifact,
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Resolve paper-context aliases through SI systematic-name headings")
    parser.add_argument("--request", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    try:
        artifact = build_contextual_alias_resolution_from_files(request_json=args.request, output_json=args.output)
    except Exception as exc:
        print(json.dumps({"error_code": "contextual_alias_resolution_failed", "exception_type": type(exc).__name__}, sort_keys=True))
        return 1
    print(
        json.dumps(
            {
                "artifact_digest": artifact.artifact_digest,
                "ready_count": artifact.ready_count,
                "rejected_count": artifact.rejected_count,
                "run_id": artifact.run_id,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
