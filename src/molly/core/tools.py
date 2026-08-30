"""Closed host-owned tool contracts for the CORE-02 AgentLoop."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Protocol, runtime_checkable

from jsonschema import Draft202012Validator, SchemaError, ValidationError

from .errors import (
    CoreContractError,
    SchemaValidationError,
    ToolAccessError,
    ToolContractError,
    ToolPolicyError,
)
from .ids import (
    canonical_json_bytes,
    freeze_json_mapping,
    normalize_timestamp,
    sha256_bytes,
    thaw_json,
    utc_timestamp,
    validate_artifact_ids,
    validate_digest_reference,
    validate_identifier,
    validate_reference,
)


class SideEffectClass(str, Enum):
    """Closed side-effect vocabulary; values do not authorize implementations."""

    PURE = "PURE"
    LOCAL_ARTIFACT = "LOCAL_ARTIFACT"
    NETWORK_READ = "NETWORK_READ"
    EXTERNAL_WRITE = "EXTERNAL_WRITE"
    REMOTE_COMPUTE = "REMOTE_COMPUTE"


SIDE_EFFECT_CLASSES = frozenset(item.value for item in SideEffectClass)


def _side_effect_value(value: str | SideEffectClass) -> str:
    candidate = value.value if isinstance(value, SideEffectClass) else value
    if not isinstance(candidate, str):
        raise ToolContractError("side_effect_class must be a string")
    normalized = candidate.strip().upper()
    if normalized not in SIDE_EFFECT_CLASSES:
        raise ToolContractError(f"unknown side_effect_class: {candidate!r}")
    return normalized


def _reject_external_schema_refs(value: Any) -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            if key in {"$ref", "$dynamicRef", "$recursiveRef"}:
                raise ToolContractError("external or indirect JSON schema references are disabled")
            _reject_external_schema_refs(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            _reject_external_schema_refs(item)


def _validated_schema(value: Mapping[str, Any], *, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ToolContractError(f"{field} must be a JSON schema object")
    try:
        _reject_external_schema_refs(value)
        frozen = freeze_json_mapping(value, field=field)
        Draft202012Validator.check_schema(thaw_json(frozen))
    except SchemaError as exc:
        raise ToolContractError(f"{field} is not a valid JSON schema") from exc
    return frozen


def _validate_json_data(value: Any, *, field: str) -> None:
    try:
        canonical_json_bytes(value)
    except CoreContractError as exc:
        raise ToolContractError(f"{field} must contain canonical JSON data") from exc


def _validate_schema_data(
    schema: Mapping[str, Any], value: Any, *, field: str
) -> None:
    try:
        Draft202012Validator(thaw_json(schema)).validate(thaw_json(value))
    except ValidationError as exc:
        path = ".".join(str(part) for part in exc.absolute_path)
        suffix = f" at {path}" if path else ""
        raise SchemaValidationError(f"{field} does not match its JSON schema{suffix}") from exc


@dataclass(frozen=True, slots=True)
class ToolSpec:
    """One exact, server-owned version of a bounded tool."""

    name: str
    version: str = "1"
    description: str = ""
    input_schema: Mapping[str, Any] = field(default_factory=lambda: {"type": "object"})
    output_schema: Mapping[str, Any] = field(default_factory=lambda: {})
    side_effect_class: str | SideEffectClass = SideEffectClass.PURE
    requires_approval: bool = False

    def __post_init__(self) -> None:
        validate_identifier(self.name, field="tool name")
        validate_identifier(self.version, field="tool version")
        if not isinstance(self.description, str) or not self.description.strip():
            raise ToolContractError("tool description is required")
        object.__setattr__(self, "side_effect_class", _side_effect_value(self.side_effect_class))
        if not isinstance(self.requires_approval, bool):
            raise ToolContractError("requires_approval must be boolean")
        object.__setattr__(
            self,
            "input_schema",
            _validated_schema(self.input_schema, field="input_schema"),
        )
        object.__setattr__(
            self,
            "output_schema",
            _validated_schema(self.output_schema, field="output_schema"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "version": self.version,
            "description": self.description,
            "input_schema": thaw_json(self.input_schema),
            "output_schema": thaw_json(self.output_schema),
            "side_effect_class": self.side_effect_class,
            "requires_approval": self.requires_approval,
        }

    @property
    def spec_digest(self) -> str:
        return sha256_bytes(canonical_json_bytes(self.to_dict()))

    @property
    def digest(self) -> str:
        return self.spec_digest

    def model_view(self) -> dict[str, Any]:
        """Return the only tool information exposed to a DecisionProvider."""

        return {
            "name": self.name,
            "description": self.description,
            "input_schema": thaw_json(self.input_schema),
        }

    @property
    def side_effect(self) -> str:
        """Descriptive alias for the closed side-effect class."""

        return self.side_effect_class

    def validate_arguments(self, arguments: Mapping[str, Any]) -> None:
        if not isinstance(arguments, Mapping):
            raise SchemaValidationError("tool arguments must be a JSON object")
        _validate_schema_data(self.input_schema, arguments, field="tool arguments")

    def validate_output(self, data: Any) -> None:
        _validate_json_data(data, field="tool output")
        _validate_schema_data(self.output_schema, data, field="tool output")


ToolExecutor = Callable[["ToolExecutionContext"], "ToolResult"]


@dataclass(frozen=True, slots=True)
class _RegisteredTool:
    spec: ToolSpec
    executor: ToolExecutor


class ToolRegistry:
    """Closed server-owned map from exact tool versions to executors."""

    def __init__(self, tools: Sequence[tuple[ToolSpec, ToolExecutor]] = ()) -> None:
        self._tools: dict[tuple[str, str], _RegisteredTool] = {}
        for spec, executor in tools:
            self.register(spec, executor)

    def register(self, spec: ToolSpec, executor: ToolExecutor) -> None:
        """Register a host callable; callers must never pass model data here."""

        if not isinstance(spec, ToolSpec):
            raise ToolContractError("registry accepts only ToolSpec")
        if not callable(executor):
            raise ToolContractError("tool executor must be callable")
        key = (spec.name, spec.version)
        if key in self._tools:
            raise ToolContractError(f"duplicate tool registration: {spec.name}@{spec.version}")
        self._tools[key] = _RegisteredTool(spec=spec, executor=executor)

    def register_tool(self, spec: ToolSpec, executor: ToolExecutor) -> None:
        """Explicit host-side alias; no model-facing registration path exists."""

        self.register(spec, executor)

    def _by_name(self, name: str) -> list[_RegisteredTool]:
        validate_identifier(name, field="tool name")
        return [entry for (tool_name, _), entry in self._tools.items() if tool_name == name]

    def resolve(self, name: str, *, version: str | None = None) -> ToolSpec:
        candidates = self._by_name(name)
        if version is not None:
            validate_identifier(version, field="tool version")
            candidates = [entry for entry in candidates if entry.spec.version == version]
        if not candidates:
            raise ToolContractError(f"unknown tool: {name!r}")
        if len(candidates) != 1:
            raise ToolContractError(f"tool name is ambiguous; exact version required: {name!r}")
        return candidates[0].spec

    def resolve_exact(self, name: str, version: str, spec_digest: str) -> ToolSpec:
        spec = self.resolve(name, version=version)
        expected = validate_digest_reference(spec_digest, field="tool_spec_digest")
        if spec.spec_digest != expected:
            raise ToolContractError("registered tool spec digest does not match the materialized call")
        return spec

    def executor_for(self, spec: ToolSpec) -> ToolExecutor:
        if not isinstance(spec, ToolSpec):
            raise ToolContractError("executor lookup requires a ToolSpec")
        try:
            entry = self._tools[(spec.name, spec.version)]
        except KeyError as exc:
            raise ToolContractError("tool is not registered") from exc
        if entry.spec.spec_digest != spec.spec_digest:
            raise ToolContractError("registered tool semantics changed")
        return entry.executor

    def model_visible_tools(self) -> tuple[Mapping[str, Any], ...]:
        return tuple(
            spec.model_view()
            for spec in sorted(
                (entry.spec for entry in self._tools.values()),
                key=lambda item: (item.name, item.version),
            )
        )

    def visible_tools(self) -> tuple[Mapping[str, Any], ...]:
        """Descriptive alias for the sanitized provider view."""

        return self.model_visible_tools()

    @property
    def specs(self) -> tuple[ToolSpec, ...]:
        return tuple(
            entry.spec
            for entry in sorted(
                self._tools.values(), key=lambda item: (item.spec.name, item.spec.version)
            )
        )


def _unique_strings(
    values: Sequence[str], *, field: str, validator: Callable[[str], str]
) -> tuple[str, ...]:
    result = tuple(validator(value) for value in values)
    if len(result) != len(set(result)):
        raise ToolContractError(f"{field} must not contain duplicates")
    return result


@dataclass(frozen=True, slots=True)
class ToolPolicy:
    """The complete immutable general execution policy for a run."""

    allowed_tools: tuple[str, ...] = ()
    allowed_side_effect_classes: tuple[str | SideEffectClass, ...] = ()
    approval_required_side_effect_classes: tuple[str | SideEffectClass, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "allowed_tools",
            _unique_strings(
                self.allowed_tools,
                field="allowed_tools",
                validator=lambda value: validate_identifier(value, field="allowed tool"),
            ),
        )
        object.__setattr__(
            self,
            "allowed_side_effect_classes",
            tuple(
                sorted(
                    _unique_strings(
                        tuple(self.allowed_side_effect_classes),
                        field="allowed_side_effect_classes",
                        validator=_side_effect_value,
                    )
                )
            ),
        )
        object.__setattr__(
            self,
            "approval_required_side_effect_classes",
            tuple(
                sorted(
                    _unique_strings(
                        tuple(self.approval_required_side_effect_classes),
                        field="approval_required_side_effect_classes",
                        validator=_side_effect_value,
                    )
                )
            ),
        )
        object.__setattr__(self, "allowed_tools", tuple(sorted(self.allowed_tools)))

    @property
    def allowed_tool_names(self) -> tuple[str, ...]:
        return self.allowed_tools

    @property
    def approval_required_classes(self) -> tuple[str, ...]:
        return self.approval_required_side_effect_classes

    def to_dict(self) -> dict[str, Any]:
        return {
            "allowed_tools": list(self.allowed_tools),
            "allowed_side_effect_classes": list(self.allowed_side_effect_classes),
            "approval_required_side_effect_classes": list(
                self.approval_required_side_effect_classes
            ),
        }

    @property
    def policy_digest(self) -> str:
        return sha256_bytes(canonical_json_bytes(self.to_dict()))

    @property
    def digest(self) -> str:
        return self.policy_digest

    def allows(self, spec: ToolSpec) -> bool:
        return (
            spec.name in self.allowed_tools
            and spec.side_effect_class in self.allowed_side_effect_classes
        )

    def is_allowed(self, spec: ToolSpec) -> bool:
        """Descriptive alias for :meth:`allows`."""

        return self.allows(spec)

    def requires_approval(self, spec: ToolSpec) -> bool:
        return spec.requires_approval or (
            spec.side_effect_class in self.approval_required_side_effect_classes
        )

    def check(self, spec: ToolSpec) -> None:
        if not self.allows(spec):
            raise ToolPolicyError(f"tool is not allowed by the run ToolPolicy: {spec.name}")


_MODEL_AUTHORITY_KEYS = frozenset(
    {
        "run_id",
        "step_id",
        "call_id",
        "tool_version",
        "tool_spec_digest",
        "policy_digest",
        "approval_id",
        "approval_digest",
        "idempotency_key",
        "executor",
        "backend",
        "path",
        "filesystem_path",
        "store_root",
        "credential",
        "credentials",
        "secret",
        "token",
        "ssh_target",
        "endpoint",
        "private_endpoint",
    }
)


def _reject_model_authority_fields(value: Any) -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            if isinstance(key, str) and key.casefold() in _MODEL_AUTHORITY_KEYS:
                raise ToolContractError(f"model action cannot provide authority field: {key}")
            _reject_model_authority_fields(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            _reject_model_authority_fields(item)


@dataclass(frozen=True, slots=True)
class ToolCallProposal:
    """The bounded model-owned portion of a future tool call."""

    tool_name: str
    arguments: Mapping[str, Any] = field(default_factory=dict)
    input_artifact_ids: tuple[str, ...] = ()
    reason_summary: str = ""

    def __post_init__(self) -> None:
        validate_identifier(self.tool_name, field="tool_name")
        if not isinstance(self.arguments, Mapping):
            raise ToolContractError("tool proposal arguments must be an object")
        _reject_model_authority_fields(self.arguments)
        object.__setattr__(
            self,
            "arguments",
            freeze_json_mapping(self.arguments, field="tool proposal arguments"),
        )
        object.__setattr__(
            self,
            "input_artifact_ids",
            validate_artifact_ids(self.input_artifact_ids, field="input_artifact_ids"),
        )
        if not isinstance(self.reason_summary, str) or len(self.reason_summary) > 2_000:
            raise ToolContractError("reason_summary must be bounded text")
        if any(char in self.reason_summary for char in "\x00"):
            raise ToolContractError("reason_summary contains NUL")

    @property
    def action_type(self) -> str:
        return "TOOL_CALL"

    def to_dict(self) -> dict[str, Any]:
        return {
            "action_type": self.action_type,
            "tool_name": self.tool_name,
            "arguments": thaw_json(self.arguments),
            "input_artifact_ids": list(self.input_artifact_ids),
            "reason_summary": self.reason_summary,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ToolCallProposal":
        if not isinstance(value, Mapping):
            raise ToolContractError("tool action must be an object")
        allowed = {"action_type", "type", "tool_name", "arguments", "input_artifact_ids", "reason_summary"}
        unknown = set(value) - allowed
        if unknown:
            raise ToolContractError(f"tool action has unknown fields: {sorted(unknown)!r}")
        try:
            return cls(
                tool_name=str(value["tool_name"]),
                arguments=dict(value.get("arguments", {})),
                input_artifact_ids=tuple(value.get("input_artifact_ids", ())),
                reason_summary=str(value.get("reason_summary", "")),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ToolContractError("tool action is malformed") from exc


@dataclass(frozen=True, slots=True)
class StopAction:
    reason: str = ""

    @property
    def action_type(self) -> str:
        return "STOP"

    def __post_init__(self) -> None:
        if not isinstance(self.reason, str) or len(self.reason) > 2_000:
            raise ToolContractError("stop reason must be bounded text")

    def to_dict(self) -> dict[str, Any]:
        return {"action_type": self.action_type, "reason": self.reason}


@dataclass(frozen=True, slots=True)
class RequestReviewAction:
    reason: str = ""
    subject_ids: tuple[str, ...] = ()

    @property
    def action_type(self) -> str:
        return "REQUEST_REVIEW"

    def __post_init__(self) -> None:
        if not isinstance(self.reason, str) or len(self.reason) > 2_000:
            raise ToolContractError("review reason must be bounded text")
        object.__setattr__(
            self,
            "subject_ids",
            _unique_strings(
                self.subject_ids,
                field="review subject_ids",
                validator=lambda value: validate_reference(value, field="review subject_id"),
            ),
        )

    @property
    def subject_artifact_ids(self) -> tuple[str, ...]:
        return self.subject_ids

    def to_dict(self) -> dict[str, Any]:
        return {
            "action_type": self.action_type,
            "reason": self.reason,
            "subject_ids": list(self.subject_ids),
        }


StructuredAction = ToolCallProposal | StopAction | RequestReviewAction


def action_from_value(value: Any) -> StructuredAction:
    """Parse a provider value while keeping the action vocabulary closed."""

    if isinstance(value, (ToolCallProposal, StopAction, RequestReviewAction)):
        return value
    if not isinstance(value, Mapping):
        raise ToolContractError("DecisionProvider returned a non-object action")
    action_type = value.get("action_type", value.get("type"))
    if not isinstance(action_type, str):
        raise ToolContractError("DecisionProvider action type is required")
    normalized = action_type.strip().upper()
    if normalized in {"TOOL_CALL", "TOOL_CALL_PROPOSAL"}:
        return ToolCallProposal.from_dict(value)
    if normalized == "STOP":
        allowed = {"action_type", "type", "reason"}
        if set(value) - allowed:
            raise ToolContractError("stop action has unknown fields")
        return StopAction(reason=str(value.get("reason", "")))
    if normalized in {"REQUEST_REVIEW", "REVIEW"}:
        allowed = {"action_type", "type", "reason", "subject_ids", "subject_artifact_ids"}
        if set(value) - allowed:
            raise ToolContractError("review action has unknown fields")
        subjects = value.get("subject_ids", value.get("subject_artifact_ids", ()))
        return RequestReviewAction(reason=str(value.get("reason", "")), subject_ids=tuple(subjects))
    raise ToolContractError(f"unknown action type: {action_type!r}")


@dataclass(frozen=True, slots=True)
class MaterializedToolCall:
    """The complete server-owned, digest-bound execution identity."""

    run_id: str
    step_id: str
    call_id: str
    tool_name: str
    tool_version: str
    tool_spec_digest: str
    policy_digest: str
    arguments: Mapping[str, Any]
    input_artifact_ids: tuple[str, ...] = ()
    created_at: str = field(default_factory=utc_timestamp)
    tool_call_digest: str | None = None

    def __post_init__(self) -> None:
        for value, field_name in (
            (self.run_id, "run_id"),
            (self.step_id, "step_id"),
            (self.call_id, "call_id"),
            (self.tool_name, "tool_name"),
            (self.tool_version, "tool_version"),
        ):
            validate_identifier(value, field=field_name)
        object.__setattr__(
            self,
            "tool_spec_digest",
            validate_digest_reference(self.tool_spec_digest, field="tool_spec_digest"),
        )
        object.__setattr__(
            self,
            "policy_digest",
            validate_digest_reference(self.policy_digest, field="policy_digest"),
        )
        if not isinstance(self.arguments, Mapping):
            raise ToolContractError("materialized arguments must be an object")
        _reject_model_authority_fields(self.arguments)
        object.__setattr__(
            self,
            "arguments",
            freeze_json_mapping(self.arguments, field="materialized arguments"),
        )
        object.__setattr__(
            self,
            "input_artifact_ids",
            validate_artifact_ids(self.input_artifact_ids, field="input_artifact_ids"),
        )
        object.__setattr__(
            self,
            "created_at",
            normalize_timestamp(self.created_at, field="created_at"),
        )
        if self.tool_call_digest is not None:
            object.__setattr__(
                self,
                "tool_call_digest",
                validate_digest_reference(self.tool_call_digest, field="tool_call_digest"),
            )
        if self.tool_call_digest is None:
            object.__setattr__(self, "tool_call_digest", self.computed_digest)
        elif self.tool_call_digest != self.computed_digest:
            raise ToolContractError("materialized tool call digest mismatch")

    def _body(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "step_id": self.step_id,
            "call_id": self.call_id,
            "tool_name": self.tool_name,
            "tool_version": self.tool_version,
            "tool_spec_digest": self.tool_spec_digest,
            "policy_digest": self.policy_digest,
            "arguments": thaw_json(self.arguments),
            "input_artifact_ids": list(self.input_artifact_ids),
            "created_at": self.created_at,
        }

    @property
    def computed_digest(self) -> str:
        return sha256_bytes(canonical_json_bytes(self._body()))

    @property
    def digest(self) -> str:
        return self.computed_digest

    @property
    def idempotency_key(self) -> str:
        return self.computed_digest

    def to_dict(self) -> dict[str, Any]:
        value = self._body()
        value["tool_call_digest"] = self.computed_digest
        return value

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "MaterializedToolCall":
        if not isinstance(value, Mapping):
            raise ToolContractError("materialized tool call must be an object")
        allowed = {
            "run_id",
            "step_id",
            "call_id",
            "tool_name",
            "tool_version",
            "tool_spec_digest",
            "policy_digest",
            "arguments",
            "input_artifact_ids",
            "created_at",
            "tool_call_digest",
        }
        unknown = set(value) - allowed
        if unknown:
            raise ToolContractError(
                f"materialized tool call has unknown fields: {sorted(unknown)!r}"
            )
        try:
            return cls(
                run_id=str(value["run_id"]),
                step_id=str(value["step_id"]),
                call_id=str(value["call_id"]),
                tool_name=str(value["tool_name"]),
                tool_version=str(value["tool_version"]),
                tool_spec_digest=str(value["tool_spec_digest"]),
                policy_digest=str(value["policy_digest"]),
                arguments=dict(value["arguments"]),
                input_artifact_ids=tuple(value.get("input_artifact_ids", ())),
                created_at=str(value["created_at"]),
                tool_call_digest=str(value["tool_call_digest"]),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ToolContractError("materialized tool call is malformed") from exc


class ToolExecutionContext:
    """A bounded input reader; no store, ledger, registry, or path is exposed."""

    __slots__ = ("run_id", "step_id", "call_id", "idempotency_key", "_input_artifact_ids", "_reader")

    def __init__(
        self,
        *,
        run_id: str,
        step_id: str,
        call_id: str,
        idempotency_key: str,
        input_artifact_ids: Sequence[str],
        reader: Callable[[str], bytes],
    ) -> None:
        for value, field_name in (
            (run_id, "run_id"),
            (step_id, "step_id"),
            (call_id, "call_id"),
        ):
            validate_identifier(value, field=field_name)
        validate_digest_reference(idempotency_key, field="idempotency_key")
        if not callable(reader):
            raise ToolContractError("bounded artifact reader must be callable")
        self.run_id = run_id
        self.step_id = step_id
        self.call_id = call_id
        self.idempotency_key = idempotency_key
        self._input_artifact_ids = validate_artifact_ids(
            tuple(input_artifact_ids), field="input_artifact_ids"
        )
        self._reader = reader

    @property
    def input_artifact_ids(self) -> tuple[str, ...]:
        return self._input_artifact_ids

    def read_artifact(self, artifact_id: str) -> bytes:
        validate_reference(artifact_id, field="artifact_id")
        if artifact_id not in self._input_artifact_ids:
            raise ToolAccessError("tool may read only declared input artifacts")
        if not artifact_id.startswith("sha256:"):
            raise ToolAccessError("tool artifact reads require an artifact identity")
        return self._reader(artifact_id)


@dataclass(frozen=True, slots=True, init=False)
class ArtifactDraft:
    """Intrinsic publication data returned by a host tool."""

    content: bytes
    media_type: str
    schema_name: str | None = None
    schema_version: str | None = None

    def __init__(
        self,
        content: bytes | bytearray | memoryview | None = None,
        media_type: str = "",
        schema_name: str | None = None,
        schema_version: str | None = None,
        *,
        bytes: bytes | bytearray | memoryview | None = None,
    ) -> None:
        if content is None:
            content = bytes
        elif bytes is not None:
            raise ToolContractError("ArtifactDraft accepts content or bytes, not both")
        if content is None:
            raise ToolContractError("ArtifactDraft content is required")
        object.__setattr__(self, "content", content)
        object.__setattr__(self, "media_type", media_type)
        object.__setattr__(self, "schema_name", schema_name)
        object.__setattr__(self, "schema_version", schema_version)
        self.__post_init__()

    def __post_init__(self) -> None:
        if not isinstance(self.content, (bytes, bytearray, memoryview)):
            raise ToolContractError("ArtifactDraft content must be bytes-like")
        object.__setattr__(self, "content", bytes(self.content))
        if not isinstance(self.media_type, str) or not self.media_type.strip():
            raise ToolContractError("ArtifactDraft media_type is required")
        if any(char in self.media_type for char in "\x00\r\n"):
            raise ToolContractError("ArtifactDraft media_type contains a control character")
        for value, field_name in (
            (self.schema_name, "schema_name"),
            (self.schema_version, "schema_version"),
        ):
            if value is not None:
                validate_identifier(value, field=field_name)

    @property
    def bytes(self) -> bytes:
        """Descriptive byte-oriented alias for the intrinsic content."""

        return self.content


@dataclass(frozen=True, slots=True)
class ToolResult:
    """Structured host output plus zero or more intrinsic artifact drafts."""

    data: Any
    artifacts: tuple[ArtifactDraft, ...] = ()

    def __post_init__(self) -> None:
        _validate_json_data(self.data, field="tool result data")
        converted = tuple(
            item if isinstance(item, ArtifactDraft) else ArtifactDraft(**item)
            for item in self.artifacts
        )
        object.__setattr__(self, "artifacts", converted)

    def to_dict(self) -> dict[str, Any]:
        return {
            "data": thaw_json(self.data),
            "artifact_count": len(self.artifacts),
        }


@runtime_checkable
class DecisionProvider(Protocol):
    """Proposal source only; it owns neither execution nor authority."""

    def next_action(
        self,
        context: Any,
        model_visible_tools: Sequence[Mapping[str, Any]],
    ) -> StructuredAction:
        ...


__all__ = [
    "ArtifactDraft",
    "DecisionProvider",
    "MaterializedToolCall",
    "RequestReviewAction",
    "SideEffectClass",
    "StopAction",
    "StructuredAction",
    "ToolCallProposal",
    "ToolExecutionContext",
    "ToolExecutor",
    "ToolPolicy",
    "ToolRegistry",
    "ToolResult",
    "ToolSpec",
    "action_from_value",
]
