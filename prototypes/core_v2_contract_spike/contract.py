"""Small, offline-only contract spike for the proposed Core v2 authority model.

This module intentionally has no provider, network, shell, credential, LLM,
remote-compute, or GPU integration. It is a test vehicle for the data model,
not a production implementation or a generic plugin framework.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Callable, Mapping


SPIKE_BOUNDARY = (
    "NOT_PRODUCTION",
    "NOT_INSTALLED",
    "NO_NETWORK",
    "NO_LLM",
    "NO_REMOTE_COMPUTE",
    "NO_GPU",
)
FORBIDDEN_CAPABILITIES = frozenset(
    {"credential", "gpu", "llm", "network", "remote_compute", "shell"}
)


class ContractViolation(ValueError):
    """Raised when a contract invariant fails closed."""


def _canonical_bytes(value: object) -> bytes:
    try:
        encoded = json.dumps(
            value,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ContractViolation("value is not canonical JSON") from exc
    return encoded


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


@dataclass(frozen=True)
class RunRequest:
    """Immutable request selecting one declared tool and immutable inputs."""

    run_id: str
    tool_name: str
    input_artifact_ids: tuple[str, ...]
    policy_digest: str
    approval_digest: str | None = None

    def __post_init__(self) -> None:
        if not self.run_id or not self.tool_name or not self.policy_digest:
            raise ContractViolation("run, tool, and policy identity are required")
        object.__setattr__(self, "input_artifact_ids", tuple(self.input_artifact_ids))
        if any(not artifact_id for artifact_id in self.input_artifact_ids):
            raise ContractViolation("input artifact identity must be non-empty")

    def _payload(self, *, include_approval: bool) -> dict[str, object]:
        payload: dict[str, object] = {
            "run_id": self.run_id,
            "tool_name": self.tool_name,
            "input_artifact_ids": list(self.input_artifact_ids),
            "policy_digest": self.policy_digest,
        }
        if include_approval:
            payload["approval_digest"] = self.approval_digest
        return payload

    @property
    def request_digest(self) -> str:
        """Digest used by an approval; it excludes the approval itself."""

        return _sha256(_canonical_bytes(self._payload(include_approval=False)))

    @property
    def digest(self) -> str:
        return _sha256(_canonical_bytes(self._payload(include_approval=True)))


@dataclass(frozen=True)
class ToolSpec:
    """A declared deterministic callable with no privileged capability labels."""

    name: str
    version: str
    handler: Callable[[bytes], bytes]
    deterministic: bool = True
    capabilities: frozenset[str] = frozenset()

    def __post_init__(self) -> None:
        if not self.name or not self.version or not callable(self.handler):
            raise ContractViolation("tool name, version, and handler are required")
        object.__setattr__(self, "capabilities", frozenset(self.capabilities))
        forbidden = self.capabilities & FORBIDDEN_CAPABILITIES
        if forbidden:
            raise ContractViolation(f"forbidden tool capability: {sorted(forbidden)}")
        if not self.deterministic:
            raise ContractViolation("the spike accepts deterministic tools only")


class ToolRegistry:
    """Closed in-memory registry for the spike's explicitly declared tools."""

    def __init__(self) -> None:
        self._tools: dict[str, ToolSpec] = {}

    def register(self, spec: ToolSpec) -> None:
        if spec.name in self._tools:
            raise ContractViolation(f"duplicate tool: {spec.name}")
        self._tools[spec.name] = spec

    def resolve(self, name: str) -> ToolSpec:
        try:
            return self._tools[name]
        except KeyError as exc:
            raise ContractViolation(f"unknown tool: {name}") from exc


@dataclass(frozen=True)
class ToolPolicy:
    """Closed allowlist plus exact tool names requiring approval."""

    allowed_tools: frozenset[str]
    approval_required: frozenset[str] = frozenset()

    def __post_init__(self) -> None:
        object.__setattr__(self, "allowed_tools", frozenset(self.allowed_tools))
        object.__setattr__(self, "approval_required", frozenset(self.approval_required))
        if not self.approval_required <= self.allowed_tools:
            raise ContractViolation("approval-required tools must be allowed")

    @property
    def digest(self) -> str:
        payload = {
            "allowed_tools": sorted(self.allowed_tools),
            "approval_required": sorted(self.approval_required),
        }
        return _sha256(_canonical_bytes(payload))

    def allows(self, tool_name: str) -> bool:
        return tool_name in self.allowed_tools

    def requires_approval(self, tool_name: str) -> bool:
        return tool_name in self.approval_required


@dataclass(frozen=True)
class ApprovalRecord:
    """Exact approval binding to one request digest and policy digest."""

    run_id: str
    tool_name: str
    request_digest: str
    policy_digest: str
    approved_by: str

    @classmethod
    def for_request(
        cls, request: RunRequest, policy: ToolPolicy, approved_by: str
    ) -> "ApprovalRecord":
        if request.policy_digest != policy.digest:
            raise ContractViolation("approval policy does not match request policy")
        if not approved_by:
            raise ContractViolation("approval actor is required")
        return cls(
            run_id=request.run_id,
            tool_name=request.tool_name,
            request_digest=request.request_digest,
            policy_digest=policy.digest,
            approved_by=approved_by,
        )

    @property
    def digest(self) -> str:
        return _sha256(
            _canonical_bytes(
                {
                    "run_id": self.run_id,
                    "tool_name": self.tool_name,
                    "request_digest": self.request_digest,
                    "policy_digest": self.policy_digest,
                    "approved_by": self.approved_by,
                }
            )
        )


@dataclass(frozen=True)
class ArtifactRecord:
    """Immutable content-addressed artifact metadata."""

    artifact_id: str
    content_sha256: str
    size: int
    content_type: str

    def to_dict(self) -> dict[str, object]:
        return {
            "artifact_id": self.artifact_id,
            "content_sha256": self.content_sha256,
            "size": self.size,
            "content_type": self.content_type,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> "ArtifactRecord":
        return cls(
            artifact_id=str(value["artifact_id"]),
            content_sha256=str(value["content_sha256"]),
            size=int(value["size"]),
            content_type=str(value["content_type"]),
        )


class ArtifactStore:
    """Minimal restartable store that never replaces bytes under an identity."""

    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self.artifacts_root = self.root / "artifacts"
        self.artifacts_root.mkdir(parents=True, exist_ok=True)

    def _paths(self, artifact_id: str) -> tuple[Path, Path]:
        if not artifact_id.startswith("sha256:"):
            raise ContractViolation("invalid artifact identity")
        digest = artifact_id.removeprefix("sha256:")
        if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
            raise ContractViolation("invalid artifact digest")
        return (
            self.artifacts_root / f"{digest}.bin",
            self.artifacts_root / f"{digest}.json",
        )

    def put_bytes(self, content: bytes, *, content_type: str) -> ArtifactRecord:
        if not isinstance(content, bytes) or not content_type:
            raise ContractViolation("artifact content and type are required")
        digest = _sha256(content)
        record = ArtifactRecord(
            artifact_id=f"sha256:{digest}",
            content_sha256=digest,
            size=len(content),
            content_type=content_type,
        )
        data_path, manifest_path = self._paths(record.artifact_id)
        if data_path.exists() or manifest_path.exists():
            if not data_path.exists() or not manifest_path.exists():
                raise ContractViolation("incomplete immutable artifact")
            existing = ArtifactRecord.from_dict(
                json.loads(manifest_path.read_text(encoding="utf-8"))
            )
            if existing != record or data_path.read_bytes() != content:
                raise ContractViolation("immutable artifact identity collision")
            return existing
        try:
            with data_path.open("xb") as stream:
                stream.write(content)
            with manifest_path.open("x", encoding="utf-8") as stream:
                json.dump(record.to_dict(), stream, sort_keys=True, separators=(",", ":"))
                stream.write("\n")
        except FileExistsError as exc:
            raise ContractViolation("concurrent artifact identity collision") from exc
        return record

    def get(self, artifact_id: str) -> bytes:
        data_path, manifest_path = self._paths(artifact_id)
        if not data_path.exists() or not manifest_path.exists():
            raise ContractViolation(f"unknown artifact: {artifact_id}")
        content = data_path.read_bytes()
        record = ArtifactRecord.from_dict(
            json.loads(manifest_path.read_text(encoding="utf-8"))
        )
        if record.artifact_id != artifact_id or record.content_sha256 != _sha256(content):
            raise ContractViolation("artifact integrity check failed")
        if record.size != len(content):
            raise ContractViolation("artifact size check failed")
        return content

    def contains(self, artifact_id: str) -> bool:
        data_path, manifest_path = self._paths(artifact_id)
        return data_path.exists() and manifest_path.exists()


class RunLedger:
    """Append-only JSONL ledger with a verifiable hash chain."""

    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.path = self.root / "run-ledger.jsonl"
        self._records: list[dict[str, object]] = []
        self._load()

    @staticmethod
    def _record_digest(record: Mapping[str, object]) -> str:
        return _sha256(_canonical_bytes(record))

    def _load(self) -> None:
        if not self.path.exists():
            return
        previous: str | None = None
        for expected_sequence, line in enumerate(
            self.path.read_text(encoding="utf-8").splitlines(), start=1
        ):
            if not line:
                raise ContractViolation("blank line in append-only ledger")
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ContractViolation("invalid ledger record") from exc
            digest = record.pop("record_digest", None)
            if record.get("sequence") != expected_sequence:
                raise ContractViolation("ledger sequence mutation")
            if record.get("previous_digest") != previous:
                raise ContractViolation("ledger hash-chain mutation")
            if not isinstance(digest, str) or digest != self._record_digest(record):
                raise ContractViolation("ledger record digest mismatch")
            record["record_digest"] = digest
            self._records.append(record)
            previous = digest

    @property
    def records(self) -> tuple[dict[str, object], ...]:
        return tuple(dict(record) for record in self._records)

    def append(
        self, *, event_type: str, run_id: str, payload: Mapping[str, object]
    ) -> str:
        if not event_type or not run_id:
            raise ContractViolation("ledger event identity is required")
        sequence = len(self._records) + 1
        previous = self._records[-1]["record_digest"] if self._records else None
        record: dict[str, object] = {
            "sequence": sequence,
            "event_type": event_type,
            "run_id": run_id,
            "payload": dict(payload),
            "previous_digest": previous,
        }
        digest = self._record_digest(record)
        record["record_digest"] = digest
        with self.path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, sort_keys=True, separators=(",", ":")))
            stream.write("\n")
        self._records.append(record)
        return digest


@dataclass(frozen=True)
class ArtifactLineage:
    """Input/output dependency record for one deterministic tool execution."""

    run_id: str
    tool_name: str
    input_artifact_ids: tuple[str, ...]
    output_artifact_ids: tuple[str, ...]

    @property
    def digest(self) -> str:
        return _sha256(_canonical_bytes(self.to_dict()))

    def to_dict(self) -> dict[str, object]:
        return {
            "run_id": self.run_id,
            "tool_name": self.tool_name,
            "input_artifact_ids": list(self.input_artifact_ids),
            "output_artifact_ids": list(self.output_artifact_ids),
        }


@dataclass(frozen=True)
class ExecutionResult:
    output_artifact: ArtifactRecord
    lineage: ArtifactLineage
    ledger_record_digest: str


def deterministic_example_tool(content: bytes) -> bytes:
    """Return canonical JSON and perform no I/O other than its argument."""

    try:
        value = json.loads(content.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ContractViolation("example tool accepts JSON artifact bytes") from exc
    return _canonical_bytes({"input": value, "tool": "deterministic.echo"})


def example_tool_registry() -> ToolRegistry:
    registry = ToolRegistry()
    registry.register(
        ToolSpec(
            name="deterministic.echo",
            version="1",
            handler=deterministic_example_tool,
        )
    )
    return registry


def execute(
    request: RunRequest,
    *,
    registry: ToolRegistry,
    policy: ToolPolicy,
    artifact_store: ArtifactStore,
    ledger: RunLedger,
    approval: ApprovalRecord | None = None,
) -> ExecutionResult:
    """Execute one declared deterministic tool after all fail-closed checks."""

    if request.policy_digest != policy.digest:
        raise ContractViolation("request policy digest mismatch")
    spec = registry.resolve(request.tool_name)
    if not policy.allows(request.tool_name):
        raise ContractViolation(f"tool is disallowed: {request.tool_name}")
    if policy.requires_approval(request.tool_name):
        if approval is None:
            raise ContractViolation("exact approval is required")
        if request.approval_digest != approval.digest:
            raise ContractViolation("approval digest mismatch")
        if (
            approval.run_id != request.run_id
            or approval.tool_name != request.tool_name
            or approval.request_digest != request.request_digest
            or approval.policy_digest != policy.digest
        ):
            raise ContractViolation("approval binding mismatch")
    elif request.approval_digest is not None:
        raise ContractViolation("unexpected approval on non-approval action")

    if len(request.input_artifact_ids) != 1:
        raise ContractViolation("the example tool requires exactly one input")
    input_id = request.input_artifact_ids[0]
    input_content = artifact_store.get(input_id)
    output_content = spec.handler(input_content)
    if not isinstance(output_content, bytes):
        raise ContractViolation("tool output must be bytes")
    output = artifact_store.put_bytes(
        output_content, content_type="application/json"
    )
    lineage = ArtifactLineage(
        run_id=request.run_id,
        tool_name=spec.name,
        input_artifact_ids=(input_id,),
        output_artifact_ids=(output.artifact_id,),
    )
    ledger_digest = ledger.append(
        event_type="tool_execution",
        run_id=request.run_id,
        payload={
            "request_digest": request.digest,
            "tool_name": spec.name,
            "tool_version": spec.version,
            "input_artifact_ids": list(lineage.input_artifact_ids),
            "output_artifact_ids": list(lineage.output_artifact_ids),
            "lineage_digest": lineage.digest,
            "approval_digest": approval.digest if approval is not None else None,
        },
    )
    return ExecutionResult(
        output_artifact=output,
        lineage=lineage,
        ledger_record_digest=ledger_digest,
    )
