from __future__ import annotations

from contextlib import AbstractContextManager
from pathlib import Path
import shutil
from typing import Any

import pytest

from ai4s_agent.harness_tracing import (
    NoopHarnessTracer,
    OpenTelemetryHarnessTracer,
)
from ai4s_agent.schemas import (
    AgentToolCallApplicationRequest,
    AgentToolCallProposalRequest,
    _agent_digest,
)
from tests.execution_agent_test_support import (
    CountingStubProvider,
    execution_agent_service,
    local_controller_execution,
    reopen_local_controller,
)


class _FailingDelegate:
    def start_as_current_span(self, *args: Any, **kwargs: Any):
        raise RuntimeError("private exporter context failure")


class _FailingProvider:
    def shutdown(self) -> None:
        raise RuntimeError("private exporter shutdown failure")


class _RecordingSpan:
    def __init__(self, record: dict[str, Any]) -> None:
        self.record = record

    def set_attribute(self, key: str, value: str | int) -> None:
        self.record["attributes"][key] = value

    def add_event(self, name: str, *, attributes: dict[str, Any]) -> None:
        self.record["events"].append((name, attributes))


class _RecordingContext(AbstractContextManager[_RecordingSpan]):
    def __init__(self, record: dict[str, Any]) -> None:
        self.record = record

    def __enter__(self) -> _RecordingSpan:
        return _RecordingSpan(self.record)

    def __exit__(self, exc_type, exc_value, traceback) -> bool:
        return False


class _RecordingDelegate:
    def __init__(self) -> None:
        self.records: list[dict[str, Any]] = []

    def start_as_current_span(
        self,
        name: str,
        *,
        attributes: dict[str, Any],
        links: list[Any],
        **_: Any,
    ) -> _RecordingContext:
        record = {
            "name": name,
            "attributes": dict(attributes),
            "events": [],
            "links": list(links),
        }
        self.records.append(record)
        return _RecordingContext(record)


class _RecordingProvider:
    def shutdown(self) -> None:
        return None


def _run_pause_turn(tmp_path: Path, tracer, *, initial=None):
    if initial is None:
        storage, _, controller, initial = local_controller_execution(tmp_path)
    else:
        storage, controller = reopen_local_controller(tmp_path / "workspace")
    service = execution_agent_service(
        storage=storage,
        controller=controller,
        tracer=tracer,
    )
    proposed = service.create_proposal(
        project_id="project-1",
        controller_execution_id=initial.execution.controller_execution_id,
        request=AgentToolCallProposalRequest(
            expected_controller_execution_digest=initial.execution.execution_digest,
            client_request_id="tracing-proposal-1",
            external_llm_approved=True,
            llm_provider={"provider": "stub"},
        ),
        provider=CountingStubProvider(
            response={
                "selected_tool_id": "agent.pause_current.v1",
                "decision_summary": "Pause this bounded turn.",
            }
        ),
        provider_binding_digest=_agent_digest({"provider": "stub"}),
    )
    proposal = proposed.publication.proposal
    applied = service.apply_proposal(
        project_id="project-1",
        controller_execution_id=initial.execution.controller_execution_id,
        tool_call_proposal_id=proposal.tool_call_proposal_id,
        request=AgentToolCallApplicationRequest(
            expected_tool_call_proposal_digest=proposal.tool_call_proposal_digest,
            client_request_id="tracing-apply-1",
        ),
    )
    return storage, proposed, applied


def _project_bytes(storage) -> dict[str, bytes]:
    root = storage.project_dir("project-1")
    return {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


@pytest.mark.pr_fast
def test_exporter_failure_preserves_all_execution_agent_authoritative_bytes(
    tmp_path,
) -> None:
    source = tmp_path / "source"
    _, _, _, initial = local_controller_execution(source)
    shutil.copytree(source, tmp_path / "noop")
    shutil.copytree(source, tmp_path / "failing")
    baseline_storage, baseline_proposal, baseline_application = _run_pause_turn(
        tmp_path / "noop",
        NoopHarnessTracer(),
        initial=initial,
    )
    failing_tracer = OpenTelemetryHarnessTracer(
        tracer=_FailingDelegate(),
        provider=_FailingProvider(),
    )
    traced_storage, traced_proposal, traced_application = _run_pause_turn(
        tmp_path / "failing",
        failing_tracer,
        initial=initial,
    )
    failing_tracer.shutdown()
    assert baseline_proposal == traced_proposal
    assert baseline_application == traced_application
    assert _project_bytes(baseline_storage) == _project_bytes(traced_storage)


def test_execution_agent_tracing_emits_only_safe_allowlisted_metadata(tmp_path) -> None:
    delegate = _RecordingDelegate()
    tracer = OpenTelemetryHarnessTracer(
        tracer=delegate,
        provider=_RecordingProvider(),
    )
    _run_pause_turn(tmp_path, tracer)
    assert {
        "execution_agent.propose",
        "execution_agent.observe",
        "execution_agent.llm_call",
        "execution_agent.validate_response",
        "execution_agent.publish_proposal",
        "execution_agent.apply",
    }.issubset({record["name"] for record in delegate.records})
    serialized = repr(delegate.records).lower()
    for forbidden in (
        "decision_summary",
        "prompt text",
        "api_key",
        "endpoint",
        "/private/",
        "conversation",
    ):
        assert forbidden not in serialized
