"""Small CLI-first operator surface for Molly Core v2 CORE-07."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, TextIO

from molly.core.approvals import ApprovalDecision
from molly.core.ids import canonical_json_bytes
from molly.core.reviews import ReviewDecision
from molly.observability import (
    JsonTraceExporter,
    LangSmithExporter,
    OpenTelemetryExporter,
)
from molly.runtime import (
    RuntimeProfileRegistry,
    RuntimeService,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="molly", description="Molly Core v2 operator surface")
    parser.add_argument(
        "--state-root",
        default=".molly",
        help="server-owned Core state root (not interpreted from model actions)",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    run = commands.add_parser("run", help="start or resume a run")
    run_commands = run.add_subparsers(dest="run_command", required=True)
    start = run_commands.add_parser("start", help="start a new run")
    start.add_argument("--profile", required=True, dest="profile_id")
    start.add_argument("--goal", required=True)
    start.add_argument("--input-artifact", action="append", default=())
    resume = run_commands.add_parser("resume", help="resume an existing run")
    resume.add_argument("run_id")

    inspect = commands.add_parser("inspect", help="read-only Core inspection")
    inspect_commands = inspect.add_subparsers(dest="inspect_command", required=True)
    inspect_run = inspect_commands.add_parser("run")
    inspect_run.add_argument("run_id")
    inspect_run.add_argument("--json", action="store_true", dest="canonical_json")
    inspect_artifact = inspect_commands.add_parser("artifact")
    inspect_artifact.add_argument("artifact_id")
    inspect_artifact.add_argument("--json", action="store_true", dest="canonical_json")

    approve = commands.add_parser("approve", help="record one exact pending-call decision")
    approve.add_argument("run_id")
    approve.add_argument("--decision", choices=tuple(item.value for item in ApprovalDecision), required=True)
    approve.add_argument("--reviewer-ref", required=True)
    approve.add_argument("--call-id")

    review = commands.add_parser("review", help="record one exact artifact review")
    review.add_argument("artifact_id")
    review.add_argument("--decision", choices=tuple(item.value for item in ReviewDecision), required=True)
    review.add_argument("--reviewer-ref", required=True)
    review.add_argument("--reason", default="")

    observe = commands.add_parser("observe", help="project and optionally export a run trace")
    observe.add_argument("run_id")
    observe.add_argument("--format", choices=("json", "human"), default="human")
    observe.add_argument("--exporter", choices=("json", "otel", "langsmith"), default="json")

    config = commands.add_parser(
        "config", help="manage server-side settings without exposing secrets to the browser"
    )
    config_commands = config.add_subparsers(dest="config_command", required=True)
    set_key = config_commands.add_parser(
        "set-key", help="save one provider key through a hidden terminal prompt"
    )
    set_key.add_argument("--profile", required=True, dest="profile_ref")
    remove_key = config_commands.add_parser("remove-key", help="remove one provider key")
    remove_key.add_argument("--profile", required=True, dest="profile_ref")

    web = commands.add_parser("web", help="start the local Molly browser interface")
    web.add_argument("--port", type=int, default=8765)
    return parser


def _human_bytes(value: Any) -> str:
    return json.dumps(value, ensure_ascii=True, sort_keys=True, indent=2) + "\n"


def _canonical_bytes(value: Any) -> str:
    return canonical_json_bytes(value).decode("utf-8") + "\n"


def _error_type(error: BaseException) -> str:
    name = type(error).__name__
    mapping = {
        "RuntimeProfileUnavailable": "RUNTIME_PROFILE_UNAVAILABLE",
        "RuntimeBindingError": "RUNTIME_BINDING_ERROR",
        "RuntimeStateError": "RUNTIME_STATE_UNAVAILABLE",
        "InspectionIntegrityError": "INSPECTION_INTEGRITY_ERROR",
        "InspectionError": "INSPECTION_ERROR",
        "ExporterUnavailableError": "EXPORTER_UNAVAILABLE",
        "ObserverIntegrityError": "OBSERVER_INTEGRITY_ERROR",
    }
    return mapping.get(name, name if name.isidentifier() else "CLI_ERROR")


def _error_payload(error: BaseException) -> dict[str, str]:
    code = _error_type(error)
    messages = {
        "RUNTIME_PROFILE_UNAVAILABLE": "runtime profile is unavailable",
        "RUNTIME_BINDING_ERROR": "runtime profile or run binding is invalid",
        "RUNTIME_STATE_UNAVAILABLE": "runtime state is unavailable",
        "INSPECTION_INTEGRITY_ERROR": "authoritative Core facts failed integrity checks",
        "INSPECTION_ERROR": "inspection could not be produced",
        "EXPORTER_UNAVAILABLE": "requested observer exporter is unavailable",
        "OBSERVER_INTEGRITY_ERROR": "observer changed authoritative Core facts",
    }
    return {"error_type": code, "message": messages.get(code, "operation failed")}


def _exporter(name: str):
    if name == "json":
        return JsonTraceExporter()
    if name == "otel":
        return OpenTelemetryExporter()
    return LangSmithExporter()


def main(
    argv: list[str] | None = None,
    *,
    service: RuntimeService | None = None,
    stdout: TextIO | None = None,
    stderr: TextIO | None = None,
) -> int:
    """Run one bounded operator command and return a process exit code."""

    output = stdout or sys.stdout
    errors = stderr or sys.stderr
    try:
        args = _parser().parse_args(argv)
        if args.command == "config":
            from getpass import getpass

            from molly.web import ProviderConfigStore

            store = ProviderConfigStore(Path(args.state_root))
            if args.config_command == "set-key":
                secret = getpass("Provider key (input hidden): ")
                store.set_secret(args.profile_ref, secret)
                output.write("密钥已保存到本机服务器端。\n")
            elif args.config_command == "remove-key":
                store.remove_secret(args.profile_ref)
                output.write("密钥已从本机服务器端移除。\n")
            else:  # pragma: no cover - argparse choices make this unreachable
                raise RuntimeError("unsupported config command")
            return 0
        if args.command == "web":
            from molly.web import create_application, serve

            return serve(
                create_application(Path(args.state_root)),
                port=args.port,
            )
        active_service = service or RuntimeService(
            Path(args.state_root), profiles=RuntimeProfileRegistry()
        )
        if args.command == "run" and args.run_command == "start":
            result = active_service.start_run(
                profile_id=args.profile_id,
                goal=args.goal,
                input_artifact_ids=tuple(args.input_artifact),
            )
            payload = result.to_dict()
        elif args.command == "run" and args.run_command == "resume":
            payload = active_service.resume_run(args.run_id).to_dict()
        elif args.command == "inspect" and args.inspect_command == "run":
            value = active_service.inspect_run(args.run_id)
            payload = value.to_dict()
            if args.canonical_json:
                output.write(_canonical_bytes(payload))
                return 0
        elif args.command == "inspect" and args.inspect_command == "artifact":
            value = active_service.inspect_artifact(args.artifact_id)
            payload = value.to_dict()
            if args.canonical_json:
                output.write(_canonical_bytes(payload))
                return 0
        elif args.command == "approve":
            payload = active_service.record_approval(
                args.run_id,
                decision=args.decision,
                reviewer_ref=args.reviewer_ref,
                call_id=args.call_id,
            ).to_dict()
        elif args.command == "review":
            payload = active_service.create_review(
                args.artifact_id,
                decision=args.decision,
                reviewer_ref=args.reviewer_ref,
                reason=args.reason,
            ).to_dict()
        elif args.command == "observe":
            outcome = active_service.observe_run(args.run_id, _exporter(args.exporter))
            payload = outcome.trace.to_dict() if args.exporter == "json" else outcome.to_dict()
            if outcome.status != "EXPORTED":
                output.write(_canonical_bytes(payload) if args.format == "json" else _human_bytes(payload))
                return 1
        else:  # pragma: no cover - argparse choices make this unreachable
            raise RuntimeError("unsupported command")
        output.write(_canonical_bytes(payload) if getattr(args, "canonical_json", False) or (args.command == "observe" and args.format == "json") else _human_bytes(payload))
        return 0
    except SystemExit:
        raise
    except Exception as exc:
        errors.write(_canonical_bytes(_error_payload(exc)))
        return 1


__all__ = ["main"]
