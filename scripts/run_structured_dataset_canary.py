from __future__ import annotations

import argparse
import json
from pathlib import Path

from ai4s_agent.storage import ProjectStorage
from ai4s_agent.structured_dataset_canary import StructuredDatasetCanaryService
from ai4s_agent.structured_dataset_private_canary import PrivateRealToolCanaryRequest


def main() -> int:
    parser = argparse.ArgumentParser(description="Run or prepare Structured Dataset Canary v1")
    parser.add_argument("--mode", choices=("ci_reference", "private_real_tool_request"), required=True)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--project-id", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--raw-csv", type=Path)
    parser.add_argument("--actor")
    parser.add_argument("--seed", type=int, default=1729)
    parser.add_argument("--top-n", type=int, default=5)
    parser.add_argument("--request-json", type=Path)
    args = parser.parse_args()

    storage = ProjectStorage(args.workspace)
    if args.mode == "ci_reference":
        if args.raw_csv is None or not args.actor:
            parser.error("ci_reference requires --raw-csv and --actor")
        if not any(item["project_id"] == args.project_id for item in storage.list_projects()):
            storage.create_project(args.project_id, name=args.project_id, created_at="runtime")
        result = StructuredDatasetCanaryService(
            storage=storage,
            trusted_actors={args.actor},
        ).run_ci_reference(
            project_id=args.project_id,
            run_id=args.run_id,
            raw_csv=args.raw_csv,
            actor=args.actor,
            seed=args.seed,
            top_n=args.top_n,
        )
        print(json.dumps({
            "status": "succeeded",
            "artifact_name": result.computational_top_n["artifact_name"],
            "evidence_digest": result.evidence["evidence_digest"],
            "topn_digest": result.computational_top_n["publication_digest"],
            "replayed": result.replayed,
        }, sort_keys=True))
        return 0

    if args.request_json is None:
        parser.error("private_real_tool_request requires --request-json")
    values = json.loads(args.request_json.read_text(encoding="utf-8"))
    request = PrivateRealToolCanaryRequest(**values).to_publication()
    # This command deliberately prepares authority input only. Dispatch remains
    # owned by approve-and-start + Harness Controller + RemoteExecutionService.
    print(json.dumps(request, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
