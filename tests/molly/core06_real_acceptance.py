"""Host-side CORE-06C fresh-real acceptance runner.

This is bounded acceptance support, not a production runtime.  It supplies a
server-owned runner to the production ``RemoteComputeBackend`` and drives the
production BR1 ToolSpecs through ``AgentLoop``.  Remote endpoints,
interpreters, and worker roots are required at invocation time and are never
written to public evidence.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import tarfile
import tempfile
from textwrap import dedent
from typing import Any, Mapping, Sequence

from molly.core import (
    AgentLoop,
    ArtifactLineage,
    ArtifactStore,
    RunLedger,
    RunRequest,
    RunStatus,
    SideEffectClass,
    StopAction,
    ToolCallProposal,
    ToolPolicy,
    ToolRegistry,
)
from molly.core.ids import canonical_json_bytes, sha256_bytes
from molly.plugins.br1_inverse_design import (
    Br1PluginConfig,
    Br1Services,
    ComputeBackedBr1Runtime,
    GenerationConfig,
    PredictionConfig,
    TrainingConfig,
    EvaluationConfig,
    migrate_real_csv,
    register_br1_tools,
)
from molly.plugins.remote_compute import (
    ComputeOutput,
    ComputeProfile,
    JobHandle,
    JobState,
    RemoteComputeBackend,
)


class RealAcceptanceError(RuntimeError):
    """The real acceptance environment or output contract failed closed."""


class RemoteCommandError(RealAcceptanceError):
    """A server-owned remote command returned a non-zero status."""


class ArtifactTransferError(RealAcceptanceError):
    """A server-owned artifact transfer could not be completed."""


@dataclass(frozen=True, slots=True)
class HostConfig:
    source_path: Path
    ssh_target: str
    remote_root: str
    unimol_python: str
    reinvent_python: str
    reinvent_repository: str
    max_rows: int = 1024
    candidate_count: int = 8
    seed: int = 20260831

    def __post_init__(self) -> None:
        if not self.source_path.is_file():
            raise RealAcceptanceError("real reviewed dataset source is unavailable")
        for value, label in (
            (self.ssh_target, "SSH target"),
            (self.remote_root, "remote root"),
            (self.unimol_python, "Uni-Mol interpreter"),
            (self.reinvent_python, "REINVENT4 interpreter"),
            (self.reinvent_repository, "REINVENT4 repository"),
        ):
            if not isinstance(value, str) or not value.strip() or any(char in value for char in "\x00\r\n"):
                raise RealAcceptanceError(f"{label} is not configured")
        if not self.remote_root.startswith("/"):
            raise RealAcceptanceError("remote root must be an absolute server-owned path")
        if not 1 <= self.max_rows <= 100_000 or not 1 <= self.candidate_count <= 1024:
            raise RealAcceptanceError("acceptance bounds are invalid")


def _canonical(value: Any) -> bytes:
    return canonical_json_bytes(value)


def _json_write(path: Path, value: Mapping[str, Any]) -> None:
    path.write_bytes(_canonical(value) + b"\n")


def _json_read(path: Path) -> Mapping[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RealAcceptanceError("acceptance output is not valid JSON") from exc
    if not isinstance(value, Mapping):
        raise RealAcceptanceError("acceptance output is not a JSON object")
    return value


def _run_checked(
    argv: Sequence[str], *, cwd: Path | None = None, operation: str = "server-owned command"
) -> None:
    try:
        subprocess.run(
            list(argv),
            cwd=None if cwd is None else str(cwd),
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
    except OSError as exc:
        # Do not persist command lines or remote stderr: either may contain a
        # host-specific path or other environment metadata.
        error_class = (
            RemoteCommandError if operation == "remote command" else ArtifactTransferError
        )
        raise error_class(f"{operation} could not start") from exc
    except subprocess.CalledProcessError as exc:
        error_class = (
            RemoteCommandError if operation == "remote command" else ArtifactTransferError
        )
        raise error_class(f"{operation} failed with exit status {exc.returncode}") from exc


def _ssh(config: HostConfig, command: Sequence[str]) -> None:
    rendered = " ".join(shlex.quote(str(item)) for item in command)
    _run_checked(("ssh", config.ssh_target, "--", rendered), operation="remote command")


def _scp(config: HostConfig, source: Path | str, target: str) -> None:
    _run_checked(("scp", str(source), f"{config.ssh_target}:{target}"), operation="artifact transfer")


def _scp_from_remote(config: HostConfig, source: str, target: Path) -> None:
    _run_checked(
        ("scp", f"{config.ssh_target}:{source}", str(target)),
        operation="artifact download",
    )


def _remote_child(root: str, name: str) -> str:
    return root.rstrip("/") + "/" + name


def _training_script() -> str:
    return dedent(
        r'''
        from __future__ import annotations
        import importlib.metadata
        import json
        from pathlib import Path
        import random
        import sys
        import tarfile

        import numpy as np
        import torch
        from unimol_tools import MolTrain

        csv_path = Path(sys.argv[1])
        model_root = Path(sys.argv[2])
        package_path = Path(sys.argv[3])
        report_path = Path(sys.argv[4])
        dataset_artifact_id = sys.argv[5]
        config_digest = sys.argv[6]
        run_id = sys.argv[7]
        step_id = sys.argv[8]
        target_property = sys.argv[9]
        seed = int(sys.argv[10])

        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        model_root.mkdir(parents=True, exist_ok=True)
        trainer = MolTrain(
            task="regression",
            data_type="molecule",
            epochs=1,
            learning_rate=0.0001,
            batch_size=16,
            early_stopping=1,
            metrics="none",
            split="random",
            kfold=1,
            save_path=str(model_root),
            smiles_col="SMILES",
            target_cols=["target_value"],
            target_normalize="auto",
            smiles_check="filter",
            use_cuda=True,
            use_amp=False,
            use_ddp=False,
            use_gpu="0",
            model_name="unimolv1",
            model_size="84m",
            conf_cache_level=0,
        )
        trainer.fit(str(csv_path))

        required = (model_root / "config.yaml", model_root / "model_0.pth", model_root / "target_scaler.ss")
        if not all(item.is_file() for item in required):
            raise RuntimeError("Uni-Mol did not produce a prediction-capable model roster")
        with tarfile.open(package_path, "w:gz") as archive:
            archive.add(model_root, arcname="model")
        files = sorted(item.relative_to(model_root).as_posix() for item in model_root.rglob("*") if item.is_file())
        report_path.write_bytes(json.dumps({
            "schema_name": "molly.br1.training-report",
            "schema_version": "1",
            "status": "SUCCEEDED",
            "runtime_kind": "unimol_tools",
            "unimol_version": importlib.metadata.version("unimol-tools"),
            "model_name": "unimolv1",
            "model_size": "84m",
            "dataset_artifact_id": dataset_artifact_id,
            "training_config_digest": config_digest,
            "target_property": target_property,
            "seed": seed,
            "fresh_training": True,
            "run_id": run_id,
            "step_id": step_id,
            "model_file_count": len(files),
            "model_files": files,
            "claim_boundary": "COMPUTATIONAL_ONLY",
        }, sort_keys=True, separators=(",", ":")).encode("utf-8") + b"\n")
        '''
    )


def _generation_script() -> str:
    return dedent(
        r'''
        from __future__ import annotations
        import csv
        import hashlib
        import importlib.metadata
        import json
        from pathlib import Path
        import subprocess
        import sys

        from rdkit import Chem

        repository = Path(sys.argv[1])
        reinvent_python = sys.argv[2]
        config_path = Path(sys.argv[3])
        output_csv = Path(sys.argv[4])
        candidate_path = Path(sys.argv[5])
        report_path = Path(sys.argv[6])
        model_artifact_id = sys.argv[7]
        config_digest = sys.argv[8]
        run_id = sys.argv[9]
        step_id = sys.argv[10]
        candidate_count = int(sys.argv[11])
        seed = int(sys.argv[12])
        task_digest = sys.argv[13]

        prior = repository / "priors" / "reinvent.prior"
        if not prior.is_file():
            raise RuntimeError("server-owned REINVENT4 prior is unavailable")
        config_path.write_text(
            "\n".join((
                'run_type = "sampling"',
                'device = "cpu"',
                f"seed = {seed}",
                f"json_out_config = {json.dumps(str(config_path) + '.effective.json')}",
                "",
                "[parameters]",
                f"model_file = {json.dumps(str(prior))}",
                f"output_file = {json.dumps(str(output_csv))}",
                f"num_smiles = {candidate_count}",
                "unique_molecules = true",
                "randomize_smiles = false",
                "temperature = 1.0",
                "",
            )),
            encoding="utf-8",
        )
        subprocess.run(
            [reinvent_python, "-m", "reinvent.Reinvent", str(config_path)],
            cwd=str(repository),
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        if not output_csv.is_file():
            raise RuntimeError("REINVENT4 did not produce a sampling CSV")
        candidates = []
        with output_csv.open(encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                smiles = str(row.get("SMILES") or "").strip()
                if not smiles or Chem.MolFromSmiles(smiles) is None:
                    continue
                index = len(candidates) + 1
                candidate_id = "candidate_" + hashlib.sha256(
                    f"{task_digest}:{index}:{smiles}".encode("utf-8")
                ).hexdigest()[:24]
                candidates.append({
                    "candidate_id": candidate_id,
                    "smiles": smiles,
                    "reinvent_smiles_state": str(row.get("SMILES_state") or "VALID"),
                    "nll": float(row.get("NLL")) if row.get("NLL") not in (None, "") else None,
                })
                if len(candidates) >= candidate_count:
                    break
        if not candidates:
            raise RuntimeError("REINVENT4 produced no valid molecules")
        body = {
            "schema_name": "molly.br1.candidate-package",
            "schema_version": "1",
            "claim_boundary": "COMPUTATIONAL_ONLY",
            "runtime_kind": "reinvent4",
            "model_artifact_id": model_artifact_id,
            "generation_config_digest": config_digest,
            "run_id": run_id,
            "step_id": step_id,
            "fresh_generation": True,
            "historical_reuse": False,
            "rows": candidates,
        }
        candidate_path.write_bytes(json.dumps(body, sort_keys=True, separators=(",", ":")).encode("utf-8") + b"\n")
        report_path.write_bytes(json.dumps({
            "schema_name": "molly.br1.generation-report",
            "schema_version": "1",
            "status": "SUCCEEDED",
            "runtime_kind": "reinvent4",
            "reinvent4_version": importlib.metadata.version("reinvent"),
            "generation_config_digest": config_digest,
            "model_artifact_id": model_artifact_id,
            "candidate_count": len(candidates),
            "requested_candidate_count": candidate_count,
            "seed": seed,
            "fresh_generation": True,
            "historical_reuse": False,
            "raw_output_sha256": hashlib.sha256(output_csv.read_bytes()).hexdigest(),
            "run_id": run_id,
            "step_id": step_id,
            "claim_boundary": "COMPUTATIONAL_ONLY",
        }, sort_keys=True, separators=(",", ":")).encode("utf-8") + b"\n")
        '''
    )


def _prediction_script() -> str:
    return dedent(
        r'''
        from __future__ import annotations
        import csv
        import importlib.metadata
        import json
        from pathlib import Path
        import shutil
        import sys
        import tarfile

        import numpy as np
        from unimol_tools import MolPredict

        model_package = Path(sys.argv[1])
        candidate_path = Path(sys.argv[2])
        extract_root = Path(sys.argv[3])
        prediction_path = Path(sys.argv[4])
        report_path = Path(sys.argv[5])
        model_artifact_id = sys.argv[6]
        candidate_artifact_id = sys.argv[7]
        config_digest = sys.argv[8]
        target_property = sys.argv[9]
        run_id = sys.argv[10]
        step_id = sys.argv[11]

        candidates = json.loads(candidate_path.read_text(encoding="utf-8"))
        rows = candidates.get("rows")
        if not isinstance(rows, list) or not rows:
            raise RuntimeError("candidate package is empty")
        extract_root.mkdir(parents=True, exist_ok=True)
        with tarfile.open(model_package, "r:gz") as archive:
            for member in archive.getmembers():
                parts = Path(member.name).parts
                if member.name.startswith("/") or ".." in parts or not (member.isfile() or member.isdir()):
                    raise RuntimeError("model package contains an unsafe member")
            archive.extractall(extract_root)
        model_root = extract_root / "model"
        if not (model_root / "config.yaml").is_file() or not (model_root / "model_0.pth").is_file():
            raise RuntimeError("model package is not prediction-capable")

        prediction_input = extract_root / "prediction-input.csv"
        with prediction_input.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=("SMILES", "target_value"))
            writer.writeheader()
            for row in rows:
                writer.writerow({"SMILES": str(row["smiles"]), "target_value": -1.0})
        predictor = MolPredict(load_model=str(model_root))
        predicted = np.asarray(
            predictor.predict(str(prediction_input), save_path=str(extract_root / "prediction-output"), metrics="none")
        ).reshape(-1)
        if len(predicted) != len(rows) or not np.isfinite(predicted).all():
            raise RuntimeError("Uni-Mol prediction output is malformed")
        prediction_rows = [
            {
                "candidate_id": str(row["candidate_id"]),
                "smiles": str(row["smiles"]),
                "target_property": target_property,
                "predicted_value": float(value),
            }
            for row, value in zip(rows, predicted, strict=True)
        ]
        prediction_path.write_bytes(json.dumps({
            "schema_name": "molly.br1.prediction-package",
            "schema_version": "1",
            "claim_boundary": "COMPUTATIONAL_ONLY",
            "runtime_kind": "unimol_tools",
            "model_artifact_id": model_artifact_id,
            "candidate_artifact_id": candidate_artifact_id,
            "prediction_config_digest": config_digest,
            "target_property": target_property,
            "run_id": run_id,
            "step_id": step_id,
            "rows": prediction_rows,
        }, sort_keys=True, separators=(",", ":")).encode("utf-8") + b"\n")
        report_path.write_bytes(json.dumps({
            "schema_name": "molly.br1.prediction-report",
            "schema_version": "1",
            "status": "SUCCEEDED",
            "runtime_kind": "unimol_tools",
            "unimol_version": importlib.metadata.version("unimol-tools"),
            "prediction_config_digest": config_digest,
            "model_artifact_id": model_artifact_id,
            "candidate_artifact_id": candidate_artifact_id,
            "prediction_count": len(prediction_rows),
            "current_run_model_binding": True,
            "run_id": run_id,
            "step_id": step_id,
            "claim_boundary": "COMPUTATIONAL_ONLY",
        }, sort_keys=True, separators=(",", ":")).encode("utf-8") + b"\n")
        '''
    )


class ServerOwnedRemoteRunner:
    """Run only fixed scientific adapters on a configured remote host."""

    def __init__(self, config: HostConfig, store: ArtifactStore) -> None:
        self.config = config
        self.store = store
        self.calls: list[str] = []

    def _prepare_remote_dir(self, task_digest: str) -> str:
        remote_dir = _remote_child(self.config.remote_root, f"core06-{task_digest[:32]}")
        _ssh(self.config, ("mkdir", "-p", remote_dir))
        return remote_dir

    def _upload_script(self, workdir: Path, remote_dir: str, name: str, content: str) -> str:
        local = workdir / name
        local.write_text(content, encoding="utf-8")
        remote = _remote_child(remote_dir, name)
        _scp(self.config, local, remote)
        return remote

    def _fetch(self, remote: str, local: Path) -> bytes:
        _scp_from_remote(self.config, remote, local)
        return local.read_bytes()

    def __call__(self, task: Mapping[str, Any], profile: ComputeProfile, workdir: Path) -> Sequence[ComputeOutput]:
        operation = str(task.get("operation") or "")
        if operation not in {"br1_train_unimol", "br1_generate_reinvent4", "br1_predict_unimol"}:
            raise RealAcceptanceError("unknown real BR1 operation")
        task_digest = sha256_bytes(_canonical(task))
        self.calls.append(operation)
        remote_dir = self._prepare_remote_dir(task_digest)
        input_ids = tuple(str(item) for item in task.get("input_artifact_ids", ()))
        parameters = task.get("parameters")
        if not isinstance(parameters, Mapping):
            raise RealAcceptanceError("BR1 task parameters are malformed")

        if operation == "br1_train_unimol":
            if len(input_ids) != 1:
                raise RealAcceptanceError("training task has the wrong input count")
            dataset = _json_read(workdir / "dataset.json") if (workdir / "dataset.json").exists() else json.loads(self.store.read(input_ids[0]).decode("utf-8"))
            rows = dataset.get("rows") if isinstance(dataset, Mapping) else None
            if not isinstance(rows, list) or not rows:
                raise RealAcceptanceError("migrated dataset has no rows")
            csv_path = workdir / "training-input.csv"
            with csv_path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=("SMILES", "target_value"))
                writer.writeheader()
                for row in rows:
                    if not isinstance(row, Mapping):
                        raise RealAcceptanceError("migrated dataset row is malformed")
                    writer.writerow({"SMILES": row["smiles"], "target_value": row["target_value"]})
            _json_write(workdir / "dataset.json", dataset)
            script = self._upload_script(workdir, remote_dir, "train.py", _training_script())
            remote_csv = _remote_child(remote_dir, csv_path.name)
            _scp(self.config, csv_path, remote_csv)
            remote_model = _remote_child(remote_dir, "model")
            remote_package = _remote_child(remote_dir, "model-package.tar.gz")
            remote_report = _remote_child(remote_dir, "training-report.json")
            _ssh(self.config, (
                self.config.unimol_python,
                script,
                remote_csv,
                remote_model,
                remote_package,
                remote_report,
                input_ids[0],
                str(task["config_digest"]),
                str(task["run_id"]),
                str(task["step_id"]),
                str(parameters["target_property"]),
                str(parameters["seed"]),
            ))
            package = self._fetch(remote_package, workdir / "model-package.tar.gz")
            report = self._fetch(remote_report, workdir / "training-report.json")
            return (
                ComputeOutput("model_package", package, "application/gzip", "molly.br1.model-package", "1"),
                ComputeOutput("training_report", report, "application/json", "molly.br1.training-report", "1"),
            )

        if operation == "br1_generate_reinvent4":
            if len(input_ids) != 1:
                raise RealAcceptanceError("generation task has the wrong input count")
            self.store.verify(input_ids[0])
            script = self._upload_script(workdir, remote_dir, "generate.py", _generation_script())
            remote_config = _remote_child(remote_dir, "sampling.toml")
            remote_csv = _remote_child(remote_dir, "sampling.csv")
            remote_candidates = _remote_child(remote_dir, "candidate-package.json")
            remote_report = _remote_child(remote_dir, "generation-report.json")
            _ssh(self.config, (
                self.config.reinvent_python,
                script,
                self.config.reinvent_repository,
                self.config.reinvent_python,
                remote_config,
                remote_csv,
                remote_candidates,
                remote_report,
                input_ids[0],
                str(task["config_digest"]),
                str(task["run_id"]),
                str(task["step_id"]),
                str(parameters["candidate_count"]),
                str(parameters["seed"]),
                task_digest,
            ))
            candidates = self._fetch(remote_candidates, workdir / "candidate-package.json")
            report = self._fetch(remote_report, workdir / "generation-report.json")
            return (
                ComputeOutput("candidate_package", candidates, "application/json", "molly.br1.candidate-package", "1"),
                ComputeOutput("generation_report", report, "application/json", "molly.br1.generation-report", "1"),
            )

        if len(input_ids) != 2:
            raise RealAcceptanceError("prediction task has the wrong input count")
        model = self.store.read(input_ids[0])
        candidates = self.store.read(input_ids[1])
        model_path = workdir / "model-package.tar.gz"
        candidate_path = workdir / "candidate-package.json"
        model_path.write_bytes(model)
        candidate_path.write_bytes(candidates)
        script = self._upload_script(workdir, remote_dir, "predict.py", _prediction_script())
        remote_model = _remote_child(remote_dir, model_path.name)
        remote_candidates = _remote_child(remote_dir, candidate_path.name)
        _scp(self.config, model_path, remote_model)
        _scp(self.config, candidate_path, remote_candidates)
        remote_extract = _remote_child(remote_dir, "prediction-model")
        remote_prediction = _remote_child(remote_dir, "prediction-package.json")
        remote_report = _remote_child(remote_dir, "prediction-report.json")
        _ssh(self.config, (
            self.config.unimol_python,
            script,
            remote_model,
            remote_candidates,
            remote_extract,
            remote_prediction,
            remote_report,
            input_ids[0],
            input_ids[1],
            str(task["config_digest"]),
            str(parameters["target_property"]),
            str(task["run_id"]),
            str(task["step_id"]),
        ))
        prediction = self._fetch(remote_prediction, workdir / "prediction-package.json")
        report = self._fetch(remote_report, workdir / "prediction-report.json")
        return (
            ComputeOutput("prediction_package", prediction, "application/json", "molly.br1.prediction-package", "1"),
            ComputeOutput("prediction_report", report, "application/json", "molly.br1.prediction-report", "1"),
        )


class RealChainProvider:
    def __init__(self, dataset_id: str, *, candidate_count: int, top_n: int = 5) -> None:
        self.dataset_id = dataset_id
        self.candidate_count = candidate_count
        self.top_n = top_n
        self.calls = 0

    def next_action(self, context: Any, _model_visible_tools: Any) -> Any:
        index = self.calls
        self.calls += 1
        if index == 0:
            return ToolCallProposal("br1_applicability_preflight", input_artifact_ids=(self.dataset_id,))
        previous = context.previous_tool_outcome
        if previous is None or previous.get("status") != "SUCCEEDED" or "data" not in previous:
            raise RealAcceptanceError("BR1 chain received no successful durable outcome for the next stage")
        data = previous["data"]
        if index == 1:
            return ToolCallProposal(
                "br1_train_unimol",
                input_artifact_ids=(self.dataset_id, data["preflight_artifact_id"]),
                arguments={"target_property": "quantum_yield"},
            )
        if index == 2:
            return ToolCallProposal(
                "br1_generate_reinvent4",
                input_artifact_ids=(data["model_artifact_id"],),
                arguments={"candidate_count": self.candidate_count},
            )
        if index == 3:
            return ToolCallProposal(
                "br1_predict_unimol",
                input_artifact_ids=(data["model_artifact_id"], data["candidate_artifact_id"]),
                arguments={"target_property": "quantum_yield"},
            )
        if index == 4:
            return ToolCallProposal(
                "br1_evaluate_top_n",
                input_artifact_ids=(data["candidate_artifact_id"], data["prediction_artifact_id"]),
                arguments={"top_n": self.top_n, "direction": "MAX", "target_property": "quantum_yield"},
            )
        return StopAction("fresh-real BR1 parity chain complete")


def _successes(ledger: RunLedger, tool_name: str) -> list[Any]:
    return [
        event
        for event in ledger.events
        if event.tool_name == tool_name and event.event_type == "TOOL_EXECUTION_SUCCEEDED"
    ]


def _event_by_tool(ledger: RunLedger, tool_name: str) -> Any:
    events = _successes(ledger, tool_name)
    if len(events) != 1:
        raise RealAcceptanceError(f"expected one successful {tool_name} occurrence")
    return events[0]


def _public_handle(handle: JobHandle) -> dict[str, Any]:
    return {
        "job_id": handle.job_id,
        "profile_id": handle.profile_id,
        "profile_digest": handle.profile_digest,
        "task_digest": handle.task_digest,
        "idempotency_key": handle.idempotency_key,
        "input_artifact_ids": list(handle.input_artifact_ids),
        "execution_config_digest": handle.execution_config_digest,
        "submitted_at": handle.submitted_at,
    }


def run_acceptance(config: HostConfig) -> dict[str, Any]:
    source_bytes = config.source_path.read_bytes()
    migrated = migrate_real_csv(
        source_bytes,
        historical_acceptance_id="br1-real-acceptance-20260809-v11",
        historical_review_basis=(
            "historical v1 runtime-verified BR1 acceptance evidence; bounded "
            "v2 import did not recreate CORE-05 ReviewRecord history"
        ),
        max_rows=config.max_rows,
    )
    temp_root = Path(tempfile.mkdtemp(prefix="molly-core06-real-", dir="/private/tmp"))
    try:
        store = ArtifactStore(temp_root / "artifacts")
        ledger = RunLedger(temp_root / "events.jsonl")
        lineage = ArtifactLineage(temp_root / "lineage.jsonl")
        dataset_record = store.put(
            migrated.content,
            media_type="application/json",
            schema_name="molly.br1.migrated-reviewed-dataset",
            schema_version="1",
        )
        plugin_config = Br1PluginConfig(
            unimol_version="unimol-tools==0.1.5",
            reinvent4_version="reinvent4==4.7.15",
            runtime_ref="br1-runtime-real-acceptance",
            training_profile_ref="profile:br1-training-real",
            generation_profile_ref="profile:br1-generation-real",
            prediction_profile_ref="profile:br1-prediction-real",
            environment_ref="environment:br1-real",
        )
        profile = ComputeProfile(
            profile_id="profile:br1-remote-real",
            profile_version="1",
            backend_kind="remote",
            host_identity="server-owned-br1-remote",
            worker_ref="worker:br1-real",
            environment_ref="environment:br1-real",
            resource_constraints={"cpu_threads": 8, "gpu_count": 1, "walltime_sec": 3600},
            credential_ref="server-material:br1-ssh",
        )
        runner = ServerOwnedRemoteRunner(config, store)
        backend = RemoteComputeBackend(
            temp_root / "compute",
            profile=profile,
            store=store,
            runner=runner,
        )
        services = Br1Services(
            store,
            ledger,
            config=plugin_config,
            runtime=ComputeBackedBr1Runtime(backend, store),
        )
        registry = ToolRegistry()
        specs = register_br1_tools(registry, services)
        policy = ToolPolicy(
            allowed_tools=tuple(spec.name for spec in specs),
            allowed_side_effect_classes=(SideEffectClass.PURE, SideEffectClass.REMOTE_COMPUTE),
        )
        top_n = min(5, config.candidate_count)
        provider = RealChainProvider(dataset_record.artifact_id, candidate_count=config.candidate_count, top_n=top_n)
        loop = AgentLoop(
            store=store,
            ledger=ledger,
            lineage=lineage,
            registry=registry,
            policy=policy,
            decision_provider=provider,
        )
        request = RunRequest.create(
            goal="CORE-06C fresh-real BR1 parity acceptance",
            tool_policy_digest=policy.digest,
            input_artifact_ids=(dataset_record.artifact_id,),
            metadata={"acceptance_id": "core06-br1-v2-real-20260831"},
        )
        try:
            result = loop.run(request)
        except Exception as exc:
            failures = [
                event
                for event in ledger.for_run(request.run_id)
                if event.event_type == "TOOL_EXECUTION_FAILED"
            ]
            if failures:
                failed = failures[-1]
                error_type = str(failed.metadata.get("error_type") or "unknown")
                compute_errors: list[str] = []
                jobs_root = temp_root / "compute" / "jobs"
                if jobs_root.is_dir():
                    for state_path in sorted(jobs_root.glob("*.json")):
                        try:
                            state = _json_read(state_path)
                        except RealAcceptanceError:
                            continue
                        if state.get("state") == "FAILED":
                            compute_errors.append(str(state.get("error_type") or "unknown"))
                detail = ", ".join(compute_errors) if compute_errors else "none"
                cause: BaseException = exc
                while cause.__cause__ is not None:
                    cause = cause.__cause__
                raise RealAcceptanceError(
                    f"BR1 AgentLoop failed at {failed.tool_name or 'unknown'}: {error_type}; "
                    f"compute_state_errors={detail}; detail={cause}"
                ) from exc
            raise
        if result.status != RunStatus.STOPPED.value or provider.calls != 6:
            raise RealAcceptanceError("fresh-real BR1 AgentLoop chain did not stop successfully")

        preflight = _event_by_tool(ledger, "br1_applicability_preflight")
        training = _event_by_tool(ledger, "br1_train_unimol")
        generation = _event_by_tool(ledger, "br1_generate_reinvent4")
        prediction = _event_by_tool(ledger, "br1_predict_unimol")
        evaluation = _event_by_tool(ledger, "br1_evaluate_top_n")
        model_id, training_report_id = training.output_artifact_ids
        candidate_id, generation_report_id = generation.output_artifact_ids
        prediction_id, prediction_report_id = prediction.output_artifact_ids
        top_n_id, evaluation_report_id = evaluation.output_artifact_ids

        for artifact_id, schema_name in (
            (model_id, "molly.br1.model-package"),
            (training_report_id, "molly.br1.training-report"),
            (candidate_id, "molly.br1.candidate-package"),
            (generation_report_id, "molly.br1.generation-report"),
            (prediction_id, "molly.br1.prediction-package"),
            (prediction_report_id, "molly.br1.prediction-report"),
            (top_n_id, "molly.br1.computational-top-n"),
            (evaluation_report_id, "molly.br1.evaluation-report"),
        ):
            if store.verify(artifact_id).schema_name != schema_name:
                raise RealAcceptanceError(f"artifact {schema_name} has an unexpected schema")

        lineage_checks = {
            "model_from_dataset_and_preflight": set(lineage.parents(model_id)) == set(training.input_artifact_ids),
            "candidate_from_current_model": lineage.parents(candidate_id) == (model_id,),
            "prediction_from_current_model_and_candidate": set(lineage.parents(prediction_id)) == set(prediction.input_artifact_ids),
            "top_n_from_candidate_and_prediction": set(lineage.parents(top_n_id)) == set(evaluation.input_artifact_ids),
            "current_run_model_producer": lineage.producer_steps(model_id) == (training.step_id,),
            "current_run_candidate_producer": lineage.producer_steps(candidate_id) == (generation.step_id,),
            "current_run_prediction_producer": lineage.producer_steps(prediction_id) == (prediction.step_id,),
        }
        if not all(lineage_checks.values()):
            raise RealAcceptanceError("BR1 current-run lineage verification failed")

        reopened_store = ArtifactStore(store.root)
        reopened_backend = RemoteComputeBackend(
            temp_root / "compute",
            profile=profile,
            store=reopened_store,
            runner=None,
        )
        canary: dict[str, Any] = {}
        for tool_name, event in (("br1_train_unimol", training), ("br1_generate_reinvent4", generation), ("br1_predict_unimol", prediction)):
            observation = event.metadata["result_data"]
            runtime_metadata = observation["runtime_metadata"]
            handle = JobHandle.from_dict(runtime_metadata["job_handle"])
            status = reopened_backend.inspect(handle)
            bundle = reopened_backend.collect(handle)
            raw_output_ids = runtime_metadata.get("job_output_artifact_ids")
            if not isinstance(raw_output_ids, Mapping):
                raise RealAcceptanceError(f"remote restart canary lacks raw output binding for {tool_name}")
            expected_raw_ids = {str(value) for value in raw_output_ids.values()}
            collected_ids = {item.artifact_id for item in bundle.outputs}
            if (
                status.state != JobState.SUCCEEDED.value
                or collected_ids != expected_raw_ids
                or event.output_artifact_ids[0] not in expected_raw_ids
            ):
                raise RealAcceptanceError(f"remote restart canary failed for {tool_name}")
            canary[tool_name] = {
                "handle": _public_handle(handle),
                "inspect_state": status.state,
                "collected_output_artifact_ids": sorted(collected_ids),
                "final_event_artifact_ids": list(event.output_artifact_ids),
                "duplicate_dispatches_after_restart": 0,
            }

        repo_root = Path(__file__).resolve().parents[2]
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        tool_bindings = {
            spec.name: {
                "tool_spec_digest": spec.spec_digest,
                "execution_config_digest": spec.execution_config_digest,
            }
            for spec in specs
        }
        training_config = TrainingConfig(
            unimol_version=plugin_config.unimol_version,
            resource_profile_ref=plugin_config.training_profile_ref,
            environment_ref=plugin_config.environment_ref,
        )
        generation_config = GenerationConfig(
            candidate_count=config.candidate_count,
            reinvent4_version=plugin_config.reinvent4_version,
            resource_profile_ref=plugin_config.generation_profile_ref,
            environment_ref=plugin_config.environment_ref,
        )
        prediction_config = PredictionConfig(
            unimol_version=plugin_config.unimol_version,
            resource_profile_ref=plugin_config.prediction_profile_ref,
            environment_ref=plugin_config.environment_ref,
        )
        evaluation_config = EvaluationConfig(top_n=top_n, direction="MAX")
        return {
            "acceptance_id": "core06-br1-v2-real-20260831",
            "status": "PASS",
            "claim_boundary": "COMPUTATIONAL_ONLY",
            "repository_commit": commit,
            "run_id": request.run_id,
            "run_request_sha256": request.request_sha256,
            "dataset": {
                "artifact_id": dataset_record.artifact_id,
                "source_content_sha256": migrated.source_content_digest,
                "transformation_sha256": migrated.transformation_digest,
                "review_status": "MIGRATED_ACCEPTED_REAL_DATASET",
                "historical_acceptance_id": "br1-real-acceptance-20260809-v11",
                "row_count": migrated.row_count,
            },
            "occurrences": {
                "applicability_preflight": {"event_id": preflight.event_id, "step_id": preflight.step_id, "artifact_ids": list(preflight.output_artifact_ids)},
                "training": {"event_id": training.event_id, "step_id": training.step_id, "input_artifact_ids": list(training.input_artifact_ids), "model_artifact_id": model_id, "training_report_artifact_id": training_report_id},
                "generation": {"event_id": generation.event_id, "step_id": generation.step_id, "input_artifact_ids": list(generation.input_artifact_ids), "candidate_artifact_id": candidate_id, "generation_report_artifact_id": generation_report_id},
                "prediction": {"event_id": prediction.event_id, "step_id": prediction.step_id, "input_artifact_ids": list(prediction.input_artifact_ids), "prediction_artifact_id": prediction_id, "prediction_report_artifact_id": prediction_report_id},
                "evaluation": {"event_id": evaluation.event_id, "step_id": evaluation.step_id, "input_artifact_ids": list(evaluation.input_artifact_ids), "top_n_artifact_id": top_n_id, "evaluation_report_artifact_id": evaluation_report_id},
            },
            "software": {
                "unimol_version": plugin_config.unimol_version,
                "reinvent4_version": plugin_config.reinvent4_version,
                "model_name": training_config.model_name,
                "model_size": training_config.model_size,
            },
            "configs": {
                "plugin_config_sha256": plugin_config.digest,
                "training_config_sha256": training_config.digest,
                "generation_config_sha256": generation_config.digest,
                "prediction_config_sha256": prediction_config.digest,
                "evaluation_config_sha256": evaluation_config.digest,
                "tool_bindings": tool_bindings,
            },
            "seeds": {"training": training_config.seed, "generation": generation_config.seed},
            "resource_profile_refs": {
                "training": training_config.resource_profile_ref,
                "generation": generation_config.resource_profile_ref,
                "prediction": prediction_config.resource_profile_ref,
            },
            "lineage": lineage_checks,
            "remote_restart_canary": canary,
            "remote_runner_dispatch_count": len(runner.calls),
            "b2": "PASS",
            "b3": "PASS",
            "b4": "PENDING_OWNER_APPROVAL",
            "core_cutover_ready": False,
        }
    finally:
        shutil.rmtree(temp_root, ignore_errors=True)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the host-owned CORE-06C fresh-real BR1 acceptance.")
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--ssh-target", default=os.environ.get("MOLLY_CORE06_SSH_TARGET", ""))
    parser.add_argument("--remote-root", default=os.environ.get("MOLLY_CORE06_REMOTE_ROOT", ""))
    parser.add_argument("--unimol-python", default=os.environ.get("MOLLY_CORE06_UNIMOL_PYTHON", ""))
    parser.add_argument("--reinvent-python", default=os.environ.get("MOLLY_CORE06_REINVENT_PYTHON", ""))
    parser.add_argument("--reinvent-repository", default=os.environ.get("MOLLY_CORE06_REINVENT_REPOSITORY", ""))
    parser.add_argument("--max-rows", type=int, default=1024)
    parser.add_argument("--candidate-count", type=int, default=8)
    parser.add_argument("--result", type=Path, required=True)
    return parser


def main() -> int:
    args = _parser().parse_args()
    result = run_acceptance(
        HostConfig(
            source_path=args.source,
            ssh_target=args.ssh_target,
            remote_root=args.remote_root,
            unimol_python=args.unimol_python,
            reinvent_python=args.reinvent_python,
            reinvent_repository=args.reinvent_repository,
            max_rows=args.max_rows,
            candidate_count=args.candidate_count,
        )
    )
    args.result.parent.mkdir(parents=True, exist_ok=True)
    _json_write(args.result, result)
    print(json.dumps(result, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
