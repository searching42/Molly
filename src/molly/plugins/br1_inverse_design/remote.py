"""Production server-owned adapters for the real BR1 remote runtime.

The adapter is intentionally a small bridge around ``ssh``/``scp``.  The
remote host, interpreters, repository, prior, and worker root are all supplied
by a registered server profile.  No value from a model action is interpolated
as a command or path.
"""

from __future__ import annotations

import asyncio
import csv
from dataclasses import dataclass, field
from dataclasses import replace
import hashlib
import importlib.metadata
import json
from pathlib import Path
import shlex
from textwrap import dedent
from typing import Any, Mapping, Sequence

from molly.core.artifacts import ArtifactStore
from molly.core.ids import canonical_json_bytes, sha256_bytes
from molly.plugins.remote_compute import (
    ComputeOutput,
    ComputeProfile,
    RemoteComputeBackend,
)

from .errors import Br1RuntimeError
from .runtime import ComputeBackedBr1Runtime
from .schema import Br1PluginConfig


class Br1RemoteError(Br1RuntimeError):
    """A server-owned remote command or transfer failed."""


DEFAULT_REMOTE_WALLTIME_SEC = 7_200
DEFAULT_REMOTE_CONNECT_TIMEOUT_SEC = 15
REMOTE_KILL_GRACE_SEC = 30
MAX_REMOTE_WALLTIME_SEC = 604_800
MAX_REMOTE_CONNECT_TIMEOUT_SEC = 300


def _bounded_seconds(
    constraints: Mapping[str, Any],
    name: str,
    *,
    default: int,
    minimum: int,
    maximum: int,
) -> int:
    value = constraints.get(name, default)
    if isinstance(value, bool) or not isinstance(value, int) or not minimum <= value <= maximum:
        raise Br1RemoteError(f"{name} must be an integer in [{minimum}, {maximum}]")
    return value


@dataclass(frozen=True, slots=True)
class Br1RemoteHost:
    """Non-secret connection and software references for one worker profile."""

    ssh_target: str
    remote_root: str
    unimol_python: str
    reinvent_python: str
    reinvent_repository: str
    host_identity: str = "server-owned-remote"
    worker_ref: str = "worker:br1"
    credential_ref: str = "server-material:br1-ssh"
    resource_constraints: Mapping[str, Any] = field(
        default_factory=lambda: {
            "cpu_threads": 8,
            "gpu_count": 1,
            "walltime_sec": DEFAULT_REMOTE_WALLTIME_SEC,
            "connect_timeout_sec": DEFAULT_REMOTE_CONNECT_TIMEOUT_SEC,
        }
    )

    def __post_init__(self) -> None:
        for value, label in (
            (self.ssh_target, "ssh_target"),
            (self.remote_root, "remote_root"),
            (self.unimol_python, "unimol_python"),
            (self.reinvent_python, "reinvent_python"),
            (self.reinvent_repository, "reinvent_repository"),
            (self.host_identity, "host_identity"),
            (self.worker_ref, "worker_ref"),
            (self.credential_ref, "credential_ref"),
        ):
            if not isinstance(value, str) or not value.strip() or any(char in value for char in "\x00\r\n"):
                raise Br1RuntimeError(f"{label} is not configured")
        if not self.remote_root.startswith("/") or not self.unimol_python.startswith("/") or not self.reinvent_python.startswith("/") or not self.reinvent_repository.startswith("/"):
            raise Br1RuntimeError("remote paths must be absolute server-owned paths")
        try:
            constraints = dict(self.resource_constraints)
        except (TypeError, ValueError) as exc:
            raise Br1RuntimeError("resource_constraints must be a mapping") from exc
        constraints["walltime_sec"] = _bounded_seconds(
            constraints,
            "walltime_sec",
            default=DEFAULT_REMOTE_WALLTIME_SEC,
            minimum=60,
            maximum=MAX_REMOTE_WALLTIME_SEC,
        )
        constraints["connect_timeout_sec"] = _bounded_seconds(
            constraints,
            "connect_timeout_sec",
            default=DEFAULT_REMOTE_CONNECT_TIMEOUT_SEC,
            minimum=1,
            maximum=MAX_REMOTE_CONNECT_TIMEOUT_SEC,
        )
        object.__setattr__(self, "resource_constraints", constraints)


def _walltime_sec(config: Br1RemoteHost) -> int:
    return _bounded_seconds(
        config.resource_constraints,
        "walltime_sec",
        default=DEFAULT_REMOTE_WALLTIME_SEC,
        minimum=60,
        maximum=MAX_REMOTE_WALLTIME_SEC,
    )


def _connect_timeout_sec(config: Br1RemoteHost) -> int:
    return _bounded_seconds(
        config.resource_constraints,
        "connect_timeout_sec",
        default=DEFAULT_REMOTE_CONNECT_TIMEOUT_SEC,
        minimum=1,
        maximum=MAX_REMOTE_CONNECT_TIMEOUT_SEC,
    )


def _subprocess_timeout_sec(config: Br1RemoteHost) -> int:
    return _walltime_sec(config) + _connect_timeout_sec(config) + REMOTE_KILL_GRACE_SEC + 5


def _connection_options(config: Br1RemoteHost) -> tuple[str, ...]:
    return (
        "-o",
        "BatchMode=yes",
        "-o",
        "RequestTTY=no",
        "-o",
        f"ConnectTimeout={_connect_timeout_sec(config)}",
        "-o",
        "ConnectionAttempts=1",
        "-o",
        "ServerAliveInterval=15",
        "-o",
        "ServerAliveCountMax=3",
    )


async def _run_checked_async(
    argv: Sequence[str], *, timeout_sec: int = DEFAULT_REMOTE_WALLTIME_SEC
) -> None:
    process = await asyncio.create_subprocess_exec(
        *(str(item) for item in argv),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        await asyncio.wait_for(process.communicate(), timeout=float(timeout_sec))
    except asyncio.TimeoutError as exc:
        try:
            process.kill()
        except ProcessLookupError:
            pass
        await process.communicate()
        raise RuntimeError("child process exceeded its wall-time limit") from exc
    if process.returncode != 0:
        raise RuntimeError("child process returned a non-zero status")


def _run_checked(
    argv: Sequence[str], *, operation: str, timeout_sec: int = DEFAULT_REMOTE_WALLTIME_SEC
) -> None:
    try:
        asyncio.run(_run_checked_async(argv, timeout_sec=timeout_sec))
    except (OSError, RuntimeError) as exc:
        # Remote output is deliberately not persisted or returned.  It may
        # contain private filesystem paths, host details, or credentials.
        raise Br1RemoteError(f"server-owned {operation} failed") from exc


def _render_remote_command(command: Sequence[str], *, walltime_sec: int) -> str:
    rendered = " ".join(shlex.quote(str(item)) for item in command)
    return (
        "timeout --signal=TERM --kill-after="
        + str(REMOTE_KILL_GRACE_SEC)
        + "s "
        + shlex.quote(str(walltime_sec))
        + "s "
        + rendered
    )


def _ssh(config: Br1RemoteHost, command: Sequence[str]) -> None:
    walltime = _walltime_sec(config)
    wrapped = _render_remote_command(command, walltime_sec=walltime)
    _run_checked(
        (
            "ssh",
            "-T",
            *_connection_options(config),
            config.ssh_target,
            "--",
            wrapped,
        ),
        operation="remote command",
        timeout_sec=_subprocess_timeout_sec(config),
    )


def _scp(config: Br1RemoteHost, source: Path | str, target: str) -> None:
    _run_checked(
        (
            "scp",
            *_connection_options(config),
            str(source),
            f"{config.ssh_target}:{target}",
        ),
        operation="artifact transfer",
        timeout_sec=_subprocess_timeout_sec(config),
    )


def _scp_from_remote(config: Br1RemoteHost, source: str, target: Path) -> None:
    _run_checked(
        (
            "scp",
            *_connection_options(config),
            f"{config.ssh_target}:{source}",
            str(target),
        ),
        operation="artifact download",
        timeout_sec=_subprocess_timeout_sec(config),
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
        import pandas as pd
        import torch
        from rdkit import Chem
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
        model_name = sys.argv[11]
        model_size = sys.argv[12]
        training_parameters = json.loads(sys.argv[13])
        expected_training_parameters = {
            "task", "epochs", "learning_rate", "batch_size", "early_stopping",
            "metrics", "split", "kfold", "smiles_col", "target_cols",
            "target_normalize", "smiles_check", "use_cuda", "use_amp", "use_ddp",
            "use_gpu", "model_name", "model_size", "conf_cache_level",
        }
        if not isinstance(training_parameters, dict) or set(training_parameters) != expected_training_parameters:
            raise RuntimeError("training parameters are not the server-owned supported set")
        if training_parameters["model_name"] != model_name or training_parameters["model_size"] != model_size:
            raise RuntimeError("training model identity is inconsistent")
        smiles_col = training_parameters["smiles_col"]
        target_cols = training_parameters["target_cols"]
        if (
            not isinstance(smiles_col, str)
            or not smiles_col.strip()
            or not isinstance(target_cols, list)
            or len(target_cols) != 1
            or not isinstance(target_cols[0], str)
            or not target_cols[0].strip()
            or any(char in smiles_col + target_cols[0] for char in "\r\n")
        ):
            raise RuntimeError("training column parameters are malformed")

        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        model_root.mkdir(parents=True, exist_ok=True)
        source_data = pd.read_csv(csv_path)
        source_row_count = len(source_data)
        valid_mask = source_data[smiles_col].map(lambda smi: Chem.MolFromSmiles(str(smi)) is not None)
        training_data = source_data.loc[valid_mask].reset_index(drop=True)
        if training_data.empty:
            raise RuntimeError("RDKit filtering removed every training molecule")
        filtered_csv = model_root.parent / "training-input-filtered.csv"
        training_data.to_csv(filtered_csv, index=False)
        invalid_smiles_filtered = source_row_count - len(training_data)

        trainer = MolTrain(
            task=training_parameters["task"],
            data_type="molecule",
            epochs=training_parameters["epochs"],
            learning_rate=training_parameters["learning_rate"],
            batch_size=training_parameters["batch_size"],
            early_stopping=training_parameters["early_stopping"],
            metrics=training_parameters["metrics"],
            split=training_parameters["split"],
            kfold=training_parameters["kfold"],
            save_path=str(model_root),
            smiles_col=training_parameters["smiles_col"],
            target_cols=training_parameters["target_cols"],
            target_normalize=training_parameters["target_normalize"],
            smiles_check=training_parameters["smiles_check"],
            use_cuda=training_parameters["use_cuda"],
            use_amp=training_parameters["use_amp"],
            use_ddp=training_parameters["use_ddp"],
            use_gpu=training_parameters["use_gpu"],
            model_name=model_name,
            model_size=model_size,
            conf_cache_level=training_parameters["conf_cache_level"],
        )
        trainer.fit(str(filtered_csv))

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
            "model_name": model_name,
            "model_size": model_size,
            "use_cuda": training_parameters["use_cuda"],
            "use_gpu": training_parameters["use_gpu"],
            "parameters": training_parameters,
            "dataset_artifact_id": dataset_artifact_id,
            "training_config_digest": config_digest,
            "target_property": target_property,
            "seed": seed,
            "fresh_training": True,
            "run_id": run_id,
            "step_id": step_id,
            "source_row_count": source_row_count,
            "training_row_count": len(training_data),
            "invalid_smiles_filtered": invalid_smiles_filtered,
            "smiles_filter_runtime": "rdkit_pre_filter_then_unimol_filter",
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
        import math
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
        generation_parameters = json.loads(sys.argv[14])
        expected_generation_parameters = {
            "unique_molecules", "randomize_smiles", "temperature", "device", "prior_model_ref",
        }
        if not isinstance(generation_parameters, dict) or set(generation_parameters) != expected_generation_parameters:
            raise RuntimeError("generation parameters are not the server-owned supported set")
        if not isinstance(generation_parameters["unique_molecules"], bool) or not isinstance(generation_parameters["randomize_smiles"], bool):
            raise RuntimeError("generation boolean parameters are malformed")
        temperature = generation_parameters["temperature"]
        if isinstance(temperature, bool) or not isinstance(temperature, (int, float)) or not math.isfinite(float(temperature)):
            raise RuntimeError("generation temperature is malformed")
        device = generation_parameters["device"]
        if not isinstance(device, str) or not device.strip():
            raise RuntimeError("generation device is malformed")
        prior_model_ref = generation_parameters["prior_model_ref"]
        if prior_model_ref != "reinvent.prior":
            raise RuntimeError("generation prior reference is not server-owned")

        prior = repository / "priors" / prior_model_ref
        if not prior.is_file():
            raise RuntimeError("server-owned REINVENT4 prior is unavailable")

        # REINVENT's requested count includes invalid/duplicate strings in
        # some versions.  Over-sample and retry with deterministic seed
        # offsets until the contract has the requested number of unique valid
        # molecules, or fail explicitly instead of silently returning a
        # partial sampling space.
        unique = {}
        attempts = 0
        attempted_samples = 0
        max_attempts = 8
        while len(unique) < candidate_count and attempts < max_attempts:
            attempts += 1
            needed = candidate_count - len(unique)
            batch_size = max(needed, int(needed * 1.25) + 8)
            attempt_csv = output_csv.with_name(output_csv.stem + f"-{attempts}" + output_csv.suffix)
            attempt_config = config_path.with_name(config_path.stem + f"-{attempts}" + config_path.suffix)
            attempt_seed = seed + attempts - 1
            attempt_config.write_text(
                "\n".join((
                    'run_type = "sampling"',
                    f"device = {json.dumps(device)}",
                    f"seed = {attempt_seed}",
                    f"json_out_config = {json.dumps(str(attempt_config) + '.effective.json')}",
                    "",
                    "[parameters]",
                    f"model_file = {json.dumps(str(prior))}",
                    f"output_file = {json.dumps(str(attempt_csv))}",
                    f"num_smiles = {batch_size}",
                    f"unique_molecules = {json.dumps(generation_parameters['unique_molecules'])}",
                    f"randomize_smiles = {json.dumps(generation_parameters['randomize_smiles'])}",
                    f"temperature = {json.dumps(temperature)}",
                    "",
                )),
                encoding="utf-8",
            )
            subprocess.run(
                [reinvent_python, "-m", "reinvent.Reinvent", str(attempt_config)],
                cwd=str(repository),
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            if not attempt_csv.is_file():
                raise RuntimeError("REINVENT4 did not produce a sampling CSV")
            with attempt_csv.open(encoding="utf-8", newline="") as handle:
                reader = csv.DictReader(handle)
                for row in reader:
                    attempted_samples += 1
                    smiles = str(row.get("SMILES") or "").strip()
                    molecule = Chem.MolFromSmiles(smiles) if smiles else None
                    if molecule is None:
                        continue
                    canonical = Chem.MolToSmiles(molecule, canonical=True)
                    if canonical in unique:
                        continue
                    index = len(unique) + 1
                    unique[canonical] = {
                        "candidate_id": "candidate_" + hashlib.sha256(
                            f"{task_digest}:{index}:{canonical}".encode("utf-8")
                        ).hexdigest()[:24],
                        "smiles": smiles,
                        "reinvent_smiles_state": str(row.get("SMILES_state") or "VALID"),
                        "nll": float(row.get("NLL")) if row.get("NLL") not in (None, "") else None,
                    }
                    if len(unique) >= candidate_count:
                        break
        if len(unique) < candidate_count:
            raise RuntimeError(
                f"REINVENT4 produced only {len(unique)} valid unique molecules; requested {candidate_count}"
            )
        candidates = list(unique.values())[:candidate_count]
        candidate_path.write_bytes(json.dumps({
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
        }, sort_keys=True, separators=(",", ":")).encode("utf-8") + b"\n")
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
            "device": device,
            "attempt_count": attempts,
            "attempted_sample_count": attempted_samples,
            "seed": seed,
            "parameters": generation_parameters,
            "fresh_generation": True,
            "historical_reuse": False,
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
        prediction_parameters = json.loads(sys.argv[12])
        expected_prediction_parameters = {"task", "smiles_col", "target_col", "target_normalize"}
        if not isinstance(prediction_parameters, dict) or set(prediction_parameters) != expected_prediction_parameters:
            raise RuntimeError("prediction parameters are not the server-owned supported set")
        if prediction_parameters["task"] != "regression" or prediction_parameters["target_normalize"] != "auto":
            raise RuntimeError("prediction semantics are not supported by the remote adapter")
        smiles_col = prediction_parameters["smiles_col"]
        target_col = prediction_parameters["target_col"]
        if (
            not isinstance(smiles_col, str)
            or not smiles_col.strip()
            or not isinstance(target_col, str)
            or not target_col.strip()
            or any(char in smiles_col + target_col for char in "\r\n")
        ):
            raise RuntimeError("prediction column parameters are malformed")

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
            writer = csv.DictWriter(handle, fieldnames=(smiles_col, target_col))
            writer.writeheader()
            for row in rows:
                writer.writerow({smiles_col: str(row["smiles"]), target_col: -1.0})
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
            "parameters": prediction_parameters,
            "current_run_model_binding": True,
            "run_id": run_id,
            "step_id": step_id,
            "claim_boundary": "COMPUTATIONAL_ONLY",
        }, sort_keys=True, separators=(",", ":")).encode("utf-8") + b"\n")
        '''
    )


class ServerOwnedBr1RemoteRunner:
    """Dispatch only the three fixed BR1 scientific operations."""

    def __init__(self, config: Br1RemoteHost, store: ArtifactStore) -> None:
        if not isinstance(config, Br1RemoteHost):
            raise TypeError("config must be a Br1RemoteHost")
        if not isinstance(store, ArtifactStore):
            raise TypeError("store must be an ArtifactStore")
        self.config = config
        self.store = store

    def _prepare_remote_dir(self, task_digest: str) -> str:
        remote_dir = _remote_child(self.config.remote_root, f"molly-br1-{task_digest[:32]}")
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

    def __call__(self, task: Mapping[str, Any], _profile: ComputeProfile, workdir: Path) -> Sequence[ComputeOutput]:
        operation = str(task.get("operation") or "")
        if operation not in {"br1_train_unimol", "br1_generate_reinvent4", "br1_predict_unimol"}:
            raise Br1RemoteError("unknown BR1 remote operation")
        task_digest = sha256_bytes(canonical_json_bytes(task))
        remote_dir = self._prepare_remote_dir(task_digest)
        input_ids = tuple(str(item) for item in task.get("input_artifact_ids", ()))
        parameters = task.get("parameters")
        if not isinstance(parameters, Mapping):
            raise Br1RemoteError("BR1 task parameters are malformed")
        stage_parameters = parameters.get("parameters", {})
        if not isinstance(stage_parameters, Mapping):
            raise Br1RemoteError("BR1 stage parameters are malformed")
        try:
            serialized_stage_parameters = json.dumps(
                dict(stage_parameters), sort_keys=True, separators=(",", ":")
            )
        except (TypeError, ValueError) as exc:
            raise Br1RemoteError("BR1 stage parameters are not JSON serializable") from exc

        if operation == "br1_train_unimol":
            if len(input_ids) != 1:
                raise Br1RemoteError("training task has the wrong input count")
            try:
                dataset = json.loads(self.store.read(input_ids[0]).decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise Br1RemoteError("training dataset is not valid JSON") from exc
            rows = dataset.get("rows") if isinstance(dataset, Mapping) else None
            if not isinstance(rows, list) or not rows:
                raise Br1RemoteError("training dataset has no rows")
            training_smiles_col = stage_parameters.get("smiles_col")
            training_target_cols = stage_parameters.get("target_cols")
            if (
                not isinstance(training_smiles_col, str)
                or not training_smiles_col.strip()
                or not isinstance(training_target_cols, list)
                or len(training_target_cols) != 1
                or not isinstance(training_target_cols[0], str)
                or not training_target_cols[0].strip()
                or any(
                    char in training_smiles_col + training_target_cols[0]
                    for char in "\r\n"
                )
            ):
                raise Br1RemoteError("training column parameters are malformed")
            csv_path = workdir / "training-input.csv"
            training_target_col = training_target_cols[0]
            with csv_path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(
                    handle, fieldnames=(training_smiles_col, training_target_col)
                )
                writer.writeheader()
                for row in rows:
                    if not isinstance(row, Mapping):
                        raise Br1RemoteError("training dataset row is malformed")
                    writer.writerow(
                        {
                            training_smiles_col: row["smiles"],
                            training_target_col: row["target_value"],
                        }
                    )
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
                str(parameters["model_name"]),
                str(parameters["model_size"]),
                serialized_stage_parameters,
            ))
            return (
                ComputeOutput("model_package", self._fetch(remote_package, workdir / "model-package.tar.gz"), "application/gzip", "molly.br1.model-package", "1"),
                ComputeOutput("training_report", self._fetch(remote_report, workdir / "training-report.json"), "application/json", "molly.br1.training-report", "1"),
            )

        if operation == "br1_generate_reinvent4":
            if len(input_ids) != 1:
                raise Br1RemoteError("generation task has the wrong input count")
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
                serialized_stage_parameters,
            ))
            return (
                ComputeOutput("candidate_package", self._fetch(remote_candidates, workdir / "candidate-package.json"), "application/json", "molly.br1.candidate-package", "1"),
                ComputeOutput("generation_report", self._fetch(remote_report, workdir / "generation-report.json"), "application/json", "molly.br1.generation-report", "1"),
            )

        if len(input_ids) != 2:
            raise Br1RemoteError("prediction task has the wrong input count")
        model_path = workdir / "model-package.tar.gz"
        candidate_path = workdir / "candidate-package.json"
        model_path.write_bytes(self.store.read(input_ids[0]))
        candidate_path.write_bytes(self.store.read(input_ids[1]))
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
            serialized_stage_parameters,
        ))
        return (
            ComputeOutput("prediction_package", self._fetch(remote_prediction, workdir / "prediction-package.json"), "application/json", "molly.br1.prediction-package", "1"),
            ComputeOutput("prediction_report", self._fetch(remote_report, workdir / "prediction-report.json"), "application/json", "molly.br1.prediction-report", "1"),
        )


def remote_br1_profile(
    root: Path | str,
    host: Br1RemoteHost,
    *,
    plugin_config: Br1PluginConfig | None = None,
    profile_id: str = "profile:br1-remote",
    display_name: str = "BR1 远程科学运行",
    description: str = "在服务端登记的远程工作站执行 Uni-Mol 与 REINVENT4",
    host_preference: str = "auto",
    gpu_count: int | None = None,
    intent_provider_resolver: Any | None = None,
) -> Any:
    """Build a BR1 RuntimeProfile backed by the durable remote backend."""

    from .workflow import br1_profile

    configured = plugin_config or Br1PluginConfig(
        unimol_version="unimol-tools==0.1.5",
        reinvent4_version="reinvent4==4.7.15",
    )
    selected_gpu_count = int(
        host.resource_constraints.get("gpu_count", 1) if gpu_count is None else gpu_count
    )
    configured = replace(
        configured,
        training_parameters={
            **dict(configured.training_parameters),
            "use_cuda": selected_gpu_count > 0,
            "use_gpu": "0",
        },
        generation_parameters={
            **dict(configured.generation_parameters),
            "device": "cuda" if selected_gpu_count > 0 else "cpu",
        },
    )
    root_path = Path(root)
    compute_profile = ComputeProfile(
        profile_id=profile_id,
        backend_kind="remote",
        host_identity=host.host_identity,
        worker_ref=host.worker_ref,
        environment_ref=configured.environment_ref,
        resource_constraints=host.resource_constraints,
        credential_ref=host.credential_ref,
    )

    def runtime_factory(store: ArtifactStore, _ledger: Any) -> ComputeBackedBr1Runtime:
        backend = RemoteComputeBackend(
            root_path / "compute" / profile_id.replace(":", "_"),
            profile=compute_profile,
            store=store,
            runner=ServerOwnedBr1RemoteRunner(host, store),
        )
        return ComputeBackedBr1Runtime(backend, store)

    return br1_profile(
        root_path,
        plugin_config=configured,
        profile_id=profile_id,
        display_name=display_name,
        description=description,
        runtime_factory=runtime_factory,
        spec_overrides={
            "host_preference": host_preference,
            "cpu_threads": int(host.resource_constraints.get("cpu_threads", 8)),
            "gpu_count": selected_gpu_count,
            "walltime_sec": int(host.resource_constraints.get("walltime_sec", 7_200)),
        },
        intent_provider_resolver=intent_provider_resolver,
        config={
            "backend_kind": "remote",
            "host_identity": host.host_identity,
            "worker_ref": host.worker_ref,
            "compute_profile_id": compute_profile.profile_id,
            "resource_constraints": dict(host.resource_constraints),
        },
    )


__all__ = [
    "Br1RemoteError",
    "Br1RemoteHost",
    "ServerOwnedBr1RemoteRunner",
    "remote_br1_profile",
]
