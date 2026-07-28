from __future__ import annotations

from flask import Flask, jsonify, request

from ai4s_agent.dataset_workflow import DatasetWorkflowService


def register_dataset_routes(app: Flask, *, datasets: DatasetWorkflowService) -> None:
    @app.get("/api/projects/<project_id>/datasets")
    def list_project_datasets(project_id: str):
        try:
            items = datasets.list_datasets(project_id)
        except (OSError, ValueError) as exc:
            return jsonify({"ok": False, "error": str(exc)}), 400
        return jsonify({"ok": True, "datasets": items})

    @app.post("/api/projects/<project_id>/datasets/inspect-attachment")
    def inspect_dataset_attachment(project_id: str):
        payload = request.get_json(silent=True)
        if not isinstance(payload, dict):
            return jsonify({"ok": False, "error": "JSON object required"}), 400
        artifact_id = str(payload.get("artifact_id") or "").strip()
        if not artifact_id:
            return jsonify({"ok": False, "error": "artifact_id required"}), 400
        try:
            dataset = datasets.inspect_attachment(project_id, artifact_id)
        except FileNotFoundError as exc:
            return jsonify({"ok": False, "error": str(exc)}), 404
        except (OSError, ValueError) as exc:
            return jsonify({"ok": False, "error": str(exc)}), 400
        return jsonify({"ok": True, "dataset": dataset}), 201

    @app.post("/api/projects/<project_id>/datasets/<dataset_id>/confirm")
    def confirm_project_dataset(project_id: str, dataset_id: str):
        payload = request.get_json(silent=True)
        if not isinstance(payload, dict):
            return jsonify({"ok": False, "error": "JSON object required"}), 400
        for key in ("smiles_column", "target_column", "property_id", "confirmed_by"):
            if not str(payload.get(key) or "").strip():
                return jsonify({"ok": False, "error": f"{key} required"}), 400
        strict = payload.get("strict_smiles_cleaning", True)
        drop_empty = payload.get("drop_empty_target_rows", True)
        if not isinstance(strict, bool) or not isinstance(drop_empty, bool):
            return jsonify(
                {
                    "ok": False,
                    "error": "strict_smiles_cleaning and drop_empty_target_rows must be booleans",
                }
            ), 400
        try:
            confirmed = datasets.confirm_dataset(
                project_id,
                dataset_id,
                smiles_column=str(payload["smiles_column"]),
                target_column=str(payload["target_column"]),
                property_id=str(payload["property_id"]),
                confirmed_by=str(payload["confirmed_by"]),
                note=str(payload.get("note") or ""),
                strict_smiles_cleaning=strict,
                drop_empty_target_rows=drop_empty,
            )
        except FileNotFoundError as exc:
            return jsonify({"ok": False, "error": str(exc)}), 404
        except (OSError, ValueError) as exc:
            return jsonify({"ok": False, "error": str(exc)}), 400
        return jsonify({"ok": True, "confirmed_dataset": confirmed}), 200


__all__ = ["register_dataset_routes"]
