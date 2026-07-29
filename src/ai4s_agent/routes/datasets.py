from __future__ import annotations

from flask import Flask, jsonify, request

from ai4s_agent.dataset_workflow import DatasetWorkflowService


def register_dataset_routes(app: Flask, *, datasets: DatasetWorkflowService) -> None:
    @app.get("/api/projects/<project_id>/datasets")
    def list_project_datasets(project_id: str):
        try:
            items = datasets.list_datasets(project_id)
        except (OSError, ValueError):
            app.logger.warning("dataset listing verification failed", exc_info=True)
            return _dataset_error(
                "dataset_verification_failed",
                "A stored dataset failed integrity verification.",
                409,
            )
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
        except FileNotFoundError:
            return _dataset_error(
                "dataset_attachment_not_found",
                "The selected dataset attachment is unavailable.",
                404,
            )
        except (OSError, ValueError):
            app.logger.warning("dataset attachment inspection failed", exc_info=True)
            return _dataset_error(
                "dataset_inspection_failed",
                "The selected attachment could not be inspected as a CSV dataset.",
                400,
            )
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
        except FileNotFoundError:
            return _dataset_error(
                "dataset_not_found",
                "The selected dataset is unavailable.",
                404,
            )
        except (OSError, ValueError):
            app.logger.warning("dataset confirmation failed", exc_info=True)
            return _dataset_error(
                "dataset_confirmation_failed",
                "The dataset could not be confirmed or failed integrity verification.",
                400,
            )
        return jsonify({"ok": True, "confirmed_dataset": confirmed}), 200


def _dataset_error(code: str, message: str, status: int):
    return jsonify({"ok": False, "error": message, "error_code": code}), status


__all__ = ["register_dataset_routes"]
