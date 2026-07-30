(function attachMollyTaskState(root, factory) {
  const api = factory();
  if (typeof module === "object" && module.exports) module.exports = api;
  if (root) root.MollyTaskState = Object.freeze(api);
})(typeof globalThis === "object" ? globalThis : this, function buildMollyTaskState() {
  "use strict";

  const PROJECTION_VERSION = "task_state_conversation_projection.v1";
  const STATUS_ALLOWLIST = new Set([
    "PENDING",
    "RUNNING",
    "WAITING_USER",
    "PAUSED_BY_USER",
    "SUCCEEDED",
    "DEGRADED",
    "FAILED",
    "SKIPPED",
    "CANCELLED",
    "DONE",
  ]);
  const STAGE_ALLOWLIST = new Set([
    "acquire_literature_sources",
    "build_dense_index",
    "build_multi_index",
    "check_public_dataset_leakage",
    "check_trainability",
    "clean_dataset",
    "confirm_extracted_dataset",
    "evaluate_extraction_benchmark",
    "execute_oled_bounded_discovery_controller",
    "execute_oled_candidate_decision",
    "execute_oled_experiment_batch_selection",
    "execute_oled_generated_candidate_evaluation",
    "execute_oled_inverse_design",
    "execute_oled_local_demo",
    "execute_oled_registry_candidate_screening",
    "extract_records",
    "filter_rank",
    "generate_candidates",
    "index_corpus",
    "inspect_dataset",
    "literature_to_dataset_workflow",
    "merge_extracted_records",
    "normalize_extracted_units",
    "parse_document",
    "parse_document_grobid",
    "parse_document_pdfplumber",
    "parse_document_pymupdf",
    "parse_pdf_corpus_pdfplumber",
    "predict_candidates",
    "prepare_literature_corpus_sources",
    "render_report",
    "retrieve_evidence",
    "run_baseline",
    "track_citation_provenance",
    "train_model",
  ]);
  const STATE_FILE_ALLOWLIST = new Set([
    "artifact_registry.json",
    "background_job_state.json",
    "gate_decisions.json",
    "job_state.json",
    "plan.json",
    "run_plan.json",
    "stage.json",
  ]);
  const ARTIFACT_ALLOWLIST = new Set([
    "acquisition_manifest",
    "audit_summary",
    "backend_recommendation",
    "baseline_report",
    "benchmark_contamination_report",
    "candidate_dataset",
    "candidate_predictions",
    "candidate_training_dataset",
    "citation_provenance_report",
    "cleaned_train_dataset",
    "cleaning_rules",
    "confirmed_training_dataset",
    "conflict_report",
    "corpus_index",
    "corpus_manifest",
    "corpus_source_manifest",
    "dataset_profile",
    "dense_index",
    "domain_model_manifest",
    "evidence_chunks",
    "evidence_hits",
    "extracted_records",
    "extraction_benchmark_report",
    "extraction_confidence_report",
    "extraction_confirmation_record",
    "generation_publication",
    "generation_report",
    "merged_records",
    "model_diagnostics_report",
    "model_manifest",
    "model_metadata",
    "model_package_review",
    "multi_index",
    "normalized_extracted_records",
    "oled_bounded_controller_execution_record",
    "oled_bounded_controller_generation_authorization",
    "oled_bounded_controller_receipt",
    "oled_bounded_controller_report",
    "oled_bounded_controller_request",
    "oled_bounded_controller_request_snapshot",
    "oled_candidate_decision_dossier",
    "oled_candidate_evaluation_exclusions",
    "oled_candidate_evaluation_execution_record",
    "oled_candidate_evaluation_predictions",
    "oled_candidate_evaluation_receipt",
    "oled_candidate_evaluation_report",
    "oled_candidate_evaluation_shortlist",
    "oled_dataset_snapshot",
    "oled_demo_bundle_markdown",
    "oled_demo_bundle_report",
    "oled_experiment_batch_execution_record",
    "oled_experiment_batch_handoff",
    "oled_experiment_batch_receipt",
    "oled_experiment_batch_report",
    "oled_final_candidate_decision_dossier",
    "oled_final_candidate_decision_execution_record",
    "oled_final_candidate_decision_receipt",
    "oled_final_candidate_decision_report",
    "oled_final_candidate_decision_top_n",
    "oled_inverse_design_candidates",
    "oled_inverse_design_exclusions",
    "oled_inverse_design_execution_record",
    "oled_inverse_design_receipt",
    "oled_inverse_design_reinvent4_config",
    "oled_inverse_design_report",
    "oled_local_demo_execution_manifest",
    "oled_phase1_execution_dir",
    "oled_registry_screening_eligible_candidates",
    "oled_registry_screening_exclusions",
    "oled_registry_screening_execution_record",
    "oled_registry_screening_predictions",
    "oled_registry_screening_receipt",
    "oled_registry_screening_report",
    "oled_registry_screening_shortlist",
    "oled_registry_snapshot",
    "parsed_corpus_manifest",
    "parsed_document",
    "parsed_tables",
    "parser_audit",
    "pdf_corpus",
    "property_catalog",
    "ranked_candidates",
    "rejected_records",
    "report_html",
    "report_markdown",
    "retrieval_log",
    "structured_datasets",
    "topn_export",
    "trainability_report",
    "trained_model",
    "unit_normalization_report",
    "workflow_report",
  ]);

  function semanticValue(value, allowlist) {
    const clean = String(value || "").trim();
    return allowlist.has(clean) ? clean : "unavailable";
  }

  function allowlistedList(value, allowlist) {
    if (!Array.isArray(value)) return [];
    return Array.from(new Set(value.map((item) => String(item || "").trim()).filter((item) => allowlist.has(item)))).sort();
  }

  function taskStateSnapshot(payload, { runId = "", projectId = "" } = {}) {
    const expectedRunId = String(runId || "").trim();
    const expectedProjectId = String(projectId || "").trim();
    const taskState = payload && typeof payload === "object" && payload.task_state && typeof payload.task_state === "object"
      ? payload.task_state
      : null;
    if (!taskState || taskState.projection_version !== PROJECTION_VERSION || !expectedRunId) return null;
    if (String(taskState.run_id || "") !== expectedRunId) return null;
    if (expectedProjectId && String(taskState.project_id || "") !== expectedProjectId) return null;
    const history = Array.isArray(taskState.history)
      ? taskState.history.slice(-8).flatMap((item) => {
          const stage = semanticValue(item?.stage, STAGE_ALLOWLIST);
          const status = semanticValue(item?.status, STATUS_ALLOWLIST);
          return stage === "unavailable" || status === "unavailable" ? [] : [{ stage, status }];
        })
      : [];
    return {
      runId: expectedRunId,
      projectId: expectedProjectId,
      currentStage: semanticValue(taskState.current_stage, STAGE_ALLOWLIST),
      status: semanticValue(taskState.status, STATUS_ALLOWLIST),
      history,
      artifactIds: allowlistedList(taskState.artifact_ids, ARTIFACT_ALLOWLIST),
      stateFiles: allowlistedList(taskState.state_files, STATE_FILE_ALLOWLIST),
    };
  }

  function taskStateConversationContent(snapshot) {
    const lines = [
      `运行状态 · ${snapshot.runId}`,
      `当前阶段：${snapshot.currentStage}`,
      `状态：${snapshot.status}`,
      `中间状态文件：${snapshot.stateFiles.length ? snapshot.stateFiles.join("、") : "尚无可展示的已持久化状态文件"}`,
    ];
    if (snapshot.history.length) {
      lines.push(`阶段记录：${snapshot.history.map((item) => `${item.stage} (${item.status})`).join(" → ")}`);
    }
    if (snapshot.artifactIds.length) lines.push(`已登记工件：${snapshot.artifactIds.join("、")}`);
    return lines.join("\n");
  }

  function sameConversationContext(expected, current) {
    return Boolean(
      expected
      && current
      && expected.projectId === current.projectId
      && expected.conversationId === current.conversationId
      && expected.generation === current.generation,
    );
  }

  function taskStateResponseSnapshot(payload, { runId, projectId, context, getContext }) {
    if (typeof getContext !== "function" || !sameConversationContext(context, getContext())) return null;
    return taskStateSnapshot(payload, { runId, projectId });
  }

  async function persistTaskStateRecord({
    payload,
    runId,
    projectId,
    context,
    getContext,
    messages,
    persistMessage,
    appendMessage,
  }) {
    if (!sameConversationContext(context, getContext())) return { status: "stale_context" };
    if (context.projectId !== projectId) return { status: "project_mismatch" };
    const snapshot = taskStateSnapshot(payload, { runId, projectId });
    if (!snapshot || (snapshot.status === "unavailable" && snapshot.currentStage === "unavailable")) {
      return { status: "invalid_task_state" };
    }
    const content = taskStateConversationContent(snapshot);
    if (messages.some((item) => item.role === "system" && item.content === content)) {
      return { status: "duplicate", snapshot };
    }
    if (!sameConversationContext(context, getContext())) return { status: "stale_context" };
    const persisted = await persistMessage({
      projectId: context.projectId,
      conversationId: context.conversationId,
      generation: context.generation,
      role: "system",
      content,
    });
    if (!sameConversationContext(context, getContext())) {
      return { status: "stale_after_persist", persisted: true, snapshot };
    }
    if (!persisted || String(persisted.content || "") !== content) {
      return { status: "invalid_persisted_message", snapshot };
    }
    messages.push({ role: "system", content });
    const chips = snapshot.stateFiles.map((name) => ({ original_name: name, artifact_id: name }));
    appendMessage("system", content, chips);
    return { status: "persisted", snapshot };
  }

  function conversationDecisionMessages(messages) {
    return Array.isArray(messages)
      ? messages.filter((item) => !(item?.role === "system" && String(item?.content || "").startsWith("运行状态 · ")))
      : [];
  }

  return {
    PROJECTION_VERSION,
    conversationDecisionMessages,
    persistTaskStateRecord,
    sameConversationContext,
    taskStateConversationContent,
    taskStateResponseSnapshot,
    taskStateSnapshot,
  };
});
