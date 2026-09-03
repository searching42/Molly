"use strict";

const state = {
  view: "new",
  bootstrap: null,
  detail: null,
  selectedRunId: null,
  editingProvider: null,
  files: [],
  goal: "",
  profileId: "",
  llmProfileRef: "",
  plan: null,
  planConfirmed: false,
  loading: false,
  pollTimer: null,
  pollInFlight: false,
  resumeInFlight: false,
  approvalInFlight: false,
  renderFingerprint: "",
};

const content = document.getElementById("app-content");
const topbarContext = document.getElementById("topbar-context");
const toast = document.getElementById("toast");
const localSessionToken = document.querySelector('meta[name="local-session-token"]')?.content || "";

const html = (value) => String(value ?? "")
  .replaceAll("&", "&amp;")
  .replaceAll("<", "&lt;")
  .replaceAll(">", "&gt;")
  .replaceAll('"', "&quot;")
  .replaceAll("'", "&#039;");

const pretty = (value) => {
  try {
    return html(JSON.stringify(value, null, 2));
  } catch {
    return html(value);
  }
};

const formatBytes = (size) => {
  if (!Number.isFinite(Number(size))) return "—";
  if (size < 1024) return `${size} B`;
  if (size < 1024 * 1024) return `${(size / 1024).toFixed(1)} KB`;
  return `${(size / (1024 * 1024)).toFixed(1)} MB`;
};

const statusClass = (status) => `status-${String(status || "").toLowerCase()}`;

const operationLabels = {
  br1_prepare_dataset: "清洗并标准化数据集",
  br1_applicability_preflight: "检查数据集适用性",
  br1_train_unimol: "训练 Uni-Mol 模型",
  br1_generate_reinvent4: "使用 REINVENT4 生成分子",
  br1_predict_unimol: "用当前模型预测",
  br1_evaluate_top_n: "筛选并输出 Top-N",
};

const executionLabels = {
  PENDING: "等待执行",
  SUCCEEDED: "已完成",
  FAILED: "执行失败",
  REJECTED: "已拒绝",
  INTERRUPTED: "已中断",
};

const operationLabel = (toolName) => operationLabels[toolName] || "系统操作";
const executionLabel = (status) => executionLabels[status] || "处理中";

const request = async (path, options = {}) => {
  const headers = { "content-type": "application/json", ...(options.headers || {}) };
  headers["X-Local-Session-Token"] = localSessionToken;
  const response = await fetch(path, {
    ...options,
    headers,
  });
  const value = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(value.message || "操作没有完成");
  return value;
};

const showToast = (message, isError = false) => {
  toast.textContent = message;
  toast.classList.toggle("is-error", isError);
  toast.classList.add("is-visible");
  window.clearTimeout(showToast.timer);
  showToast.timer = window.setTimeout(() => toast.classList.remove("is-visible"), 3200);
};

const setBusy = (busy) => {
  state.loading = busy;
  document.querySelectorAll("button").forEach((button) => {
    if (button.dataset.keepEnabled !== "true") button.disabled = busy;
  });
};

const selectedProfile = () => {
  const profiles = state.bootstrap?.runtime_profiles || [];
  return profiles.find((profile) => profile.profile_id === state.profileId) || profiles[0] || null;
};

const selectedWorkflow = () => selectedProfile()?.workflow || "core";
const selectedModel = () => (state.bootstrap?.model_profiles || []).find(
  (profile) => profile.profile_ref === state.llmProfileRef
);
const workflowNeedsFile = () => selectedWorkflow() === "br1";
const workflowNeedsModel = () => workflowNeedsFile();
const invalidatePlan = () => {
  state.plan = null;
  state.planConfirmed = false;
};

const loadBootstrap = async () => {
  state.bootstrap = await request("/api/bootstrap");
  const profiles = state.bootstrap.runtime_profiles || [];
  if (!profiles.some((profile) => profile.profile_id === state.profileId)) {
    state.profileId = profiles.find((profile) => profile.available)?.profile_id || "";
    invalidatePlan();
  }
  if (state.selectedRunId && !state.bootstrap.runs.some((run) => run.run_id === state.selectedRunId)) {
    state.selectedRunId = null;
    state.detail = null;
  }
  render({ force: true, preserveInteractive: true });
};

const loadRun = async (runId) => {
  state.selectedRunId = runId;
  state.detail = await request(`/api/runs/${encodeURIComponent(runId)}`);
  state.view = "runs";
  render({ force: true, preserveInteractive: true });
  startRunPolling();
};

const stopRunPolling = () => {
  if (state.pollTimer !== null) {
    window.clearInterval(state.pollTimer);
    state.pollTimer = null;
  }
};

const terminalStatuses = new Set(["STOPPED", "REJECTED", "FAILED", "BUDGET_EXHAUSTED"]);

const pollSelectedRun = async () => {
  if (!state.selectedRunId || state.pollInFlight) return;
  state.pollInFlight = true;
  try {
    const detail = await request(`/api/runs/${encodeURIComponent(state.selectedRunId)}`);
    state.detail = detail;
    const effectiveStatus = detail.effective_status || detail.ui_status || detail.status;
    if (state.view === "runs") render({ preserveInteractive: true });
    if (terminalStatuses.has(effectiveStatus) && !detail.background_pending) {
      stopRunPolling();
      return;
    }
    if (effectiveStatus === "ACTIVE" && detail.workflow === "br1" && !detail.pending_call && !detail.background_pending && !detail.background_error_type && !state.resumeInFlight) {
      state.resumeInFlight = true;
      try {
        const result = await request(`/api/runs/${encodeURIComponent(detail.run_id)}/resume`, { method: "POST", body: "{}" });
        state.detail = result.inspection || result;
        if (state.view === "runs") render({ force: true, preserveInteractive: true });
      } finally {
        state.resumeInFlight = false;
      }
      return;
    }
    if (effectiveStatus === "WAITING_APPROVAL" && !detail.background_pending) {
      stopRunPolling();
    }
  } catch (error) {
    showToast(error.message, true);
  } finally {
    state.pollInFlight = false;
  }
};

const startRunPolling = () => {
  stopRunPolling();
  if (!state.selectedRunId) return;
  state.pollTimer = window.setInterval(pollSelectedRun, 1200);
  pollSelectedRun();
};

const readFileAsBase64 = (file) => new Promise((resolve, reject) => {
  const reader = new FileReader();
  reader.onload = () => resolve(String(reader.result).split(",")[1] || "");
  reader.onerror = () => reject(new Error(`无法读取文件：${file.name}`));
  reader.readAsDataURL(file);
});

const parseCsvHeader = (text) => {
  const line = String(text || "").split(/\r?\n/, 1)[0] || "";
  const fields = [];
  let field = "";
  let quoted = false;
  for (let index = 0; index < line.length; index += 1) {
    const char = line[index];
    if (char === '"') {
      if (quoted && line[index + 1] === '"') { field += '"'; index += 1; }
      else quoted = !quoted;
    } else if (char === "," && !quoted) {
      fields.push(field.trim()); field = "";
    } else field += char;
  }
  fields.push(field.trim());
  return fields.filter(Boolean);
};

const validateWorkflowFile = async (file) => {
  const extension = String(file.name || "").toLowerCase().match(/\.[a-z0-9]+$/)?.[0] || "";
  if (!new Set([".json", ".csv"]).has(extension)) {
    throw new Error("工作流数据文件必须是 .json 或 .csv 格式");
  }
  let columns = [];
  const text = await file.text();
  if (extension === ".json") {
    let value;
    try { value = JSON.parse(text); } catch { throw new Error("JSON 数据文件无法解析"); }
    if (value && Array.isArray(value.columns)) columns = value.columns;
    else if (Array.isArray(value) && value.length && value[0] && typeof value[0] === "object") columns = Object.keys(value[0]);
    else throw new Error("JSON 数据文件必须是 pandas split JSON 或对象数组");
  } else {
    columns = parseCsvHeader(text);
  }
  const normalized = new Set(columns.map((column) => String(column).trim().toLowerCase()));
  if (!["canonical_smiles", "smiles", "chromophore"].some((name) => normalized.has(name))) {
    throw new Error("数据文件缺少必需的分子结构列（SMILES）");
  }
  const directTargetColumns = ["quantum_yield", "quantum yield", "homo_lumo_gap", "target_value", "target"];
  const hasOrbitalPair = normalized.has("energies_occ_pbe0_vac_tier2") && normalized.has("energies_unocc_pbe0_vac_tier2");
  if (!directTargetColumns.some((name) => normalized.has(name)) && !hasOrbitalPair) {
    throw new Error("数据文件缺少受支持的目标属性列");
  }
};

const uploadFiles = async (fileList) => {
  const files = Array.from(fileList || []);
  if (files.length !== 1) throw new Error("该工作流必须选择且只能选择一个数据文件");
  const file = files[0];
  if (file.size > 128 * 1024 * 1024) throw new Error(`${file.name} 超过 128 MB，暂不能上传`);
  if (workflowNeedsFile()) await validateWorkflowFile(file);
  const encoded = await readFileAsBase64(file);
  const uploaded = await request("/api/artifacts", {
    method: "POST",
    body: JSON.stringify({
      file_name: file.name,
      media_type: file.type || "application/octet-stream",
      workflow: selectedWorkflow(),
      content_base64: encoded,
    }),
  });
  if (workflowNeedsFile()) state.files = [uploaded];
  else state.files.push(uploaded);
  invalidatePlan();
};

const renderFiles = () => {
  if (!state.files.length) return `<div class="empty-files">${workflowNeedsFile() ? "请添加一个数据文件" : "还没有添加文件"}</div>`;
  return state.files.map((file, index) => `
    <div class="file-row">
      <div class="file-info"><span class="file-name">${html(file.name)}</span><span class="file-id">${html(file.artifact_id.slice(0, 20))} · ${formatBytes(file.size_bytes)}</span></div>
      <div class="file-actions"><button class="text-button" data-replace-file="${index}" type="button">替换</button><button class="text-button danger-text" data-remove-file="${index}" type="button">移除</button></div>
    </div>
  `).join("");
};

const formatDuration = (seconds) => {
  const value = Number(seconds);
  if (!Number.isFinite(value)) return "—";
  if (value < 3600) return `${Math.round(value / 60)} 分钟`;
  return `${(value / 3600).toFixed(value % 3600 ? 1 : 0)} 小时`;
};

const renderIntentPlan = (intent, options = {}) => {
  const spec = intent?.spec;
  if (!spec) return "";
  const rows = [
    ["目标属性", spec.target_property],
    ["优化方向", spec.direction],
    ["候选数量", spec.candidate_count],
    ["Top-N", spec.top_n],
    ["骨架约束", spec.scaffold_constraint],
    ["随机种子", spec.seed],
    ["工作站偏好", spec.host_preference],
    ["资源", `CPU ${spec.cpu_threads} · GPU ${spec.gpu_count}`],
    ["最长运行时间", formatDuration(spec.walltime_sec)],
  ];
  const confirmation = options.confirm
    ? `<label class="plan-confirm"><input id="confirm-plan" type="checkbox" ${state.planConfirmed ? "checked" : ""} /> 我已核对以上计划，确认开始执行</label>`
    : "";
  return `
    <section class="plan-card" aria-labelledby="plan-title">
      <div class="plan-head"><div><h3 id="plan-title">${html(options.title || "结构化执行计划")}</h3><p class="field-hint">以下参数由结构化 LLM 解析后冻结，开始后不会因轮询或恢复而改变。</p></div><span class="status-pill status-active">已冻结</span></div>
      <div class="plan-grid">${rows.map(([label, value]) => `<div class="plan-item"><span>${html(label)}</span><strong>${html(value)}</strong></div>`).join("")}</div>
      ${intent.warnings?.length ? `<div class="plan-warning">提示：${html(intent.warnings.join("；"))}</div>` : ""}
      ${options.digest ? `<div class="plan-digests"><span>intent digest</span><code>${html(options.digest)}</code><span>spec digest</span><code>${html(options.specDigest || "—")}</code></div>` : ""}
      ${confirmation}
    </section>
  `;
};

const renderNew = () => {
  const profiles = (state.bootstrap?.runtime_profiles || []).filter((profile) => profile.available);
  const profileOptions = profiles.length
    ? profiles.map((profile) => {
      const constraints = profile.resource_constraints || {};
      const resource = profile.workflow === "br1"
        ? ` · CPU ${html(constraints.cpu_threads ?? "—")} · GPU ${html(constraints.gpu_count ?? "—")}`
        : "";
      const selected = profile.profile_id === state.profileId || (!state.profileId && profile === profiles[0]);
      return `<option value="${html(profile.profile_id)}" ${selected ? "selected" : ""}>${html(profile.name)}${resource}</option>`;
    }).join("")
    : '<option value="">暂无可用运行配置</option>';
  const noProfile = !profiles.length;
  const currentProfile = selectedProfile();
  const workflow = currentProfile?.workflow || "core";
  const needsFile = workflow === "br1";
  const needsModel = workflowNeedsModel();
  const modelProfiles = state.bootstrap?.model_profiles || [];
  const modelOptions = `<option value="">请选择解析模型</option>${modelProfiles.map((profile) => `<option value="${html(profile.profile_ref)}" ${profile.profile_ref === state.llmProfileRef ? "selected" : ""}>${html(profile.name)} · ${html(profile.model_identifier)}${profile.credential_configured ? "" : "（未配置）"}</option>`).join("")}`;
  const model = selectedModel();
  const modelReady = !needsModel ? true : Boolean(model?.credential_configured);
  const fileReady = !needsFile || state.files.length === 1;
  const planReady = !needsFile || Boolean(state.plan?.preview_token && state.planConfirmed);
  const cannotStart = noProfile || !modelReady || !fileReady || !planReady;
  const modelHint = needsModel
    ? "任务目标必须经过已配置的结构化 LLM 解析；API Key 只由本机服务端读取。"
    : "当前工作流不需要自然语言解析；执行策略和资源配置由服务器端登记。";
  const workflowHint = currentProfile
    ? "工作流会按服务器端登记的资源配置，在需要确认的阶段暂停。"
    : "运行配置由本机服务器端登记。";
  const modelNotice = !needsModel
    ? ""
    : !modelProfiles.length
    ? '<div class="notice failure-notice"><div class="notice-icon" aria-hidden="true">!</div><div><strong>尚未配置模型服务</strong>开始任务前请先添加模型服务并保存 API Key。<button class="text-button" data-view="providers" type="button">去配置模型服务</button></div></div>'
    : (needsModel && state.llmProfileRef && !modelReady
      ? '<div class="notice"><div class="notice-icon" aria-hidden="true">ⓘ</div><div><strong>所选模型服务尚未就绪</strong>请先保存 API Key，再解析执行计划。<button class="text-button" data-view="providers" type="button">去配置模型服务</button></div></div>'
      : "");
  return `
    <div class="page-heading">
      <div>
        <div class="eyebrow">科学任务 / 开始</div>
        <h1>你想完成什么任务？</h1>
        <p class="lead">写下目标，系统会按服务器端允许的范围执行，并把每一步留下可查看的记录。</p>
      </div>
      <button class="secondary-button" data-view="providers" type="button">管理模型服务</button>
    </div>
    ${noProfile ? `
      <div class="notice mb-20">
        <div class="notice-icon" aria-hidden="true">ⓘ</div>
        <div><strong>当前没有可执行的运行配置</strong>请先由服务器端登记运行配置，再开始任务。</div>
      </div>
    ` : ""}
    <div class="layout-two">
      <form class="card card-pad stack" id="new-task-form">
        <div class="field">
          <label class="field-label" for="goal">任务目标</label>
          <span class="field-hint">尽量用一句或几句话说明你希望得到什么结果。</span>
          <textarea id="goal" name="goal" required placeholder="例如：以 HOMO-LUMO gap 为目标，不限制骨架，采样空间 1000，筛选较小的分子，输出 top 5">${html(state.goal)}</textarea>
        </div>
        ${modelNotice}
        <div class="field-row">
          <div class="field">
            <label class="field-label" for="profile-id">运行配置</label>
            <select id="profile-id" name="profile_id" ${noProfile ? "disabled" : ""} required>${profileOptions}</select>
          </div>
          <div class="field">
            <label class="field-label">开始方式</label>
            <div class="notice mode-note">${html(workflowHint)}</div>
          </div>
        </div>
        <div class="field-row">
          <div class="field">
            <label class="field-label" for="llm-profile">自然语言解析模型服务</label>
            <select id="llm-profile" ${!modelProfiles.length ? "disabled" : ""}>${modelOptions}</select>
            <span class="field-hint">${html(modelHint)}</span>
          </div>
          <div class="field">
            <label class="field-label">资源选择</label>
            <div class="notice mode-note">CPU/GPU、工作站和路径均来自运行配置，网页不能改写。</div>
          </div>
        </div>
        <div class="upload-card">
          <div class="upload-head">
            <div>
              <div class="upload-title">数据文件 <span class="optional-label">${needsFile ? "（必填，单个）" : "（可选）"}</span></div>
              <div class="upload-subtitle">${needsFile ? "仅支持一个 .json 或 .csv 文件；选择后会先检查格式和必需字段。" : "文件会保存到本机的不可变数据区，单个文件不超过 128 MB。"}</div>
            </div>
            <label class="file-button" for="file-input">选择或替换文件<input class="visually-hidden" id="file-input" type="file" accept="${needsFile ? ".json,.csv" : "*/*"}" ${needsFile ? "aria-required=\"true\"" : ""} /></label>
          </div>
          <div class="file-list" id="file-list">${renderFiles()}</div>
        </div>
        ${needsFile ? `<div class="plan-actions"><button class="secondary-button" data-preview-plan type="button" ${(!fileReady || !modelReady || noProfile) ? "disabled" : ""}>解析并预览执行计划</button><span class="field-hint">先确认结构化计划，再开始任务。</span></div>${state.plan ? renderIntentPlan(state.plan.intent, { confirm: true, digest: state.plan.intent_digest, specDigest: state.plan.spec_digest, title: "开始前确认执行计划" }) : ""}` : ""}
        <div class="form-actions">
          <button class="primary-button" type="submit" ${cannotStart ? "disabled" : ""}>开始任务</button>
          <span class="field-hint">${needsModel && !modelProfiles.length ? "请先配置模型服务。" : needsModel && !modelReady ? "所选模型服务未配置 API Key。" : needsFile && !fileReady ? "请先添加一个数据文件。" : needsFile && !state.plan ? "请先预览并确认执行计划。" : "开始后仍可在需要时确认关键操作。"}</span>
        </div>
      </form>
      <div>
        <div class="card side-card">
          <div class="side-title">这次会发生什么</div>
          <div class="side-list">
            <div class="side-list-item">先检查目标和数据文件</div>
            <div class="side-list-item">按允许的系统操作逐步执行</div>
            <div class="side-list-item">需要外部影响时先等待你的确认</div>
            <div class="side-list-item">保存结果和完整操作记录</div>
          </div>
        </div>
        <div class="card side-card">
          <div class="side-title">密钥安全</div>
          <p class="side-copy">模型密钥由本机服务端保存和读取。网页输入后只发送到本机服务端，不写入浏览器存储；模型请求再由本机服务端按已保存的服务配置发送。</p>
        </div>
      </div>
    </div>
  `;
};

const renderRunList = () => {
  const runs = state.bootstrap?.runs || [];
  if (!runs.length) return '<div class="empty-state"><h3>还没有任务记录</h3><p>完成第一个任务后，记录会显示在这里。</p></div>';
  return `<div class="run-list">${runs.map((run) => `
    <button class="run-item ${state.selectedRunId === run.run_id ? "is-selected" : ""}" data-run-id="${html(run.run_id)}" type="button">
      <div class="run-item-goal">${html(run.goal)}</div>
      <div class="run-item-meta"><span>${html(run.step_count)} 个步骤 · ${html(run.artifact_count)} 个结果</span><span class="status-pill ${statusClass(run.status)}">${html(run.status_label)}</span></div>
    </button>
  `).join("")}</div>`;
};

const callStatus = (call) => {
  if (call.execution_status === "SUCCEEDED") return "success";
  if (call.execution_status === "FAILED" || call.execution_status === "REJECTED") return "failed";
  return "pending";
};

const renderTimeline = (detail) => {
  const calls = detail.materialized_calls || [];
  if (!calls.length) return '<div class="empty-files">任务还没有产生系统操作。</div>';
  return `<div class="timeline">${calls.map((call, index) => `
    <div class="timeline-item ${callStatus(call)}">
      <div class="timeline-main"><span class="timeline-name">第 ${index + 1} 步 · ${html(operationLabel(call.tool_name))}</span><span class="field-hint">${html(executionLabel(call.execution_status))}</span></div>
      <div class="timeline-sub">${call.output_artifact_ids?.length ? `${call.output_artifact_ids.length} 个结果文件` : "没有新增结果文件"}</div>
      ${call.reason_summary ? `<div class="timeline-reason">${html(call.reason_summary)}</div>` : ""}
      ${call.result_data ? `<details class="timeline-result"><summary>查看系统摘要</summary><pre>${pretty(call.result_data)}</pre></details>` : ""}
      <details class="call-technical"><summary>查看技术标识</summary><code>${html(call.tool_name)}@${html(call.tool_version)}</code></details>
    </div>
  `).join("")}</div>`;
};

const workflowActionContext = (detail, pending) => {
  const spec = detail.frozen_intent?.spec || {};
  const resources = detail.runtime_profile?.resource_constraints || {};
  const cpu = spec.cpu_threads ?? resources.cpu_threads ?? "—";
  const gpu = spec.gpu_count ?? resources.gpu_count ?? "—";
  const walltime = spec.walltime_sec ?? resources.walltime_sec;
  const outputs = {
    br1_prepare_dataset: "一个清洗后的数据集和清洗报告",
    br1_applicability_preflight: "适用性检查结果",
    br1_train_unimol: "当前运行绑定的模型包和训练报告",
    br1_generate_reinvent4: "候选分子包和生成报告",
    br1_predict_unimol: "当前模型对候选分子的预测包和报告",
    br1_evaluate_top_n: "最终 Top-N 排名结果和评估报告",
  };
  return [
    ["为什么做", pending.reason_summary || "执行当前工作流的下一步"],
    ["预计资源", `CPU ${cpu} · GPU ${gpu}${walltime ? ` · 最长 ${formatDuration(walltime)}` : ""}`],
    ["会产生什么", outputs[pending.tool_name] || "一个新的系统产物"],
    ["拒绝后果", "任务会停止，不会执行这一步；已产生的只读记录会保留"],
  ];
};

const renderPendingAction = (detail) => {
  const pending = detail.pending_call;
  if (!pending || detail.background_pending || state.approvalInFlight) return "";
  const context = workflowActionContext(detail, pending);
  return `
    <div class="action-card">
      <h3>这一步需要你的确认</h3>
      <p class="action-copy">系统已经准备好执行下面的操作。确认后才会真正执行；长时间计算会在后台运行，页面会自动刷新。</p>
      <div class="operation-name">系统操作：${html(operationLabel(pending.tool_name))}</div>
      <div class="action-context">${context.map(([label, value]) => `<div class="action-context-row"><span>${html(label)}</span><strong>${html(value)}</strong></div>`).join("")}</div>
      <details class="operation-args"><summary>查看精确参数</summary><pre>${pretty(pending.arguments)}</pre></details>
      <details class="call-technical"><summary>查看技术标识</summary><code>${html(pending.tool_name)}@${html(pending.tool_version)}</code></details>
      <div class="action-buttons">
        <button class="primary-button" data-approval="APPROVED" type="button">确认并执行</button>
        <button class="danger-button" data-approval="REJECTED" type="button">拒绝并停止任务</button>
      </div>
    </div>
  `;
};

const renderArtifactGroup = (title, artifacts, emptyText) => {
  if (!artifacts?.length) return `<section class="artifact-group"><h3>${html(title)}</h3><p class="empty-files">${html(emptyText)}</p></section>`;
  return `<section class="artifact-group"><h3>${html(title)}</h3><div class="artifact-list">${artifacts.map((artifact) => `
    <div class="artifact-download"><div><strong>${html(artifact.name || artifact.download_name)}</strong><span class="artifact-meta">${html(artifact.media_type)} · ${formatBytes(artifact.size_bytes)}</span><code>${html(artifact.artifact_id)}</code></div><a href="${html(artifact.download_path)}">下载文件</a></div>
  `).join("")}</div></section>`;
};

const renderTopNResult = (result) => {
  if (!result?.rows?.length) return "";
  return `<section class="result-card"><div class="result-head"><div><h3>最终结果：Top-N</h3><p class="field-hint">目标属性：${html(result.target_property || "—")} · 仅表示计算预测结果</p></div><a class="secondary-link" href="/api/artifacts/${encodeURIComponent(result.artifact_id)}/content">下载 ${html(result.download_name)}</a></div><div class="table-wrap"><table><thead><tr><th>排名</th><th>SMILES</th><th>预测值</th><th>候选编号</th></tr></thead><tbody>${result.rows.map((row) => `<tr><td>${html(row.rank ?? "—")}</td><td><code>${html(row.smiles ?? "—")}</code></td><td>${html(row.predicted_property ?? row.proxy_utility ?? "—")}</td><td>${html(row.candidate_id ?? "—")}</td></tr>`).join("")}</tbody></table></div></section>`;
};

const renderRunDetail = () => {
  const detail = state.detail;
  if (!detail) return '<div class="empty-state"><h3>选择一个任务</h3><p>从左侧选择任务记录，查看执行过程。</p></div>';
  const effectiveStatus = detail.effective_status || detail.ui_status || detail.status;
  const backgroundPending = Boolean(detail.background_pending);
  const canContinue = !backgroundPending && ["ACTIVE", "INTERRUPTED"].includes(effectiveStatus);
  const groups = detail.artifact_groups || { inputs: [], intermediate: [], final: [] };
  const observability = state.bootstrap?.observability || {};
  const monitorButtons = [
    ["json", "下载 JSON 追踪", true],
    ["otel", "导出 OpenTelemetry", observability.otel?.available],
    ["langsmith", "导出 LangSmith", observability.langsmith?.available],
  ].map(([name, label, available]) => `<button class="secondary-button" data-observe="${name}" type="button" ${available ? "" : "disabled"}>${label}</button>`).join("");
  const failureNotice = (detail.failure_summary || []).length
    ? `<div class="notice failure-notice mb-20"><div class="notice-icon" aria-hidden="true">!</div><div><strong>有步骤执行失败</strong><pre>${pretty(detail.failure_summary)}</pre></div></div>`
    : "";
  const rejectionNotice = effectiveStatus === "REJECTED"
    ? '<div class="notice failure-notice mb-20"><div class="notice-icon" aria-hidden="true">×</div><div><strong>任务已拒绝/已取消</strong>你拒绝了待确认操作，任务已停止；没有后续步骤会被执行。</div></div>'
    : "";
  const backgroundNotice = backgroundPending
    ? '<div class="notice mb-20"><div class="notice-icon" aria-hidden="true">↻</div><div><strong>执行中</strong>后台正在处理当前阶段，审批和恢复操作暂不可用，页面会自动刷新。</div></div>'
    : detail.background_error_type
      ? `<div class="notice failure-notice mb-20"><div class="notice-icon" aria-hidden="true">!</div><div><strong>后台执行异常</strong>错误类型：<code>${html(detail.background_error_type)}</code>。请查看操作记录。</div></div>`
      : "";
  return `
    <div class="card run-detail">
      <div class="run-detail-head">
        <div class="run-detail-title">
          <div><div class="run-goal">${html(detail.goal)}</div><div class="run-id">任务编号：${html(detail.run_id)}</div></div>
          <span class="status-pill ${statusClass(effectiveStatus)}">${html(detail.status_label)}</span>
        </div>
        <div class="run-metrics">
          <div class="metric"><span class="metric-value">${html(detail.step_count)}</span><span class="metric-label">步骤</span></div>
          <div class="metric"><span class="metric-value">${html(detail.tool_call_count)}</span><span class="metric-label">系统操作</span></div>
          <div class="metric"><span class="metric-value">${html(detail.artifact_count)}</span><span class="metric-label">结果文件</span></div>
        </div>
      </div>
      <div class="run-detail-body">
        ${backgroundNotice}
        ${rejectionNotice}
        ${failureNotice}
        ${detail.frozen_intent ? renderIntentPlan(detail.frozen_intent, { title: "已冻结的执行计划", digest: detail.intent_digest, specDigest: detail.spec_digest }) : ""}
        ${renderPendingAction(detail)}
        ${canContinue ? '<div class="notice mb-20"><div class="notice-icon" aria-hidden="true">↻</div><div><strong>任务还可以继续</strong>恢复后会读取已冻结计划，不会重新解析目标。</div></div><button class="secondary-button mb-22" data-resume="true" type="button">继续任务</button>' : ""}
        ${renderTopNResult(detail.top_n_result)}
        <div class="artifact-groups">
          ${renderArtifactGroup("输入数据", groups.inputs, "没有输入文件")}
          ${renderArtifactGroup("最终结果", groups.final, "任务完成后，最终结果会显示在这里")}
          ${renderArtifactGroup("中间产物", groups.intermediate, "还没有中间产物")}
        </div>
        <div class="timeline-title">操作记录</div>
        ${renderTimeline(detail)}
        <div class="monitor-actions"><span class="field-hint">只读监控：</span>${monitorButtons}</div>
        <details class="technical">
          <summary>技术详情</summary>
          <div class="technical-body">
            <div class="technical-row"><span>运行配置</span><code>${html(detail.runtime_profile_ref || "—")}</code></div>
            <div class="technical-row"><span>工作流</span><code>${html(detail.workflow || "—")}</code></div>
            <div class="technical-row"><span>任务校验码</span><code>${html(detail.request_digest)}</code></div>
            <div class="technical-row"><span>权限校验码</span><code>${html(detail.policy_digest)}</code></div>
            <div class="technical-row"><span>意图校验码</span><code>${html(detail.intent_digest || "—")}</code></div>
          </div>
        </details>
      </div>
    </div>
  `;
};

const renderRuns = () => `
  <div class="page-toolbar">
    <div><h2>任务记录</h2><p class="page-toolbar-copy">每个任务的状态、操作记录和结果都在这里。</p></div>
    <button class="primary-button" data-view="new" type="button">新建任务</button>
  </div>
  <div class="run-layout">
    <div>${renderRunList()}</div>
    <div class="run-detail">${renderRunDetail()}</div>
  </div>
`;

const providerSlug = (name) => {
  const slug = String(name || "provider").toLowerCase().replace(/[^a-z0-9]+/g, "-").replace(/^-|-$/g, "");
  return `provider-${slug || Date.now().toString(36)}`.slice(0, 80);
};

const renderProviderCards = () => {
  const profiles = state.bootstrap?.model_profiles || [];
  if (!profiles.length) return '<div class="empty-state"><h3>还没有模型服务</h3><p>在右侧添加服务并在本页输入 API Key，密钥只会保存到本机服务端。</p></div>';
  return `<div class="provider-list">${profiles.map((profile) => `
    <div class="provider-card">
      <div class="provider-card-head">
        <div><div class="provider-name">${html(profile.name)}</div><div class="provider-model">${html(profile.model_identifier)}</div></div>
        <span class="status-pill ${profile.credential_configured ? "status-stopped" : "status-waiting_approval"}">${html(profile.credential_status)}</span>
      </div>
      <div class="provider-card-meta"><span>地址：${html(profile.endpoint)}</span><span>超时：${html(profile.timeout_seconds)} 秒</span></div>
      <div class="provider-actions"><button class="secondary-button" data-edit-provider="${html(profile.profile_ref)}" type="button">编辑</button><button class="secondary-button" data-check-provider="${html(profile.profile_ref)}" type="button">检查本机配置</button><button class="secondary-button" data-test-provider="${html(profile.profile_ref)}" type="button" ${profile.credential_configured ? "" : "disabled"}>测试连接</button></div>
      <details class="technical"><summary>技术详情</summary><div class="technical-body"><div class="technical-row"><span>服务标识</span><code>${html(profile.profile_ref)}</code></div><div class="technical-row"><span>密钥状态</span><span>${html(profile.credential_status)}</span></div></div></details>
    </div>
  `).join("")}</div>`;
};

const renderProviders = () => {
  const editing = state.editingProvider;
  return `
    <div class="page-heading">
      <div><div class="eyebrow">科学任务 / 设置</div><h1>模型与服务</h1><p class="lead">网页可配置地址、模型、超时和 API Key；密钥只保存到本机服务端，不会返回到页面。</p></div>
    </div>
    <div class="provider-grid">
      <div>${renderProviderCards()}</div>
      <form class="card provider-form stack" id="provider-form">
        <h3>${editing ? "编辑模型服务" : "添加模型服务"}</h3>
        <div class="field"><label class="field-label" for="provider-name">服务名称</label><input id="provider-name" type="text" required maxlength="80" value="${html(editing?.name || "")}" placeholder="例如：我的模型服务" /></div>
        <div class="field"><label class="field-label" for="provider-endpoint">服务地址</label><input id="provider-endpoint" type="text" required value="${html(editing?.endpoint || "https://api.example.com/v1")}" placeholder="https://..." /><span class="field-hint">必须是 HTTPS 地址。</span></div>
        <div class="field"><label class="field-label" for="provider-model">模型名称</label><input id="provider-model" type="text" required value="${html(editing?.model_identifier || "")}" placeholder="例如：model-name" /></div>
        <div class="field"><label class="field-label" for="provider-timeout">请求超时（秒）</label><input id="provider-timeout" type="number" min="1" max="300" value="${html(editing?.timeout_seconds || 30)}" /></div>
        ${editing ? `<div class="provider-key-note"><label class="field-label" for="provider-api-key">API Key</label><span>当前状态：${html(editing.credential_status)}</span><span class="field-hint">输入内容只发送到本机服务端，保存后不会回显、写入任务记录或保存到浏览器。</span><div class="provider-key-form"><input id="provider-api-key" type="password" autocomplete="new-password" placeholder="输入 API Key" /><button class="secondary-button" data-save-provider-key="${html(editing.profile_ref)}" type="button">保存 API Key</button></div></div>` : '<div class="provider-key-note"><strong>API Key</strong><span>先保存服务，再在这里输入 API Key。</span></div>'}
        <div class="form-actions"><button class="primary-button" type="submit">保存服务</button>${editing ? '<button class="secondary-button" data-clear-provider="true" type="button">取消编辑</button>' : ""}</div>
        ${editing ? '<div class="provider-check" id="provider-check"></div>' : ""}
      </form>
    </div>
  `;
};

const render = ({ force = false, preserveInteractive = false } = {}) => {
  if (!state.bootstrap) return;
  const fingerprint = JSON.stringify({ view: state.view, bootstrap: state.bootstrap, detail: state.detail, files: state.files, plan: state.plan, planConfirmed: state.planConfirmed, editingProvider: state.editingProvider });
  if (!force && fingerprint === state.renderFingerprint) return;
  const focusedId = preserveInteractive ? document.activeElement?.id : "";
  const openDetails = preserveInteractive
    ? Array.from(content.querySelectorAll("details")).map((detail) => detail.open)
    : [];
  const scrollPosition = preserveInteractive ? window.scrollY : 0;
  const labels = { new: "新建任务", runs: "任务记录", providers: "模型与服务" };
  topbarContext.textContent = labels[state.view] || "科学任务工作台";
  document.querySelectorAll(".nav-item").forEach((item) => item.classList.toggle("is-active", item.dataset.view === state.view));
  if (state.view === "new") content.innerHTML = renderNew();
  if (state.view === "runs") content.innerHTML = renderRuns();
  if (state.view === "providers") content.innerHTML = renderProviders();
  state.renderFingerprint = fingerprint;
  bindPageEvents();
  if (preserveInteractive) {
    content.querySelectorAll("details").forEach((detail, index) => {
      if (openDetails[index] !== undefined) detail.open = openDetails[index];
    });
    if (focusedId) document.getElementById(focusedId)?.focus({ preventScroll: true });
    window.scrollTo(0, scrollPosition);
  }
  if (state.loading) document.querySelectorAll("button").forEach((button) => { button.disabled = true; });
};

const bindPageEvents = () => {
  document.querySelectorAll("[data-view]").forEach((button) => button.addEventListener("click", () => {
    state.view = button.dataset.view;
    if (state.view !== "runs") state.detail = null;
    if (state.view !== "providers") state.editingProvider = null;
    render({ force: true, preserveInteractive: true });
  }));

  const fileInput = document.getElementById("file-input");
  if (fileInput) fileInput.addEventListener("change", async () => {
    try {
      setBusy(true);
      await uploadFiles(fileInput.files);
      showToast("文件已保存到本机");
      render({ force: true, preserveInteractive: true });
    } catch (error) {
      showToast(error.message, true);
    } finally {
      setBusy(false);
    }
  });

  document.querySelectorAll("[data-remove-file]").forEach((button) => button.addEventListener("click", () => {
    const index = Number(button.dataset.removeFile);
    if (!Number.isInteger(index)) return;
    state.files.splice(index, 1);
    invalidatePlan();
    render({ force: true, preserveInteractive: true });
  }));
  document.querySelectorAll("[data-replace-file]").forEach((button) => button.addEventListener("click", () => {
    document.getElementById("file-input")?.click();
  }));

  const goalInput = document.getElementById("goal");
  if (goalInput) goalInput.addEventListener("input", () => {
    state.goal = goalInput.value;
    invalidatePlan();
  });
  const profileInput = document.getElementById("profile-id");
  if (profileInput) profileInput.addEventListener("change", () => {
    state.profileId = profileInput.value;
    invalidatePlan();
    render({ force: true, preserveInteractive: true });
  });
  const modelInput = document.getElementById("llm-profile");
  if (modelInput) modelInput.addEventListener("change", () => {
    state.llmProfileRef = modelInput.value;
    invalidatePlan();
    render({ force: true, preserveInteractive: true });
  });

  const confirmPlan = document.getElementById("confirm-plan");
  if (confirmPlan) confirmPlan.addEventListener("change", () => {
    state.planConfirmed = confirmPlan.checked;
    render({ force: true, preserveInteractive: true });
  });

  const previewButton = document.querySelector("[data-preview-plan]");
  if (previewButton) previewButton.addEventListener("click", async () => {
    const goal = document.getElementById("goal")?.value.trim() || state.goal;
    const profileId = document.getElementById("profile-id")?.value || state.profileId;
    const llmProfileRef = document.getElementById("llm-profile")?.value || state.llmProfileRef;
    state.goal = goal; state.profileId = profileId; state.llmProfileRef = llmProfileRef;
    if (!goal || !profileId || !llmProfileRef || !state.files.length) return showToast("请先填写目标、选择模型服务并添加一个数据文件", true);
    try {
      setBusy(true);
      state.plan = await request("/api/workflows/preview", {
        method: "POST",
        body: JSON.stringify({ goal, profile_id: profileId, input_artifact_ids: state.files.map((file) => file.artifact_id), llm_profile_ref: llmProfileRef }),
      });
      state.planConfirmed = false;
      render({ force: true, preserveInteractive: true });
      showToast("执行计划已生成，请核对后确认");
    } catch (error) { showToast(error.message, true); } finally { setBusy(false); }
  });

  const taskForm = document.getElementById("new-task-form");
  if (taskForm) taskForm.addEventListener("submit", async (event) => {
    event.preventDefault();
    const goal = document.getElementById("goal").value.trim();
    const profileId = document.getElementById("profile-id").value;
    const llmProfileRef = document.getElementById("llm-profile")?.value || "";
    state.goal = goal; state.profileId = profileId; state.llmProfileRef = llmProfileRef;
    const workflow = selectedProfile()?.workflow || "core";
    if (!goal || !profileId || (workflow === "br1" && !llmProfileRef)) return showToast("请填写任务目标并选择运行配置和解析模型", true);
    if (workflow === "br1" && state.files.length !== 1) return showToast("该工作流必须上传且只能上传一个数据文件", true);
    if (workflow === "br1" && !state.plan?.preview_token) return showToast("请先解析并预览执行计划", true);
    if (workflow === "br1" && !state.planConfirmed) return showToast("请先确认执行计划", true);
    try {
      setBusy(true);
      const result = await request("/api/runs", {
        method: "POST",
          body: JSON.stringify({
          goal,
          profile_id: profileId,
          input_artifact_ids: state.files.map((file) => file.artifact_id),
          llm_profile_ref: llmProfileRef || undefined,
          workflow_intent_preview_token: state.plan?.preview_token,
        }),
      });
      state.selectedRunId = result.run_id;
      state.detail = result.inspection;
      state.files = [];
      state.plan = null;
      state.planConfirmed = false;
      state.view = "runs";
      await loadBootstrap();
      showToast(result.status_label || "任务已开始");
      startRunPolling();
    } catch (error) {
      showToast(error.message, true);
    } finally {
      setBusy(false);
    }
  });

  document.querySelectorAll("[data-run-id]").forEach((button) => button.addEventListener("click", async () => {
    try { await loadRun(button.dataset.runId); } catch (error) { showToast(error.message, true); }
  }));

  document.querySelectorAll("[data-approval]").forEach((button) => button.addEventListener("click", async () => {
    if (!state.detail?.pending_call || state.detail.background_pending || state.approvalInFlight) return;
    const decision = button.dataset.approval;
    if (decision === "REJECTED" && !window.confirm("拒绝后任务会立即停止，后续步骤不会执行。确定拒绝吗？")) return;
    const runId = state.detail.run_id;
    const callId = state.detail.pending_call.call_id;
    state.approvalInFlight = true;
    state.detail = { ...state.detail, background_pending: true, status: "ACTIVE", ui_status: "ACTIVE", effective_status: "ACTIVE", status_label: "执行中" };
    render({ force: true, preserveInteractive: true });
    try {
      setBusy(true);
      const result = await request(`/api/runs/${encodeURIComponent(runId)}/approval`, {
        method: "POST",
        body: JSON.stringify({ decision, reviewer_ref: "local-user", call_id: callId }),
      });
      state.detail = result.result?.inspection || result.inspection || await request(`/api/runs/${encodeURIComponent(runId)}`);
      await loadBootstrap();
      state.view = "runs";
      render({ force: true, preserveInteractive: true });
      startRunPolling();
      showToast(decision === "APPROVED" ? "已确认这一步，后台正在执行" : "已拒绝，任务已停止");
    } catch (error) {
      state.detail = await request(`/api/runs/${encodeURIComponent(runId)}`).catch(() => state.detail);
      render({ force: true, preserveInteractive: true });
      showToast(error.message, true);
    } finally {
      state.approvalInFlight = false;
      setBusy(false);
      render({ force: true, preserveInteractive: true });
      const status = state.detail?.effective_status || state.detail?.ui_status || state.detail?.status;
      if (state.detail?.run_id && !state.detail.background_pending && !terminalStatuses.has(status)) {
        startRunPolling();
      }
    }
  }));

  const resume = document.querySelector("[data-resume]");
  if (resume) resume.addEventListener("click", async () => {
    if (state.detail?.background_pending || state.resumeInFlight) return;
    state.resumeInFlight = true;
    state.detail = { ...state.detail, background_pending: true, status: "ACTIVE", ui_status: "ACTIVE", effective_status: "ACTIVE", status_label: "执行中" };
    render({ force: true, preserveInteractive: true });
    try {
      setBusy(true);
      const result = await request(`/api/runs/${encodeURIComponent(state.detail.run_id)}/resume`, { method: "POST", body: "{}" });
      state.detail = result.inspection || result;
      await loadBootstrap();
      render({ force: true, preserveInteractive: true });
      showToast(result.status_label || "任务状态已更新");
      startRunPolling();
    } catch (error) { showToast(error.message, true); } finally { state.resumeInFlight = false; setBusy(false); }
  });

  document.querySelectorAll("[data-observe]").forEach((button) => button.addEventListener("click", async () => {
    if (!state.detail?.run_id) return;
    try {
      setBusy(true);
      const result = await request(`/api/runs/${encodeURIComponent(state.detail.run_id)}/observe`, {
        method: "POST",
        body: JSON.stringify({ exporter: button.dataset.observe }),
      });
      if (button.dataset.observe === "json") {
        const blob = new Blob([JSON.stringify(result.trace || result, null, 2)], { type: "application/json" });
        const link = document.createElement("a");
        link.href = URL.createObjectURL(blob);
        link.download = `${state.detail.run_id}-trace.json`;
        link.click();
        URL.revokeObjectURL(link.href);
      }
      showToast(result.status === "EXPORTED" ? "监控追踪已导出" : "监控导出未完成", result.status !== "EXPORTED");
    } catch (error) { showToast(error.message, true); } finally { setBusy(false); }
  }));

  const providerForm = document.getElementById("provider-form");
  if (providerForm) providerForm.addEventListener("submit", async (event) => {
    event.preventDefault();
    const name = document.getElementById("provider-name").value.trim();
    const editingRef = state.editingProvider?.profile_ref;
    try {
      setBusy(true);
      const result = await request("/api/model-profiles", {
        method: "POST",
        body: JSON.stringify({
          profile_ref: editingRef || providerSlug(name),
          display_name: name,
          endpoint: document.getElementById("provider-endpoint").value.trim(),
          model_identifier: document.getElementById("provider-model").value.trim(),
          model_version: "1",
          timeout_seconds: Number(document.getElementById("provider-timeout").value),
          max_response_bytes: 262144,
        }),
      });
      state.editingProvider = result.profile;
      await loadBootstrap();
      state.view = "providers";
      render({ force: true, preserveInteractive: true });
      showToast("模型服务已保存");
    } catch (error) { showToast(error.message, true); } finally { setBusy(false); }
  });

  document.querySelectorAll("[data-edit-provider]").forEach((button) => button.addEventListener("click", () => {
    state.editingProvider = state.bootstrap.model_profiles.find((profile) => profile.profile_ref === button.dataset.editProvider) || null;
    render({ force: true, preserveInteractive: true });
  }));
  document.querySelectorAll("[data-clear-provider]").forEach((button) => button.addEventListener("click", () => {
    state.editingProvider = null;
    render({ force: true, preserveInteractive: true });
  }));
  document.querySelectorAll("[data-check-provider]").forEach((button) => button.addEventListener("click", async () => {
    try {
      setBusy(true);
      const result = await request(`/api/model-profiles/${encodeURIComponent(button.dataset.checkProvider)}/check`, { method: "POST", body: "{}" });
      showToast(result.message, !result.ready);
    } catch (error) { showToast(error.message, true); } finally { setBusy(false); }
  }));

  document.querySelectorAll("[data-test-provider]").forEach((button) => button.addEventListener("click", async () => {
    try {
      setBusy(true);
      const result = await request(`/api/model-profiles/${encodeURIComponent(button.dataset.testProvider)}/test`, { method: "POST", body: "{}" });
      showToast(result.message, !result.ready);
    } catch (error) { showToast(error.message, true); } finally { setBusy(false); }
  }));

  document.querySelectorAll("[data-save-provider-key]").forEach((button) => button.addEventListener("click", async () => {
    const input = document.getElementById("provider-api-key");
    const apiKey = input?.value || "";
    if (!apiKey.trim()) return showToast("请输入 API Key", true);
    try {
      setBusy(true);
      await request(`/api/model-profiles/${encodeURIComponent(button.dataset.saveProviderKey)}/credential`, {
        method: "POST",
        body: JSON.stringify({ api_key: apiKey }),
      });
      if (input) input.value = "";
      await loadBootstrap();
      state.editingProvider = state.bootstrap.model_profiles.find((profile) => profile.profile_ref === button.dataset.saveProviderKey) || null;
      render({ force: true, preserveInteractive: true });
      showToast("API Key 已保存到本机服务端");
    } catch (error) { showToast(error.message, true); } finally { setBusy(false); }
  }));
};

const applyTheme = (theme) => {
  document.documentElement.dataset.theme = theme;
  const label = document.getElementById("theme-label");
  if (label) label.textContent = theme === "dark" ? "切换浅色" : "切换深色";
};

document.getElementById("theme-toggle").addEventListener("click", () => {
  const next = document.documentElement.dataset.theme === "dark" ? "light" : "dark";
  localStorage.setItem("workbench-theme", next);
  applyTheme(next);
});

document.getElementById("refresh-button").addEventListener("click", async () => {
  try { await loadBootstrap(); if (state.view === "runs" && state.selectedRunId) await loadRun(state.selectedRunId); showToast("已刷新"); }
  catch (error) { showToast(error.message, true); }
});

applyTheme(localStorage.getItem("workbench-theme") || "light");
loadBootstrap().catch((error) => {
  content.innerHTML = `<div class="empty-state"><h3>本机服务暂不可用</h3><p>${html(error.message)}</p></div>`;
});
