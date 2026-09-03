"use strict";

const state = {
  view: "new",
  bootstrap: null,
  detail: null,
  selectedRunId: null,
  editingProvider: null,
  files: [],
  loading: false,
  pollTimer: null,
  pollInFlight: false,
  resumeInFlight: false,
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

const loadBootstrap = async () => {
  state.bootstrap = await request("/api/bootstrap");
  if (state.selectedRunId && !state.bootstrap.runs.some((run) => run.run_id === state.selectedRunId)) {
    state.selectedRunId = null;
    state.detail = null;
  }
  render();
};

const loadRun = async (runId) => {
  state.selectedRunId = runId;
  state.detail = await request(`/api/runs/${encodeURIComponent(runId)}`);
  state.view = "runs";
  render();
  startRunPolling();
};

const stopRunPolling = () => {
  if (state.pollTimer !== null) {
    window.clearInterval(state.pollTimer);
    state.pollTimer = null;
  }
};

const terminalStatuses = new Set(["STOPPED", "FAILED", "BUDGET_EXHAUSTED"]);

const pollSelectedRun = async () => {
  if (!state.selectedRunId || state.pollInFlight) return;
  state.pollInFlight = true;
  try {
    const detail = await request(`/api/runs/${encodeURIComponent(state.selectedRunId)}`);
    state.detail = detail;
    if (state.view === "runs") render();
    if (terminalStatuses.has(detail.status)) {
      stopRunPolling();
      return;
    }
    if (detail.status === "ACTIVE" && detail.workflow === "br1" && !detail.pending_call && !detail.background_pending && !detail.background_error_type && !state.resumeInFlight) {
      state.resumeInFlight = true;
      try {
        const result = await request(`/api/runs/${encodeURIComponent(detail.run_id)}/resume`, { method: "POST", body: "{}" });
        state.detail = result.inspection || result;
        if (state.view === "runs") render();
      } finally {
        state.resumeInFlight = false;
      }
      return;
    }
    if (detail.status === "WAITING_APPROVAL" && !detail.background_pending) {
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

const uploadFiles = async (fileList) => {
  const files = Array.from(fileList || []);
  for (const file of files) {
    if (file.size > 128 * 1024 * 1024) {
      throw new Error(`${file.name} 超过 128 MB，暂不能上传`);
    }
    const encoded = await readFileAsBase64(file);
    const uploaded = await request("/api/artifacts", {
      method: "POST",
      body: JSON.stringify({
        file_name: file.name,
        media_type: file.type || "application/octet-stream",
        content_base64: encoded,
      }),
    });
    state.files.push(uploaded);
  }
};

const renderFiles = () => {
  if (!state.files.length) return '<div class="empty-files">还没有添加文件（可选）</div>';
  return state.files.map((file) => `
    <div class="file-row">
      <span class="file-name">${html(file.name)}</span>
      <span class="file-id">${html(file.artifact_id.slice(0, 20))} · ${formatBytes(file.size_bytes)}</span>
    </div>
  `).join("");
};

const renderNew = () => {
  const profiles = (state.bootstrap?.runtime_profiles || []).filter((profile) => profile.available);
  const profileOptions = profiles.length
    ? profiles.map((profile) => {
      const constraints = profile.resource_constraints || {};
      const resource = profile.workflow === "br1"
        ? ` · CPU ${html(constraints.cpu_threads ?? "—")} · GPU ${html(constraints.gpu_count ?? "—")}`
        : "";
      return `<option value="${html(profile.profile_id)}">${html(profile.name)}${resource}</option>`;
    }).join("")
    : '<option value="">暂无可用运行配置</option>';
  const noProfile = !profiles.length;
  const modelProfiles = state.bootstrap?.model_profiles || [];
  const modelOptions = `<option value="">请选择解析模型</option>${modelProfiles.map((profile) => `<option value="${html(profile.profile_ref)}">${html(profile.name)} · ${html(profile.model_identifier)}</option>`).join("")}`;
  const br1 = profiles.find((profile) => profile.workflow === "br1");
  const br1Hint = br1
    ? "BR1 会按服务器端登记的资源配置，在需要确认的阶段暂停。"
    : "运行配置由本机服务器端登记。";
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
          <textarea id="goal" name="goal" required placeholder="例如：以 HOMO-LUMO gap 为目标，不限制骨架，采样空间 1000，筛选较小的分子，输出 top 5"></textarea>
        </div>
        <div class="field-row">
          <div class="field">
            <label class="field-label" for="profile-id">运行配置</label>
            <select id="profile-id" name="profile_id" ${noProfile ? "disabled" : ""}>${profileOptions}</select>
          </div>
          <div class="field">
            <label class="field-label">开始方式</label>
            <div class="notice mode-note">${html(br1Hint)}</div>
          </div>
        </div>
        <div class="field-row">
          <div class="field">
            <label class="field-label" for="llm-profile">自然语言解析模型服务</label>
            <select id="llm-profile">${modelOptions}</select>
            <span class="field-hint">任务目标必须经过已配置的结构化 LLM 解析；API Key 只由本机服务端读取。</span>
          </div>
          <div class="field">
            <label class="field-label">资源选择</label>
            <div class="notice mode-note">CPU/GPU、工作站和路径均来自运行配置，网页不能改写。</div>
          </div>
        </div>
        <div class="upload-card">
          <div class="upload-head">
            <div>
              <div class="upload-title">数据文件 <span class="optional-label">（可选）</span></div>
              <div class="upload-subtitle">文件会保存到本机的不可变数据区，单个文件不超过 128 MB。</div>
            </div>
            <label class="file-button">添加文件<input id="file-input" type="file" multiple /></label>
          </div>
          <div class="file-list" id="file-list">${renderFiles()}</div>
        </div>
        <div class="form-actions">
          <button class="primary-button" type="submit" ${noProfile ? "disabled" : ""}>开始任务</button>
          <span class="field-hint">开始后仍可在需要时确认关键操作。</span>
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
      ${call.result_data ? `<div class="timeline-result"><pre>${pretty(call.result_data)}</pre></div>` : ""}
      <details class="call-technical"><summary>查看技术标识</summary><code>${html(call.tool_name)}@${html(call.tool_version)}</code></details>
    </div>
  `).join("")}</div>`;
};

const renderPendingAction = (detail) => {
  const pending = detail.pending_call;
  if (!pending) return "";
  return `
    <div class="action-card">
      <h3>这一步需要你的确认</h3>
      <p class="action-copy">系统已经准备好执行下面的操作。确认后才会真正执行；长时间计算会在后台运行，页面会自动刷新。</p>
      <div class="operation-name">系统操作：${html(operationLabel(pending.tool_name))}</div>
      <div class="operation-args"><pre>${pretty(pending.arguments)}</pre></div>
      <details class="call-technical"><summary>查看技术标识</summary><code>${html(pending.tool_name)}@${html(pending.tool_version)}</code></details>
      <div class="action-buttons">
        <button class="primary-button" data-approval="APPROVED" type="button">确认并执行</button>
        <button class="danger-button" data-approval="REJECTED" type="button">拒绝这一步</button>
      </div>
    </div>
  `;
};

const renderRunDetail = () => {
  const detail = state.detail;
  if (!detail) return '<div class="empty-state"><h3>选择一个任务</h3><p>从左侧选择任务记录，查看执行过程。</p></div>';
  const canContinue = ["ACTIVE", "INTERRUPTED"].includes(detail.status);
  const finalFiles = (detail.final_artifact_ids || []).map((id) => `<div class="artifact-download"><code>${html(id)}</code><a href="/api/artifacts/${encodeURIComponent(id)}/content">下载文件</a></div>`).join("");
  const observability = state.bootstrap?.observability || {};
  const monitorButtons = [
    ["json", "下载 JSON 追踪", true],
    ["otel", "导出 OpenTelemetry", observability.otel?.available],
    ["langsmith", "导出 LangSmith", observability.langsmith?.available],
  ].map(([name, label, available]) => `<button class="secondary-button" data-observe="${name}" type="button" ${available ? "" : "disabled"}>${label}</button>`).join("");
  const failureNotice = (detail.failure_summary || []).length
    ? `<div class="notice failure-notice mb-20"><div class="notice-icon" aria-hidden="true">!</div><div><strong>有步骤执行失败</strong><pre>${pretty(detail.failure_summary)}</pre></div></div>`
    : "";
  const backgroundNotice = detail.background_pending
    ? '<div class="notice mb-20"><div class="notice-icon" aria-hidden="true">↻</div><div><strong>后台执行中</strong>远程工作站正在处理当前阶段，页面会自动刷新。</div></div>'
    : detail.background_error_type
      ? `<div class="notice failure-notice mb-20"><div class="notice-icon" aria-hidden="true">!</div><div><strong>后台执行异常</strong>错误类型：<code>${html(detail.background_error_type)}</code>。请查看操作记录。</div></div>`
      : "";
  return `
    <div class="card run-detail">
      <div class="run-detail-head">
        <div class="run-detail-title">
          <div><div class="run-goal">${html(detail.goal)}</div><div class="run-id">任务编号：${html(detail.run_id)}</div></div>
          <span class="status-pill ${statusClass(detail.status)}">${html(detail.status_label)}</span>
        </div>
        <div class="run-metrics">
          <div class="metric"><span class="metric-value">${html(detail.step_count)}</span><span class="metric-label">步骤</span></div>
          <div class="metric"><span class="metric-value">${html(detail.tool_call_count)}</span><span class="metric-label">系统操作</span></div>
          <div class="metric"><span class="metric-value">${html(detail.artifact_count)}</span><span class="metric-label">结果文件</span></div>
        </div>
      </div>
      <div class="run-detail-body">
        ${backgroundNotice}
        ${failureNotice}
        ${renderPendingAction(detail)}
        ${canContinue ? '<div class="notice mb-20"><div class="notice-icon" aria-hidden="true">↻</div><div><strong>任务还可以继续</strong>确认或恢复后，点击下面的按钮继续下一步。</div></div><button class="secondary-button mb-22" data-resume="true" type="button">继续任务</button>' : ""}
        <div class="timeline-title">操作记录</div>
        ${renderTimeline(detail)}
        <div class="monitor-actions"><span class="field-hint">只读监控：</span>${monitorButtons}</div>
        <details class="technical">
          <summary>技术详情</summary>
          <div class="technical-body">
            <div class="technical-row"><span>运行配置</span><code>${html(detail.runtime_profile_ref || "—")}</code></div>
            <div class="technical-row"><span>任务校验码</span><code>${html(detail.request_digest)}</code></div>
            <div class="technical-row"><span>权限校验码</span><code>${html(detail.policy_digest)}</code></div>
            <div class="technical-row"><span>结果文件编号</span><div>${finalFiles || "—"}</div></div>
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
      <div class="provider-actions"><button class="secondary-button" data-edit-provider="${html(profile.profile_ref)}" type="button">编辑</button><button class="secondary-button" data-check-provider="${html(profile.profile_ref)}" type="button">检查配置</button></div>
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
        ${editing ? `<div class="provider-key-note"><strong>API Key</strong><span>当前状态：${html(editing.credential_status)}</span><span class="field-hint">输入内容只发送到本机服务端，保存后不会回显、写入任务记录或保存到浏览器。</span><div class="provider-key-form"><input id="provider-api-key" type="password" autocomplete="new-password" placeholder="输入 API Key" /><button class="secondary-button" data-save-provider-key="${html(editing.profile_ref)}" type="button">保存 API Key</button></div></div>` : '<div class="provider-key-note"><strong>API Key</strong><span>先保存服务，再在这里输入 API Key。</span></div>'}
        <div class="form-actions"><button class="primary-button" type="submit">保存服务</button>${editing ? '<button class="secondary-button" data-clear-provider="true" type="button">取消编辑</button>' : ""}</div>
        ${editing ? '<div class="provider-check" id="provider-check"></div>' : ""}
      </form>
    </div>
  `;
};

const render = () => {
  if (!state.bootstrap) return;
  const labels = { new: "新建任务", runs: "任务记录", providers: "模型与服务" };
  topbarContext.textContent = labels[state.view] || "科学任务工作台";
  document.querySelectorAll(".nav-item").forEach((item) => item.classList.toggle("is-active", item.dataset.view === state.view));
  if (state.view === "new") content.innerHTML = renderNew();
  if (state.view === "runs") content.innerHTML = renderRuns();
  if (state.view === "providers") content.innerHTML = renderProviders();
  bindPageEvents();
};

const bindPageEvents = () => {
  document.querySelectorAll("[data-view]").forEach((button) => button.addEventListener("click", () => {
    state.view = button.dataset.view;
    if (state.view !== "runs") state.detail = null;
    if (state.view !== "providers") state.editingProvider = null;
    render();
  }));

  const fileInput = document.getElementById("file-input");
  if (fileInput) fileInput.addEventListener("change", async () => {
    try {
      setBusy(true);
      await uploadFiles(fileInput.files);
      showToast("文件已保存到本机");
      render();
    } catch (error) {
      showToast(error.message, true);
    } finally {
      setBusy(false);
    }
  });

  const taskForm = document.getElementById("new-task-form");
  if (taskForm) taskForm.addEventListener("submit", async (event) => {
    event.preventDefault();
    const goal = document.getElementById("goal").value.trim();
    const profileId = document.getElementById("profile-id").value;
    const llmProfileRef = document.getElementById("llm-profile")?.value || "";
    if (!goal || !profileId || !llmProfileRef) return showToast("请填写任务目标并选择运行配置和解析模型", true);
    try {
      setBusy(true);
      const result = await request("/api/runs", {
        method: "POST",
          body: JSON.stringify({
          goal,
          profile_id: profileId,
          input_artifact_ids: state.files.map((file) => file.artifact_id),
          llm_profile_ref: llmProfileRef,
        }),
      });
      state.selectedRunId = result.run_id;
      state.detail = result.inspection;
      state.files = [];
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
    if (!state.detail?.pending_call) return;
    try {
      setBusy(true);
      const result = await request(`/api/runs/${encodeURIComponent(state.detail.run_id)}/approval`, {
        method: "POST",
        body: JSON.stringify({ decision: button.dataset.approval, reviewer_ref: "local-user", call_id: state.detail.pending_call.call_id }),
      });
      state.detail = result.result?.inspection || result.inspection || await request(`/api/runs/${encodeURIComponent(state.detail.run_id)}`);
      await loadBootstrap();
      state.view = "runs";
      render();
      startRunPolling();
      showToast(button.dataset.approval === "APPROVED" ? "已确认这一步" : "已拒绝这一步");
    } catch (error) { showToast(error.message, true); } finally { setBusy(false); }
  }));

  const resume = document.querySelector("[data-resume]");
  if (resume) resume.addEventListener("click", async () => {
    try {
      setBusy(true);
      const result = await request(`/api/runs/${encodeURIComponent(state.detail.run_id)}/resume`, { method: "POST", body: "{}" });
      state.detail = result.inspection || result;
      await loadBootstrap();
      render();
      showToast(result.status_label || "任务状态已更新");
      startRunPolling();
    } catch (error) { showToast(error.message, true); } finally { setBusy(false); }
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
      render();
      showToast("模型服务已保存");
    } catch (error) { showToast(error.message, true); } finally { setBusy(false); }
  });

  document.querySelectorAll("[data-edit-provider]").forEach((button) => button.addEventListener("click", () => {
    state.editingProvider = state.bootstrap.model_profiles.find((profile) => profile.profile_ref === button.dataset.editProvider) || null;
    render();
  }));
  document.querySelectorAll("[data-clear-provider]").forEach((button) => button.addEventListener("click", () => {
    state.editingProvider = null;
    render();
  }));
  document.querySelectorAll("[data-check-provider]").forEach((button) => button.addEventListener("click", async () => {
    try {
      setBusy(true);
      const result = await request(`/api/model-profiles/${encodeURIComponent(button.dataset.checkProvider)}/check`, { method: "POST", body: "{}" });
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
      render();
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
