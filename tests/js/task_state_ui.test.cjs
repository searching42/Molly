"use strict";

const assert = require("node:assert/strict");
const test = require("node:test");
const taskState = require("../../src/ai4s_agent/static/task_state.js");

function payload(overrides = {}) {
  return {
    artifacts: {},
    job: { status: "RUNNING" },
    task_state: {
      projection_version: taskState.PROJECTION_VERSION,
      project_id: "proj-a",
      run_id: "run-a",
      status: "RUNNING",
      current_stage: "train_model",
      history: [{ stage: "inspect_dataset", status: "SUCCEEDED" }],
      artifact_ids: [],
      state_files: ["stage.json"],
      ...overrides,
    },
  };
}

test("missing artifact registry is not inferred from an empty artifacts object", () => {
  const snapshot = taskState.taskStateSnapshot(payload(), { runId: "run-a", projectId: "proj-a" });
  assert.deepEqual(snapshot.stateFiles, ["stage.json"]);
  assert.equal(snapshot.stateFiles.includes("artifact_registry.json"), false);
});

test("only actual server-projected job state filenames are displayed", () => {
  const snapshot = taskState.taskStateSnapshot(
    payload({
      state_files: ["job_state.json", "background_job_state.json", "job.json", "stage.json"],
    }),
    { runId: "run-a", projectId: "proj-a" },
  );
  assert.deepEqual(snapshot.stateFiles, ["background_job_state.json", "job_state.json", "stage.json"]);
});

async function assertSwitchDoesNotWrite({ switchedContext }) {
  const expected = Object.freeze({ projectId: "proj-a", conversationId: "conv-a", generation: 7 });
  let current = expected;
  let resolvePersist;
  let persistedTarget = null;
  const messages = [];
  const appended = [];
  const persisted = new Promise((resolve) => { resolvePersist = resolve; });
  const pending = taskState.persistTaskStateRecord({
    payload: payload(),
    runId: "run-a",
    projectId: "proj-a",
    context: expected,
    getContext: () => current,
    messages,
    persistMessage: async (target) => {
      persistedTarget = target;
      return persisted;
    },
    appendMessage: (...args) => appended.push(args),
  });
  await Promise.resolve();
  current = switchedContext;
  resolvePersist({ content: taskState.taskStateConversationContent(taskState.taskStateSnapshot(payload(), { runId: "run-a", projectId: "proj-a" })) });
  const result = await pending;
  assert.equal(result.status, "stale_after_persist");
  assert.equal(persistedTarget.projectId, "proj-a");
  assert.equal(persistedTarget.conversationId, "conv-a");
  assert.equal(persistedTarget.generation, 7);
  assert.deepEqual(messages, []);
  assert.deepEqual(appended, []);
}

test("project switch during persistence cannot write into the new project UI", async () => {
  await assertSwitchDoesNotWrite({
    switchedContext: { projectId: "proj-b", conversationId: "conv-b", generation: 8 },
  });
});

test("conversation switch during persistence cannot write into the new conversation UI", async () => {
  await assertSwitchDoesNotWrite({
    switchedContext: { projectId: "proj-a", conversationId: "conv-b", generation: 8 },
  });
});

async function assertSwitchBeforeResponseIsRejected(switchedContext) {
  const expected = Object.freeze({ projectId: "proj-a", conversationId: "conv-a", generation: 7 });
  let current = expected;
  let resolveRequest;
  const response = new Promise((resolve) => { resolveRequest = resolve; });
  const pending = response.then((value) => taskState.taskStateResponseSnapshot(value, {
    runId: "run-a",
    projectId: "proj-a",
    context: expected,
    getContext: () => current,
  }));
  current = switchedContext;
  resolveRequest(payload());
  assert.equal(await pending, null);
}

test("project switch before the task-state response rejects the old request", async () => {
  await assertSwitchBeforeResponseIsRejected({ projectId: "proj-b", conversationId: "conv-b", generation: 8 });
});

test("conversation switch before the task-state response rejects the old request", async () => {
  await assertSwitchBeforeResponseIsRejected({ projectId: "proj-a", conversationId: "conv-b", generation: 8 });
});

test("semantic allowlists remove hostname and IP shaped values", () => {
  const snapshot = taskState.taskStateSnapshot(
    payload({
      status: "private.compute.invalid",
      current_stage: "10.0.0.1",
      history: [
        { stage: "internal-node_42", status: "RUNNING" },
        { stage: "train_model", status: "internal-node_42" },
      ],
      artifact_ids: ["model_metadata", "private.compute.invalid", "10.0.0.1"],
      state_files: ["stage.json", "internal-node_42", "job.json"],
    }),
    { runId: "run-a", projectId: "proj-a" },
  );
  assert.equal(snapshot.status, "unavailable");
  assert.equal(snapshot.currentStage, "unavailable");
  assert.deepEqual(snapshot.history, []);
  assert.deepEqual(snapshot.artifactIds, ["model_metadata"]);
  assert.deepEqual(snapshot.stateFiles, ["stage.json"]);
  assert.equal(
    taskState.taskStateSnapshot(payload(), { runId: "private.compute.invalid", projectId: "proj-a" }),
    null,
  );
});

test("persisted task state messages never enter later LLM decision context", () => {
  const taskMessage = taskState.taskStateConversationContent(
    taskState.taskStateSnapshot(payload(), { runId: "run-a", projectId: "proj-a" }),
  );
  const messages = [
    { role: "user", content: "train a model" },
    { role: "system", content: taskMessage },
    { role: "assistant", content: "ready" },
  ];
  assert.deepEqual(taskState.conversationDecisionMessages(messages), [messages[0], messages[2]]);
});
