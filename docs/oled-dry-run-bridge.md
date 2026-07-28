# OLED Discovery Dry-Run Bridge Request

## Purpose

`OLEDDiscoveryDryRunPacketAgent` creates a review-only packet that describes what a future dry-run bridge would need to inspect. `OLEDDiscoveryDryRunBridgeRequestAgent` converts that packet into the last review artifact before any real controlled dry-run bridge is introduced.

This PR does not add execution. The bridge request records placeholder adapter invocation intent, bridge mode, reviewer confirmation requirements, snapshot-binding requirements, blockers, and audit notes while keeping `would_execute=false` and `executable=false`.

## Packet Versus Bridge Request

The dry-run packet captures reviewed planning material: task/adapter intent, payload template, dry-run mode, snapshot material, preconditions, and checklist.

The bridge request reshapes that material into what a future bridge would receive:

- selected tool id
- resolved atomic task id
- resolved adapter name
- placeholder adapter invocation
- payload template
- required gates and permissions
- missing inputs and blocked reasons
- snapshot-binding requirements
- reviewer confirmations

## Bridge Modes

- `auto_eligible_bridge_request`: packet is `auto_eligible_preview`, has no gates or missing inputs, auto eligibility is allowed, and reviewer confirmation is not required.
- `gated_bridge_request`: packet is `gated_review_packet`, has gates, has no missing inputs, gated requests are allowed, and reviewer confirmation is not required.
- `manual_bridge_request`: packet is `manual_review_packet`, has no missing inputs or blockers, and reviewer confirmation is not required.
- `blocked`: packet is blocked, not ready, has missing inputs, has blockers, lacks task/adapter mapping, has disallowed gates or auto eligibility, or reviewer confirmation is required.

Eligibility is not execution. `eligible_for_bridge=true` only means a future controlled bridge may review the request. It does not approve gates, call adapters, or mutate state.

## Reviewer Confirmation

The default is conservative. With `require_confirmed_reviewer=True`, bridge requests include `reviewer_confirmation_required` and remain blocked. Synthetic tests and future review flows may pass `require_confirmed_reviewer=False` to show what an eligible request would look like after reviewer confirmation is supplied.

## Adapter Invocation

The adapter invocation is a placeholder:

```json
{
  "schema_version": 1,
  "run_id": "demo",
  "task_id": "retrieve_evidence",
  "adapter": "retrieve_evidence_adapter",
  "payload_template": {
    "run_id": "demo",
    "review_only": true
  },
  "dry_run": true,
  "review_only": true,
  "would_execute": false
}
```

No adapter is imported or called.

## Snapshot Binding Requirements

Every bridge request records requirements a future bridge or executor would need to verify:

- bind bridge request to dry-run packet id
- bind payload template to reviewed snapshot material
- verify artifact paths before future execution
- verify gate approval snapshot before future execution
- verify adapter policy has not changed
- verify no registry/promotion/publication mutation
- verify dry-run bridge uses non-mutating mode

Gated requests add a human gate approval requirement. Auto-eligible requests add a check that auto eligibility still holds at bridge time.

## CLI Example

```bash
PYTHONPATH=src python -m ai4s_agent.agents.dry_run_bridge \
  --run-id demo \
  --goal "Find OLED emitters with high PLQY" \
  --selected-tool retrieve_evidence \
  --no-require-confirmed-reviewer
```

The CLI prints compact JSON only and does not execute tools.

## Safety Boundary

The bridge request does not execute adapters, call or instantiate `RunPlanExecutor`, call `/api/run-plan/resume`, approve or resume gates, mutate `stage.json`, mutate `gate_decisions.json`, read or hash artifact paths, train models, run prediction, validate benchmarks, call LLMs, call MinerU, read PDFs/images, use external network access, or mutate registry/promotion/publication/release/global append artifacts.

It prepares a future real controlled dry-run bridge while keeping this PR strictly review-only.
