# OLED Discovery Dry-Run Packet

## Purpose

`OLEDDiscoveryExecutionPreviewAgent` resolves an action handoff into read-only atomic task, adapter, risk, gate, and approval-mode metadata. `OLEDDiscoveryDryRunPacketAgent` packages that preview into the final review artifact before any future controlled dry-run or executor bridge.

This PR does not add execution. The packet explains what would be considered later, which payload template and snapshot material should be reviewed, and why the current artifact remains non-executable.

## Preview Versus Packet

The execution preview answers whether a selected tool maps to an atomic task and which policy metadata applies.

The dry-run packet answers what a reviewer should inspect before a future dry-run bridge is allowed to consider the step. It records:

- would-run tool, atomic task, and adapter intent
- approval mode and dry-run mode
- input, missing, and output artifact references
- required gates and permissions
- execution preconditions
- payload template
- dry-run snapshot material
- review checklist

## Dry-Run Modes

- `auto_eligible_preview`: the preview is low-risk, has no gates, has no missing inputs, and auto eligibility is allowed.
- `gated_review_packet`: the preview requires gates, has no missing inputs, and gated review packets are allowed.
- `manual_review_packet`: the preview is a manual-review planning case with no missing inputs.
- `blocked`: the preview is blocked, not ready for controlled planning, has missing inputs, requires disallowed gates, has disallowed auto eligibility, or lacks an execution-like atomic task mapping.

`auto_eligible_preview` is not execution. It only means a later controlled dry-run bridge may inspect the packet as an auto-eligible candidate after binding approvals and payloads to a concrete snapshot.

## Snapshot Material

The packet builds deterministic review snapshot material:

- schema version
- run id
- source preview id
- selected tool id
- resolved atomic task id
- resolved adapter name
- approval mode and dry-run mode
- risk level
- required gates
- input and missing artifact names
- payload template

The packet does not hash files, inspect artifact paths, or read artifact contents.

## Review Checklist

Every packet includes reviewer checks:

- confirm selected tool/task mapping
- confirm payload placeholders are correct
- confirm required artifacts are available
- confirm missing inputs are resolved
- confirm required gates are understood
- confirm no registry/promotion/publication mutation
- confirm no adapter execution in this PR
- confirm future executor will bind approval to execution snapshot

Gated packets add a human gate approval check. Auto-eligible packets add a reminder that auto eligibility applies only to a future dry-run bridge.

## CLI Example

```bash
PYTHONPATH=src python -m ai4s_agent.agents.dry_run_packet \
  --run-id demo \
  --goal "Find OLED emitters with high PLQY" \
  --selected-tool candidate_generation_or_prediction
```

The CLI prints compact JSON only and does not execute tools.

## Safety Boundary

The packet does not execute adapters, call or instantiate `RunPlanExecutor`, approve or resume gates, mutate `stage.json`, mutate `gate_decisions.json`, read or hash artifacts, run model training, run prediction, validate benchmarks, call LLMs, call MinerU, read PDFs/images, use external network access, or mutate registry/promotion/publication/release/global append artifacts.

It prepares future controlled dry-run bridge work while keeping this PR strictly review-only.
