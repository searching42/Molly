# Molly — Long-Horizon AI4S Agent

> **Public development repository.** This repository is the canonical source
> for Molly development from 2026-07-27 onward. The complete pre-migration
> history remains in a private audit archive and is intentionally not mirrored
> here.

Molly is a long-horizon AI4S agent for literature-grounded scientific
discovery. It turns conversational intent and reviewed evidence into controlled,
provenance-preserving workflows that can survive restarts, publish replayable
results, and expose an auditable trajectory without treating UI or telemetry as
scientific authority.

Molly is an execution and evidence system, not a claim that generated materials
are experimentally valid. Recommendations, predictions, computational
validation, and experimental validation remain distinct claim levels.

## Capability overview

- Natural-language intent capture, durable project conversations, immutable
  uploads, and frozen execution requests.
- Literature source planning, controlled intake, PDF parsing, extraction,
  review, and provenance-bound dataset preparation.
- `RunPlanExecutor` execution with immutable snapshots, explicit human gates,
  task policy, artifact registration, and fail-closed resume checks.
- Bounded multi-round discovery sessions with deterministic child runs,
  publication replay, crash recovery, and reconciliation.
- Local and remote scientific execution through logical resource and
  environment profiles kept outside the repository.
- Observer-only control-plane projection and SSE, plus trajectory integrity and
  audit work that cannot advance or approve scientific state.
- Evidence-aware modeling, diagnostics, candidate generation and screening,
  review, and controlled asset promotion.

The public implementation, evidence-maturity, and milestone status of these
areas is kept in the [public roadmap](docs/roadmap.md), not duplicated here.

## Trusted execution boundaries

1. Conversation and planning artifacts are non-executable until an immutable
   execution request and RunPlan are created.
2. A gate approval applies only to the current frozen execution snapshot; it
   cannot authorize changed inputs or future work.
3. Gated scientific adapters run through `RunPlanExecutor`, never through a
   direct-adapter shortcut.
4. Session revisions, artifact registry records, execution records, and bound
   publications establish execution facts. UI state, caches, SSE events, and
   observer projections do not.
5. Recovery may adopt a completed child only after exact publication replay and
   state reconciliation; it must not infer success from mutable telemetry.
6. External literature, LLM, remote compute, and promotion actions require their
   own explicit authorization boundaries.

## Quickstart

Molly requires Python 3.10 or newer.

```bash
python -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -e ".[dev]"

PYTHONPATH=src .venv/bin/python -m flask \
  --app 'ai4s_agent.app:create_app' run --host 127.0.0.1 --port 8792
```

Open `http://127.0.0.1:8792/` or verify the service with:

```bash
curl http://127.0.0.1:8792/healthz
```

Run the test suite with:

```bash
PYTHONPATH=src .venv/bin/python -m pytest -q
```

## Local and remote resources

The default local workspace is the repository checkout; ignored `projects/`
and `runs/` directories hold runtime state. Set `MOLLY_WORKSPACE` only when a
different local workspace is required, and never commit its resolved path.

User-specific configuration belongs in the platform configuration directory or
an explicit `MOLLY_CONFIG_DIR`. Keep real hostnames, SSH aliases, usernames,
known-hosts files, interpreter paths, credentials, private papers, and runtime
outputs there—not in Git.

### Configure server-owned remote resource authority

Remote compute is configured from the trusted server process, never from a
project request. First create and probe the private connection and execution
profiles. Then save one policy entry for each exact logical task/profile pair
through `RemoteResourceAuthorityPolicyStore`. The client continues to send only
the logical profile ID; it cannot submit a host, endpoint, SSH option, path,
credential, resource count, or worker override.

Use the following pattern in a private server bootstrap or one-shot operator
script. Values shown in angle brackets are logical IDs or reviewed ceilings,
not infrastructure locators:

```python
import os
from pathlib import Path

from ai4s_agent.remote_resource_authority import (
    RemoteResourceAuthorityPolicyStore,
)
from ai4s_agent.schemas import (
    AgentConfiguredRemoteResources,
    AgentRemoteResourceBudgetLimits,
    RemoteResourceAuthorityPolicy,
    RemoteResourceAuthorityPolicyEntry,
)

config_dir = Path(os.environ["MOLLY_CONFIG_DIR"])
entries = [
    RemoteResourceAuthorityPolicyEntry(
        policy_id=os.environ["MOLLY_POLICY_ID"],
        enabled=True,
        connection_id=os.environ["MOLLY_LOGICAL_CONNECTION_ID"],
        execution_profile_id=os.environ["MOLLY_EXECUTION_PROFILE_ID"],
        remote_task_type=os.environ["MOLLY_REMOTE_TASK_TYPE"],
        allowed_task_ids=[os.environ["MOLLY_PLANNER_TASK_ID"]],
        configured_resources=AgentConfiguredRemoteResources(
            gpu_count=int(os.environ["MOLLY_POLICY_GPU_COUNT"]),
            cpu_threads=int(os.environ["MOLLY_POLICY_CPU_THREADS"]),
            walltime_sec=int(os.environ["MOLLY_POLICY_WALLTIME_SEC"]),
        ),
        budget_limits=AgentRemoteResourceBudgetLimits(
            max_runtime_sec=int(os.environ["MOLLY_POLICY_MAX_RUNTIME_SEC"]),
            max_gpu_hours=float(os.environ["MOLLY_POLICY_MAX_GPU_HOURS"]),
        ),
    ),
]
RemoteResourceAuthorityPolicyStore(config_dir=config_dir).save(
    RemoteResourceAuthorityPolicy(entries=entries)
)
```

For the BR1 real-tool workflow, create three entries rather than a shared
wildcard entry. Each must bind its own exact task
ID, worker task type, logical connection, execution profile and resource
ceiling:

| Planner task ID | Remote task type | Execution profile ID |
|---|---|---|
| `train_private_unimol_v1` | `model_training` | `unimol-train-br1-v2` |
| `generate_private_reinvent4_v1` | `molecular_generation` | `reinvent4-br1-v2` |
| `predict_private_unimol_v1` | `model_inference` | `unimol-predict-br1-v1` |

The v2 Uni-Mol profile publishes `config.yaml`, `model_0.pth` and
`target_scaler.ss` as one exact prediction-capable model roster; the prediction
profile must consume those current-run artifacts together with the current
candidate CSV. The REINVENT4 v2 profile publishes both the candidate CSV and a
provider/config audit. Before approval, require an enabled exact connection, a current
capability probe, an execution-profile/provider match and a current immutable
AuthoritySet. A missing, disabled, ambiguous, over-ceiling or stale binding is
a denial. Policy configuration does not make a `local_executor` task remote;
the Planner/Registry contract must already declare the remote route and worker
protocol.

The store writes `resource_authority_policies.json` under `MOLLY_CONFIG_DIR`
with private permissions and atomic replacement. Do not hand-edit the file or
commit it. Public evidence may include safe logical IDs, policy/AuthoritySet
digests and outcome classes only. See the full
[server-owned authority contract](docs/server-owned-remote-resource-authority-v1.md).

Private BR1 dataset preparation also uses a server-owned catalog boundary.
Construct `private_structured_dataset_task_registry_v2()` during trusted
private service bootstrap and inject that same registry into Planner,
Permission/authorization, Controller and `RunPlanExecutor`. The default
registry deliberately retains only the frozen CI/synthetic v1 prepare task.
The private registry exposes the required-input
`prepare_private_structured_dataset_canary_v2` task and removes the v1 prepare
node, so omitted provenance inputs cannot downgrade private CSV to synthetic
v1. The exact plan authorization—not a JSON `owner_approved` flag—is owner
authority for the mapping-policy digest.

Formal BR1 real-tool execution uses the additional, versioned
`private_structured_dataset_real_tool_task_registry_v3()` boundary. Inject the
same registry instance into Planner, Permission/authorization, Controller and
`RunPlanExecutor`, or set the trusted server process variable below before
startup:

```bash
AI4S_SCIENTIFIC_TASK_REGISTRY=br1-private-real-tool-v3
```

This setting is server-owned and is not accepted in a project or client JSON
request. It enables only the reviewed v3 task catalog and the three v3 logical
execution profiles; an unknown value fails service startup. The default server
continues to expose the frozen v1 profiles/catalog, while the private v2
preparation-only registry remains available for exact replay.

Private v2 review verification is derivational, not merely structural. Every
prepare, confirmation, publication and recovery boundary re-reads the exact
registered Raw CSV, deterministically rebuilds the complete v2 review with the
same molecule inspector and frozen `created_at`, and requires exact publication
equality. A coherently re-signed replacement of molecule identity, normalized
condition, observed target payload, duplicate/conflict decisions, findings or
action rosters therefore fails closed even when all nested digests agree.

- [Local deployment and private configuration](docs/local_deployment.md)
- [Private remote worker setup](docs/remote_worker_setup.md)
- [Remote execution lifecycle](docs/stage6b-remote-execution-lifecycle.md)

## Documentation

Start with the [documentation map](docs/README.md). Repository-wide public
contracts are separated from local working context:

- [Development guidance](docs/development-guidance.md)
- [Public roadmap and status](docs/roadmap.md)
- [Literature intake](docs/literature-intake.md)
- [Bounded discovery sessions and recovery](docs/oled-bounded-discovery-session.md)
- [Control-plane event projection and SSE](docs/control-plane-event-projection.md)
- [Resume-intent validation](docs/resume-intent-validation-semantics.md)
- [Document parsing providers](docs/document-parsing-providers.md)
- [Security policy](SECURITY.md)

Schemas, sanitized evidence summaries, public examples, and operator runbooks
live under `docs/`. Dated implementation checklists are not maintained as a
second project history.

## Roadmap and status

[`docs/roadmap.md`](docs/roadmap.md) is the normative public source for roadmap
scope, milestone status, priorities, acceptance boundaries, and execution
order. Topic documents explain contracts and operations but must not maintain a
competing current-status table or decision log.

Local `todo.md` and `CLAUDE.md` are intentionally Git-ignored working-context
files. They may contain private scratch planning or machine-specific agent
instructions, but public CI, documentation, and runtime behavior must not depend
on their presence or contents.

References written as `legacy-private PR N` identify authorized records in the
pre-migration private audit archive. They are not links to public pull requests.
Public PR numbering restarted with this repository. Unlinked PR numbers in
dated pre-migration technical evidence have the same legacy-private meaning
unless an explicit public GitHub URL says otherwise.

## Public repository boundary

This repository may contain source code, public documentation, synthetic
fixtures, machine-readable schemas, and reviewed sanitized evidence. It must not
contain credentials, real user or project data, private papers, unpublished
full-text material, runtime bundles, private Git history, personal operating
instructions, or concrete infrastructure identities and paths.

Before publishing evidence, replace infrastructure locators with logical IDs,
retain content digests only when they are safe and useful, and verify the claim
boundary. See [SECURITY.md](SECURITY.md) for reporting and privacy guidance.
