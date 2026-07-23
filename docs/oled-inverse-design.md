# PR-AS — OLED inverse-design execution

PR-AS is the bounded generation stage in the OLED execution loop. It does not
claim that a generated structure has any target property, and it does not
write the Registry, Gold records, dataset views, models, or experimental state.

It may run only after an exact replay of PR-ARb proves all of the following:

- the batch is `not_ready`;
- the candidate shortage is a real pre-Pareto property-supply shortage;
- `inverse_design_should_trigger` is true; and
- the required return route is `inverse_design → controlled_prediction → filter
  → rank → candidate_decision_dossier`.

Pareto-only, budget-only, diversity-only, and ready-batch shortfalls are not
generation authorization.

## Inputs and execution boundary

The gated `execute_oled_inverse_design` task consumes the PR-ARb receipt, the
exact PR-AP screening publication, PR-AO execution, PR-AI dataset snapshot,
Registry snapshot, and a REINVENT4 configuration artifact. A candidate-cost
manifest is required only when the PR-ARb receipt bound one. `existing_output`
also requires a supplied raw REINVENT4 CSV; it imports and normalizes evidence
but does not claim that this invocation ran a generator.

Before gate approval, the executor creates run-local immutable copies of every
input. At resume, it replays the route and rechecks the source-to-frozen binding
before dispatching the adapter. The adapter receives only frozen paths.

When PR-AU routes the task, the frozen input roster additionally includes the
exact controller request, controller receipt, generation authorization, and
controller report. The authorization's requested candidate count replaces the
ordinary PR-ARb shortfall only for that routed invocation. PR-AS fails closed
if normalization produces more independent accepted candidates than that grant,
and executor registration exact-replays the same frozen controller bundle
against the publication receipt.

`remote` mode is restricted to the checked-in
`workstation2-node45-reinvent4-v2` and
`workstation1-node221-reinvent4-v1` transport profiles. Both profiles require an
executor-frozen `oled_inverse_design_remote_known_hosts` artifact and use
`StrictHostKeyChecking=yes`. Before it creates a remote work directory, its
strict SSH session verifies the configured remote short hostname (`node45` or
`node221`).
Arbitrary SSH hosts, repositories, Python paths, and conda environments are
not task options. Both executable profiles are deliberately CPU-only and use
the same fixed repository and environment paths. Each launches one
low-priority process with `nice -n 19` and fixes the common BLAS/OpenMP thread
counts to one, so the canary cannot allocate GPU memory or contend with the
existing GPU workload. The legacy v1 profile remains verifier-only for
previously persisted receipts and cannot be selected for a new run. Its
effective configuration is replayed with the original v1 placeholder and
active `molly_design_request_sha256` assignment rules; v2's sampling, CPU, and
`json_out_config` requirements are not retroactively applied to v1 artifacts.

The remote configuration template must contain all four placeholders:

```text
{{molly_output_csv}}
{{molly_design_request_id}}
{{molly_seed}}
{{molly_design_request_sha256}}
```

The template must be valid REINVENT4 sampling TOML with `device = "cpu"`.
Because REINVENT4 rejects unknown top-level fields, the exact request identity
is bound through the REINVENT-consumed `json_out_config` path, for example:

```toml
run_type = "sampling"
device = "cpu"
seed = {{molly_seed}}
json_out_config = "{{molly_output_csv}}.{{molly_design_request_id}}.{{molly_design_request_sha256}}.json"

[parameters]
model_file = "priors/reinvent.prior"
output_file = "{{molly_output_csv}}"
num_smiles = 2
unique_molecules = true
randomize_smiles = true
```

This binds the exact PR-ARb properties, directions, and bounds to the submitted
configuration. It does not claim that PR-AS can independently interpret an
opaque user-supplied REINVENT scoring implementation; PR-AT remains the sole
controlled-prediction/filter/rank authority.

It allocates a one-shot remote `/tmp/molly-pr-as-…` directory before copying the
configuration. The remote output and config must both remain inside that
attempt-owned directory.

PR-AS uses two distinct identities. `design_request_id` is fixed before
execution and binds the exact PR-ARb route, configuration template, transport
contract, seed, and requested count; it is the identity embedded in the remote
configuration and attempt namespace. `publication_id` is derived only after
the generator returns and additionally binds the raw generator-output SHA-256,
effective-configuration SHA-256, and canonical transport-provenance SHA-256.
The immutable output-directory basename is the `publication_id`.

## Publication and replay

Each successful invocation creates a no-replace versioned directory containing:

- `inverse_design.json`
- `reinvent4_config_template.toml`
- `reinvent4_effective_config.toml`
- `raw_generator_output.csv`
- `generated_candidates.csv`
- `excluded_candidates.jsonl`
- `report.md`

Every raw CSV row is independently parsed and structure-normalized. PR-AS
excludes invalid structures, duplicate generated identities, and overlap with
training or Registry canonical SMILES, Standard InChI, or InChIKey identities.
It assigns `oled-generated:` IDs; it never assigns Registry material IDs.

`verify_oled_inverse_design_publication_from_files()` is the mandatory
downstream replay anchor. It exact-replays PR-ARb and the upstream PR-AP/PR-AO/
PR-AI/Registry chain, reconstructs identity filtering from the raw CSV, renders
the effective REINVENT4 config again, rederives both identities (including the
raw-output-bound publication identity), and compares every persisted byte. It
uses a pinned no-symlink directory descriptor while reading the publication;
the RunPlan executor keeps that descriptor pinned through a single atomic
artifact-registry update and accepts adapter paths only for its four fixed
registered publication files (receipt, candidates, exclusions, and report).
For a remote publication, the verifier also requires the same pinned
known-hosts input so its host-key fingerprint remains bound to the replay.
For a controller-authorized publication, it also requires the exact four-file
PR-AU bundle; a receipt's self-declared controller fields are not a sufficient
replay anchor.

The only permitted next execution step is PR-AT controlled prediction,
filtering, and ranking of the generated candidates.
