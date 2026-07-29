# CI simplification baseline and evidence

This report records the pre-change baseline at `e989d16` (`main`) and the
evidence used to scope the functional-equivalence simplification. Marker-only
collection hooks do not alter application behavior; the slow-test hotspot
recheck below was taken before any production or test assertion was edited.

## Baseline commands

| Command | Result |
| --- | --- |
| `python -m pytest -q --durations=100` | 5,998 passed in 1,166.36 s (19m26s) |
| `python -m pytest --collect-only -q` | 5,998 collected in 2.16 s |
| `python -m compileall -q src tests` | passed |
| `git diff --check` | passed |

Additional inventory:

- 378 `test_*.py` files;
- 148,182 Python lines under `tests/` and 226,983 under `src/`;
- two PR CI pytest shards, assigned by alternating lexicographically sorted
  filenames rather than measured duration;
- 308 `create_app(...)` calls across 48 test files;
- no registered semantic pytest markers before this change (only the built-in
  `parametrize` marker was present).

## Existing GitHub PR CI

PR #4's final CI run (`30415743692`) started at 02:02:21Z and completed at
02:21:28Z: 19m07s wall time. Shard 0 spent 18m34s in pytest and shard 1 spent
17m45s. Each shard independently installed Poppler and the editable `dev`
dependencies, compiled all Python, ran `git diff --check`, and uploaded results.
The repeated setup cost was measurable (roughly 20-25 seconds per shard), but
pytest dominated the wall time. The original workflow also ran the same full
two-shard path on pull requests and pushes to `main`.

## Slowest baseline tests

The full baseline run and a focused hotspot recheck identified the same long
tail. The table gives the focused recheck's reproducible node-level timings;
small machine-load differences explain the few-second variation from the full
run (whose slowest node was 44.82 s).

| Rank | Seconds | Test node |
| ---: | ---: | --- |
| 1 | 50.34 | `test_oled_bounded_discovery_session_api.py::test_pr_atb_interrupted_action_is_recovered_through_project_api` |
| 2 | 34.24 | `test_oled_scientific_agent_trajectory_audit_metrics.py::test_multi_round_audit_reports_cumulative_budget_and_gates` |
| 3 | 33.13 | `test_oled_scientific_agent_trajectory_verifier.py::test_external_anchor_verifier_replays_multi_round_projection` |
| 4 | 26.50 | `test_oled_scientific_agent_trajectory_projection.py::test_multi_round_projection_contains_cumulative_children` |
| 5 | 22.25 | `test_oled_bounded_discovery_session.py::test_second_round_cumulative_evaluation_registration_is_adopted_without_redispatch` |
| 6 | 22.22 | `test_oled_bounded_discovery_session.py::test_second_round_generation_success_is_reconciled_before_session_revision` |
| 7 | 21.38 | `test_oled_bounded_discovery_session.py::test_second_round_shortfall_stops_at_pr_au_budget_without_third_generation` |
| 8 | 20.14 | `test_oled_bounded_discovery_session.py::test_second_round_consumes_controller_grant_and_cumulative_roster` |
| 9 | 9.38 | `test_oled_categorical_dataset_execution.py::test_reformatted_admission_and_artifact_tamper_fail_closed` |
| 10 | 9.36 | `test_oled_bounded_discovery_session_api.py::test_second_generation_interrupted_action_is_recovered_without_rewriting_request` |
| 11 | 9.26 | `test_oled_categorical_dataset_execution.py::test_execution_materializes_only_pr_ah_admitted_rows_and_runs_smoke_baseline` |
| 12 | 8.22 | `test_oled_material_registry_successor_postwrite_verifier.py::test_postwrite_verifier_replays_seven_entry_publication` |
| 13 | 8.12 | `test_oled_categorical_dataset_execution.py::test_concurrent_target_created_before_commit_survives_unchanged` |
| 14 | 8.07 | `test_oled_categorical_dataset_execution.py::test_output_parent_replacement_or_symlink_redirect_fails_closed` |
| 15 | 8.03 | `test_oled_categorical_dataset_execution.py::test_material_group_split_produces_holdout_metrics_when_data_supports_it` |
| 16 | 7.96 | `test_oled_real_paper_vertical_run.py::test_complete_decisions_resume_chain_through_dataset_baseline` |
| 17 | 7.66 | `test_oled_categorical_gold_dataset_admission.py::test_admission_binds_complete_snapshot_and_only_publishes_roster` |
| 18 | 7.45 | `test_control_plane_event_projector.py::test_projection_does_not_materialize_second_round_generation_roster` |
| 19 | 7.40 | `test_oled_categorical_gold_dataset_admission.py::test_decision_tamper_fails_after_outer_rehash` |
| 20 | 7.12 | `test_oled_categorical_gold_dataset_admission.py::test_reformatted_input_fails_exact_sha[snapshot]` |

The slowest files are therefore the bounded discovery session/API, scientific
trajectory audit/projection/verifier, categorical dataset execution/admission,
material-registry successor verifier, real-paper vertical run, and control
plane projector suites. These remain in Full and Scheduled CI.

## Duplicate and maintenance audit

Confirmed low-risk duplication:

- 17 read-only index-page contract tests each created a fresh Flask app and
  fetched the same immutable HTML;
- five generic RunPlan acceptance modules each repeated an identical
  `phase3_executor` import-boundary test, while the same modules repeated the
  same AST inspection helper for their own no-network contract;
- the dataset workflow tests repeated the same isolated app/client setup;
- seven private production helpers have exactly one repository token reference
  (their own definition), including an obsolete non-bound SHA-256 reader in
  `dataset_workflow.py` superseded by FD/inode-bound verification.

Potential duplication deliberately left alone:

- canonical JSON and SHA-256 helpers whose exact prefix, return format, or
  publication schema differs by domain;
- fake SSH/SCP/transport fixtures whose protocol assertions differ;
- repeated tampering tests that protect different artifact identities;
- dynamic adapters, routes, schemas, artifact IDs, and historical readers.

## Implemented optimization scope

- register `unit`, `integration`, `acceptance`, `adversarial`, `slow`, and
  `remote_mock`, plus an explicit `pr_fast` canary marker;
- enforce marker spelling and exactly one semantic primary layer during pytest
  collection;
- keep all cheap unit tests in PR Fast and add reviewed dataset, Gate, UI,
  redaction, local-policy, REINVENT4, publication, and remote-lifecycle canaries;
- split static PR checks, PR Fast tests, complete main/manual tests, and weekly
  security replay;
- use measured slow-file weights for deterministic Full CI sharding;
- share immutable index rendering, centralize duplicate AST contract checks,
  and reuse dataset app/client setup without sharing mutable test state;
- remove only private helpers proven to have no code, test, documentation,
  configuration, or string-reflection caller.

## Rejected high-risk optimizations

- `pytest-xdist`: not introduced because the suite mutates environment
  variables and exercises directory replacement, locks, subprocesses, and
  process recovery; proving cross-worker isolation is outside this PR.
- weakening or sampling security checks in Full CI: rejected. Full and weekly
  runs still collect the entire suite.
- consolidating hardened filesystem/publication readers: rejected because small
  semantic differences could alter trusted-byte or historical compatibility
  boundaries.
- deleting dynamic adapters or historical schema/publication readers: rejected
  because static reachability is insufficient evidence of no compatibility
  responsibility.
- large `oled_*.py` moves, naming rewrites, UI changes, storage migrations, and
  executor changes: out of scope.

## Post-change evidence

Final counts, marker timings, PR Fast timing, Full-suite timing, changed-line
totals, and GitHub Actions wall time are recorded here before the Draft PR is
marked ready for review.
