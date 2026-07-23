# Paper018 PR-AW local control-plane canary — 2026-07-23

## Scope

This canary exercised the merged PR-AW control plane over the complete local
paper018 `existing_output` session. It used Flask's real project-scoped HTTP
routes to create, inspect, advance, approve, and poll the bounded session. The
scientific child tasks were therefore reached through PR-AW rather than by
calling PR-AV directly.

The run did not use node45, execute REINVENT4, mutate the material Registry, or
claim experimental or computational validation. Browser rendering and remote
transport remain separate acceptance boundaries.

## Runtime provenance and exact inputs

The ignored runtime tree is retained locally under
`runs/_pr_ax_paper018_control_plane_canary_20260723/`. The exact identities
below are recorded here so cleanup of `runs/` does not erase the evidence
boundary.

| Input | Exact identity | SHA-256 / digest |
|---|---|---|
| Molly code | commit `64704f9fab582dc4014a674df89e1b000c9a7d6e` (merged PR #389) | commit identity |
| SessionSpec | `oled-bounded-session-0e5b27f53f774f0f7245f7520f4ff21a498db4c8781b2243859f5a847a434869` | `0e175bf3d96cfeae899097fd47d606e464f6d449a4f0d4c1c0004ce3780df709` |
| paper016 PR-AO execution | `oled-real-phase1-execution:e161bfedf82a661fef3254e70cdeb9bb21521704348da7fd1020480348fedfd0` | receipt `cb6169ee27d4931f9a222512c0efe5585dc8f4d3fc5899025a5e9c04170941b2`; directory manifest `1b28805feb1d7167d3d5dc07c0fa7a92d2973b6af066c8609d13c35bb6b5ad56` |
| paper016 dataset snapshot | `oled-categorical-dataset-snapshot:97616f119d74cd4c375dcf47e70807b3c90af606ce46f8170174b60f5d9e23fe` | file `fc1ddbad38dd94e014b8ffd6cd64a6f3cd44cc7f6be96202868bfbd4af9ef8bb` |
| material Registry snapshot | `molly-material-registry`, version `successor-3cfef8ab0cd6202a08237863` | file `17b78ea89066f5e41234d00ed291de3e2562d54de4a0bb74753986dd10f57dab` |
| local REINVENT4 template | existing-output-only config | `56446f8f037767c81ffda48d08af4438e4ce9e0bbf2eb857e53d8746db7432a4` |
| round-1 existing output | CBP-1 and CCO-1 | `0c49e10eb4a70dd6a4dd7b53948b524ffef8f6e9e3b6651d559b4c2c848cb9e0` |
| reserved round-2 output | CCO-2; not consumed | `2500bb4af88de93fb0d1391764feacbb2c7eaa73a867310b1bfd904f843b463c` |

The same SessionSpec and exact scientific inputs were used by the earlier
PR-AV canary. Publication-scoped generated IDs and timestamps are expected to
differ between independent projects; the chemical identities, ranking order,
and prediction values are compared below instead of claiming cross-run byte
identity.

## Control-plane trace

The client created project `paper018-control-plane-canary-verified`, then used
only the following route families:

- `POST /api/projects/<project>/oled-bounded-sessions`;
- `GET /api/projects/<project>/oled-bounded-sessions/<session>`;
- `POST .../actions/advance` and `POST .../actions/approve`;
- `GET /api/projects/<project>/oled-bounded-session-actions/<action>` until
  each asynchronous action reached a terminal state.

Ten revision-CAS actions succeeded: seven `advance` actions and three gate
approvals. The immutable revision sequence was:

```text
0 ACTIVE/screening
1 WAITING_USER/screening
2 ACTIVE/initial_decision
3 WAITING_USER/initial_decision
4 ACTIVE/generation
5 WAITING_USER/generation
6 ACTIVE/evaluation
7 ACTIVE/candidate_decision
8 ACTIVE/controller
9 ACTIVE/controller
10 COMPLETED_TOP_N/controller
```

After revision 1, the Flask application and PR-AW service were recreated. The
new process exact-replayed the same waiting session before approval. Every
action directory contained exactly the immutable `request.json` and mutable
`action.json`; all ten were complete. The canonical manifest over the ten
action records has SHA-256
`ac6cffafe0358eb74e2ac27d59030d5973b8f2a5a1e06966996d53bd6addc251`.

The dedicated UI route `/oled-bounded-sessions` returned HTTP 200. Its 12,231
response bytes had SHA-256
`46cb32363523fe583f8640ddd305ffc6088d9558ddc04143ec45019fdc65e222`
and contained the create, advance, gate-approval, and terminal Top-N controls.

## Durable result

The session reached revision 10 with result ID
`oled-bounded-session-result:52988a391aa73f91d695a2bd9a071c5b38bcae89daf447ba362cb7b589c5704b`.
It used one generation round, two generated candidates, and one controller
iteration.

| Artifact | SHA-256 |
|---|---|
| local canary summary | `50118eb5fcb78e4744392ba9f0a9e8896f951f2888b03db2fa581b7f7045cc38` |
| `session_state.json` | `da0df3778a17ef615dc8d610b6cd0981cdcc826c4946258c34f16eb9daf571cc` |
| `session_result.json` | `9f049906b1792fb5cfd853976bf69f445aea5df03607ad9cb6a426d2bfb0b953` |
| final candidate-decision receipt | `27852ad2f0abf5e3b7a5ac33a710086684ef17c843d38636a3b1928d7f4ffde1` |
| `top_candidates.csv` | `fb06cd4736f4edcf49be0ac712ff14af573d770a8831b73d1b6abb88719912ef` |
| candidate dossier | `576d924c9607b4dd713f595a55a1fbf431cd53b81ed1deab8b84c48f79bb8876` |
| final report | `60c44be87b7cd8aa2dc4c4345383a4681b1bf84300bbd7a4f0bed05d12e627f0` |
| execution record | `582077c3de289eb3ddc3fe7aaedd74eed96bd5c15ed0a2b661fc56c7c02e237e` |

## Explainable Top-4

| Order | Source | Chemical identity | Predicted S1 (eV) | Predicted delta E_ST (eV) | Reason |
|---:|---|---|---:|---:|---|
| 1 | generated / paper018 CCO-1 | `AHESUVKREFCROS-UHFFFAOYSA-N` | 3.254842 | 0.497579 | rank-anchored greedy max-min Tanimoto |
| 2 | Registry, mTDBA-Ph | `XQCPKMMDIGKHNJ-UHFFFAOYSA-N` | 3.222349 | 0.478826 | rank-anchored greedy max-min Tanimoto |
| 3 | generated / paper018 CBP-1 | `AWNQKZDWLDGQQN-UHFFFAOYSA-N` | 3.232934 | 0.488533 | rank-anchored greedy max-min Tanimoto |
| 4 | Registry, TDBA | `RTUFWKHMCQMJJH-UHFFFAOYSA-N` | 3.248169 | 0.495916 | rank-anchored greedy max-min Tanimoto |

This is the same chemical Top-4, order, and prediction values as the direct
PR-AV canary. All four passed the requested `s1_ev >= 0.0 eV` and
`delta_e_st_ev <= 1.0 eV` constraints.

## Acceptance conclusion

PR-AW's local control-plane boundary is accepted for the bounded
`existing_output` path: asynchronous actions, revision-CAS transitions,
restart inspection, exact gate approvals, durable terminal replay, and
explainable Top-N display all closed without bypassing the HTTP control plane.

The result remains a recommendation. The persisted claims are
`experimental_validation_claimed=false`,
`computational_validation_claimed=false`,
`human_candidate_adjudication_performed=false`, and `registry_mutated=false`.
The next runtime acceptance action remains a non-disruptive repeat with
node45's remote REINVENT4 transport when sufficient GPU capacity is available.
