# Paper018 node221 remote REINVENT4 session canary — 2026-07-23

## Scope

This canary exercised the complete bounded discovery session through the PR-AW
HTTP control plane with a real remote PR-AS generation on `node221`. The run
used the `workstation1-node221-reinvent4-v1` transport profile, then locally
exact-replayed the remote publication through PR-AT controlled prediction,
PR-ARb v2 Top-N selection, and PR-AU termination.

The result is recommendation-only. It does not claim experimental or
computational validation, mutate the material Registry, or perform human
candidate adjudication.

## Runtime provenance and exact inputs

The ignored runtime tree is retained locally under
`runs/_pr_ay_paper018_remote_reinvent4_canary_20260723/`. The table preserves
the audit boundary if that tree is later removed.

| Input | Exact identity | SHA-256 / digest |
|---|---|---|
| Molly code | commit `fe2328c` on top of merged PR #391 | commit identity |
| SessionSpec | `oled-bounded-session-b28a718ae2c048dac2688e9d597f0f4e3a4acbf29016f594e8d8f9a0744f61ec` | `381a18565a327c82905884e385178585fbc85190c71c29b86900325fa5f128b0` |
| paper016 PR-AO execution | `oled-real-phase1-execution:e161bfedf82a661fef3254e70cdeb9bb21521704348da7fd1020480348fedfd0` | receipt `cb6169ee27d4931f9a222512c0efe5585dc8f4d3fc5899025a5e9c04170941b2`; directory manifest `1b28805feb1d7167d3d5dc07c0fa7a92d2973b6af066c8609d13c35bb6b5ad56` |
| paper016 dataset snapshot | `oled-categorical-dataset-snapshot:97616f119d74cd4c375dcf47e70807b3c90af606ce46f8170174b60f5d9e23fe` | `fc1ddbad38dd94e014b8ffd6cd64a6f3cd44cc7f6be96202868bfbd4af9ef8bb` |
| material Registry snapshot | `molly-material-registry`, version `successor-3cfef8ab0cd6202a08237863` | `17b78ea89066f5e41234d00ed291de3e2562d54de4a0bb74753986dd10f57dab` |
| REINVENT4 template | sampling, CPU, two requested SMILES | `52dba1049ba393e71e6feb92014d785a62c6fe15aa7fc930c867ba5a475f8159` |
| pinned known-hosts file | `workstation1` / `node221` Ed25519 binding | `e8b4ef61457460748443d991e524134dfef76554fd5ca79d53646b918a7e1cda` |

The node221 deployment used REINVENT 4.7.15 from source commit
`5e67f40eedbb4c617710f8156b1db585cd789770`, Python environment
`/home/lbh/miniconda3/envs/reinvent4`, prior SHA-256
`b6513ec6dbc54c87ea45cdbf9b4aaefadd7652548b74175366b27f12ec5732fe`,
and pip-freeze SHA-256
`2ead2e82155c694ec2629ce43b0cfba08db0cb75e73a9483f0262784ba71c862`.

## Non-disruptive transport execution

Immediately before approval, node221 reported load averages
`0.00, 0.02, 0.05`, no compute applications, 0% GPU utilization, and
32,069 MiB free GPU memory. The transport nevertheless used the profile's
`cpu_only` and `nice_19_single_thread` policies, so the canary did not allocate
GPU memory.

The exact generation identities were:

| Item | Identity / SHA-256 |
|---|---|
| design request | `oled-inverse-design-request:00fd06f0453450dc841617203aceec99d4ee74cdd859a45be452b1a6dcbd4ac3` |
| immutable publication | `oled-inverse-design-publication:61dd0f55bbd41520674c85834dd2ca676fb68bd233906b697e9105d954c65ee4` |
| rendered config | `ce9b13f189fe96dab7be11ed3e4df7e9c90c03f8441598da1441d7c5fc523e77` |
| raw remote CSV | `9e972c194c13ca0551b25957c0ce6b49e718fd85f7f83f7a83fa3718cdf76447` |
| transport provenance | `d4ca95ba487f19e82ccce66e65ad3242e704b2909ba7d41ce1209b9efd1aed92` |
| accepted / excluded | 2 / 0 |

Strict pinned-host verification succeeded and the receipt records
`endpoint_hostname_verified=true`, `expected_hostname=node221`, and an
invocation-isolated remote attempt directory. No automatic remote cleanup or
process manipulation was performed.

## Provenance propagation defect found and closed

The first real remote attempt successfully published PR-AS but the following
PR-AT child failed closed. Diagnosis showed that PR-AV passed the pinned
known-hosts anchor to generation only; downstream children therefore could not
exact-replay the remote PR-AS publication. The failed session remains immutable
and is not counted as acceptance evidence.

Commit `fe2328c` propagates the same bound known-hosts artifact through every
round child. A remote-mode integration test now executes a fake transport and
requires evaluation to succeed with the external anchor present. The accepted
canary used a new project and reran the full path after this fix.

## Durable result

The project
`paper018-node221-remote-reinvent4-canary-provenance-fixed-20260723` reached
revision 10 with status `COMPLETED_TOP_N`. A new process then inspected and
exact-replayed the terminal session successfully.

| Artifact | Identity / SHA-256 |
|---|---|
| session result | `oled-bounded-session-result:d0e6853cf27132a704e70de00950f6cb8e661f949527672e7133265ed0f39127`; file `f109f40d3f4b4819371f482864c7586af9a29a862fb26db71a2b32742fd38822` |
| terminal session state | `bad0b6c680c83220c7d5b6e2c467cefedb729de1b3412b20806a12e69ae0c820` |
| PR-AT evaluation | `oled-generated-evaluation:ed00cf56a2ccb16407bc6138fc35df1175010a724321bbe1759b1c573d02dfdf`; receipt `0a0c9fea746a4009e5b532f9862111c073a2c973e6374b814e4439bde3e90d6b` |
| PR-ARb v2 decision | `oled-candidate-decision:68c565ecc710330ec76ed463b63a1a618dc934f517ccdcf4368e57b029c3ea10`; receipt `3d9a5e1bf3e9ffefa6a15faa3793077f4e95282aecb5e9157a9e6408cf2b84c0` |
| Top-N CSV | `8c14cc095000d6d7c4e2b1bf7bc93c2487994c3461f1bb2b381890017213ee3a` |
| PR-AU controller | `oled-bounded-controller:8262e5f9348b1914f481b1c6e99f0164595fae3ac7922099b7e0070b7698a3c0`; receipt `8bff8dd2b27188d21f3b081de35cf9575d3cd15417893ba59f31b32715dc73bd` |
| terminal summary | `a145ce3514051c21e2e25a646d040cbe0f1732126e0b19b4f3d3246f97e1268f` |

PR-AT produced one single-round Registry-plus-generated evaluation with four
complete predictions: two Registry candidates and two generated candidates.
This is PR-AT v1, not a multi-publication cumulative PR-ATb evaluation.

## Explainable Top-4

| Order | Source | Chemical identity | Predicted S1 (eV) | Predicted delta E_ST (eV) |
|---:|---|---|---:|---:|
| 1 | generated | `LRFINQRWLQDDMQ-UHFFFAOYSA-N` | 3.254813 | 0.497593 |
| 2 | Registry / TDBA | `RTUFWKHMCQMJJH-UHFFFAOYSA-N` | 3.248169 | 0.495916 |
| 3 | generated | `MINRLIGJRHIXHD-UHFFFAOYSA-N` | 3.226405 | 0.486798 |
| 4 | Registry / mTDBA-Ph | `XQCPKMMDIGKHNJ-UHFFFAOYSA-N` | 3.222349 | 0.478826 |

All four passed the requested `s1_ev >= 0.0 eV` and
`delta_e_st_ev <= 1.0 eV` prediction constraints and were selected by the
rank-anchored greedy max-min Tanimoto policy. The session used one generation
round, two generated candidates, and one controller iteration.

## Acceptance conclusion

The node221 remote transport and the complete single-round bounded discovery
session are accepted as an executable canary. The persisted claims remain
`experimental_validation_claimed=false`,
`computational_validation_claimed=false`,
`human_candidate_adjudication_performed=false`, and
`registry_mutated=false`.
