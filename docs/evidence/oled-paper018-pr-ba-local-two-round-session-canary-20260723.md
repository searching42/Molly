# Paper018 PR-BA local two-round Session canary

## Claim boundary

This evidence validates the already-implemented bounded two-round Molly workflow through the PR-AW project control plane. It does not claim experimental or computational validation, Registry mutation, Gold publication, dataset mutation, model registration, or remote REINVENT4 transport.

The run used deterministic local `existing_output` inputs so that the scientific outcome was frozen before execution. The accepted outcome was only:

```text
Round 1: Registry 2 + generated 1 -> incomplete Top-4
Round 2: Registry 2 + generated 2 -> complete Top-4
PR-AU: stop / target_top_n_complete
usage: iterations=2, generation_rounds=2, generated_candidates=2
```

## Runtime provenance

| Field | Exact value |
|---|---|
| Molly commit used for the canary | `86f554c7510d5c92b7f8cb91cfcb90094d27632f` |
| Scientific implementation baseline | merged PR #392, `3eb548240a014acb8a9168aa36021a3bcc1c10cc` |
| Session ID | `oled-bounded-session-935a932d7d6cc1ffa995eef0d5e8e3a878df63d3caae76832d19902e46aed32f` |
| SessionSpec SHA-256 | `fa8896ef7828e555f3969a46d9ad47e6dd8c078f8c940ce273b2263727f623b1` |
| PR-AO execution ID | `oled-real-phase1-execution:e161bfedf82a661fef3254e70cdeb9bb21521704348da7fd1020480348fedfd0` |
| PR-AO receipt SHA-256 | `cb6169ee27d4931f9a222512c0efe5585dc8f4d3fc5899025a5e9c04170941b2` |
| PR-AO directory manifest SHA-256 | `1b28805feb1d7167d3d5dc07c0fa7a92d2973b6af066c8609d13c35bb6b5ad56` |
| Dataset snapshot ID | `oled-categorical-dataset-snapshot:97616f119d74cd4c375dcf47e70807b3c90af606ce46f8170174b60f5d9e23fe` |
| Dataset snapshot SHA-256 | `fc1ddbad38dd94e014b8ffd6cd64a6f3cd44cc7f6be96202868bfbd4af9ef8bb` |
| Registry identity/version | `molly-material-registry` / `successor-3cfef8ab0cd6202a08237863` |
| Registry snapshot SHA-256 | `17b78ea89066f5e41234d00ed291de3e2562d54de4a0bb74753986dd10f57dab` |
| `s1_ev` model SHA-256 | `7bfb3291e7a082d635a2b7357108f38ba6172d996c7a4c73e649714f2ed4440c` |
| `delta_e_st_ev` model SHA-256 | `f9eb984cc0de640927b21392821d58cf28c7bb5e357ac2bb37a47647bc03227f` |
| `t1_ev` model SHA-256 | `bafeeb407c092e1afb4bc2ff62357414c6a361abe92c6384f9b9e786c1ccd7e1` |
| Local REINVENT4 template SHA-256 | `56446f8f037767c81ffda48d08af4438e4ce9e0bbf2eb857e53d8746db7432a4` |
| Round-1 raw CSV SHA-256 | `79520a32b697522062cba3744dbe5b3b982d01886c6c1678eb11edddbf8c3d45` |
| Round-2 raw CSV SHA-256 | `d13eebacc5240941645751adc0e738d6d23ff2de5b2e650baf14c5029edfe014` |

The ignored runtime directory retains the exact local artifacts. Absolute local paths are intentionally excluded from this durable evidence.

## Control-plane and restart procedure

The complete workflow was first exercised through the visible PR-AW web UI. A second canary used the same project-level PR-AW action API and exact inputs, but split execution across two independent Python processes:

1. Process 1 advanced and approved the Session through revision 10, then exited while the second generation child was `WAITING_USER`.
2. Process 2 reconstructed the Session from immutable revisions, approved the exact second-generation gate, and continued to terminal revision 15.
3. Process 3 loaded the terminal Session and exact-replayed its externally bound result without dispatching a child again.

The restart canary contains 15 immutable action request/state pairs. Every action finished `SUCCEEDED`; there is exactly one approval at expected revision 10. The sorted request/state manifest SHA-256 is `8deacfc8f28bfdb46d66e867a46bdf183d5d836840281615f52254dbe781948e`.

## Round 1: bounded shortfall

| Field | Exact result |
|---|---|
| Candidate decision ID | `oled-candidate-decision:caf0189d4ea5e0074e7570b0b342f86b79359f54aa873b60805320a017b2e28b` |
| Receipt SHA-256 | `036e54b5490d2e8b9300cbb3e03ad1b7cc7571a9e6ec4aea50ad6d3bfbb4ab85` |
| Status/counts | `incomplete`, evaluated `3`, selected `0`, target `4` |
| Selected candidates | empty |
| Top-N CSV | header only; SHA-256 `dd3af9bef6d751ad556f1a862123ccb0b66b4d1af2e6a0b842caa1d797089a91` |

No partial recommendation was exposed as final.

PR-AU then produced:

| Field | Exact result |
|---|---|
| Controller ID | `oled-bounded-controller:6b82f077909d231841cdb00ce9ebd8c4f0e145ea4982b9e80f314fd0b6164704` |
| Controller receipt SHA-256 | `a408b3b0d7a24e0ec18ae1329f549bf065a0f95b307491f5ff9883244753c108` |
| Route | `request_generation_approval` / `property_eligible_candidate_shortfall` |
| Requested count and gate | `1` / `gate_5_final_threshold` |
| Authorization ID | `oled-bounded-generation-authorization:1d3b020dff88327d405596f113963fe03739ad7fae425adfe7237246526c2a53` |
| Authorization SHA-256 | `2ca9e6a56352f7649966fe30538f1a3b1eeef32c533b5213497f96eb200bef57` |
| Loop/state fingerprints | `oled-bounded-loop:3141a4abf7ccba70db47dba9484cd847905493506d7828cd835fc40207d818e6` / `oled-bounded-loop-state:3cd353b58f3242accea2a8018d97da2a6f5277b87741f91e2cee104bb93f556d` |

The second PR-AS publication is not a direct/root publication. Its receipt binds this controller ID, authorization ID, loop/state fingerprints, requested count, target task, and required gate.

## Round 2: cumulative evaluation

| Field | Exact result |
|---|---|
| Root PR-AS publication | `oled-inverse-design-publication:f2bdff43889aefdff89793891aa1bfaf6052537538af3704261c93e3ec5f95a4` |
| Authorized PR-AS publication | `oled-inverse-design-publication:964fe410049b10a821c7e57571488624ee561c29ce55c7e3f9aea8f094453a1c` |
| Ordered roster SHA-256 | `f8ccf04e1f14886014aa6de5965fd332fc1a86561ca0e02bdea13b8624876b18` |
| PR-ATb evaluation ID | `oled-generated-evaluation:bd90fa150d1f4690da5dad7cff9b300d3835c51e1b506ab1e08edd962379142f` |
| PR-ATb receipt SHA-256 | `d17c4c2e855bde6b832896d2a4586d2a62717e9334ce431febd56de5ed3a386d` |
| Version/status | `oled_generated_candidate_evaluation.v2` / `completed` |
| Counts | Registry `2`, generated source `2`, generated predicted `2`, excluded `0`, complete pool `4`, shortlist `4` |
| Complete predictions SHA-256 | `d42e8eb55617ea8a8d50b07354c2e6eaeb38b39cc1779e4aebe0ee0df76966ac` |
| Ranked shortlist SHA-256 | `0f12a4a4996b18b011c0ee32385cd164c0f170efe83adb73480575260e59e33d` |

PR-ATb cumulatively exact-replayed both ordered publications, applied chemical identity exclusion across canonical isomeric SMILES, Standard InChI, and InChIKey, and globally recalculated constraints, predictions, percentiles, Pareto status, and ranking for the Registry-plus-generated pool.

## Terminal result

| Field | Exact result |
|---|---|
| Final decision ID | `oled-candidate-decision:cdf8393d902917b43ed10aeb54cb4e52e83ea4124cf35ea339070bccb83b56e2` |
| Final decision receipt SHA-256 | `58b5cc9f790a6975c7402ea644079e6b4396867c2c12d815f4bce8722b78d19c` |
| Final Top-N CSV SHA-256 | `845237c64ecdee52f8cb17a7d9e956e99235833e33bc2b278a117829dc9901b4` |
| Final controller ID | `oled-bounded-controller:1de7309cc1a0418a573890b0e5df6fbfea2629c50409a54b42c75ff3673d692e` |
| Session result ID | `oled-bounded-session-result:e71415b12f59d87bf2af5426245e9bf9274b0d347f225effcf3af837c73f253d` |
| Session result SHA-256 | `71f3dd583b5bed205643db97fa423b0e44af1aff7afdd6d3a53c401cf8024ffb` |
| Terminal revision/state SHA-256 | `15` / `92acac11747894805878def1b397605e4b039feb1b7c3da19b78633c64d1789a` |
| Terminal outcome | `COMPLETED_TOP_N`, `target_top_n_complete` |
| Cumulative usage | iterations `2`, generation rounds `2`, generated candidates `2` |

Final selection order:

| Order | Source | Name/identity | `s1_ev` | `delta_e_st_ev` | Reason |
|---:|---|---|---:|---:|---|
| 1 | generated | `AHESUVKREFCROS-UHFFFAOYSA-N` | 3.2548418025 | 0.4975790988 | `selected_by_rank_anchored_greedy_max_min_tanimoto` |
| 2 | Registry | `mTDBA-Ph` | 3.2223489933 | 0.4788255034 | same |
| 3 | generated | `AWNQKZDWLDGQQN-UHFFFAOYSA-N` | 3.2329338447 | 0.4885330777 | same |
| 4 | Registry | `TDBA` | 3.2481687440 | 0.4959156280 | same |

All persisted claims remain recommendation-only: no experimental validation, computational validation, Registry mutation, Gold/dataset write, or model registration is claimed.

## Verification

```text
PYTHONPATH=src:. .venv/bin/pytest \
  tests/test_oled_bounded_discovery_session.py \
  tests/test_oled_bounded_discovery_session_api.py -q

25 passed in 47.63s
```

A focused cumulative-round test also passed independently:

```text
tests/test_oled_bounded_discovery_session.py::test_second_round_consumes_controller_grant_and_cumulative_roster
1 passed in 18.48s
```

## Acceptance

PR-BA satisfies the frozen M1 local runtime acceptance path. It validates a real two-round bounded Session, exact second-round controller authorization, PR-ATb cumulative roster replay, process restart at the second generation gate, terminal replay in a fresh process, and the unique predeclared Top-4 outcome. It does not close the separate PR-BB fault-injection tasks for post-child/pre-revision reconciliation or post-registration restart.
