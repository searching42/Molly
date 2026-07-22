# Paper018 local existing-output session canary — 2026-07-22

## Scope

This canary exercised the merged PR-AV bounded session from screening through
final Top-N publication. It imported two exact paper018 structures through the
local PR-AS `existing_output` boundary; it did not execute REINVENT4 or claim
that those literature structures were generated de novo.

The property models and immutable dataset/Registry anchors are the previously
validated paper016 execution. Paper018 is therefore the candidate source for
this canary, not the model-training corpus. This run establishes orchestration
and artifact closure, not experimental or computational validation.

## Inputs

The session requested Top-4 with `s1_ev >= 0.0 eV`,
`delta_e_st_ev <= 1.0 eV`, no cost limit, and a maximum pairwise Tanimoto of
`1.0`. The bounded limits remained three iterations, two generation rounds,
and 512 generated candidates.

### Runtime provenance and exact inputs

The ignored runtime paths are intentionally omitted. The identities and hashes
below are the canonical values bound by the persisted SessionSpec and its
upstream receipts, so the run remains auditable after local `runs/` cleanup.

| Input | Exact identity | SHA-256 / digest |
|---|---|---|
| Molly code | commit `7db57d1a0ff56a138289da88131f83ae2b0ac139` (merged PR #386) | commit identity |
| paper016 PR-AO execution | `oled-real-phase1-execution:e161bfedf82a661fef3254e70cdeb9bb21521704348da7fd1020480348fedfd0` | receipt SHA `cb6169ee27d4931f9a222512c0efe5585dc8f4d3fc5899025a5e9c04170941b2`; 8-entry directory manifest `1b28805feb1d7167d3d5dc07c0fa7a92d2973b6af066c8609d13c35bb6b5ad56` |
| paper016 dataset snapshot | `oled-categorical-dataset-snapshot:97616f119d74cd4c375dcf47e70807b3c90af606ce46f8170174b60f5d9e23fe` | artifact digest `8c93641288c6c1bcdcadb1c6ee7b969c71f3a74ec0e56fdccc6a12aba34b893a`; file SHA `fc1ddbad38dd94e014b8ffd6cd64a6f3cd44cc7f6be96202868bfbd4af9ef8bb` |
| material Registry snapshot | `molly-material-registry`, version `successor-3cfef8ab0cd6202a08237863` | snapshot digest `649a391e44b3fc7506ca6a5b633861642e62fef910228415417449f52aa1bcb4`; file SHA `17b78ea89066f5e41234d00ed291de3e2562d54de4a0bb74753986dd10f57dab` |
| `delta_e_st_ev` model | `model__delta_e_st_ev.json` | `f9eb984cc0de640927b21392821d58cf28c7bb5e357ac2bb37a47647bc03227f` |
| `s1_ev` model | `model__s1_ev.json` | `7bfb3291e7a082d635a2b7357108f38ba6172d996c7a4c73e649714f2ed4440c` |
| `t1_ev` model | `model__t1_ev.json` | `bafeeb407c092e1afb4bc2ff62357414c6a361abe92c6384f9b9e786c1ccd7e1` |
| local REINVENT4 template | existing-output-only config, 73 bytes | `56446f8f037767c81ffda48d08af4438e4ce9e0bbf2eb857e53d8746db7432a4` |
| round-1 existing output | CBP-1 and CCO-1, 224 bytes | `0c49e10eb4a70dd6a4dd7b53948b524ffef8f6e9e3b6651d559b4c2c848cb9e0` |
| reserved round-2 existing output | CCO-2, 134 bytes | `2500bb4af88de93fb0d1391764feacbb2c7eaa73a867310b1bfd904f843b463c` |

Round one contained the three-way OPSIN/RDKit/reference exact matches already
reported by the paper018 contextual-alias canary, split as follows:

| Local source ID | InChIKey | Disposition |
|---|---|---|
| `paper018-CBP-1` | `AWNQKZDWLDGQQN-UHFFFAOYSA-N` | round 1, consumed |
| `paper018-CCO-1` | `AHESUVKREFCROS-UHFFFAOYSA-N` | round 1, consumed |
| `paper018-CCO-2` | `NDQAPBNMEZKOSW-UHFFFAOYSA-N` | reserved round 2, not consumed |

CCO-3 was excluded before session creation because its SI systematic-name
heading resolves to CCO-2 while its reviewed structure has a different
InChIKey.

## Durable result

The successful session is
`oled-bounded-session-0e5b27f53f774f0f7245f7520f4ff21a498db4c8781b2243859f5a847a434869`.
It reached `COMPLETED_TOP_N` at immutable revision 10 after one generation
round and one controller iteration. Its six succeeded children were:

1. Registry screening;
2. initial candidate decision;
3. gated local inverse-design import;
4. generated-candidate evaluation;
5. final candidate decision;
6. bounded controller decision.

Three user gates were approved: screening, initial decision, and generation.
The second generation round was not dispatched because the first
single-round Registry-plus-generated evaluation formed the complete Top-4.
This path used PR-AT v1 with one PR-AS publication; it did not create a
generation roster or execute the multi-publication PR-ATb cumulative
successor.

| Artifact | SHA-256 |
|---|---|
| `session_spec.json` | `0e175bf3d96cfeae899097fd47d606e464f6d449a4f0d4c1c0004ce3780df709` |
| `session_state.json` | `14fe8df266093ff341777a75379b6efe7a3d51f96ad16484e2b147498b1cfb19` |
| `session_result.json` | `85c7083ff9258788a661e37357e9f9000b0e8bf3ae99d7ac35729b24c4f98d57` |
| `top_candidates.csv` | `b4fb8171b9771a3130b2d74b73ce3e3e663ac9ff2f80e528cf49cbd3e098f9f3` |
| `candidate_decision_dossier.csv` | `6efaa1cd6712cd9470d95bacddba796863ad2d4bb993a1154fe786587ca6609e` |
| final `report.md` | `9a79a0e0aba7c80b80a95d45fc06f3825911147c463e519c39e5870906664d07` |

The session result ID is
`oled-bounded-session-result:8b13a043aca69dd642cf9f4e62ab628672f5577371e0ff72c59e5157baf56db4`.
An independent terminal inspection exact-replayed the external child facts,
returned the same revision and result, and left the mutable head byte-for-byte
unchanged.

## Explainable Top-4

| Order | Source | Candidate | Predicted S1 (eV) | Predicted ΔE_ST (eV) | Why selected |
|---:|---|---|---:|---:|---|
| 1 | paper018 existing output | CCO-1 | 3.254842 | 0.497579 | rank-anchored greedy max-min Tanimoto |
| 2 | Registry | mTDBA-Ph | 3.222349 | 0.478826 | rank-anchored greedy max-min Tanimoto |
| 3 | paper018 existing output | CBP-1 | 3.232934 | 0.488533 | rank-anchored greedy max-min Tanimoto |
| 4 | Registry | TDBA | 3.248169 | 0.495916 | rank-anchored greedy max-min Tanimoto |

All four candidates passed both requested property constraints. The final
artifact explicitly records `experimental/computational validation claimed =
false`; these values are model predictions, not measured results.

## Acceptance conclusion

The local `existing_output` acceptance step is complete. It demonstrates the
end-to-end path from exact-bound model/dataset/Registry inputs through gated
candidate import, combined-pool prediction, explainable mixed-source Top-N,
and a durable terminal session result. The next execution acceptance step is
the same SessionSpec contract with the node45 remote REINVENT4 transport.
