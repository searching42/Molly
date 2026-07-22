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
The second generation round was not dispatched because the first cumulative
evaluation formed the complete Top-4.

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
candidate import, cumulative prediction, explainable mixed-source Top-N, and a
durable terminal session result. The next execution acceptance step is the
same SessionSpec contract with the node45 remote REINVENT4 transport.

