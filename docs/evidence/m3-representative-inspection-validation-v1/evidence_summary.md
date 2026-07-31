# Evidence summary

Machine evidence is complete (8 executed / 8 passed). The immutable machine
manifest and case records retain the `pending` / `not_yet_claimed` state captured
when they were generated. The independent `owner_review.json` is approved, and
the verifier computes the committed package as `m3_v_eligible=true`. M3 is
therefore `I/T/V / DONE`, and M4 is `READY`.

The M3 `V` is limited to representative runtime validation of trajectory
integrity, deterministic audit, failure attribution, and read-only inspection.
It does not claim scientific performance, M4 benchmark attribution accuracy, or
experimental validation.

| Case | Source class | Machine status |
|---|---|---|
| `single_round_success` | `representative_local_runtime` | `passed` |
| `multi_round_success` | `representative_local_runtime` | `passed` |
| `known_hosts_propagation` | `representative_fault_injection` | `passed` |
| `history_truncation` | `representative_fault_injection` | `passed` |
| `duplicate_dispatch` | `representative_fault_injection` | `passed` |
| `stale_state` | `representative_fault_injection` | `passed` |
| `multiple_equal_first_cause_candidates` | `representative_fault_injection` | `passed` |
| `causal_link_not_proven` | `representative_fault_injection` | `passed` |

All eight cases execute through production Session/source construction, PR-BD, PR-BF, PR-BG, and two fresh-process calls to the project-scoped PR-BH GET route. PR #12 authoritative receipts and typed failure evidence are summarized without private paths or infrastructure values.
