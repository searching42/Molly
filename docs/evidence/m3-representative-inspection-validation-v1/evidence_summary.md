# Evidence summary

Machine evidence is incomplete; human review is pending; M3 remains I/T/—.

| Case | Source class | Machine status |
|---|---|---|
| `single_round_success` | `representative_local_runtime` | `passed` |
| `multi_round_success` | `representative_local_runtime` | `passed` |
| `known_hosts_propagation` | `representative_fault_injection` | `not_executed` |
| `history_truncation` | `representative_fault_injection` | `passed` |
| `duplicate_dispatch` | `representative_fault_injection` | `not_executed` |
| `stale_state` | `representative_fault_injection` | `passed` |
| `multiple_equal_first_cause_candidates` | `representative_fault_injection` | `not_executed` |
| `causal_link_not_proven` | `representative_fault_injection` | `not_executed` |

Four source-contract preflights are design-analysis blocked: exact PR-BD production module bytes and inspected symbol digests show that v1 does not persist the required transport, duplicate-dispatch, multi-family, or recovered-failure linkage facts. They are not described as executed runtime validations. The runner does not weaken replay or modify PR-BD–PR-BH to manufacture them.
