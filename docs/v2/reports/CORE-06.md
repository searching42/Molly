# CORE-06 — BR1 parity macro

Status: CORE-06A, CORE-06B, CORE-06C, B2, B3, PR Fast, CodeQL, and Full CI
PASS. B4 remains an explicit Owner gate; CORE-07 has not started. This single
macro branch contains CORE-06A, CORE-06B, and CORE-06C.

## Base and checkpoints

- Macro base: merged Scientific Intake `main` at
  `d533d35751ebd6c88befaf63f540da9b10dbaedc`.
- CORE-06A: optional BR1 scientific plugin, offline/local contract tests;
  report `docs/v2/reports/CORE-06A.md`.
- CORE-06B: durable local/remote compute seam with JobHandle idempotency,
  restart inspection, and verified collection; report
  `docs/v2/reports/CORE-06B.md`.
- CORE-06C: fresh-real BR1 plus remote restart canary; report
  `docs/v2/reports/CORE-06C.md`.

The final fresh-real acceptance executable/test commit is
`6f2340e91ae45224a3a19b6e6c9dcd19343c07a2`.

## Macro status

| Gate | Status | Evidence |
| --- | --- | --- |
| CORE-06A | PASS | `docs/v2/reports/CORE-06A.md` and focused plugin tests |
| CORE-06B | PASS | `docs/v2/reports/CORE-06B.md` and durable backend tests |
| B2 fresh-real BR1 | PASS | `docs/v2/evidence/core-06/CORE06C_BR1_ACCEPTANCE.json` |
| B3 restart canary | PASS | Same evidence; three JobHandles inspected/collected |
| B4 cutover approval | PENDING_OWNER_APPROVAL | Explicit future Owner gate |

The exact B2/B3 dataset, stage artifacts, RunLedger occurrences, config
digests, lineage checks, and remote JobHandle evidence are in the public-safe
CORE-06C evidence manifest. The scientific claim boundary remains
`COMPUTATIONAL_ONLY`.

## Final verification status

The final executable/test HEAD is
`74bb3cfad82c4dc5d65faf7d76ce45b3e0d9e3c0`.

| Check | Result | Evidence |
| --- | --- | --- |
| PR Fast CI | PASS | workflow `33375489761`; compile/diff `99435924446`, pytest `99435969468` |
| CodeQL | PASS | workflow `33375487331`; analyses `99435919198`, `99435919069`, `99435919174`; check `99436052699` |
| Full CI compile/shard policy | PASS | run `33375992342`; job `99437484872` |
| Full CI weighted shard 0 | PASS | job `99437526721` |
| Full CI weighted shard 1 | PASS | job `99437526742` |
| Full CI weighted shard 2 | PASS | job `99437526826` |
| Full CI weighted shard 3 | PASS | job `99437526768` |

The report-only update after this tested HEAD contains no executable, test, or
acceptance-evidence changes. The CI evidence remains bound to the exact
executable/test HEAD above.

## Safety boundary

`B4` remains pending and `core_cutover_ready = false`. CORE-06 does not start
CORE-07, does not migrate the v1 authority machinery, and does not claim
experimental validation or numerical reproduction of the historical v1 run.
