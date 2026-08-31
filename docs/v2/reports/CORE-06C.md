# CORE-06C — fresh-real BR1 parity and restart canary

Status: PASS for the executed fresh-real acceptance. This report records
actual remote execution evidence; it does not authorize CORE-07 or CORE-08.

## Acceptance

Acceptance ID: `core06-br1-v2-real-20260831`

The bounded dataset was derived from the exact reviewed real BR1 source and
stored as `MIGRATED_ACCEPTED_REAL_DATASET`. The migration recorded the source
content digest and deterministic transformation digest without recreating a
CORE-05 ReviewRecord. The bounded v2 dataset contains 1,024 rows.

The production BR1 plugin path executed this chain:

```text
migrated accepted-real dataset
→ applicability preflight
→ fresh Uni-Mol training
→ current-run model artifact
→ fresh REINVENT4 generation
→ current-run candidate artifact
→ prediction using the current-run model
→ deterministic evaluation
→ computational Top-N
```

The acceptance used Uni-Mol Tools `0.1.5`, REINVENT4 `4.7.15`, Uni-Mol
`unimolv1`/`84m`, and server-owned seeds/configuration. Exact artifact IDs,
occurrence IDs, config digests, lineage checks, and public-safe JobHandle
evidence are frozen in:

`docs/v2/evidence/core-06/CORE06C_BR1_ACCEPTANCE.json`

The executable/test commit recorded by the acceptance is
`6f2340e91ae45224a3a19b6e6c9dcd19343c07a2`. The evidence file SHA-256 is
`63868bbdd0a4499139208e0732f21fc62ab340623a3d743524e04c032ce97d8e`.

## B2 — fresh-real BR1

`PASS`.

The run performed fresh training and fresh generation. Prediction consumed the
model artifact produced by the same current run, and deterministic evaluation
consumed the current-run candidate and prediction artifacts. No historical
model, candidate, prediction, or Top-N artifact was used as a scientific
result. The final claim boundary is `COMPUTATIONAL_ONLY`.

## B3 — remote restart/inspect/collect canary

`PASS`.

Each of the training, generation, and prediction JobHandles was reconstructed
in a new backend object after the run, inspected read-only, and collected
against its durable output manifest. Profile, task, idempotency, input, and
execution-config bindings matched; the canary recorded zero duplicate
dispatches after restart.

The canary also distinguishes raw compute-manifest outputs from final
AgentLoop artifacts whose reports receive current-run binding fields. This is
why the evidence records both collected manifest outputs and final event
outputs without conflating the two authorities.

## B4 and cutover

`B4 = PENDING_OWNER_APPROVAL`.

`core_cutover_ready = false`. CORE-06 does not authorize CORE-08. CORE-07 has
not started.

## Limitations

The dataset is a bounded deterministic migration/import of the accepted real
v1 dataset lineage, not a claim that the new CORE-05 review workflow reviewed
the historical source. The acceptance proves the computational BR1 contract
and current-run provenance, not experimental validation, numerical identity
with the historical v1 run, or general scientific performance.

Final PR Fast, CodeQL, and Full CI remain macro merge-candidate checks to be
recorded after the final executable/test HEAD is pushed.
