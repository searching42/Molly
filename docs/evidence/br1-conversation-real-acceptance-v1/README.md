# BR1 conversation-driven real acceptance v1

This directory is the privacy-safe checked-in summary for the BR1 real
conversation acceptance. It preserves the earlier fail-closed preparation
attempts while making the completed v11 runtime result the current record.

## Current v11 result

The fresh project, conversation, and run were driven from the natural-language
conversation front door. The server resolved and auto-bound the sole eligible
owner-approved input bundle. No browser call to the input-binding or remote-resource-authority operator APIs was used.

The run completed the reviewed BR1 chain:

```text
conversation goal
→ plan with evaluation top_n=3
→ exact owner-approved input binding
→ dataset confirmation Gate
→ Uni-Mol training
→ REINVENT4 generation
→ current-model prediction
→ final deterministic evaluation
→ verified Computational Top-3
→ scientific_result.available
```

The final task was
`evaluate_private_structured_dataset_canary_v1`, with output contract
`computational-top-n-v1`. The server-generated projection is:

```text
projection_id: result-054adbf946ce441dabf477d222965031
projection_digest: sha256:53ea970ceb890a39217151e406d268834fa7740787811591690b26a24fc7cae7
verified candidates: 6
requested Top-N: 3
returned Top-N: 3
scores: 0.766931, 0.452630, 0.283465
```

These are model-predicted values in a verified deterministic Computational
Top-N, not experimental measurements. The projection is bound to the verified
model, validation, ranking, and evidence publications and does not guarantee
synthesis, experimental performance, or applicability outside the verified
scope.

The result was emitted as one durable `scientific_result.available` event. The
projection was replayed and produced the same digest. The checked-in summary
contains only logical identifiers, digests, aggregate counts, resource
envelopes, provider versions, and verification flags. It contains no private
rows, SMILES, raw files, host information, commands, credentials, model
weights, or raw worker output.

## Restart and replay evidence

An isolated real canary covered the required remote continuation boundary. The
control-plane process was restarted while the Uni-Mol training request was
running. The worker completed while the control plane was down. After restart,
the same Controller execution and remote request were restored; the control
plane performed `refresh_remote_task` and `adopt_remote_outputs`, both with
`dispatch_occurred=false`. The dispatch count stayed at one, the publication
was verified, and the run continued to model packaging and the next Gate.

This canary proves restart continuation and exactly-once remote adoption for a
real training stage. It intentionally did not rerun the full scientific chain;
the terminal v11 run supplies the full result evidence.

## Deployment and preflight

The v11 runtime used reviewed control-plane commit
`15a4365fb6597a231cbff5b083af23ad8b783c32` and the recorded clean worker
implementation digest. The authoritative 1999-row applicability preflight was
`PASS`: 1999 supported, 0 unsupported, 0 unresolved. It completed in 824.0
seconds with no swap and without training, generation, prediction, or
Controller side effects.

The resolved remote profiles were:

```text
reinvent4-br1-v2       0 GPU / 1 CPU / 21600 seconds
unimol-predict-br1-v1  1 GPU / 8 CPU / 43200 seconds
unimol-train-br1-v2    1 GPU / 8 CPU / 86400 seconds
```

The exact owner was `searching42`; the recorded decision was
`ACCEPT_EXACT_PROPOSAL` for the v11 proposal. The final result projection is
server-generated and does not allow the LLM to invent scientific content.

## Historical fail-closed attempts

Earlier evidence remains part of the history rather than being overwritten:

- The initial natural-language attempt stopped at `BR1_INPUT_BUNDLE_REQUIRED`
  before proposal, Controller, or remote dispatch.
- The follow-up deployment/preflight attempt stopped on a missing provider
  capability asset and did not create acceptance authority.
- The complete authoritative preflight later passed for all 1999 rows.
- The first freeze implementation was corrected to derive live canonical
  identities independently from the frozen Raw bytes; forged source/provider
  digest tests fail closed.
- A new freeze/proposal and exact owner approval then enabled the v11 run.

CI status is external to this immutable acceptance snapshot. The authoritative
current status is recorded on PR #43 and GitHub Actions; this snapshot does not
copy a pending/success CI state, and no `full-ci` pull-request label is required.
