# BR1 conversation-driven real acceptance v1

This directory records a privacy-safe blocked acceptance attempt. It is not
evidence of a completed real BR1 run and it makes no scientific-result claim.

The attempt used the exact natural-language front door on control-plane
commit `ce1f4b6e0c849e6037dd3dc42944dccde7284557` with the server-owned
`br1-private-real-tool-v3` registry. A fresh project, conversation, and
conversation run were created. The user goal was submitted as conversation
content, and the first `agent-session/turn` was allowed to resolve BR1 inputs
server-side. No browser or client call bound an input bundle, created a plan
proposal, published remote resource authority, approved execution, or
dispatched a worker.

The server returned `needs_input` with
`BR1_INPUT_BUNDLE_REQUIRED`: there was no eligible owner-approved BR1 input
bundle. The owner-approval boundary therefore stopped the attempt before a
Controller execution existed. This is an expected fail-closed result, not a
runtime success.

The deployment preflight also found an independent resource-policy mismatch:
the configured GPU REINVENT4 policy names `reinvent4-br1-gpu-v1`, which is not
an allowlisted control-plane execution profile. The deployed worker was
observed separately as a dirty working tree at commit
`bf82cfd75d23abfe17328f362bd76b79b134b138`; its actual implementation digest
was recorded, but it is not treated as a clean reviewed deployment.

No prior Desktop artifacts, synthetic outputs, or historical run outputs were
used. No Uni-Mol training, REINVENT4 generation, prediction, final evaluation,
restart continuation, result projection, or `scientific_result.available`
event was produced. The required next attempt needs a fresh clean worker
deployment, an exact owner-approved bundle bound to the current deployment
identity, and a corrected server-owned GPU generation profile/policy pair.

All files here contain logical identifiers, aggregate status, digests, and
privacy-safe metadata only. They contain no private rows, SMILES, raw files,
hostnames, accounts, commands, credentials, model weights, or raw worker
output.

## Follow-up v2 status

The same Draft PR records a follow-up preparation attempt under
`br1-real-acceptance-20260809-v2`. The policy-bound worker was replaced with a
clean packaged runtime whose implementation digest matches reviewed control
plane commit `ce1f4b6e0c849e6037dd3dc42944dccde7284557`. Its public capability
probe reported CPU, GPU, Uni-Mol, and REINVENT4 availability with provider
versions `unimol-tools 0.1.5` and `REINVENT4 4.7.15`.

The isolated acceptance configuration now resolves the allowlisted
`reinvent4-br1-v2` CPU-only profile (`0 GPU`, `1 CPU`, `21600 seconds`) and
uniquely resolves the BR1 training, inference, and generation profiles. A
fresh deployment-bound source authority chain was materialized against the
new worker digest and reviewed profile.

The applicability preflight still failed closed because the configured
`unimol-tools 0.1.5` environment has no `mol.dict.txt` provider asset. The
result was `BLOCKED` for all 1999 rows with provider capability/adapter
availability reason codes. No freeze package, owner proposal, exact owner
approval, Controller execution, or remote dispatch was created. The
REINVENT4 template was independently verified to contain the required output
and seed placeholders, but it was not admitted into a bundle.

This follow-up therefore remains deployment-blocked, not runtime-success
evidence. The original v1 attempt remains the historical front-door
fail-closed record; these v2 files document only the subsequent deployment,
policy, authority, and preflight checks.

After the initial v2 preflight, the missing public Uni-Mol dictionary asset was
repaired from the reviewed `dptech/Uni-Mol-Models` source and its deployed
91-byte digest was verified. Provider adapter discovery and bounded
one-, 100-, and 500-row preprocessing checks all returned supported results.
The complete 1999-row applicability preflight was then retried, but it did not
write a new summary within the bounded eight-minute operator window and was
stopped before any acceptance side effect. The prior blocked summary is not
used as the result of this retry. The current gate is therefore full
applicability-preflight completion; freeze, owner approval, Controller, remote
dispatch, and scientific result projection remain unstarted.

## Full preflight and freeze follow-up v4

The existing authoritative full path was then allowed to run without manual
termination. It completed `PASS` after 898.3 seconds for all 1999 rows:
1999 supported, 0 unsupported, 0 unresolved, and no reason codes. During the
run, observed CPU utilization was 69.1--140.5%, RSS was 1039.5--1221.1 MiB,
swap remained 0, the provider subprocess remained alive during processing and
ended with the preflight, and the preflight exited 0. No training, generation, prediction,
Controller, or owner-approval side effect occurred.

The first live freeze attempt exposed a real readiness blocker: the CLI
default compared the fresh report against stale historical raw/source
identities. The readiness code now derives live stable identities from the
already verified report and raw bytes when no explicit historical identity is
provided; explicit identity mismatch checks remain strict. Focused readiness
and conversation-bridge tests pass.

The corrected CLI produced a private freeze package and owner proposal with
status `FROZEN_WAITING_OWNER`. The REINVENT4 template has not yet been added,
no exact owner approval exists, no eligible bundle has been assembled or
counted, and no new project/conversation/run has started. The next permitted
step is exact owner approval, followed by one-template bundle assembly and a
fresh natural-language front-door run.

## Independent live identity follow-up v5

Review of the v4 implementation found that its default live identity path
still copied the report's observed canonical source/provider digests into both
sides of the comparison. That detected missing fields but did not independently
bind those digests to the current Raw bytes. The old v4 package and proposal
remain pre-fix readiness evidence and were not approved.

The freeze implementation now uses the shared Raw CSV parser and the existing
canonical source/provider serializers to derive `input_row_count`, the raw
digest, the canonical source digest, and the canonical provider-input digest
from the exact stable Raw bytes read for the freeze. The report's corresponding
observed identities remain the `actual` side of the comparison. Explicit
`expected_stable_identities` callers, including historical/trusted mode, retain
their strict comparison behavior.

Two adversarial tests now forge and re-sign a source canonical digest and a
coherently re-signed provider canonical digest. Both are rejected by the
default live freeze path. The existing 1999-row PASS report was not rerun; it
remains bound by report digest
`sha256:fc4a060583b63609c10f83d093d620db7c4d04f09b1a3b23123ae706cba00cd6`.

The fixed implementation regenerated a new private package and proposal:
package `br1-real-acceptance-20260809-v2-freeze-v2` with digest
`sha256:6af7a5e844a4852b5b15681970b05e91e770b9929d40322f872905b0794ba652`,
and proposal `br1-real-acceptance-20260809-v2-owner-proposal-v2` with digest
`sha256:17a04dd465587963355da86791f3edaa9fc9a675a5a1bc97fb4c321e34ee756b`.
It is `FROZEN_WAITING_OWNER`; no approval, REINVENT4 template, eligible bundle,
Controller execution, remote dispatch, or new conversation run exists.
