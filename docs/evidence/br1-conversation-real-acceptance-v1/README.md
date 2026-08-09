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
