# BR1 Private Real-Tool Canary v1 preflight evidence

This directory records a privacy-safe blocked preflight. It is not evidence of
a completed acceptance run. No acceptance ID, project, run, Controller
execution, Gate decision, model, generation, Top-N, restart, or replay was
created.

The preflight used the PR #27 contract merged at
`e9316ad64fd219a26740b3424f541e00a9409d39`. Both repository-owned logical
execution profiles had a current available capability probe, but the
server-owned remote resource authority policy was not configured. The supplied
source CSV also did not satisfy the BR1 Raw Dataset column contract as-is. Its
dataset name/version, source URL, license, and download date were unavailable,
so source provenance remains explicitly missing rather than inferred.

Within the 13,978-row numeric-QY subset, 10,697 rows belong to repeated raw
chromophore groups, 2,309 raw chromophores occur in multiple solvent conditions,
and 12 rows cannot produce a Standard InChIKey. The current review policy treats
InChIKey alone as duplicate identity, which is unresolved against the frozen
no-silent-condition-merge policy. None of those rows may be materialized as a
BR1 Raw Dataset until an authoritative mapping and condition-aware identity
policy are approved.

Molly therefore failed closed before formal execution. It did not dispatch
Uni-Mol or REINVENT4, did not use `existing_output`, and did not create public
evidence that could be mistaken for real-tool runtime validation.

The canonical blocker roster, in fixed order across the acceptance-manifest
status, blocked evidence, preflight summary, and runtime findings, is:

1. `REMOTE_RESOURCE_AUTHORITY_POLICY_MISSING`
2. `SOURCE_PROVENANCE_MISSING`
3. `DATA_CONTRACT_INCOMPLETE`
4. `CONDITION_AWARE_IDENTITY_POLICY_UNRESOLVED`

GitHub PR Fast, CodeQL, and Full CI run `30779825797` passed at exact HEAD
`7935f22b3a7df2f128e0f8e5e4c24bf02169dbc9`. Full CI included static
validation and all four weighted shards.

The blocker must be resolved by the repository owner outside the project
request: configure the server-owned resource authority policy and provide an
authoritatively mapped BR1 Raw Dataset that preserves material role, emission
mechanism, measurement conditions, comparability, and paper evidence. A future
attempt must use a new clean acceptance ID and run ID.
