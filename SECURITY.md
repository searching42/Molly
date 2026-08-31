# Security policy

## Reporting a vulnerability or privacy issue

Do not open a public issue for credentials, personal data, private
infrastructure details, or an exploitable security defect. Use GitHub's
private vulnerability reporting for this repository and include only the
minimum reproduction detail. Do not paste live credentials, private datasets,
private papers, or machine-specific configuration into a report.

## Current Core v2 boundaries

- `ToolRegistry` and `ToolPolicy` are closed, server-owned execution
  boundaries. Model proposals cannot select shells, arbitrary paths,
  credentials, hosts, or worker commands.
- An approval is exact and digest-bound to the immutable materialized call.
  Conversation text, UI state, caches, and telemetry are not authority.
- Runtime, provider, and compute profiles are server-owned. Secret values do
  not enter prompts, materialized calls, artifacts, ledger events, or tool
  observations.
- Acquisition uses configured providers/routes, HTTPS and address validation,
  redirect and response limits, licensing/access metadata, and cache
  provenance. It does not provide arbitrary model URL fetching or access
  control bypass.
- `ArtifactStore` verifies immutable content digests and uses safe publication;
  `RunLedger` is append-only; `ArtifactLineage` is bounded provenance rather
  than a causal or recovery authority.
- OpenTelemetry and LangSmith, when installed, are observer-only. Exporter
  failure or missing telemetry cannot alter authoritative scientific state.
- The legacy v1 implementation is available only through the immutable
  rollback refs documented in the Core v2 cutover evidence.

## Repository privacy boundary

The public repository may contain source code, public documentation, synthetic
fixtures, machine-readable contracts, and sanitized evidence. Keep runtime
state, real user/project data, private papers, secrets, concrete hostnames,
usernames, absolute infrastructure paths, and machine-specific profiles in
private server-owned storage.

Local working-context files are not public documentation authority. Public
tests, runtime code, packaging, and documentation must not depend on their
presence or contents.

## Optional exact-value audit

Authorized maintainers may run an additional exact-value scan with a
newline-delimited denylist stored outside the checkout or under an explicitly
ignored path:

```bash
MOLLY_PRIVATE_DENYLIST_PATH=/path/outside/checkout/private-denylist.txt \
  python scripts/audit_private_denylist.py
```

The optional scan reports only denylist entry numbers and matching tracked
paths. Never commit the denylist or expose its contents to untrusted CI jobs.

## Rollback and public history

Deleting an old file from the current tree does not erase existing Git objects,
clones, caches, or mirrors. Core v2 rollback depends on the immutable v1 tag
and branch, not on a compatibility package in the default install. Any public
history rewrite requires separate owner approval and downstream remediation.
