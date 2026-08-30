# Molly Core v2 CORE-03 compliant literature acquisition

Status: `IMPLEMENTED — offline network-mock acceptance`

This report records the CORE-03 acquisition milestone only.  CORE-04 and all
later milestones have not started.

```text
repository: searching42/Molly
base: origin/main@f67ed5441b075c0259b088f20690424c1a799aae
branch: codex/molly-core-v2-core-03-acquisition
draft PR: pending final local verification
accepted CORE-02 ancestor: 90423ca6edd2fa0df2c7f2540d2db37fee8486ef
accepted CORE-02A ancestor: 341f3dd99ff993693b92161f0ef4f434bf9c35eb
```

The base is the merged CORE-02 main line.  Readiness remains the frozen
pre-implementation state: C0-C7 are `PASS`, `core_goal_mode_ready=true`, and
`core_cutover_ready=false`.

## Provider and package boundary

The production package is `src/molly/acquisition/`, with the dependency
direction:

```text
molly.acquisition -> molly.core
```

It adds:

```text
models.py    server-owned ProviderConfig/ProviderRoute, normalized records,
             access-profile material, status/classification values
providers.py OpenAlex and Crossref metadata adapters, Unpaywall resolver,
             configured full-text request interface
policy.py    DOI/query bounds, exact URL/route policy, SSRF address policy,
             content-family and redaction helpers
transport.py pinned-IP HTTPS transport with manual redirects and bounded retry
cache.py     integrity-checked, no-replace response cache
service.py   cache-first metadata/OA/full-text orchestration
tools.py     bounded AgentLoop ToolSpecs and host-owned executors
errors.py    fail-closed acquisition error vocabulary
```

The model-facing operations are only:

```text
literature_metadata_search
literature_metadata_lookup
literature_acquire_full_text
```

Their schemas accept semantic query/DOI/identifier/limit values only.  URLs,
hosts, ports, providers, proxies, filesystem paths, headers, access profiles,
and credentials are not model inputs.  Acquisition executors capture the
server-owned `AcquisitionService`; `ToolExecutionContext` exposes only the
already-materialized arguments and declared Core artifact reader.

## Server-owned configuration and exact tool binding

`ProviderConfig` and `ProviderRoute` are frozen values.  Configuration binds
exact provider IDs, provider classes, DNS hostnames, HTTPS port, route/path
templates, allowed GET query/header fields, accepted content types, access and
redistribution policy, timeout ceilings, response-size ceiling, concurrency,
rate, retry, and logical access-profile reference.

`AcquisitionConfig.config_digest` is canonical JSON SHA-256 over all of those
non-secret semantics.  `ToolSpec.execution_config_digest` includes this digest
in the exact `ToolSpec.spec_digest` while remaining absent from the
model-facing view.  Actual API keys, email values, bearer tokens, cookies, and
other secret bytes are not configuration-digest inputs.  A changed configured
route therefore changes the exact tool semantics used by materialized calls.

`MetadataProvider`, `OpenAccessResolver`, and the configured
`FullTextFetcher` request interface are typed provider boundaries.  Provider
responses are normalized to small source-neutral records; raw provider JSON is
not placed in `ToolResult.data`.

## Access material and privacy

`EphemeralAccessMaterial` is host-only, redacted in `repr`, and recursively
held behind immutable mappings.  It is resolved by a server-owned
`AccessProfileResolver` and never exposed to a `DecisionProvider`.  Secret
values are passed only to the request path and exact reflection checks.

Credential-bearing query fields such as `email`, `mailto`, `api_key`, and
token-like names are removed from recorded URLs.  Sanitized request/resolved
URLs, redirect chains, cache manifests, provenance artifacts, ToolResult data,
and cache identities contain no access material.  An exact reflected secret
fails before cache publication or an ArtifactDraft is returned.  Error text
for transport failures uses the same URL redaction path.

## URL, DNS, redirect, and transport policy

The production transport is `SafeNetworkTransport`.  It uses only configured
HTTPS routes, rejects userinfo, fragments, alternate ports, IP-literal and
non-canonical hosts, traversal, unconfigured hosts, and unsafe query fields.
Each connection resolves the server-owned hostname, rejects non-global/private,
loopback, link-local, multicast, unspecified, and reserved IPv4/IPv6 results,
connects to the validated address, retains the configured hostname for TLS SNI
and certificate verification, and checks the connected peer when available.
The HTTP client is manual and does not inherit proxy/environment routing or
perform an uncontrolled second hostname lookup.

Redirects are followed manually, revalidated against the exact configured
route, bounded at five hops, loop-checked, and independently resolved/pinned
on each request.  Unsupported transfer/content encodings are rejected.
Response bytes are streamed under a 25 MiB maximum (or a stricter provider
limit), with identity encoding, route-allowed JSON/XML/HTML/PDF content types,
10-second connect, 30-second read, and 60-second total operation ceilings.
Per-provider concurrency is at most two and each configured host is limited
to one request per second.  Idempotent GET transport failures and 408/429/5xx
responses use at most three bounded retries with exponential delay, jitter,
and bounded `Retry-After`; policy, content, access, and cache-integrity
failures are not retried.

The live transport implementation is present behind the closed server-owned
configuration, but this milestone makes no external provider, credential, DNS,
or publisher availability claim.  Acceptance is deterministic offline
network-mock acceptance.

## Cache and artifact/provenance model

`AcquisitionCache` is a network optimization and reproducibility store, not
the scientific artifact authority.  Its identity is canonical JSON SHA-256
over provider, semantic request identity, canonical identifier, provider
configuration digest, route policy, request shape, sanitized source URL, and
logical access-profile reference.  Actual secret values are excluded.  Body
and manifest are published through fsynced temporary files and no-replace
links; partial, symlinked, malformed, contradictory, or digest-mismatched
entries fail closed.  A valid second request is a verified cache hit without a
network call.

Full-text candidates from Unpaywall remain untrusted.  Only exact configured
full-text routes can be selected.  Eligible sources are ranked deterministically
as JATS/XML, other XML, HTML, then PDF; the body is not parsed.  The output
contains one content `ArtifactDraft` and one JSON acquisition-provenance
`ArtifactDraft`.  Content identity remains the exact immutable SHA-256
artifact ID.  `PUBLIC_ARTIFACT` versus `PRIVATE_ARTIFACT` is an acquisition
occurrence decision based on explicit route/access/redistribution policy and
is not added to `ArtifactRecord`; runtime secrets and credential references
are not cacheable or publishable.

`AcquisitionProvenance` binds provider/config/route/request/canonical DOI,
sanitized source/resolved/redirect URLs, response status/time, access/license
basis, occurrence classification, content type/family/digest/artifact ID,
cache identity/status, access-profile reference, and evaluated candidates.
Different acquisition occurrences can therefore reuse one content artifact
identity while retaining distinct sanitized provenance.

## Network-mock acceptance and tests

The dedicated suite is:

```text
tests/molly/test_core03_acquisition.py
```

It covers:

```text
OpenAlex/Crossref/Unpaywall fixture normalization
AgentLoop -> NETWORK_READ tool -> mock OA resolution -> configured full text
ArtifactStore publication and bounded RunLedger/ArtifactLineage projection
cache miss/hit, restart, corruption, partial entry, symlink escape
exact config-digest ToolSpec binding and hidden model view
DOI/query/schema bounds and no arbitrary URL authority
exact host/port/userinfo/scheme/path/query route policy
IPv4/IPv6 denied-address, peer-pinning, redirect, loop, and target checks
stream/content-length/size/content-type/transfer/content-encoding checks
timeout, Retry-After/backoff, retry class, and authentication behavior
synthetic credential reflection and durable URL/cache/provenance redaction
public/private occurrence classification without ArtifactRecord mutation
```

The current focused local evidence is:

```text
CORE-03 suite: 37 passed
CORE-01 + CORE-02 + CORE-03 focused regression: 77 passed
C4/readiness/fixture regression: 20 passed
repository privacy: 40 passed
legacy literature/phase3 smoke: 57 passed
git diff --check: PASS
python -m compileall -q src tests prototypes: PASS
uv lock --check: PASS (185 packages resolved)
PR Fast: pending final coherent branch validation
Full CI: pending Draft PR and final pushed HEAD
CodeQL: pending Draft PR checks
```

The refined import boundary keeps `molly.core` network-free while permitting
bounded network primitives only under `molly.acquisition`.  All v2 production
files remain free of `ai4s_agent`, the C4 prototype, and subprocess/shell
authority.  The frozen synthetic fixture bytes were reused; no copyrighted
full text or live credentials were added.

## Explicit non-goals and remaining gates

CORE-03 acquires bytes and records provenance only.  It does not create
`CanonicalDocument`, parse XML/HTML/PDF, invoke MinerU, extract OLED facts,
run LLMs, train or run BR1, use remote compute, add UI/API/observability, or
restore legacy Controller/Permission/Autonomy authority.  It does not alter
the C0-C7 readiness manifest.

```text
B0: unchanged
B1: unchanged
B2: PENDING — no fresh-real BR1 v2 evidence
B3: PENDING — no remote restart canary
B4: PENDING — no Owner cutover approval
core_cutover_ready: false
```

The next milestone is CORE-04 document normalization/parsing, subject to
review of this acquisition implementation and its final CI evidence.  No
CORE-04 work has started.

## Final CI synchronization

This section is updated after the final executable/test HEAD is pushed.  A
later report-only update must identify the exact executable/test HEAD covered
by Full CI and must not be treated as new runtime or test evidence.
