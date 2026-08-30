# Acquisition, security, and provenance contract

Status: `PASS` (`FROZEN_INITIAL_C3_CONTRACT`)

This is the conservative initial boundary for a future Core v2 acquisition
layer. It is a contract and test target, not an implemented fetcher. It does
not authorize network-live acceptance, publisher access, or any production
acquisition code in this readiness Goal.

## Authority and provider boundary

Provider configuration is closed and server-owned. A provider may be used
only when its logical name, allowed host set, request shape, rate limit, and
credential profile are present in the server configuration. A model, prompt,
document, or remote response may not add a provider or host at runtime.

The initial provider classes are:

| Class | Initial providers or source route | Boundary |
| --- | --- | --- |
| Metadata | OpenAlex; Crossref | Use only the configured HTTPS API host and declared request parameters. |
| Open-access resolution | Unpaywall | Use only the configured HTTPS API host and server-owned logical access profile. |
| Full text | Verified legal open-access source; explicitly configured publisher/TDM adapter; repository-hosted or otherwise explicitly permitted source | Each host and access route is explicitly configured and license/access status is recorded before materialization. |

The domain allowlist is an exact configuration, not a suffix or wildcard
allowlist. Publisher, repository, and TDM hosts are added individually after
legal/access review. A DNS answer, redirect target, or alternate URL is not
trusted merely because the initial URL was allowed.

The existing repository rules in [`SECURITY.md`](../../../SECURITY.md) remain
authoritative for public/private data, secrets, reports, local context, and
history. This contract adds acquisition-specific rules and does not weaken
that policy.

## URL, DNS, and network restrictions

- HTTPS is required wherever the provider supports it; plain HTTP is denied
  unless a future server-owned exception is explicitly documented and tested.
- Userinfo, embedded credentials, unsupported schemes, fragments used as
  routing instructions, and non-canonical host forms are rejected.
- Loopback, RFC1918/private, link-local, multicast, unspecified, and other
  non-public destination ranges are denied for both IPv4 and IPv6.
- DNS is resolved by the server-side client. Every resolved address is
  checked against the denied ranges before connection, and each redirect is
  resolved and checked again. DNS rebinding and an address/hostname mismatch
  fail closed.
- Only the configured exact host and port are allowed. A suffix match,
  wildcard, alternate port, proxy-provided destination, or model-provided
  override is not sufficient.
- Redirects are limited to 5 hops. Every `Location` target is parsed,
  allowlist-checked, DNS-resolved, and range-checked before following it.
- The initial response-size ceiling is 25 MiB, measured while streaming. A
  response that exceeds the ceiling is discarded and is not an artifact.
- Accepted content types are configured per route from
  `application/json`, `application/xml`, `text/xml`, `text/html`, and
  `application/pdf`. Missing, contradictory, or unconfigured content types
  fail closed; sniffing cannot broaden the allowlist.
- Initial timeout limits are 10 seconds for connection, 30 seconds for a
  read, and 60 seconds total. Providers may be assigned stricter limits.
- A response body is never passed to a shell, interpreter, or executable
  loader as part of acquisition.

## Rate, retry, and cache policy

The initial per-provider default is at most 2 in-flight requests and 1 request
per second per configured host, subject to a lower provider-specific limit.
The implementation must honor `Retry-After` when it is valid and must cap a
server-requested delay at the configured maximum. It must use exponential
backoff with bounded jitter for retryable failures (initial delay 1 second,
maximum delay 32 seconds, maximum 3 retries). Authentication failures,
allowlist failures, content/type violations, and policy denials are not
retryable.

Cache-first behavior is mandatory. A cache lookup uses provider, normalized
request/query identity, canonical identifier, route policy version, and
request shape. A cache hit is revalidated against its recorded provenance and
content SHA-256 before reuse. An immutable content collision, changed bytes
under an existing identity, or incomplete provenance record fails closed; it
must not silently overwrite or fork an artifact identity.

## Credentials and model boundary

Credentials are server-owned and selected by logical profile. Secrets must not
appear in prompts, model-visible tool arguments, response payloads, cache
keys, artifacts, ledger records, exception text, or ordinary logs. A
`CREDENTIAL_REFERENCE` may identify a server-side profile without containing
the secret.

The future acquisition API accepts a typed provider request or canonical
identifier, not an arbitrary model-provided URL. The model cannot choose a
host, proxy, redirect policy, credential, local path, shell command, or
network transport.

The following behavior is explicitly prohibited:

- CAPTCHA bypass or solving for access-control evasion;
- residential proxy rotation;
- IP rotation to evade a block or rate limit;
- browser fingerprint or anti-bot evasion;
- access-control bypass, paywall circumvention, or unauthorized publisher
  retrieval;
- credential leakage, credential logging, or credential inclusion in an
  artifact or prompt.

## Artifact classes

Every durable output is labeled with exactly one of these classes:

| Class | Meaning | Allowed public-repository treatment |
| --- | --- | --- |
| `PUBLIC_ARTIFACT` | Public-safe, redistributable content or derived evidence with a verified access/license basis. | May be checked in only when the license and provenance permit redistribution. |
| `PRIVATE_ARTIFACT` | User/project data, non-redistributable source content, or restricted derived material. | Remains in private storage; repository evidence contains only a sanitized manifest or digest where allowed. |
| `RUNTIME_SECRET` | Secret bytes such as API keys, tokens, cookies, private keys, or passwords. | Never enters repository, model context, artifact bytes, ledger payload, or logs. |
| `CREDENTIAL_REFERENCE` | A non-secret logical reference to a server-owned credential/profile. | May identify the profile class; it must not reveal secret material or private endpoint details. |

An artifact class is assigned before publication. A public-looking URL does
not by itself make the content public or redistributable.

## Minimum acquisition provenance

The minimum provenance record for a retrieved or resolved item contains:

```text
provider
query/request identity
DOI or canonical identifier
source URL
resolved URL
retrieved_at
license/access status
content type
content SHA-256
cache identity
```

The record also binds the route-policy version, redirect chain, response
status, parser/content-family selection, and the logical credential profile
when one was used. It never stores the credential value. A derived document,
evidence packet, or dataset row must retain its source artifact digest and
source locator.

## Required adversarial tests before implementation acceptance

The future implementation must have isolated tests for:

1. loopback, private, link-local, multicast, IPv6, alternate-port, userinfo,
   and DNS-rebinding targets;
2. an allowed first URL redirecting to an unallowlisted host or denied IP;
3. redirect-loop and redirect-limit behavior;
4. response-size overflow, missing or conflicting content type, and timeout;
5. provider concurrency/rate limits, valid and invalid `Retry-After`, bounded
   exponential backoff, and non-retryable policy failures;
6. cache-first reuse, cache provenance mismatch, and immutable-content
   collision/no-replace behavior;
7. credential absence from prompts, model-visible arguments, artifacts,
   ledgers, exceptions, and logs;
8. refusal of arbitrary model-provided URLs and all prohibited access-evasion
   behaviors; and
9. provenance completeness, artifact-class enforcement, and public/private
   repository hygiene.

These tests may use deterministic network mocks and fixture responses. They
must not require live network access to close Core readiness.

## C3 decision

This contract is complete for the initial Core v2 boundary. It freezes the
provider, network, credential, artifact-class, provenance, and adversarial
test requirements while leaving the production fetcher for CORE-03. C3 is
therefore `PASS` when the readiness manifest points to this file and its
digest; no network-live or licensing claim is made by this closure.
