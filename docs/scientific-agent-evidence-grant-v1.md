# Scientific Agent EvidenceGrant v1

This document defines the durable structured-feedback contract for
`M3.5-AUT-FEEDBACK`. The public status and execution order remain in
[`roadmap.md`](roadmap.md); this is a technical contract, not a second status
table.

## Authority boundary

Conversation text is a review surface only. Messages such as `yes`,
`approved`, `use it`, `looks good`, `确认`, and `继续` do not create an
`EvidenceGrant`, approve a plan, approve a Gate or remote request, change a
scientific boundary, or create authorization. The LLM and Execution Agent may
explain that review is required, but neither may mint a grant, select
provenance, choose a source digest, or write authority records.

Plan, Gate, and remote approval are separate typed Controller operations. In
particular, an ordinary `ConversationAgent` turn never calls a Controller
approval method; the server-resolved actor and the exact typed snapshot/request
are supplied only by the corresponding structured route.

The pending plan action is:

```text
POST /api/projects/{project_id}/conversations/{conversation_id}/agent-session/approve
```

Gate and remote approval use the existing Controller routes:

```text
POST /api/projects/{project_id}/agent-harness-controller-executions/{controller_execution_id}/gates/{gate_id}/approve
POST /api/projects/{project_id}/agent-harness-controller-executions/{controller_execution_id}/remote-approvals
```

The explicit server action is:

```text
POST /api/projects/{project_id}/conversations/{conversation_id}/agent-session/
     evidence/candidate_raw_dataset/confirm
```

Its only client-controlled fields are:

```json
{
  "expected_source_digest": "sha256:...",
  "confirmed": true,
  "client_request_id": "confirm-..."
}
```

The request must contain the literal boolean `true`. Actor identity comes
from `resolve_authenticated_actor`; body, query, form, and `X-Actor` values
are not accepted as provenance. The route does not resolve or call an LLM.

## EvidenceGrantV1

`EvidenceGrantV1` is a frozen, extra-field-forbidden Pydantic model with a
deterministic `grant_id` and `grant_digest`. The digest covers all semantic
authority material: project, source identity and digest, closed-world scope,
actor/provenance, optional run/conversation binding, evidence type, and the
explicit coverage mode. `issued_at` is audit metadata and is excluded from
semantic identity so a replay does not mint a new semantic grant.

The v1 scope is exactly `extracted_dataset_confirmation`. Its coverage mode is
exact-source. The v1 implementation intentionally does not offer an item-level
or source-wildcard scope; an empty item list therefore means the one fully
identified source artifact, never “all evidence.” The BR2 source identity is
`candidate_raw_dataset`, and its server-computed digest binds the exact bytes
of both the candidate package and its review artifact, together with their
schema and paper bindings.

This scope does not grant retries, training, generation, publication,
promotion, goal changes, new remote resources, dataset replacement, or access
to another source or digest. `EvidenceGrant` is separate from `AutonomyGrant`,
Permission, Authorization, StartIntent, and Controller state.

## Issuance, storage, and consumption

`EvidenceGrantStore` stores immutable grant and request-checkpoint files below
the project-owned `evidence-grants` directory. Publication uses confined
directory descriptors, `O_NOFOLLOW`, atomic no-replace publication, and an
exclusive process lock. A request checkpoint is bound to the project, run,
conversation, source ID, expected/current digest, action, scope, client
request ID, and trusted actor provenance. Same-request replay returns the
original bytes; conflicting replay fails closed. A crash after grant
publication but before checkpoint publication is reconciled by reusing the
existing grant and publishing only the missing checkpoint. Old grant files are
never deleted or replaced.

The explicit BR2 route performs this sequence:

1. The conversation service verifies that the existing Controller is still at
   the exact BR2 confirmation boundary and that the active
   proposal/controller digests still match. The compiled BR2 plan includes a
   server-bound, non-planner-visible
   `consume_oled_candidate_evidence_admission` continuation after the review
   package.
2. The evidence service rereads the registered candidate and review artifacts
   as stable regular files and revalidates their review-only boundary.
3. The server compares the client’s expected digest with the newly computed
   source digest. A stale or unsafe source produces no grant and no admission.
4. The server publishes the typed EvidenceGrant and idempotency checkpoint.
5. The exact grant is consumed into a separate immutable
   `ScientificEvidenceAdmissionV1`, which repeats the project/run/conversation,
   source/package/review digests, grant binding, actor provenance, and
   `SCIENTIFIC_CONFIRMATION` boundary. The admission is registered as an
   immutable run artifact under both its auditable ID and the stable logical
   binding consumed by the downstream task.
6. The structured confirmation resumes the existing Controller. Before the
   consumer task can enter `RUNNING`, `RunPlanExecutor` calls
   `verify_br2_admission()` against the current project/run/conversation,
   grant, registry, and source bytes. Only then may the consumer publish the
   immutable `confirmed_oled_evidence` receipt. A stale, foreign, retargeted,
   or malformed admission fails closed before the task stage or adapter effect.

Admission verification rereads the current source and the grant. It rechecks
every authority-relevant semantic binding carried by the producer: project,
run, conversation, source ID and digest, package/review/paper identity, scope,
actor and actor source, issuer, evidence type, exact-source coverage, and the
closed-world item roster. It rejects a different source ID, a new
package/review digest, a different conversation or project, an unknown scope,
a mismatched grant digest, or a corrupted/symlinked artifact. Downstream code
must verify the grant and source binding rather than trusting `grant_id` alone.

The first production consumer is the local
`consume_oled_candidate_evidence_admission` task. Its typed
`ScientificEvidenceConsumptionReceiptV1` binds the consumer output to the
verified admission, grant, exact source, actor provenance, and retained
`SCIENTIFIC_CONFIRMATION` boundary.

## Semantic boundary and compatibility

Before the structured action, BR2 remains at the human-required
`SCIENTIFIC_CONFIRMATION` boundary. After admission, only that exact reviewed
source may continue according to this narrow scope; the boundary is retained
in the admission and privacy-safe conversation projection rather than being
globally downgraded to `NONE`.

Historical BR1 confirmation artifacts and existing BR2 review-only artifacts
remain readable through their existing paths and schemas. New v1 writes use
the versioned grant/admission formats. No grant is fabricated from historical
chat text or retroactively inferred from an old generic confirmation record.

Conversation and durable event projections expose only source/grant/admission
identities, digests, scope, replay/consumed flags, and the semantic boundary.
They do not expose raw paper text, extracted rows, provider payloads, paths,
credentials, or arbitrary message text.
