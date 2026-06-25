from __future__ import annotations

from pathlib import Path


SEMANTICS_DOC = Path("docs/queued-canary-retry-requeue-semantics.md")
ROLLOUT_DOC = Path("docs/queued-execute-canary-rollout-policy.md")
WORKER_QUEUE_DOC = Path("docs/worker-queue-skeleton.md")
HARDENING_DOC = Path("docs/post-open-hardening.md")
MILESTONE_DOC = Path("docs/phase-1-4-milestone-status.md")


def _text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def test_retry_requeue_semantics_doc_exists_and_distinguishes_core_terms() -> None:
    assert SEMANTICS_DOC.exists()
    text = _text(SEMANTICS_DOC)

    for required in [
        "Lease attempt",
        "Stale lease recovery",
        "Explicit retry",
        "Explicit requeue",
        "Rerun / new execution",
        "The current `WorkerQueue.attempts` field counts acquisitions",
        "A lease attempt is not an explicit retry count",
        "Stale recovery keeps the same `job_id`",
        "Explicit retry creates a new `job_id`",
        "The original failed job remains immutable",
    ]:
        assert required in text


def test_retry_requeue_semantics_doc_defines_eligibility_lineage_and_idempotency() -> None:
    text = _text(SEMANTICS_DOC)

    for required in [
        "Only a terminal failed job may be considered for explicit retry",
        "a succeeded job is not retryable",
        "a cancelled job is not retryable",
        "a queued job is not retryable",
        "a running job is not retryable",
        "`WAITING_USER` is not a retry condition",
        "Retry remains restricted to the current queued-canary allowlist",
        "No literature/mining, generation, `train_model`, or unknown task chain becomes retryable through this policy",
        "`retry_of_job_id`",
        "`retry_root_job_id`",
        "`retry_index`",
        "`retry_request_id`",
        "retry_request_id provides idempotency",
        "repeating the same `retry_request_id` must not create duplicate retry jobs",
        "at most one active queued/running retry may exist for the same source job",
        "the initial canary implementation should permit at most one explicit retry",
    ]:
        assert required in text


def test_retry_requeue_semantics_doc_forbids_automatic_retry_and_route_mutation() -> None:
    text = _text(SEMANTICS_DOC)

    for required in [
        "This PR does not implement an API route or queue mutation",
        "It does not add `WorkerQueue.retry`, `WorkerQueue.requeue`, automatic retry, timers, or any change to `/api/run-plan/execute` or `/api/run-plan/resume`",
        "no automatic retry loop",
        "no retry from inside `WorkerQueuePoller`",
        "stale recovery must not consume the explicit retry allowance",
        "a cancelled job cannot be converted into a retry job",
        "`WAITING_USER` uses the existing gate/resume path, not retry",
        "default-route migration remains blocked",
    ]:
        assert required in text


def test_retry_requeue_docs_and_status_pages_do_not_claim_retry_is_implemented() -> None:
    combined = "\n".join(
        [
            _text(SEMANTICS_DOC),
            _text(ROLLOUT_DOC),
            _text(WORKER_QUEUE_DOC),
            _text(HARDENING_DOC),
            _text(MILESTONE_DOC),
        ]
    ).lower()

    for forbidden in [
        "retry/requeue is implemented",
        "automatic retry is enabled",
        "cancelled jobs are retryable",
        "waiting_user uses retry instead of resume",
        "queued execution is now default",
        "default migration completed",
    ]:
        assert forbidden not in combined
