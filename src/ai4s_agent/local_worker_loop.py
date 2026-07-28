from __future__ import annotations

from dataclasses import dataclass

from ai4s_agent.worker_queue_poller import WorkerQueuePollResult, WorkerQueuePoller


@dataclass(frozen=True)
class LocalWorkerLoopResult:
    iterations: int
    results: list[WorkerQueuePollResult]


class LocalWorkerLoop:
    """Bounded local loop around WorkerQueuePoller.

    This wrapper does not start background threads, expose API routes, call
    RunPlanExecutor, contact remote workers, or change queue storage.
    """

    def __init__(self, poller: WorkerQueuePoller) -> None:
        self.poller = poller

    def run_once(self, *, now: str = "") -> WorkerQueuePollResult:
        return self.poller.poll_once(now=now)

    def run_until_idle(self, *, max_iterations: int, now: str = "") -> LocalWorkerLoopResult:
        if max_iterations <= 0:
            return LocalWorkerLoopResult(iterations=0, results=[])
        results: list[WorkerQueuePollResult] = []
        for _index in range(max_iterations):
            result = self.run_once(now=now)
            results.append(result)
            if result.action == "idle":
                break
        return LocalWorkerLoopResult(iterations=len(results), results=results)
