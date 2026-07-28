from __future__ import annotations

from dataclasses import dataclass

from .execution_binding import ExecutionBinding


@dataclass(frozen=True)
class ArtifactRegistrationResult:
    allowed: bool
    registry: "ArtifactRegistry"
    errors: tuple[str, ...] = ()


@dataclass(frozen=True)
class ArtifactRegistry:
    records: tuple[ExecutionBinding, ...] = ()

    @classmethod
    def empty(cls) -> "ArtifactRegistry":
        return cls(())

    def register(self, binding: ExecutionBinding) -> ArtifactRegistrationResult:
        errors: list[str] = []
        if not isinstance(binding, ExecutionBinding):
            return ArtifactRegistrationResult(False, self, ("invalid_execution_binding",))

        for record in self.records:
            if record.transition_id == binding.transition_id:
                errors.append("transition_already_bound_to_artifact")
            if record.artifact_hash == binding.artifact_hash:
                errors.append("artifact_hash_already_bound")
            if record.to_record() == binding.to_record():
                errors.append("duplicate_artifact_binding")

        if errors:
            return ArtifactRegistrationResult(False, self, tuple(dict.fromkeys(errors)))

        return ArtifactRegistrationResult(True, ArtifactRegistry((*self.records, binding)))
