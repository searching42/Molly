"""Controlled writer execution state-machine primitives."""

from .execution_state import ExecutionState
from .transitions import TransitionResult, validate_transition

__all__ = ["ExecutionState", "TransitionResult", "validate_transition"]
