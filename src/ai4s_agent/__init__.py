__all__ = ["__version__"]
__version__ = "0.1.0"

from ai4s_agent.execution_confirmation import install_execution_confirmation_audit
from ai4s_agent.snapshot_material import install_run_plan_executor_snapshot_builder

install_run_plan_executor_snapshot_builder()
install_execution_confirmation_audit()
