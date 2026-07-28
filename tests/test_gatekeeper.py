from ai4s_agent.gatekeeper import Gatekeeper
from ai4s_agent.schemas import GateName


def test_gatekeeper_blocks_without_approval() -> None:
    gk = Gatekeeper()
    assert gk.can_advance("r1", GateName.TASK_PARSE) is False
