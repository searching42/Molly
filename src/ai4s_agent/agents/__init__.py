"""Expert agent contracts for AI4S orchestration."""

from ai4s_agent.agents.critic import CriticAgent
from ai4s_agent.agents.evaluation import compute_autonomy_metrics
from ai4s_agent.agents.generation import GenerationAgent
from ai4s_agent.agents.conversation import ConversationAgent
from ai4s_agent.agents.modeling import ModelingAgent
from ai4s_agent.agents.oled_discovery import OLEDDiscoveryLoopAgent
from ai4s_agent.agents.observer import ObserverAgent
from ai4s_agent.agents.planner import PlannerAgent
from ai4s_agent.agents.prediction import PredictionPreparationAgent
from ai4s_agent.agents.recovery import RecoveryAgent
from ai4s_agent.agents.report import ReportAgent
from ai4s_agent.agents.research import ResearchAgent
from ai4s_agent.agents.tool_registry import AgentToolRegistry
from ai4s_agent.agents.verifier import VerifierAgent

__all__ = [
    "CriticAgent",
    "GenerationAgent",
    "ConversationAgent",
    "ModelingAgent",
    "OLEDDiscoveryLoopAgent",
    "ObserverAgent",
    "PlannerAgent",
    "PredictionPreparationAgent",
    "RecoveryAgent",
    "ReportAgent",
    "ResearchAgent",
    "AgentToolRegistry",
    "VerifierAgent",
    "compute_autonomy_metrics",
]
