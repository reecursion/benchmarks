"""Base classes for specialized agents."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from openhands.sdk import Agent, Conversation, get_logger
from openhands.sdk.workspace import RemoteWorkspace

logger = get_logger(__name__)


@dataclass
class AgentState:
    """State shared between agents."""

    task_name: str
    paper_path: str
    submission_path: str
    agents_completed: List[str] = field(default_factory=list)
    agent_outputs: Dict[str, Any] = field(default_factory=dict)
    errors: List[Dict[str, str]] = field(default_factory=dict)
    models_loaded: List[str] = field(default_factory=list)
    datasets_loaded: List[str] = field(default_factory=list)
    methods_implemented: List[str] = field(default_factory=list)
    experiments_configured: List[Dict[str, Any]] = field(default_factory=list)
    experiments_executed: List[Dict[str, Any]] = field(default_factory=list)
    metrics_calculated: List[Dict[str, Any]] = field(default_factory=list)
    results_analyzed: Dict[str, Any] = field(default_factory=dict)
    reports_generated: List[str] = field(default_factory=list)


class BaseAgent(ABC):
    """Base class for specialized agents."""

    def __init__(
        self,
        workspace: RemoteWorkspace,
        agent: Agent,
        shared_state: Dict[str, Any],
    ):
        """
        Initialize the agent.

        Args:
            workspace: The workspace where the agent operates
            agent: The OpenHands agent instance
            shared_state: Shared state dictionary for agent communication
        """
        self.workspace = workspace
        self.agent = agent
        self.shared_state = shared_state
        self.name = self.__class__.__name__

    @abstractmethod
    def get_system_prompt(self, context: Dict[str, Any]) -> str:
        """
        Get the system prompt for this agent.

        Args:
            context: Context dictionary with task information

        Returns:
            System prompt string
        """
        pass

    @abstractmethod
    def get_instruction(self, context: Dict[str, Any]) -> str:
        """
        Get the instruction for this agent.

        Args:
            context: Context dictionary with task information

        Returns:
            Instruction string
        """
        pass

    async def run(
        self, context: Dict[str, Any], max_iterations: int = 50
    ) -> Dict[str, Any]:
        """
        Run the agent on the task.

        Args:
            context: Context dictionary with task information
            max_iterations: Maximum number of iterations

        Returns:
            Dictionary with agent output and status
        """
        logger.info(f"Running agent: {self.name}")

        try:
            # Get system prompt and instruction
            system_prompt = self.get_system_prompt(context)
            instruction = self.get_instruction(context)

            # Build full instruction with system context
            full_instruction = f"""{system_prompt}

{instruction}

Please work on this task step by step. Document your progress and save any outputs to the appropriate locations.
"""

            # Create conversation
            conversation = Conversation(
                agent=self.agent,
                workspace=self.workspace,
                callbacks=[lambda ev: logger.debug(f"{self.name} event: %s", ev)],
                max_iteration_per_run=max_iterations,
            )

            # Send instruction and run
            conversation.send_message(full_instruction)
            conversation.run()

            # Extract results from conversation
            result = self._extract_results(conversation, context)

            logger.info(f"Agent {self.name} completed successfully")
            return {
                "status": "success",
                "agent": self.name,
                "result": result,
                "conversation_history": len(conversation.state.events),
            }

        except Exception as e:
            logger.error(f"Agent {self.name} failed: {e}")
            return {
                "status": "error",
                "agent": self.name,
                "error": str(e),
                "fatal": False,
            }

    def _extract_results(
        self, conversation: Conversation, context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Extract results from conversation.

        Args:
            conversation: The conversation object
            context: Context dictionary

        Returns:
            Dictionary with extracted results (may include callback_agent and callback_reason)
        """
        # Extract information from conversation events
        events = conversation.state.events
        result = {
            "events_count": len(events),
            "last_messages": [],
        }

        # Get last few messages for context
        message_events = [
            e for e in events if hasattr(e, "source") and e.source == "agent"
        ]
        if message_events:
            result["last_messages"] = [
                str(e)[:200] for e in message_events[-5:]
            ]  # Last 5 messages

        # Parse callback requests from agent messages
        callback_agent = None
        callback_reason = None
        
        # Look through all events for callback requests
        for event in reversed(events):  # Start from most recent
            event_str = str(event)
            
            # Check for callback request
            if "CALLBACK_REQUEST:" in event_str:
                lines = event_str.split('\n')
                for line in lines:
                    if "CALLBACK_REQUEST:" in line:
                        callback_agent = line.split("CALLBACK_REQUEST:")[-1].strip()
                    if "CALLBACK_REASON:" in line:
                        callback_reason = line.split("CALLBACK_REASON:")[-1].strip()
                
                if callback_agent:
                    result["callback_agent"] = callback_agent
                    result["callback_reason"] = callback_reason or "No reason provided"
                    logger.info(f"🔄 Agent {self.name} requests callback to {callback_agent}")
                    break

        return result

    def _update_shared_state(self, key: str, value: Any) -> None:
        """Update shared state."""
        self.shared_state[key] = value

    def _get_previous_agent_output(self, agent_name: str) -> Optional[Dict[str, Any]]:
        """Get output from a previous agent."""
        agent_outputs = self.shared_state.get("agent_outputs", {})
        return agent_outputs.get(agent_name)

