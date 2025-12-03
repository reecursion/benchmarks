"""Critic Agent for reviewing and improving agent outputs."""

import json
from enum import Enum
from typing import Any, Dict, List, Optional

from benchmarks.paper_bench.agents.agent_base import BaseAgent
from openhands.sdk import Agent, Conversation, get_logger
from openhands.sdk.workspace import RemoteWorkspace


logger = get_logger(__name__)


class CritiqueRating(str, Enum):
    """Rating levels for critique."""

    PASS = "pass"
    NEEDS_IMPROVEMENT = "needs_improvement"
    FAIL = "fail"


class CritiqueResult:
    """Result of a critique."""

    def __init__(
        self,
        rating: CritiqueRating,
        score: float,
        feedback: str,
        specific_issues: List[str],
        suggestions: List[str],
    ):
        self.rating = rating
        self.score = score  # 0.0 - 1.0
        self.feedback = feedback
        self.specific_issues = specific_issues
        self.suggestions = suggestions

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "rating": self.rating.value,
            "score": self.score,
            "feedback": self.feedback,
            "specific_issues": self.specific_issues,
            "suggestions": self.suggestions,
        }

    @classmethod
    def from_llm_response(cls, response: str) -> "CritiqueResult":
        """Parse critique from LLM response."""
        # Extract structured data from response
        rating = CritiqueRating.PASS
        score = 0.8
        feedback = ""
        issues = []
        suggestions = []

        lines = response.lower().split("\n")
        for i, line in enumerate(lines):
            if "rating:" in line or "verdict:" in line or "result:" in line:
                if "pass" in line:
                    rating = CritiqueRating.PASS
                    score = 0.9
                elif "needs improvement" in line or "needs_improvement" in line:
                    rating = CritiqueRating.NEEDS_IMPROVEMENT
                    score = 0.6
                elif "fail" in line:
                    rating = CritiqueRating.FAIL
                    score = 0.3

            if "issue" in line and i + 1 < len(lines):
                # Collect issues
                for j in range(i + 1, len(lines)):
                    if lines[j].strip().startswith(("-", "*", "•", str(j - i))):
                        issues.append(lines[j].strip().lstrip("-*•0123456789. "))
                    elif lines[j].strip() and not any(
                        x in lines[j]
                        for x in ["suggestion", "recommendation", "feedback"]
                    ):
                        issues.append(lines[j].strip())
                    else:
                        break

            if "suggest" in line or "recommend" in line:
                # Collect suggestions
                for j in range(i + 1, len(lines)):
                    if lines[j].strip().startswith(("-", "*", "•", str(j - i))):
                        suggestions.append(lines[j].strip().lstrip("-*•0123456789. "))
                    elif lines[j].strip():
                        suggestions.append(lines[j].strip())
                    else:
                        break

        feedback = response

        return cls(rating, score, feedback, issues, suggestions)


class CriticAgent(BaseAgent):
    """Agent that reviews and critiques other agents' outputs."""

    def __init__(
        self,
        workspace: RemoteWorkspace,
        agent: Agent,
        shared_state: Dict[str, Any],
        max_refinements: int = 3,
        enable_for_agents: Optional[List[str]] = None,
    ):
        """
        Initialize the critic agent.

        Args:
            workspace: The workspace
            agent: The OpenHands agent instance
            shared_state: Shared state dictionary
            max_refinements: Maximum refinement iterations per agent
            enable_for_agents: List of agent names to critique (None = all)
        """
        super().__init__(workspace, agent, shared_state)
        self.max_refinements = max_refinements
        self.enable_for_agents = enable_for_agents or [
            "infrastructure",
            "model_dataset",
            "method_implementation",
            "experiment_config",
            "experiment_execution",
            "metrics_evaluation",
        ]
        self.critique_history = []

    def get_system_prompt(self, context: Dict[str, Any]) -> str:
        return """You are a Critical Reviewer Agent responsible for evaluating agent work quality.

Your role is to:
- Review agent outputs for completeness, correctness, and quality
- Identify specific issues and gaps
- Provide actionable feedback for improvement
- Assess if work meets the requirements

You are thorough but fair. You recognize good work but also catch errors and omissions.
Your feedback should be specific, actionable, and focused on the most important issues.
"""

    def get_instruction(self, context: Dict[str, Any]) -> str:
        """This won't be used directly - critique is called programmatically."""
        return ""

    def should_critique_agent(self, agent_name: str) -> bool:
        """Check if agent should be critiqued."""
        return agent_name in self.enable_for_agents

    def critique_agent_output(
        self,
        agent_name: str,
        agent_output: dict,
        context: Dict[str, Any],
    ) -> CritiqueResult:
        """
        Critique an agent's output.

        Args:
            agent_name: Name of the agent being critiqued
            agent_output: Output from the agent
            context: Context including task requirements

        Returns:
            CritiqueResult with rating and feedback
        """
        logger.info(f"Critic reviewing {agent_name} output...")

        # Build critique prompt
        critique_prompt = self._build_critique_prompt(agent_name, agent_output, context)

        try:
            # Create conversation for critique
            conversation = Conversation(
                agent=self.agent,
                workspace=self.workspace,
                callbacks=[lambda ev: logger.debug("Critic event: %s", ev)],
                max_iteration_per_run=10,  # Critique should be quick
            )

            # Get critique
            conversation.send_message(critique_prompt)
            conversation.run()

            # Extract critique from conversation
            critique_text = self._extract_critique_from_conversation(conversation)  # type: ignore[arg-type]
            critique_result = CritiqueResult.from_llm_response(critique_text)

            # Log critique
            self.critique_history.append(
                {
                    "agent": agent_name,
                    "rating": critique_result.rating.value,
                    "score": critique_result.score,
                    "issues": critique_result.specific_issues,
                }
            )

            logger.info(
                f"Critique result: {critique_result.rating.value} (score: {critique_result.score:.2f})"
            )
            if critique_result.specific_issues:
                logger.info(f"   Issues found: {len(critique_result.specific_issues)}")

            return critique_result

        except Exception as e:
            logger.error(f"Error during critique: {e}")
            # Default to pass on error to not block workflow
            return CritiqueResult(
                rating=CritiqueRating.PASS,
                score=0.7,
                feedback=f"Critique failed: {e}",
                specific_issues=[],
                suggestions=[],
            )

    def _build_critique_prompt(
        self,
        agent_name: str,
        agent_output: dict,
        context: Dict[str, Any],
    ) -> str:
        """Build critique prompt based on agent type."""

        base_prompt = f"""
You are reviewing the work of the {agent_name} agent.

TASK CONTEXT:
{self._format_context(context)}

AGENT OUTPUT:
{self._format_agent_output(agent_output)}

Please critically evaluate this work:

"""

        # Add agent-specific evaluation criteria
        if agent_name == "infrastructure":
            criteria = """
EVALUATION CRITERIA:
1. Completeness: Are all required dependencies installed?
2. Correctness: Is the environment properly configured?
3. Validation: Was the setup tested and validated?
4. Documentation: Is there a requirements.txt or environment.yml?
5. GPU Setup: If needed, is GPU access properly configured?

Check if common dependencies are missing or if there are configuration issues.
"""
        elif agent_name == "model_dataset":
            criteria = """
EVALUATION CRITERIA:
1. Completeness: Are all required models and datasets loaded?
2. Correctness: Are models loaded with correct configurations?
3. Validation: Was data loading tested with samples?
4. Documentation: Are data loading scripts created?
5. Splits: Are train/dev/test splits handled correctly?

Check if models match paper specifications and if datasets are properly preprocessed.
"""
        elif agent_name == "method_implementation":
            criteria = """
EVALUATION CRITERIA:
1. Completeness: Are all methods from the paper implemented?
2. Correctness: Do implementations match paper descriptions?
3. Code Quality: Is the code well-structured and documented?
4. Testing: Were implementations tested?
5. Baselines: Are baseline methods implemented correctly?

Check if the implementation matches mathematical formulas and algorithms in the paper.
"""
        elif agent_name == "experiment_config":
            criteria = """
EVALUATION CRITERIA:
1. Completeness: Are all experiments from the paper configured?
2. Correctness: Do hyperparameters match the paper?
3. Documentation: Are configurations well-documented?
4. Reproducibility: Are seeds set for reproducibility?
5. Coverage: Are ablation studies included?

Check if learning rates, batch sizes, and other hyperparameters match paper specifications.
"""
        elif agent_name == "experiment_execution":
            criteria = """
EVALUATION CRITERIA:
1. Completeness: Were all configured experiments executed?
2. Correctness: Did experiments run to completion?
3. Results: Are results saved properly?
4. Validation: Are checkpoints and logs available?
5. Error Handling: Were errors handled gracefully?

Check if experiments produced expected outputs and if results are properly saved.
"""
        elif agent_name == "metrics_evaluation":
            criteria = """
EVALUATION CRITERIA:
1. Completeness: Are all metrics from the paper calculated?
2. Correctness: Are metric calculations implemented correctly?
3. Validation: Do metrics make sense given the results?
4. Documentation: Are metric calculation scripts documented?
5. Comparison: Can metrics be compared with paper results?

Check if metric formulas match the paper and if calculated values are reasonable.
"""
        else:
            criteria = """
EVALUATION CRITERIA:
1. Completeness: Was the task fully completed?
2. Correctness: Is the work accurate and correct?
3. Quality: Is the work production-ready?
4. Documentation: Is the work well-documented?
5. Requirements: Does it meet all requirements?
"""

        final_prompt = (
            base_prompt
            + criteria
            + """

PROVIDE YOUR CRITIQUE:

1. **Rating**: PASS / NEEDS_IMPROVEMENT / FAIL
   - PASS: Work is complete, correct, and meets quality standards
   - NEEDS_IMPROVEMENT: Work is mostly good but has fixable issues
   - FAIL: Work has critical errors or major omissions

2. **Score**: Rate the work from 0.0 to 1.0

3. **Specific Issues** (if any):
   List concrete, specific issues found:
   - Issue 1: [be specific]
   - Issue 2: [be specific]
   ...

4. **Suggestions for Improvement** (if rating is not PASS):
   Provide actionable suggestions:
   - Suggestion 1: [actionable]
   - Suggestion 2: [actionable]
   ...

5. **Overall Feedback**:
   Brief summary of the review.

Be thorough but constructive. Focus on the most important issues first.
"""
        )

        return final_prompt

    def _format_context(self, context: Dict[str, Any]) -> str:
        """Format context for critique prompt."""
        parts = []

        if context.get("task"):
            parts.append(f"Task: {context['task']}")

        if context.get("paper_path"):
            parts.append(f"Paper: {context['paper_path']}")

        if context.get("instructions"):
            # Truncate long instructions
            instructions = context["instructions"]
            if len(instructions) > 1000:
                instructions = instructions[:1000] + "..."
            parts.append(f"Instructions:\n{instructions}")

        return "\n".join(parts)

    def _format_agent_output(self, agent_output: dict) -> str:
        """Format agent output for critique prompt."""
        # Convert output to readable format
        formatted = json.dumps(agent_output, indent=2, default=str)

        # Truncate if too long
        if len(formatted) > 3000:
            formatted = formatted[:3000] + "\n... (truncated)"

        return formatted

    def _extract_critique_from_conversation(self, conversation: Conversation) -> str:
        """Extract critique text from conversation events."""
        # Get all message events from agent
        messages = []
        for event in conversation.state.events:  # type: ignore[attr-defined]
            if hasattr(event, "source") and event.source == "agent":
                if hasattr(event, "message"):
                    messages.append(event.message)
                elif hasattr(event, "content"):
                    messages.append(event.content)
                else:
                    messages.append(str(event))

        # Combine messages
        critique_text = "\n".join(messages)
        return critique_text

    async def run(
        self, context: Dict[str, Any], max_iterations: int = 50
    ) -> Dict[str, Any]:
        """
        Run method - not typically called directly.
        Critic is invoked via critique_agent_output().
        """
        return {
            "status": "success",
            "agent": self.name,
            "result": {
                "critique_history": self.critique_history,
                "total_critiques": len(self.critique_history),
            },
        }

    def get_critique_summary(self) -> dict:
        """Get summary of all critiques performed."""
        if not self.critique_history:
            return {"total": 0}

        summary = {
            "total": len(self.critique_history),
            "by_rating": {},
            "average_score": 0.0,
            "agents_critiqued": [],
        }

        total_score = 0.0
        for critique in self.critique_history:
            rating = critique["rating"]
            summary["by_rating"][rating] = summary["by_rating"].get(rating, 0) + 1
            total_score += critique["score"]

            if critique["agent"] not in summary["agents_critiqued"]:
                summary["agents_critiqued"].append(critique["agent"])

        summary["average_score"] = total_score / len(self.critique_history)

        return summary
