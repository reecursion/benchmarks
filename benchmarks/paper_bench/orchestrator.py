"""
Multi-agent orchestrator for PaperBench evaluation.

This orchestrator coordinates specialized agents to reproduce research papers
following the workflow defined in agent_rollout_plan.md.
"""

import asyncio
import json
import os
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

from openhands.sdk import Agent, Conversation, Event, LLM, Message, get_logger
from openhands.sdk.workspace import RemoteWorkspace
from openhands.tools.preset.default import get_default_tools

from benchmarks.paper_bench.agents import (
    AgentState,
    ExperimentConfigAgent,
    ExperimentExecutionAgent,
    InfrastructureAgent,
    MethodImplementationAgent,
    MetricsEvaluationAgent,
    ModelDatasetAgent,
    ReportingAgent,
    ResultAnalysisAgent,
)

logger = get_logger(__name__)


class AgentRole(str, Enum):
    """Specialized agent roles in the reproduction workflow."""

    INFRASTRUCTURE = "infrastructure"
    MODEL_DATASET = "model_dataset"
    METHOD_IMPLEMENTATION = "method_implementation"
    EXPERIMENT_CONFIG = "experiment_config"
    EXPERIMENT_EXECUTION = "experiment_execution"
    METRICS_EVALUATION = "metrics_evaluation"
    RESULT_ANALYSIS = "result_analysis"
    REPORTING = "reporting"


class MultiAgentOrchestrator:
    """
    Orchestrates multiple specialized agents to reproduce a research paper.

    The orchestrator manages the workflow:
    1. Infrastructure Agent → Set up environment
    2. Model & Dataset Agent → Load models and datasets
    3. Method Implementation Agent → Implement paper methods
    4. Experiment Config Agent → Configure experiments
    5. Experiment Execution Agent → Run experiments
    6. Metrics & Evaluation Agent → Calculate metrics
    7. Result Analysis Agent → Analyze results
    8. Reporting Agent → Generate reports
    """

    def __init__(
        self,
        workspace: RemoteWorkspace,
        llm: LLM,
        task_name: str,
        paper_path: str = "/workspace/paper",
        submission_path: str = "/workspace/submission",
        shared_state_path: str = "/workspace/shared_state/shared_state.json",
    ):
        """
        Initialize the orchestrator.

        Args:
            workspace: The workspace where agents will operate
            llm: LLM configuration for agents
            task_name: Name of the task/paper
            paper_path: Path to paper files
            submission_path: Path for final submission
            shared_state_path: Path to shared state file for agent communication
        """
        self.workspace = workspace
        self.llm = llm
        self.task_name = task_name
        self.paper_path = paper_path
        self.submission_path = submission_path
        self.shared_state_path = shared_state_path

        # Initialize shared state
        self.shared_state: Dict[str, Any] = {
            "task_name": task_name,
            "paper_path": paper_path,
            "submission_path": submission_path,
            "agents_completed": [],
            "agent_outputs": {},
            "errors": [],
        }

        # Initialize agents
        tools = get_default_tools(enable_browser=True)
        self.agents: Dict[AgentRole, Any] = {
            AgentRole.INFRASTRUCTURE: InfrastructureAgent(
                workspace=workspace,
                agent=Agent(llm=llm, tools=tools),
                shared_state=self.shared_state,
            ),
            AgentRole.MODEL_DATASET: ModelDatasetAgent(
                workspace=workspace,
                agent=Agent(llm=llm, tools=tools),
                shared_state=self.shared_state,
            ),
            AgentRole.METHOD_IMPLEMENTATION: MethodImplementationAgent(
                workspace=workspace,
                agent=Agent(llm=llm, tools=tools),
                shared_state=self.shared_state,
            ),
            AgentRole.EXPERIMENT_CONFIG: ExperimentConfigAgent(
                workspace=workspace,
                agent=Agent(llm=llm, tools=tools),
                shared_state=self.shared_state,
            ),
            AgentRole.EXPERIMENT_EXECUTION: ExperimentExecutionAgent(
                workspace=workspace,
                agent=Agent(llm=llm, tools=tools),
                shared_state=self.shared_state,
            ),
            AgentRole.METRICS_EVALUATION: MetricsEvaluationAgent(
                workspace=workspace,
                agent=Agent(llm=llm, tools=tools),
                shared_state=self.shared_state,
            ),
            AgentRole.RESULT_ANALYSIS: ResultAnalysisAgent(
                workspace=workspace,
                agent=Agent(llm=llm, tools=tools),
                shared_state=self.shared_state,
            ),
            AgentRole.REPORTING: ReportingAgent(
                workspace=workspace,
                agent=Agent(llm=llm, tools=tools),
                shared_state=self.shared_state,
            ),
        }

    def save_shared_state(self) -> None:
        """Save shared state to file."""
        try:
            state_json = json.dumps(self.shared_state, indent=2)
            # Create directory first
            dir_path = os.path.dirname(self.shared_state_path)
            result = self.workspace.execute_command(f"mkdir -p {dir_path}")
            if result.exit_code != 0:
                logger.warning(f"Failed to create directory: {result.stderr}")
                return
            
            # Write state to file using Python for proper JSON handling
            # Use a Python script to write JSON safely
            import json as json_module
            state_json_str = json_module.dumps(self.shared_state)
            # Create a Python script that writes the JSON
            python_script = f"""import json
state = {repr(self.shared_state)}
with open('{self.shared_state_path}', 'w') as f:
    json.dump(state, f, indent=2)
"""
            # Write Python script to temp file and execute it
            result = self.workspace.execute_command(
                f'''python3 << 'PYTHONSCRIPT'
{python_script}
PYTHONSCRIPT'''
            )
            # Fallback: try using echo with base64 encoding if Python script fails
            if result.exit_code != 0:
                import base64
                state_b64 = base64.b64encode(state_json_str.encode()).decode()
                result = self.workspace.execute_command(
                    f"echo '{state_b64}' | base64 -d > {self.shared_state_path}"
                )
            if result.exit_code != 0:
                logger.warning(f"Failed to save shared state: {result.stderr}")
        except Exception as e:
            logger.warning(f"Failed to save shared state: {e}")

    def load_shared_state(self) -> None:
        """Load shared state from file."""
        try:
            result = self.workspace.execute_command(f"cat {self.shared_state_path} 2>/dev/null || echo '{{}}'")
            if result.exit_code == 0 and result.stdout:
                try:
                    loaded_state = json.loads(result.stdout)
                    self.shared_state.update(loaded_state)
                except json.JSONDecodeError as e:
                    logger.warning(f"Failed to parse shared state JSON: {e}")
        except Exception as e:
            logger.warning(f"Failed to load shared state: {e}")

    async def run(
        self,
        instructions_path: str = "/workspace/instructions.md",
        rubric_path: Optional[str] = None,
        max_iterations_per_agent: int = 50,
    ) -> Dict[str, Any]:
        """
        Run the multi-agent reproduction workflow.

        Args:
            instructions_path: Path to task instructions
            rubric_path: Optional path to rubric.json
            max_iterations_per_agent: Maximum iterations per agent

        Returns:
            Dictionary with workflow results and agent outputs
        """
        logger.info(f"Starting multi-agent reproduction for task: {self.task_name}")

        # Load initial shared state if exists
        self.load_shared_state()

        # Read instructions
        try:
            result = self.workspace.execute_command(f"cat {instructions_path}")
            instructions = result.stdout if result.exit_code == 0 else ""
        except Exception as e:
            logger.error(f"Failed to read instructions: {e}")
            instructions = ""

        # Load rubric if provided
        rubric = None
        if rubric_path:
            try:
                result = self.workspace.execute_command(f"cat {rubric_path}")
                if result.exit_code == 0:
                    rubric = json.loads(result.stdout)
            except Exception as e:
                logger.warning(f"Failed to load rubric: {e}")

        # Define workflow sequence
        workflow = [
            AgentRole.INFRASTRUCTURE,
            AgentRole.MODEL_DATASET,
            AgentRole.METHOD_IMPLEMENTATION,
            AgentRole.EXPERIMENT_CONFIG,
            AgentRole.EXPERIMENT_EXECUTION,
            AgentRole.METRICS_EVALUATION,
            AgentRole.RESULT_ANALYSIS,
            AgentRole.REPORTING,
        ]

        # Track callback history to prevent infinite loops
        callback_count = {}  # agent_role -> count
        max_callbacks_per_agent = 3
        max_total_callbacks = 10
        total_callbacks = 0
        
        # Initialize callback counts
        for role in workflow:
            callback_count[role] = 0

        # Execute workflow with callback support
        i = 0
        while i < len(workflow):
            agent_role = workflow[i]
            
            try:
                logger.info(f"Running agent: {agent_role.value} (run #{callback_count[agent_role] + 1})")
                agent = self.agents[agent_role]

                # Create context for agent
                context = self._build_agent_context(
                    agent_role, instructions, rubric, workflow
                )

                # Run agent
                result = await agent.run(
                    context=context, max_iterations=max_iterations_per_agent
                )

                # Update shared state
                self.shared_state["agent_outputs"][agent_role.value] = result
                if agent_role.value not in self.shared_state["agents_completed"]:
                    self.shared_state["agents_completed"].append(agent_role.value)
                self.save_shared_state()

                # Check if agent requests a callback to another agent
                if result.get("callback_agent"):
                    callback_target = result.get("callback_agent")
                    callback_reason = result.get("callback_reason", "No reason provided")
                    
                    # Find the target agent role
                    target_role = None
                    for role in workflow:
                        if role.value == callback_target:
                            target_role = role
                            break
                    
                    if target_role and total_callbacks < max_total_callbacks:
                        if callback_count[target_role] < max_callbacks_per_agent:
                            logger.info(f"🔄 Agent {agent_role.value} requests callback to {callback_target}")
                            logger.info(f"   Reason: {callback_reason}")
                            
                            # Insert the callback agent right after current position
                            workflow.insert(i + 1, target_role)
                            callback_count[target_role] += 1
                            total_callbacks += 1
                            
                            # Log callback event
                            self.shared_state["callback_events"] = self.shared_state.get("callback_events", [])
                            self.shared_state["callback_events"].append({
                                "from_agent": agent_role.value,
                                "to_agent": callback_target,
                                "reason": callback_reason,
                                "callback_number": callback_count[target_role]
                            })
                            self.save_shared_state()
                        else:
                            logger.warning(f"⚠️  Agent {callback_target} has reached max callbacks ({max_callbacks_per_agent}), ignoring request")
                    elif total_callbacks >= max_total_callbacks:
                        logger.warning(f"⚠️  Max total callbacks ({max_total_callbacks}) reached, ignoring request")
                    elif not target_role:
                        logger.warning(f"⚠️  Unknown callback target: {callback_target}")

                # Check if agent indicates we should stop
                if result.get("status") == "error" and result.get("fatal", False):
                    logger.error(
                        f"Fatal error in {agent_role.value}, stopping workflow"
                    )
                    break

                # Move to next agent
                i += 1

            except Exception as e:
                logger.error(f"Error running {agent_role.value}: {e}")
                self.shared_state["errors"].append(
                    {"agent": agent_role.value, "error": str(e)}
                )
                self.save_shared_state()
                i += 1

        # Finalize submission
        await self._finalize_submission()

        logger.info("Multi-agent reproduction workflow completed")
        return self.shared_state

    def _build_agent_context(
        self,
        agent_role: AgentRole,
        instructions: str,
        rubric: Optional[Dict],
        workflow: List[AgentRole],
    ) -> Dict[str, Any]:
        """Build context for an agent based on its role and workflow state."""
        # Get previous agent outputs
        previous_outputs = {}
        for prev_role in workflow:
            if prev_role == agent_role:
                break
            if prev_role.value in self.shared_state["agent_outputs"]:
                previous_outputs[prev_role.value] = self.shared_state[
                    "agent_outputs"
                ][prev_role.value]

        context = {
            "task_name": self.task_name,
            "instructions": instructions,
            "rubric": rubric,
            "paper_path": self.paper_path,
            "submission_path": self.submission_path,
            "previous_agent_outputs": previous_outputs,
            "shared_state": self.shared_state,
        }

        # Add role-specific context
        if agent_role == AgentRole.INFRASTRUCTURE:
            context["task"] = "Set up Python environment, install dependencies, configure GPU access"
        elif agent_role == AgentRole.MODEL_DATASET:
            context["task"] = "Load pre-trained models and set up datasets"
        elif agent_role == AgentRole.METHOD_IMPLEMENTATION:
            context["task"] = "Implement research methods from the paper"
        elif agent_role == AgentRole.EXPERIMENT_CONFIG:
            context["task"] = "Configure experiments and hyperparameters"
        elif agent_role == AgentRole.EXPERIMENT_EXECUTION:
            context["task"] = "Run experiments and train models"
        elif agent_role == AgentRole.METRICS_EVALUATION:
            context["task"] = "Calculate evaluation metrics"
        elif agent_role == AgentRole.RESULT_ANALYSIS:
            context["task"] = "Analyze results and validate against paper"
        elif agent_role == AgentRole.REPORTING:
            context["task"] = "Generate reports and documentation"

        return context

    async def _finalize_submission(self) -> None:
        """Finalize the submission by creating reproduce.sh and README.md."""
        logger.info("Finalizing submission")

        # Create reproduce.sh script
        reproduce_script = """#!/bin/bash
set -e

# Reproduction script for paper reproduction
# This script should reproduce all results from the paper

echo "Starting reproduction..."

# Add reproduction commands here based on agent outputs
# This will be populated by the agents during the workflow

echo "Reproduction completed!"
"""

        try:
            # Create directory
            result = self.workspace.execute_command(f"mkdir -p {self.submission_path}")
            if result.exit_code != 0:
                logger.warning(f"Failed to create submission directory: {result.stderr}")
                return
            
            # Create reproduce.sh using heredoc
            result = self.workspace.execute_command(
                f'''cat > {self.submission_path}/reproduce.sh << 'REPRODUCE_EOF'
{reproduce_script}
REPRODUCE_EOF
chmod +x {self.submission_path}/reproduce.sh'''
            )
            if result.exit_code != 0:
                logger.warning(f"Failed to create reproduce.sh: {result.stderr}")
        except Exception as e:
            logger.warning(f"Failed to create reproduce.sh: {e}")

        # Create README.md
        readme_content = f"""# Paper Reproduction: {self.task_name}

This repository contains the reproduction of the research paper.

## What was accomplished

This reproduction was completed using a multi-agent system with the following agents:
{', '.join(self.shared_state.get('agents_completed', []))}

## Running the reproduction

Run the reproduction script:

```bash
bash reproduce.sh
```

## Agent Outputs

See shared_state.json for detailed outputs from each agent.
"""

        try:
            # Create README.md using heredoc
            result = self.workspace.execute_command(
                f'''cat > {self.submission_path}/README.md << 'README_EOF'
{readme_content}
README_EOF'''
            )
            if result.exit_code != 0:
                logger.warning(f"Failed to create README.md: {result.stderr}")
        except Exception as e:
            logger.warning(f"Failed to create README.md: {e}")

