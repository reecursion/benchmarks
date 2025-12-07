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

        # Initialize shared state with enhanced memory structure
        self.shared_state: Dict[str, Any] = {
            "task_name": task_name,
            "paper_path": paper_path,
            "submission_path": submission_path,
            "agents_completed": [],
            "agent_outputs": {},
            "errors": [],
            # Enhanced shared memory for cross-agent context
            "paper_context": {
                "title": "",
                "summary": "",
                "dependencies": [],  # List of required packages
                "models": [],  # Models mentioned in paper
                "datasets": [],  # Datasets mentioned in paper
                "methods": [],  # Key methods to implement
                "experiments": [],  # Experiments to reproduce
                "metrics": [],  # Metrics to calculate
                "key_results": [],  # Key results from paper tables/figures
                # NEW: Track all algorithms for completeness checking
                "all_algorithms": [],  # ALL algorithms mentioned in paper (main + baselines)
                "implemented_algorithms": [],  # Algorithms actually implemented
                "missing_algorithms": [],  # Algorithms not implemented (with reasons)
                "baseline_algorithms": [],  # Specifically baseline/comparison methods
            },
            "files_created": {},  # Dict mapping agent_name -> list of files created
            "reproduce_steps": [],  # Ordered list of commands for reproduce.sh
            # NEW: Track implementation completeness
            "implementation_status": {
                "main_method": None,
                "baselines_required": [],
                "baselines_implemented": [],
                "completion_percentage": 0,
            },
            # NEW: Track critical failures for recovery
            "infrastructure_failed": False,
            "critical_failure": None,
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

    def format_shared_context(self) -> str:
        """Format the shared context as a string for agent prompts."""
        lines = []
        
        paper_context = self.shared_state.get("paper_context", {})
        files_created = self.shared_state.get("files_created", {})
        reproduce_steps = self.shared_state.get("reproduce_steps", [])
        
        # Paper context section
        if paper_context.get("summary"):
            lines.append("## Paper Summary (from previous agents)")
            lines.append(paper_context["summary"])
            lines.append("")
        
        if paper_context.get("dependencies"):
            lines.append("## Known Dependencies")
            for dep in paper_context["dependencies"][:20]:  # Limit to avoid too long
                lines.append(f"  - {dep}")
            lines.append("")
        
        if paper_context.get("models"):
            lines.append("## Models to Use")
            for model in paper_context["models"][:10]:
                lines.append(f"  - {model}")
            lines.append("")
        
        if paper_context.get("datasets"):
            lines.append("## Datasets to Use")
            for dataset in paper_context["datasets"][:10]:
                lines.append(f"  - {dataset}")
            lines.append("")
        
        if paper_context.get("methods"):
            lines.append("## Methods to Implement")
            for method in paper_context["methods"][:10]:
                lines.append(f"  - {method}")
            lines.append("")
        
        if paper_context.get("experiments"):
            lines.append("## Experiments to Run")
            for exp in paper_context["experiments"][:10]:
                lines.append(f"  - {exp}")
            lines.append("")
        
        if paper_context.get("metrics"):
            lines.append("## Metrics to Calculate")
            for metric in paper_context["metrics"][:10]:
                lines.append(f"  - {metric}")
            lines.append("")
        
        # Files created by previous agents
        if files_created:
            lines.append("## Files Already Created (by previous agents)")
            for agent_name, files in files_created.items():
                if files:
                    lines.append(f"  {agent_name}:")
                    for f in files[:10]:  # Limit per agent
                        lines.append(f"    - {f}")
            lines.append("")
        
        # Reproduce steps so far
        if reproduce_steps:
            lines.append("## Reproduction Steps (add yours to this list)")
            for i, step in enumerate(reproduce_steps[:20], 1):
                lines.append(f"  {i}. {step}")
            lines.append("")
        
        return "\n".join(lines) if lines else ""

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

        # Retry configuration for transient errors
        max_retries = 3
        retry_delay_base = 5  # seconds

        # Execute workflow with callback support
        i = 0
        while i < len(workflow):
            agent_role = workflow[i]
            
            # Retry loop for transient errors
            last_error = None
            for retry_attempt in range(max_retries):
                try:
                    if retry_attempt > 0:
                        logger.info(f"Retry attempt {retry_attempt + 1}/{max_retries} for agent: {agent_role.value}")
                    else:
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

                    # Success - break out of retry loop
                    last_error = None
                    break

                except Exception as e:
                    last_error = e
                    error_str = str(e)
                    
                    # Check if this is a retryable error (500 errors, timeouts, connection issues)
                    is_retryable = any(x in error_str.lower() for x in [
                        '500', 'internal server error', 'timeout', 'timed out',
                        'connection', 'temporarily unavailable', 'service unavailable'
                    ])
                    
                    if is_retryable and retry_attempt < max_retries - 1:
                        # Calculate delay with exponential backoff
                        delay = retry_delay_base * (2 ** retry_attempt)
                        logger.warning(f"⚠️  Retryable error in {agent_role.value}: {error_str[:200]}")
                        logger.info(f"   Waiting {delay}s before retry...")
                        await asyncio.sleep(delay)
                    else:
                        # Non-retryable error or max retries reached
                        if retry_attempt >= max_retries - 1:
                            logger.error(f"❌ Max retries ({max_retries}) reached for {agent_role.value}")
                        logger.error(f"Error running {agent_role.value}: {e}")
                        self.shared_state["agent_outputs"][agent_role.value] = {
                            "status": "error",
                            "agent": agent.__class__.__name__,
                            "error": str(e),
                            "fatal": False,
                            "retries_attempted": retry_attempt + 1
                        }
                        self.shared_state["errors"].append(
                            {"agent": agent_role.value, "error": str(e), "retries": retry_attempt + 1}
                        )
                        self.save_shared_state()
                        break
            
            # Move to next agent
            i += 1

        # Check for critical failures and set recovery flags
        self._check_critical_failures()

        # Finalize submission
        await self._finalize_submission()

        logger.info("Multi-agent reproduction workflow completed")
        return self.shared_state

    def _check_critical_failures(self) -> None:
        """Check for critical agent failures and set appropriate flags for recovery."""
        critical_agents = [AgentRole.INFRASTRUCTURE, AgentRole.METHOD_IMPLEMENTATION]
        
        for critical_role in critical_agents:
            output = self.shared_state["agent_outputs"].get(critical_role.value, {})
            if output.get("status") == "error":
                logger.warning(f"⚠️  Critical agent {critical_role.value} failed - submission may be incomplete")
                self.shared_state["critical_failure"] = critical_role.value
                
                # For infrastructure failure, set recovery flag for subsequent agents
                if critical_role == AgentRole.INFRASTRUCTURE:
                    logger.info("Infrastructure failed - downstream agents will attempt recovery")
                    self.shared_state["infrastructure_failed"] = True
                    # Set default dependencies that most papers need
                    if not self.shared_state["paper_context"]["dependencies"]:
                        self.shared_state["paper_context"]["dependencies"] = [
                            "torch", "numpy", "scipy", "transformers", "datasets"
                        ]
                    self.save_shared_state()
                    
                # For method implementation failure, log what might be missing
                if critical_role == AgentRole.METHOD_IMPLEMENTATION:
                    logger.warning("Method implementation failed - reproduction will likely be incomplete")
                    self.shared_state["implementation_status"]["completion_percentage"] = 0

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

        # Format shared context for inclusion in prompts
        shared_context_str = self.format_shared_context()

        context = {
            "task_name": self.task_name,
            "instructions": instructions,
            "rubric": rubric,
            "paper_path": self.paper_path,
            "submission_path": self.submission_path,
            "previous_agent_outputs": previous_outputs,
            "shared_state": self.shared_state,
            "shared_context_summary": shared_context_str,  # New: formatted context string
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
        """Finalize the submission by creating reproduce.sh and README.md if they don't exist."""
        logger.info("Finalizing submission")

        # Fallback reproduce.sh script (only used if agents didn't create one)
        fallback_reproduce_script = """#!/bin/bash
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
            
            # Check if reproduce.sh already exists and has meaningful content
            check_result = self.workspace.execute_command(
                f"test -s {self.submission_path}/reproduce.sh && wc -l < {self.submission_path}/reproduce.sh"
            )
            reproduce_exists = check_result.exit_code == 0
            reproduce_lines = 0
            if reproduce_exists:
                try:
                    reproduce_lines = int(check_result.stdout.strip())
                except (ValueError, AttributeError):
                    reproduce_lines = 0
            
            # Only create reproduce.sh if it doesn't exist or has very little content (< 10 lines)
            if not reproduce_exists or reproduce_lines < 10:
                if reproduce_exists and reproduce_lines > 0:
                    logger.info(f"reproduce.sh exists but only has {reproduce_lines} lines, keeping agent version")
                else:
                    logger.info("No reproduce.sh found, creating fallback template")
                    result = self.workspace.execute_command(
                        f'''cat > {self.submission_path}/reproduce.sh << 'REPRODUCE_EOF'
{fallback_reproduce_script}
REPRODUCE_EOF
chmod +x {self.submission_path}/reproduce.sh'''
                    )
                    if result.exit_code != 0:
                        logger.warning(f"Failed to create reproduce.sh: {result.stderr}")
            else:
                logger.info(f"Keeping existing reproduce.sh ({reproduce_lines} lines)")
                # Just ensure it's executable
                self.workspace.execute_command(f"chmod +x {self.submission_path}/reproduce.sh")
                
        except Exception as e:
            logger.warning(f"Failed to handle reproduce.sh: {e}")

        # Fallback README.md content
        fallback_readme = f"""# Paper Reproduction: {self.task_name}

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
            # Check if README.md already exists and has meaningful content
            check_result = self.workspace.execute_command(
                f"test -s {self.submission_path}/README.md && wc -l < {self.submission_path}/README.md"
            )
            readme_exists = check_result.exit_code == 0
            readme_lines = 0
            if readme_exists:
                try:
                    readme_lines = int(check_result.stdout.strip())
                except (ValueError, AttributeError):
                    readme_lines = 0
            
            # Only create README.md if it doesn't exist or has very little content (< 10 lines)
            if not readme_exists or readme_lines < 10:
                if readme_exists and readme_lines > 0:
                    logger.info(f"README.md exists but only has {readme_lines} lines, keeping agent version")
                else:
                    logger.info("No README.md found, creating fallback template")
                    result = self.workspace.execute_command(
                        f'''cat > {self.submission_path}/README.md << 'README_EOF'
{fallback_readme}
README_EOF'''
                    )
                    if result.exit_code != 0:
                        logger.warning(f"Failed to create README.md: {result.stderr}")
            else:
                logger.info(f"Keeping existing README.md ({readme_lines} lines)")
                
        except Exception as e:
            logger.warning(f"Failed to handle README.md: {e}")

