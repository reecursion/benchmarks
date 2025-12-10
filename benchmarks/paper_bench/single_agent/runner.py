"""
Single-agent runner for PaperBench benchmark.

This module provides a simple single-agent approach to paper reproduction,
as opposed to the multi-agent orchestrator approach.
"""

import json
from typing import Any, Dict, Optional

from openhands.sdk import Agent, Conversation, LLM, get_logger
from openhands.sdk.workspace import RemoteWorkspace
from openhands.tools.preset.default import get_default_tools

from benchmarks.paper_bench.single_agent.prompts import (
    SINGLE_AGENT_SYSTEM_PROMPT,
    get_single_agent_instruction,
)

logger = get_logger(__name__)


class SingleAgentRunner:
    """
    Single-agent runner for PaperBench paper reproduction.
    
    Unlike the multi-agent orchestrator, this runner uses a single agent
    to handle the entire paper reproduction task from start to finish.
    """

    def __init__(
        self,
        workspace: RemoteWorkspace,
        llm: LLM,
        task_name: str,
        paper_path: str = "/workspace/paper",
        submission_path: str = "/workspace/submission",
    ):
        """
        Initialize the single-agent runner.

        Args:
            workspace: The workspace where the agent will operate
            llm: LLM configuration for the agent
            task_name: Name of the task/paper
            paper_path: Path to paper files
            submission_path: Path for final submission
        """
        self.workspace = workspace
        self.llm = llm
        self.task_name = task_name
        self.paper_path = paper_path
        self.submission_path = submission_path
        
        # Store results
        self.result: Dict[str, Any] = {
            "task_name": task_name,
            "status": "not_started",
            "iterations_used": 0,
            "errors": [],
        }

    def _setup_workspace(self) -> None:
        """Set up the workspace directories and initial files."""
        logger.info("Setting up workspace for single-agent run")
        
        # Ensure directories exist
        self.workspace.execute_command(
            f"mkdir -p {self.submission_path}"
        )
        
        # Initialize git repo in submission directory
        result = self.workspace.execute_command(
            f"cd {self.submission_path} && git init && git config user.email 'agent@paperbench.ai' && git config user.name 'PaperBench Agent'"
        )
        if result.exit_code != 0:
            logger.warning(f"Git init warning: {result.stderr}")
        
        # Create initial README
        self.workspace.execute_command(
            f"""cat > {self.submission_path}/README.md << 'EOF'
# Paper Reproduction

This repository contains the reproduction of the research paper.

## Status
In progress...

## How to Run
```bash
bash reproduce.sh
```
EOF"""
        )
        
        # Create initial reproduce.sh
        self.workspace.execute_command(
            f"""cat > {self.submission_path}/reproduce.sh << 'EOF'
#!/bin/bash
# Paper Reproduction Script
# This script should set up the environment and run all experiments

set -e

echo "Starting paper reproduction..."

# Install dependencies
if [ -f requirements.txt ]; then
    pip install -r requirements.txt
fi

# TODO: Add experiment execution commands here

echo "Reproduction complete!"
EOF"""
        )
        self.workspace.execute_command(f"chmod +x {self.submission_path}/reproduce.sh")
        
        logger.info("Workspace setup complete")

    async def run(
        self,
        instructions_path: str = "/workspace/instructions.md",
        max_iterations: int = 500,
    ) -> Dict[str, Any]:
        """
        Run the single-agent paper reproduction.

        Args:
            instructions_path: Path to instructions file
            max_iterations: Maximum iterations for the agent

        Returns:
            Dictionary with results
        """
        logger.info(f"Starting single-agent run for task: {self.task_name}")
        logger.info(f"Max iterations: {max_iterations}")
        
        # Setup workspace
        self._setup_workspace()
        
        # Create the agent (no system_prompt in constructor - it goes in the message)
        agent = Agent(
            llm=self.llm,
            tools=get_default_tools(),
        )
        
        # Build the full instruction with system prompt included
        instruction = get_single_agent_instruction(
            paper_path=self.paper_path,
            submission_path=self.submission_path,
        )
        
        # Combine system prompt with instruction (same pattern as multi-agent)
        full_instruction = f"""{SINGLE_AGENT_SYSTEM_PROMPT}

{instruction}

Please work on this task step by step. Document your progress and save any outputs to the appropriate locations.
"""
        
        # Track events
        events_received = []
        
        def event_callback(event):
            events_received.append(event)
            if len(events_received) % 50 == 0:
                logger.info(f"Progress: {len(events_received)} events received")
        
        # Create conversation (OpenHands SDK style)
        conversation = Conversation(
            agent=agent,
            workspace=self.workspace,
            callbacks=[event_callback],
            max_iteration_per_run=max_iterations,
        )
        
        self.result["status"] = "running"
        
        try:
            logger.info("Starting agent execution")
            
            # Send instruction and run
            conversation.send_message(full_instruction)
            conversation.run()
            
            # Get iteration count from conversation state
            iterations_used = len(conversation.state.events) if hasattr(conversation, 'state') else len(events_received)
            
            self.result["status"] = "completed"
            self.result["iterations_used"] = iterations_used
            
            logger.info(f"Agent finished after {iterations_used} events")
            
        except Exception as e:
            logger.error(f"Agent execution error: {e}")
            self.result["status"] = "error"
            self.result["errors"].append(str(e))
            self.result["iterations_used"] = len(events_received)
        
        # Finalize submission
        await self._finalize_submission()
        
        logger.info(f"Single-agent run completed with status: {self.result['status']}")
        return self.result

    async def _finalize_submission(self) -> None:
        """Finalize the submission by committing and cleaning up."""
        logger.info("Finalizing submission")
        
        # Commit all changes
        commit_result = self.workspace.execute_command(
            f"cd {self.submission_path} && git add -A && git commit -m 'Final submission' --allow-empty"
        )
        if commit_result.exit_code != 0:
            logger.warning(f"Git commit warning: {commit_result.stderr}")
        
        # Clean up any large files that shouldn't be in submission
        cleanup_cmds = [
            f"find {self.submission_path} -name '*.pt' -delete 2>/dev/null || true",
            f"find {self.submission_path} -name '*.pth' -delete 2>/dev/null || true",
            f"find {self.submission_path} -name '*.ckpt' -delete 2>/dev/null || true",
            f"find {self.submission_path} -name '*.safetensors' -delete 2>/dev/null || true",
            f"find {self.submission_path} -name '__pycache__' -type d -exec rm -rf {{}} + 2>/dev/null || true",
            f"find {self.submission_path} -name '.cache' -type d -exec rm -rf {{}} + 2>/dev/null || true",
            f"find {self.submission_path} -name 'wandb' -type d -exec rm -rf {{}} + 2>/dev/null || true",
        ]
        for cmd in cleanup_cmds:
            self.workspace.execute_command(cmd)
        
        # Get submission stats
        size_result = self.workspace.execute_command(f"du -sh {self.submission_path}")
        files_result = self.workspace.execute_command(f"find {self.submission_path} -type f | wc -l")
        
        if size_result.exit_code == 0:
            self.result["submission_size"] = size_result.stdout.strip().split()[0]
        if files_result.exit_code == 0:
            self.result["file_count"] = int(files_result.stdout.strip())
        
        # List files in submission
        ls_result = self.workspace.execute_command(f"ls -la {self.submission_path}")
        if ls_result.exit_code == 0:
            logger.info(f"Submission contents:\n{ls_result.stdout}")
        
        logger.info("Submission finalized")

