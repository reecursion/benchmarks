"""
PaperBench inference with single-agent approach.

This module implements the inference phase using a SINGLE agent
to reproduce research papers, as opposed to the multi-agent orchestrator.
"""

import argparse
import asyncio
import json
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Optional

from openhands.sdk import LLM, get_logger
from openhands.sdk.workspace import RemoteWorkspace
from openhands.workspace import DockerWorkspace

from benchmarks.paper_bench.single_agent.runner import SingleAgentRunner
from benchmarks.utils.args_parser import get_parser
from benchmarks.utils.build_utils import build_image
from benchmarks.utils.constants import EVAL_AGENT_SERVER_IMAGE
from benchmarks.utils.evaluation_utils import construct_eval_output_dir
from benchmarks.utils.version import SDK_SHORT_SHA

logger = get_logger(__name__)

# Image configuration (same as multi-agent)
BASE_IMAGE = "leandermaben7/pb-env:1.0.0"
BUILD_TARGET = "source"


def extract_custom_tag(base_image: str) -> str:
    """Extract custom tag from base image name."""
    return base_image.replace("/", "_s_").replace(":", "_tag_")


def get_config_for_task(
    task_name: str,
    base_image: str = BASE_IMAGE,
    workspace_dir: str = "/workspace",
    enable_gpu: bool = False,
) -> DockerWorkspace:
    """Create workspace - builds image with --platform linux/amd64."""
    logger.info(f"Building agent-server image for task: {task_name}")
    
    custom_tag = extract_custom_tag(base_image)
    agent_server_image = f"{EVAL_AGENT_SERVER_IMAGE}:{SDK_SHORT_SHA}-{custom_tag}-{BUILD_TARGET}"
    
    output = build_image(
        base_image=base_image,
        target_image=EVAL_AGENT_SERVER_IMAGE,
        custom_tag=custom_tag,
        target=BUILD_TARGET,
        push=False,
    )
    
    if output.error:
        raise RuntimeError(f"Build failed: {output.error}")
    
    return DockerWorkspace(
        server_image=agent_server_image,
        working_dir=workspace_dir,
        enable_gpu=enable_gpu,
    )


def init_task_environment(
    workspace: RemoteWorkspace, 
    task_name: str, 
    instructions_path: str,
    paper_cache_dir: Optional[str] = None
) -> None:
    """
    Initialize the task environment by setting up paper files.
    
    This is a simplified version of the multi-agent init that doesn't need
    shared state directories.
    """
    logger.info(f"Initializing task environment for: {task_name}")

    # Create directories
    workspace.execute_command("mkdir -p /workspace/paper /workspace/submission")
    
    # Clear paper directory if it exists
    workspace.execute_command("rm -rf /workspace/paper/* /workspace/paper/.* 2>/dev/null || true")
    workspace.execute_command("mkdir -p /workspace/paper")

    # Check for local paper cache
    local_paper_path = None
    if paper_cache_dir:
        candidate = os.path.join(paper_cache_dir, task_name)
        if os.path.exists(candidate) and os.path.isdir(candidate):
            local_paper_path = candidate
            logger.info(f"📁 Found local paper cache at: {local_paper_path}")
    
    # Also check default cache location
    if not local_paper_path:
        default_cache = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))), 
            "paper_cache", 
            task_name
        )
        if os.path.exists(default_cache) and os.path.isdir(default_cache):
            local_paper_path = default_cache
            logger.info(f"📁 Found paper in default cache: {local_paper_path}")

    if local_paper_path:
        # Upload files from local cache
        logger.info(f"Using pre-downloaded paper files from: {local_paper_path}")
        
        for root, dirs, files in os.walk(local_paper_path):
            rel_root = os.path.relpath(root, local_paper_path)
            dest_root = f"/workspace/paper/{rel_root}" if rel_root != "." else "/workspace/paper"
            
            # Create subdirectories
            if rel_root != ".":
                workspace.execute_command(f"mkdir -p {dest_root}")
            
            for file in files:
                # Skip rubric.json - agents shouldn't see evaluation criteria
                if file == "rubric.json":
                    continue
                    
                src_path = os.path.join(root, file)
                dest_path = f"{dest_root}/{file}"
                
                try:
                    result = workspace.file_upload(src_path, dest_path)
                    if not result.success:
                        logger.warning(f"Failed to upload {file}: {result}")
                except Exception as e:
                    logger.warning(f"Upload failed for {file}: {e}")
                    # Fallback to base64
                    try:
                        import base64
                        with open(src_path, 'rb') as f:
                            content = f.read()
                        b64 = base64.b64encode(content).decode('ascii')
                        workspace.execute_command(f"echo '{b64}' | base64 -d > {dest_path}")
                    except Exception as e2:
                        logger.error(f"Base64 fallback also failed for {file}: {e2}")
    else:
        # Fall back to git clone approach
        logger.warning("No local paper cache found, attempting git clone...")
        clone_cmd = f"""cd /tmp && \\
rm -rf frontier-evals 2>/dev/null || true && \\
git clone --depth 1 --filter=blob:none --sparse https://github.com/openai/frontier-evals.git && \\
cd frontier-evals && \\
git sparse-checkout set project/paperbench/data/papers/{task_name} && \\
git lfs pull --include="project/paperbench/data/papers/{task_name}/**" && \\
cp -r project/paperbench/data/papers/{task_name}/* /workspace/paper/ && \\
rm -f /workspace/paper/rubric.json 2>&1 || true
"""
        result = workspace.execute_command(clone_cmd, timeout=300)
        if result.exit_code != 0:
            logger.error(f"Failed to clone paper: {result.stderr}")

    # Verify files
    file_count_result = workspace.execute_command("ls /workspace/paper | wc -l")
    if file_count_result.exit_code == 0:
        file_count = int(file_count_result.stdout.strip()) if file_count_result.stdout.strip().isdigit() else 0
        logger.info(f"📁 Files in /workspace/paper: {file_count}")
        
        # List files
        ls_result = workspace.execute_command("ls -la /workspace/paper/")
        if ls_result.stdout:
            logger.info(f"Paper files:\n{ls_result.stdout}")

    # Copy instructions if provided
    if instructions_path and os.path.exists(instructions_path):
        with open(instructions_path, "r") as f:
            instructions_content = f.read()
        
        # Modify instructions for single-agent mode
        single_agent_note = """
---
NOTE: You are running in SINGLE-AGENT mode. You must handle all tasks yourself:
- Environment setup
- Model and dataset loading  
- Method implementation (main AND baselines)
- Experiment configuration and execution
- Metrics evaluation
- Documentation and submission

Work efficiently - you have limited iterations to complete everything.
---

"""
        instructions_content = single_agent_note + instructions_content
        
        # Remove multi-agent section if present
        if "MULTI-AGENT SYSTEM" in instructions_content:
            # Keep everything except the multi-agent section
            parts = instructions_content.split("MULTI-AGENT SYSTEM")
            if len(parts) > 1:
                # Find the next section (starts with all caps line)
                rest = parts[1]
                lines = rest.split('\n')
                for i, line in enumerate(lines):
                    if line.strip() and line.strip().isupper() and not line.startswith('-'):
                        rest = '\n'.join(lines[i:])
                        break
                else:
                    rest = ""
                instructions_content = parts[0] + rest
        
        result = workspace.execute_command(
            f'''cat > /workspace/instructions.md << 'INSTRUCTIONSEOF'
{instructions_content}
INSTRUCTIONSEOF'''
        )
        if result.exit_code == 0:
            logger.info("Copied instructions to /workspace/instructions.md")

    logger.info("Task environment initialized successfully")


async def run_single_agent_inference(
    workspace: RemoteWorkspace,
    llm: LLM,
    task_name: str,
    max_iterations: int = 500,
) -> dict:
    """
    Run single-agent inference for a paper reproduction task.

    Args:
        workspace: The workspace where the agent will operate
        llm: LLM configuration
        task_name: Name of the task/paper
        max_iterations: Maximum iterations for the agent

    Returns:
        Dictionary with results
    """
    logger.info(f"Starting single-agent inference for task: {task_name}")
    logger.info(f"Max iterations: {max_iterations}")

    # Initialize runner
    runner = SingleAgentRunner(
        workspace=workspace,
        llm=llm,
        task_name=task_name,
        paper_path="/workspace/paper",
        submission_path="/workspace/submission",
    )

    # Run
    result = await runner.run(
        instructions_path="/workspace/instructions.md",
        max_iterations=max_iterations,
    )

    logger.info(f"Single-agent inference completed for task: {task_name}")
    return result


def extract_submission(workspace: RemoteWorkspace, submission_dir: str, task_name: str) -> None:
    """Extract submission from workspace to local directory."""
    logger.info(f"Extracting submission for task: {task_name}")
    
    local_submission_path = os.path.join(submission_dir, task_name)
    os.makedirs(local_submission_path, exist_ok=True)
    
    # Get list of files
    result = workspace.execute_command("find /workspace/submission -type f")
    if result.exit_code != 0:
        logger.error(f"Failed to list submission files: {result.stderr}")
        return
    
    files = result.stdout.strip().split('\n') if result.stdout.strip() else []
    logger.info(f"Found {len(files)} files to extract")
    
    for remote_path in files:
        if not remote_path.strip():
            continue
            
        rel_path = remote_path.replace("/workspace/submission/", "")
        local_path = os.path.join(local_submission_path, rel_path)
        
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        
        try:
            # Try to download file
            content_result = workspace.execute_command(f"cat '{remote_path}'")
            if content_result.exit_code == 0:
                with open(local_path, 'w') as f:
                    f.write(content_result.stdout)
        except Exception as e:
            logger.warning(f"Failed to extract {rel_path}: {e}")
    
    logger.info(f"Submission extracted to: {local_submission_path}")


def main():
    """Main entry point for single-agent PaperBench inference."""
    parser = argparse.ArgumentParser(
        description="Run single-agent PaperBench inference",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
  python -m benchmarks.paper_bench.single_agent.run_infer \\
    --task-name bam \\
    --llm-config-path configs/llm/gpt5_mini.json \\
    --max-iterations 500 \\
    --output-dir ./outputs/single_agent_run_1
"""
    )
    
    parser.add_argument(
        "--task-name",
        type=str,
        required=True,
        help="Task name(s) to run, comma-separated (e.g., 'bam' or 'bam,pinn,rice')",
    )
    parser.add_argument(
        "--llm-config-path",
        type=str,
        required=True,
        help="Path to LLM configuration JSON file",
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=500,
        help="Maximum iterations for the agent (default: 500)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./outputs/single_agent",
        help="Base directory for outputs",
    )
    parser.add_argument(
        "--submission-dir",
        type=str,
        default="./outputs/submissions_single_agent",
        help="Directory to save submissions",
    )
    parser.add_argument(
        "--base-image",
        type=str,
        default=BASE_IMAGE,
        help=f"Base Docker image (default: {BASE_IMAGE})",
    )
    parser.add_argument(
        "--enable-gpu",
        action="store_true",
        help="Enable GPU support",
    )
    parser.add_argument(
        "--paper-cache-dir",
        type=str,
        default=None,
        help="Directory containing pre-downloaded paper files",
    )
    parser.add_argument(
        "--instructions-path",
        type=str,
        default="benchmarks/paper_bench/instructions.md",
        help="Path to instructions file",
    )
    parser.add_argument(
        "--note",
        type=str,
        default=None,
        help="Optional note for output directory naming",
    )
    parser.add_argument(
        "--log-completions",
        action="store_true",
        help="Enable logging of LLM API calls",
    )

    args = parser.parse_args()

    # Load LLM config
    if not os.path.isfile(args.llm_config_path):
        raise ValueError(f"LLM config file {args.llm_config_path} does not exist")
    
    with open(args.llm_config_path, "r") as f:
        llm_config = f.read()
    llm = LLM.model_validate_json(llm_config)
    
    if args.log_completions:
        llm.log_completions = True
        llm.log_completions_folder = "/workspace/logs/completions"
    
    logger.info(f"Using LLM: {llm.model}")

    # Parse task names
    task_names = [t.strip() for t in args.task_name.split(",") if t.strip()]
    if not task_names:
        raise ValueError("No valid task names provided")
    
    logger.info(f"Will run single-agent inference for {len(task_names)} task(s): {task_names}")

    # Resolve instructions path
    instructions_path = args.instructions_path
    if not os.path.isabs(instructions_path):
        project_root = Path(__file__).parent.parent.parent.parent
        potential_path = project_root / instructions_path
        if potential_path.exists():
            instructions_path = str(potential_path)
        elif not os.path.exists(instructions_path):
            # Try relative to paper_bench
            script_dir = Path(__file__).parent.parent
            potential_path = script_dir / "instructions.md"
            if potential_path.exists():
                instructions_path = str(potential_path)

    # Track results
    all_results = {}
    
    # Run inference for each task
    for task_idx, task_name in enumerate(task_names, 1):
        logger.info(f"\n{'='*60}")
        logger.info(f"Starting task {task_idx}/{len(task_names)}: {task_name} (SINGLE-AGENT)")
        logger.info(f"{'='*60}")
        
        # Construct output directory
        dataset_description = f"paperbench-{task_name}"
        note = f"single_agent_{args.note}" if args.note else "single_agent"
        task_output_dir = construct_eval_output_dir(
            base_dir=args.output_dir,
            dataset_name=dataset_description,
            model_name=llm.model,
            max_iterations=args.max_iterations,
            eval_note=note,
        )
        
        # Create workspace
        workspace = get_config_for_task(
            task_name=task_name,
            base_image=args.base_image,
            enable_gpu=args.enable_gpu,
        )

        try:
            with workspace:
                # Initialize environment
                init_task_environment(
                    workspace, 
                    task_name, 
                    instructions_path,
                    paper_cache_dir=args.paper_cache_dir
                )

                # Run single-agent inference
                result = asyncio.run(run_single_agent_inference(
                    workspace=workspace,
                    llm=llm,
                    task_name=task_name,
                    max_iterations=args.max_iterations,
                ))

                # Extract submission
                extract_submission(workspace, args.submission_dir, task_name)

                # Save results
                results_file = os.path.join(task_output_dir, "results.json")
                os.makedirs(task_output_dir, exist_ok=True)
                with open(results_file, "w") as f:
                    json.dump(result, f, indent=2)

                logger.info(f"Results saved to: {results_file}")
                
                all_results[task_name] = {
                    "status": "success",
                    "result": result,
                    "output_dir": task_output_dir,
                }
                
        except Exception as e:
            logger.error(f"Task {task_name} failed with error: {e}")
            all_results[task_name] = {
                "status": "error",
                "error": str(e),
            }
            continue
        
        logger.info(f"Completed task {task_idx}/{len(task_names)}: {task_name}")

    # Summary
    logger.info(f"\n{'='*60}")
    logger.info("All tasks completed!")
    logger.info(f"{'='*60}")
    for task_name, task_result in all_results.items():
        status = task_result.get('status', 'unknown')
        if status == 'success':
            iterations = task_result.get('result', {}).get('iterations_used', 'N/A')
            logger.info(f"  ✅ {task_name}: {status} (iterations: {iterations}) -> {task_result.get('output_dir', 'N/A')}")
        else:
            logger.info(f"  ❌ {task_name}: {status} - {task_result.get('error', 'Unknown error')}")


if __name__ == "__main__":
    main()

