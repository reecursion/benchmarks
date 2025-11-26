"""
PaperBench inference with multi-agent orchestration.

This module implements the inference phase using a multi-agent system
to reproduce research papers using the leandermaben7/pb-env:1.0.0 Docker image.
"""

import asyncio
import json
import os
import shutil
import tempfile
from pathlib import Path

from openhands.sdk import LLM, get_logger
from openhands.sdk.workspace import RemoteWorkspace
from openhands.workspace import DockerWorkspace

from benchmarks.paper_bench.orchestrator import MultiAgentOrchestrator
from benchmarks.utils.args_parser import get_parser
from benchmarks.utils.build_utils import build_image
from benchmarks.utils.constants import EVAL_AGENT_SERVER_IMAGE
from benchmarks.utils.evaluation_utils import construct_eval_output_dir
from benchmarks.utils.version import SDK_SHORT_SHA

logger = get_logger(__name__)

# Image configuration
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
    workspace: RemoteWorkspace, task_name: str, instructions_path: str
) -> None:
    """
    Initialize the task environment by setting up paper files and instructions.

    Args:
        workspace: The workspace to initialize
        task_name: Name of the task/paper
        instructions_path: Path to instructions file on host
    """
    logger.info(f"Initializing task environment for: {task_name}")

    # Create directories in workspace (which we have full control over)
    workspace.execute_command("mkdir -p /workspace/paper /workspace/submission /workspace/shared_state")
    
    # Clear paper directory if it exists (safer than rm -rf)
    workspace.execute_command("rm -rf /workspace/paper/* /workspace/paper/.* 2>/dev/null || true")
    workspace.execute_command("mkdir -p /workspace/paper")

    # Install git-lfs if not already installed (needed for cloning paper files)
    logger.info("Installing git-lfs...")
    install_cmd = """bash -c '
        if ! command -v git-lfs >/dev/null 2>&1; then
            sudo apt-get update -qq && \
            sudo apt-get install -y git-lfs && \
            git lfs install
        else
            echo "git-lfs already installed"
        fi
    '"""
    install_result = workspace.execute_command(install_cmd, timeout=60)
    if install_result.exit_code != 0:
        logger.warning(f"git-lfs installation had issues: {install_result.stderr}, continuing anyway...")
    else:
        logger.info("git-lfs ready")
    
    # Clone paper files with proper Git LFS handling
    logger.info(f"Cloning paper files for {task_name}...")
    
    # Step 1: Clone repository
    clone_cmd = f"""rm -rf /tmp/frontier-evals && \\
git clone --filter=blob:none --no-checkout https://github.com/openai/frontier-evals.git /tmp/frontier-evals
"""
    result = workspace.execute_command(clone_cmd)
    if result.exit_code != 0:
        logger.error(f"Failed to clone repository: {result.stderr}")
        raise RuntimeError(f"Failed to clone repository: {result.stderr}")
    
    # Step 2: Set up sparse checkout and checkout files
    checkout_cmd = f"""cd /tmp/frontier-evals && \\
git sparse-checkout init --cone && \\
git sparse-checkout set project/paperbench/data/papers/{task_name} && \\
git checkout main
"""
    result = workspace.execute_command(checkout_cmd)
    if result.exit_code != 0:
        logger.error(f"Failed to checkout files: {result.stderr}")
        raise RuntimeError(f"Failed to checkout files: {result.stderr}")
    
    # Step 3: Install and use Git LFS to pull actual file content (not pointers)
    # Increase timeout to 300 seconds for large files
    lfs_cmd = f"""cd /tmp/frontier-evals && \\
(git lfs install 2>&1 || true) && \\
(timeout 300 git lfs pull --include="project/paperbench/data/papers/{task_name}/*" 2>&1 || echo "LFS pull completed or not needed")
"""
    result = workspace.execute_command(lfs_cmd, timeout=150)
    if result.exit_code != 0:
        logger.warning(f"Git LFS pull had issues: {result.stderr}, continuing...")
    
    # Step 4: Copy files to workspace
    copy_cmd = f"""cp -r /tmp/frontier-evals/project/paperbench/data/papers/{task_name}/* /workspace/paper/ 2>&1
"""
    result = workspace.execute_command(copy_cmd)
    if result.exit_code != 0:
        logger.error(f"Failed to copy files: {result.stderr}")
        raise RuntimeError(f"Failed to copy files: {result.stderr}")

    # Check if paper.md has actual content or is an LFS pointer
    check_content = workspace.execute_command("head -5 /workspace/paper/paper.md 2>&1 || echo 'file not found'")
    logger.info(f"Checking paper.md content (first 5 lines):\n{check_content.stdout[:200]}")
    
    if "version https://git-lfs.github.com/spec/v1" in check_content.stdout:
        logger.warning("⚠️  paper.md is a Git LFS pointer! Attempting to download actual content...")
        # Try to get LFS files directly - use fetch then checkout (with longer timeout)
        lfs_fetch_cmd = f"""cd /tmp/frontier-evals && \\
timeout 120 git lfs fetch --all 2>&1 && \\
timeout 60 git lfs checkout 'project/paperbench/data/papers/{task_name}/*' 2>&1 && \\
cp -rf project/paperbench/data/papers/{task_name}/* /workspace/paper/ 2>&1
"""
        result = workspace.execute_command(lfs_fetch_cmd, timeout=200)
        logger.info(f"Git LFS fetch/checkout result (exit {result.exit_code}):")
        logger.info(f"  stdout: {result.stdout[-300:] if result.stdout else 'empty'}")
        if result.stderr:
            logger.warning(f"  stderr: {result.stderr[-300:]}")
        
        # Verify the file was replaced
        check_again = workspace.execute_command("head -10 /workspace/paper/paper.md 2>&1")
        if "version https://git-lfs.github.com/spec/v1" not in check_again.stdout:
            logger.info("✅ Git LFS files successfully downloaded - paper.md now has actual content")
        else:
            logger.warning("⚠️  Git LFS files still not downloaded - paper.md is still a pointer")
            logger.warning("The agent will work with available files (rubric.json, config.yaml, etc.)")
    
    # Final verification with detailed logging
    file_count_result = workspace.execute_command("ls /workspace/paper | wc -l")
    file_count = 0
    if file_count_result.exit_code == 0 and file_count_result.stdout.strip().isdigit():
        file_count = int(file_count_result.stdout.strip())
    
    # List all files with sizes
    file_list_cmd = workspace.execute_command("ls -lh /workspace/paper/ | tail -n +2")
    logger.info(f"📁 Files in /workspace/paper ({file_count} items):")
    if file_list_cmd.stdout:
        for line in file_list_cmd.stdout.strip().split('\n')[:15]:  # Show first 15 files
            logger.info(f"  {line}")
    
    if file_count == 0:
        logger.error(f"❌ No files found in /workspace/paper")
        raise RuntimeError(f"Failed to set up paper files: No files found")
    else:
        logger.info(f"✅ Paper files setup completed ({file_count} items)")
        
        # Check file sizes - LFS pointers are tiny (~100 bytes), real files are larger
        size_check = workspace.execute_command("du -sh /workspace/paper/* 2>&1 | head -10")
        if size_check.stdout:
            logger.info(f"📊 File sizes:")
            for line in size_check.stdout.strip().split('\n')[:10]:
                logger.info(f"  {line}")

    # Copy instructions
    if os.path.exists(instructions_path):
        # Read instructions file
        with open(instructions_path, "r") as f:
            instructions_content = f.read()

        # Write to workspace
        instructions_file = "/workspace/instructions.md"
        result = workspace.execute_command(f"mkdir -p $(dirname {instructions_file})")
        if result.exit_code == 0:
            # Write instructions using heredoc
            result = workspace.execute_command(
                f'''cat > {instructions_file} << 'INSTRUCTIONSEOF'
{instructions_content}
INSTRUCTIONSEOF'''
            )
            if result.exit_code == 0:
                logger.info(f"Copied instructions to {instructions_file}")
            else:
                logger.warning(f"Failed to write instructions: {result.stderr}")
        else:
            logger.warning(f"Failed to create instructions directory: {result.stderr}")
    else:
        logger.warning(f"Instructions file not found: {instructions_path}")

    logger.info("Task environment initialized successfully")


async def run_multi_agent_inference(
    workspace: RemoteWorkspace,
    llm: LLM,
    task_name: str,
    instructions_path: str = "/home/instructions.md",
    rubric_path: str | None = None,
    max_iterations_per_agent: int = 50,
) -> dict:
    """
    Run multi-agent inference for a paper reproduction task.

    Args:
        workspace: The workspace where agents will operate
        llm: LLM configuration for agents
        task_name: Name of the task/paper
        instructions_path: Path to instructions file
        rubric_path: Optional path to rubric.json
        max_iterations_per_agent: Maximum iterations per agent

    Returns:
        Dictionary with workflow results
    """
    logger.info(f"Starting multi-agent inference for task: {task_name}")

    # Initialize orchestrator
    orchestrator = MultiAgentOrchestrator(
        workspace=workspace,
        llm=llm,
        task_name=task_name,
        paper_path="/workspace/paper",
        submission_path="/workspace/submission",
        shared_state_path="/workspace/shared_state/shared_state.json",
    )

    # Run orchestrator
    result = await orchestrator.run(
        instructions_path=instructions_path,
        rubric_path=rubric_path,
        max_iterations_per_agent=max_iterations_per_agent,
    )

    logger.info(f"Multi-agent inference completed for task: {task_name}")
    return result


def extract_submission(
    workspace: RemoteWorkspace, submission_dir: str, task_name: str
) -> None:
    """
    Extract submission from workspace to host.

    Args:
        workspace: The workspace
        submission_dir: Directory on host to save submission
        task_name: Name of the task
    """
    logger.info(f"Extracting submission for task: {task_name}")

    # Create submission directory
    os.makedirs(submission_dir, exist_ok=True)
    task_submission_dir = os.path.join(submission_dir, task_name)
    os.makedirs(task_submission_dir, exist_ok=True)

    # Copy submission from workspace
    try:
        # List files in submission directory
        result = workspace.execute_command("find /workspace/submission -type f 2>/dev/null || true")
        if result.exit_code == 0:
            files = [f for f in result.stdout.strip().split("\n") if f]
            logger.info(f"Found {len(files)} files in submission directory")

            # Create tar archive
            archive_command = "cd /workspace/submission && tar -czf /tmp/submission.tar.gz . 2>/dev/null || true"
            archive_result = workspace.execute_command(archive_command)
            if archive_result.exit_code == 0:
                logger.info("Created submission archive")
                # Note: Download tar file
    except Exception as e:
        logger.warning(f"Failed to extract submission: {e}")


def main() -> None:
    """Main entry point for PaperBench inference."""
    parser = get_parser(add_llm_config=True)
    parser.add_argument(
        "--task-name",
        type=str,
        required=True,
        help="Task name (paper ID)",
    )
    parser.add_argument(
        "--instructions-path",
        type=str,
        default="benchmarks/paper_bench/instructions.md",
        help="Path to instructions file",
    )
    parser.add_argument(
        "--rubric-path",
        type=str,
        default=None,
        help="Optional path to rubric.json file",
    )
    parser.add_argument(
        "--base-image",
        type=str,
        default=BASE_IMAGE,
        help="Base Docker image to use",
    )
    parser.add_argument(
        "--max-iterations-per-agent",
        type=int,
        default=50,
        help="Maximum iterations per agent",
    )
    parser.add_argument(
        "--log-completions",
        action="store_true",
        help="Enable logging of LLM API calls and responses to JSON files",
    )
    parser.add_argument(
        "--log-completions-dir",
        type=str,
        default=None,
        help="Directory to store completion logs (default: <output_dir>/logs/completions)",
    )
    parser.add_argument(
        "--submission-dir",
        type=str,
        default="./outputs/submissions",
        help="Directory to save submissions",
    )
    parser.add_argument(
        "--enable-gpu",
        action="store_true",
        help="Enable GPU support (requires nvidia-docker)",
    )

    args = parser.parse_args()

    # Load LLM config first (needed for output directory construction)
    llm_config_path = args.llm_config_path
    if not os.path.isfile(llm_config_path):
        raise ValueError(f"LLM config file {llm_config_path} does not exist")
    with open(llm_config_path, "r") as f:
        llm_config = f.read()
    llm = LLM.model_validate_json(llm_config)
    
    # Construct output directory
    dataset_description = f"paperbench-{args.task_name}"
    structured_output_dir = construct_eval_output_dir(
        base_dir=args.output_dir,
        dataset_name=dataset_description,
        model_name=llm.model,
        max_iterations=args.max_iterations,
        eval_note=args.note,
    )
    
    # Enable API call logging if requested
    if args.log_completions:
        log_completions_dir = args.log_completions_dir
        if log_completions_dir is None:
            log_completions_dir = os.path.join(structured_output_dir, "logs", "completions")
        
        # Create the directory NOW before any LLM calls
        os.makedirs(log_completions_dir, exist_ok=True)
        
        # Verify it exists and is writable
        if not os.path.isdir(log_completions_dir):
            logger.warning(f"Failed to create log directory: {log_completions_dir}")
            logger.info("ℹ️  API call logging disabled due to directory creation failure")
        else:
            llm.log_completions = True
            llm.log_completions_folder = log_completions_dir
            logger.info(f"✅ API call logging enabled: {log_completions_dir}")
            logger.info(f"   All LLM API requests and responses will be saved as JSON files")
    else:
        logger.info("ℹ️  API call logging disabled (use --log-completions to enable)")
    
    logger.info("Using LLM config: %s", llm.model_dump_json(indent=2))

    # Create workspace (always builds with --platform linux/amd64)
    workspace = get_config_for_task(
        task_name=args.task_name,
        base_image=args.base_image,
        enable_gpu=args.enable_gpu,
    )

    # Resolve instructions path
    instructions_path = args.instructions_path
    if not os.path.isabs(instructions_path):
        # Try relative to project root first
        project_root = Path(__file__).parent.parent.parent
        potential_path = project_root / instructions_path
        if potential_path.exists():
            instructions_path = str(potential_path)
        elif os.path.exists(instructions_path):
            instructions_path = os.path.abspath(instructions_path)
        else:
            # Try relative to this file
            script_dir = Path(__file__).parent
            potential_path = script_dir / "instructions.md"
            if potential_path.exists():
                instructions_path = str(potential_path)
            else:
                logger.warning(f"Instructions file not found: {args.instructions_path}")

    # Run inference using workspace context manager (synchronous)
    with workspace:
        # Initialize task environment
        init_task_environment(workspace, args.task_name, instructions_path)

        # Resolve rubric path if provided
        rubric_path = None
        if args.rubric_path:
            if os.path.exists(args.rubric_path):
                # Copy rubric to workspace
                with open(args.rubric_path, "r") as f:
                    rubric_content = f.read()
                result = workspace.execute_command(
                    f'''cat > /workspace/paper/rubric.json << "RUBRICEOF"
{rubric_content}
RUBRICEOF'''
                )
                if result.exit_code == 0:
                    rubric_path = "/workspace/paper/rubric.json"
                else:
                    logger.warning(f"Failed to copy rubric: {result.stderr}")
            else:
                logger.warning(f"Rubric file not found: {args.rubric_path}")

        # Run multi-agent inference (async)
        result = asyncio.run(run_multi_agent_inference(
            workspace=workspace,
            llm=llm,
            task_name=args.task_name,
            instructions_path="/workspace/instructions.md",
            rubric_path=rubric_path,
            max_iterations_per_agent=args.max_iterations_per_agent,
        ))

        # Extract submission
        extract_submission(workspace, args.submission_dir, args.task_name)

        # Save results
        results_file = os.path.join(structured_output_dir, "results.json")
        os.makedirs(structured_output_dir, exist_ok=True)
        with open(results_file, "w") as f:
            json.dump(result, f, indent=2)

        logger.info(f"Results saved to: {results_file}")
        
        # Log completion log location if enabled
        if args.log_completions:
            log_completions_dir = args.log_completions_dir
            if log_completions_dir is None:
                log_completions_dir = os.path.join(structured_output_dir, "logs", "completions")
            logger.info(f"📝 API call logs saved to: {log_completions_dir}")
            logger.info(f"   Each API request/response is saved as a separate JSON file")

    logger.info("Inference completed!")


if __name__ == "__main__":
    main()
