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
import time
from pathlib import Path
from typing import Optional

import httpx
from urllib.parse import quote

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
TRANSFER_DIR = "/workspace/.transfer"


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


def _copy_instructions(workspace: RemoteWorkspace, instructions_path: str) -> None:
    """Copy instructions file to workspace."""
    if os.path.exists(instructions_path):
        with open(instructions_path, "r") as f:
            instructions_content = f.read()

        instructions_file = "/workspace/instructions.md"
        result = workspace.execute_command(f"mkdir -p $(dirname {instructions_file})")
        if result.exit_code == 0:
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


def init_task_environment(
    workspace: RemoteWorkspace, task_name: str, instructions_path: str,
    paper_cache_dir: Optional[str] = None
) -> None:
    """
    Initialize the task environment by setting up paper files and instructions.

    Args:
        workspace: The workspace to initialize
        task_name: Name of the task/paper
        instructions_path: Path to instructions file on host
        paper_cache_dir: Optional path to pre-downloaded paper files (avoids Git LFS issues)
    """
    logger.info(f"Initializing task environment for: {task_name}")

    # Create directories in workspace (which we have full control over)
    workspace.execute_command("mkdir -p /workspace/paper /workspace/submission /workspace/shared_state")
    
    # Clear paper directory if it exists (safer than rm -rf)
    workspace.execute_command("rm -rf /workspace/paper/* /workspace/paper/.* 2>/dev/null || true")
    workspace.execute_command("mkdir -p /workspace/paper")

    # Check for local paper cache first (bypasses Git LFS issues)
    local_paper_path = None
    if paper_cache_dir:
        candidate = os.path.join(paper_cache_dir, task_name)
        if os.path.exists(candidate) and os.path.isdir(candidate):
            local_paper_path = candidate
            logger.info(f"📁 Found local paper cache at: {local_paper_path}")
    
    # Also check default cache location
    if not local_paper_path:
        default_cache = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "paper_cache", task_name)
        if os.path.exists(default_cache) and os.path.isdir(default_cache):
            local_paper_path = default_cache
            logger.info(f"📁 Found paper in default cache: {local_paper_path}")

    if local_paper_path:
        # Use local cache - copy files to workspace using file_upload API
        logger.info(f"Using pre-downloaded paper files from: {local_paper_path}")
        
        def upload_file_to_workspace(src_path: str, dest_path: str) -> bool:
            """Upload a file to workspace using the file_upload API."""
            try:
                result = workspace.file_upload(src_path, dest_path)
                return result.success
            except Exception as e:
                logger.warning(f"file_upload failed for {src_path}, falling back to base64: {e}")
                # Fallback to base64 method
                try:
                    import base64
                    with open(src_path, 'rb') as f:
                        content = f.read()
                    b64_content = base64.b64encode(content).decode('ascii')
                    
                    # Split large base64 content into chunks
                    chunk_size = 50000
                    if len(b64_content) <= chunk_size:
                        cmd_result = workspace.execute_command(
                            f"echo '{b64_content}' | base64 -d > {dest_path}"
                        )
                    else:
                        workspace.execute_command(f"rm -f {dest_path}.b64 {dest_path}")
                        for i in range(0, len(b64_content), chunk_size):
                            chunk = b64_content[i:i+chunk_size]
                            workspace.execute_command(f"echo -n '{chunk}' >> {dest_path}.b64")
                        cmd_result = workspace.execute_command(
                            f"base64 -d {dest_path}.b64 > {dest_path} && rm -f {dest_path}.b64"
                        )
                    return cmd_result.exit_code == 0
                except Exception as e2:
                    logger.warning(f"Fallback also failed for {src_path}: {e2}")
                    return False
        
        # Copy each file from local cache to workspace
        # Skip rubric.json - agents should not have access to evaluation criteria
        EXCLUDED_FILES = {"rubric.json"}
        
        for filename in os.listdir(local_paper_path):
            if filename in EXCLUDED_FILES:
                logger.info(f"  Skipped (excluded): {filename}")
                continue
            src_path = os.path.join(local_paper_path, filename)
            if os.path.isfile(src_path):
                dest_path = f"/workspace/paper/{filename}"
                if upload_file_to_workspace(src_path, dest_path):
                    logger.info(f"  Copied: {filename}")
                else:
                    logger.warning(f"  Failed to copy: {filename}")
            elif os.path.isdir(src_path):
                # Handle subdirectories (like assets/)
                workspace.execute_command(f"mkdir -p /workspace/paper/{filename}")
                for subfile in os.listdir(src_path):
                    sub_src = os.path.join(src_path, subfile)
                    if os.path.isfile(sub_src):
                        dest_path = f"/workspace/paper/{filename}/{subfile}"
                        if upload_file_to_workspace(sub_src, dest_path):
                            logger.debug(f"  Copied: {filename}/{subfile}")
                        else:
                            logger.warning(f"  Failed to copy: {filename}/{subfile}")
                logger.info(f"  Copied: {filename}/ ({len(os.listdir(src_path))} files)")
        
        # Verify files were copied
        verify_result = workspace.execute_command("ls -la /workspace/paper/")
        logger.info(f"Paper files in workspace:\n{verify_result.stdout}")
        
        # Check paper.md exists and has actual content
        check_result = workspace.execute_command("wc -c /workspace/paper/paper.md 2>&1")
        if "No such file" in check_result.stdout or check_result.exit_code != 0:
            logger.error(f"❌ paper.md was not copied successfully! Falling back to Git LFS...")
            # Don't return - fall through to Git LFS download
        else:
            head_result = workspace.execute_command("head -5 /workspace/paper/paper.md 2>&1")
            if "git-lfs" in head_result.stdout:
                logger.error("❌ Local cache contains LFS pointers! Please re-download with git lfs pull")
            else:
                logger.info(f"✅ Paper files loaded from local cache successfully ({check_result.stdout.strip()})")
                # Copy instructions and return early
                _copy_instructions(workspace, instructions_path)
                return

    # Fall back to Git LFS download (original method)
    logger.info("No local paper cache found, falling back to Git LFS download...")

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
    # Use a more robust approach with retries and proper timeout coordination
    logger.info("Attempting to download Git LFS files...")
    
    lfs_success = False
    for attempt in range(3):  # Try up to 3 times
        logger.info(f"Git LFS attempt {attempt + 1}/3...")
        
        # Method 1: Try git lfs pull with proper include pattern
        lfs_cmd = f"""cd /tmp/frontier-evals && \\
git lfs install --skip-smudge 2>&1 || true
git lfs pull --include="project/paperbench/data/papers/{task_name}/*" 2>&1
"""
        result = workspace.execute_command(lfs_cmd, timeout=180)
        
        # Check if paper.md has actual content
        check_result = workspace.execute_command(f"head -1 /tmp/frontier-evals/project/paperbench/data/papers/{task_name}/paper.md 2>&1")
        if check_result.exit_code == 0 and "git-lfs.github.com" not in check_result.stdout:
            logger.info(f"✅ Git LFS pull succeeded on attempt {attempt + 1}")
            lfs_success = True
            break
        
        # Method 2: Try git lfs fetch + checkout
        if not lfs_success:
            logger.info("Trying git lfs fetch + checkout...")
            fetch_cmd = f"""cd /tmp/frontier-evals && \\
git lfs fetch --include="project/paperbench/data/papers/{task_name}/*" 2>&1 && \\
git lfs checkout "project/paperbench/data/papers/{task_name}/*" 2>&1
"""
            result = workspace.execute_command(fetch_cmd, timeout=180)
            
            check_result = workspace.execute_command(f"head -1 /tmp/frontier-evals/project/paperbench/data/papers/{task_name}/paper.md 2>&1")
            if check_result.exit_code == 0 and "git-lfs.github.com" not in check_result.stdout:
                logger.info(f"✅ Git LFS fetch+checkout succeeded on attempt {attempt + 1}")
                lfs_success = True
                break
        
        if attempt < 2:
            logger.warning(f"LFS attempt {attempt + 1} failed, retrying...")
            time.sleep(2)  # Small delay before retry
    
    if not lfs_success:
        logger.warning("⚠️ Git LFS download failed after 3 attempts. Trying direct download fallback...")
        
        # Method 3: Fallback - download paper.md directly via raw GitHub URL
        # For LFS files, we need to use the GitHub API or download the actual blob
        fallback_cmd = f"""cd /tmp/frontier-evals/project/paperbench/data/papers/{task_name} && \\
# Read the LFS pointer to get the OID
if head -1 paper.md | grep -q "git-lfs"; then
    OID=$(grep "^oid sha256:" paper.md | cut -d: -f2)
    if [ -n "$OID" ]; then
        echo "Attempting to download LFS object: $OID"
        # Try GitHub LFS batch API
        curl -sL "https://github.com/openai/frontier-evals.git/info/lfs/objects/batch" \\
            -H "Accept: application/vnd.git-lfs+json" \\
            -H "Content-Type: application/vnd.git-lfs+json" \\
            -d '{{"operation": "download", "transfers": ["basic"], "objects": [{{"oid": "'$OID'", "size": 0}}]}}' \\
            2>&1 || echo "LFS API call completed"
    fi
fi
"""
        workspace.execute_command(fallback_cmd, timeout=60)
    
    # Step 4: Copy files to workspace (excluding rubric.json - agents shouldn't see evaluation criteria)
    copy_cmd = f"""cp -r /tmp/frontier-evals/project/paperbench/data/papers/{task_name}/* /workspace/paper/ 2>&1 && \\
rm -f /workspace/paper/rubric.json 2>&1 || true
"""
    result = workspace.execute_command(copy_cmd)
    if result.exit_code != 0:
        logger.error(f"Failed to copy files: {result.stderr}")
        raise RuntimeError(f"Failed to copy files: {result.stderr}")
    
    # Step 5: Verify files and check for LFS pointers
    verify_result = workspace.execute_command("ls -la /workspace/paper/ | head -20")
    file_list = verify_result.stdout if verify_result.exit_code == 0 else ""
    
    # Check if paper.md has actual content or is an LFS pointer
    check_content = workspace.execute_command("head -5 /workspace/paper/paper.md 2>&1 || echo 'file not found'")
    logger.info(f"Checking paper.md content (first 5 lines):\n{check_content.stdout[:200]}")
    
    if "version https://git-lfs.github.com/spec/v1" in check_content.stdout:
        logger.warning("⚠️  paper.md is still a Git LFS pointer after initial attempts!")
        
        # Final fallback: try one more time with fetch --all (slower but more reliable)
        logger.info("Trying final fallback with git lfs fetch --all...")
        lfs_fetch_cmd = f"""cd /tmp/frontier-evals && \\
git lfs fetch --all 2>&1 && \\
git lfs checkout 2>&1 && \\
cp -rf project/paperbench/data/papers/{task_name}/* /workspace/paper/ 2>&1 && \\
rm -f /workspace/paper/rubric.json 2>&1 || true
"""
        result = workspace.execute_command(lfs_fetch_cmd, timeout=300)
        
        # Verify the file was replaced
        check_again = workspace.execute_command("head -10 /workspace/paper/paper.md 2>&1")
        if "version https://git-lfs.github.com/spec/v1" not in check_again.stdout:
            logger.info("✅ Git LFS files successfully downloaded - paper.md now has actual content")
        else:
            logger.warning("⚠️  Git LFS files still not downloaded - paper.md is still a pointer")
            logger.warning("The agent will work with available files (rubric.json, config.yaml, etc.)")
            # Log what files ARE available so we know what the agent can use
            available_files = workspace.execute_command("ls -la /workspace/paper/")
            logger.info(f"Available files for agent:\n{available_files.stdout}")
    
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
    _copy_instructions(workspace, instructions_path)

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


def download_remote_file(
    workspace: RemoteWorkspace,
    remote_path: str,
    local_path: str,
    timeout: float = 60.0,
) -> bool:
    """Download a file from the remote workspace with redirect-safe fallback."""
    try:
        client = getattr(workspace, "client", None)
        if client is None:
            raise RuntimeError("Workspace client unavailable for file download")

        encoded_path = quote(remote_path, safe="")
        request = client.build_request(
            "GET",
            f"/api/file/download/{encoded_path}",
            timeout=timeout,
        )
        response = client.send(request, follow_redirects=True)
        response.raise_for_status()

        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        with open(local_path, "wb") as f:
            f.write(response.content)
        logger.info(f"Downloaded {remote_path} via agent-server API")
        return True
    except Exception as error:
        logger.error(
            "File download failed for %s: %s",
            remote_path,
            error,
        )
        return False


def extract_logs_from_container(
    workspace: RemoteWorkspace, output_dir: str, container_log_dir: str = "/workspace/logs/completions"
) -> None:
    """
    Extract LLM completion logs from workspace container to host.

    Args:
        workspace: The workspace
        output_dir: Directory on host to save logs
        container_log_dir: Path to logs directory inside container
    """
    logger.info(f"Extracting completion logs from container: {container_log_dir}")

    try:
        workspace.execute_command(f"mkdir -p {TRANSFER_DIR}")
        # Check if log directory exists and has files
        result = workspace.execute_command(f"find {container_log_dir} -type f -name '*.json' 2>/dev/null | wc -l")
        if result.exit_code != 0 or not result.stdout.strip().isdigit():
            logger.warning(f"Could not check log directory: {result.stderr}")
            return
        
        file_count = int(result.stdout.strip())
        if file_count == 0:
            logger.info("No completion logs found in container")
            return
        
        logger.info(f"Found {file_count} completion log files in container")

        # Create tar archive of logs (use longer timeout for many files)
        archive_path = f"{TRANSFER_DIR}/completion_logs.tar.gz"
        archive_cmd = f"cd {container_log_dir} && tar -czf {archive_path} . 2>&1"
        archive_result = workspace.execute_command(archive_cmd, timeout=300.0)
        if archive_result.exit_code != 0:
            logger.error(f"Failed to create logs archive: {archive_result.stderr}")
            return
        
        # Download the archive
        local_archive_path = os.path.join(output_dir, "completion_logs.tar.gz")
        os.makedirs(output_dir, exist_ok=True)
        if not download_remote_file(workspace, archive_path, local_archive_path):
            logger.error("Failed to download logs archive via workspace APIs")
            return
        
        # Extract locally
        import tarfile
        try:
            with tarfile.open(local_archive_path, "r:gz") as tar:
                tar.extractall(path=output_dir)
            logger.info(f"✅ Extracted {file_count} completion logs to {output_dir}")
            
            # Remove the tar archive after extraction
            os.remove(local_archive_path)
        except tarfile.TarError as e:
            logger.error(f"Failed to extract logs archive: {e}")
            
    except Exception as e:
        logger.warning(f"Failed to extract completion logs: {e}")


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

    # Allowed file extensions for submission (source code, scripts, docs, configs only)
    ALLOWED_EXTENSIONS = (
        ".py", ".sh", ".md", ".txt", ".yaml", ".yml", ".json", 
        ".cfg", ".toml", ".ini", ".rst", ".csv"
    )
    # Directories to exclude from submission
    EXCLUDED_DIRS = (
        "__pycache__", ".git", ".cache", "wandb", "mlruns", ".ipynb_checkpoints",
        "node_modules", ".venv", "venv", "env", ".env", "conda", ".conda",
        "checkpoints", "outputs", "results", "logs", "data", "datasets"
    )
    
    # Copy submission from workspace
    try:
        workspace.execute_command(f"mkdir -p {TRANSFER_DIR}")
        
        # First, clean up the submission directory in the container
        # Remove excluded directories
        for excluded_dir in EXCLUDED_DIRS:
            workspace.execute_command(
                f"find /workspace/submission -type d -name '{excluded_dir}' -exec rm -rf {{}} + 2>/dev/null || true"
            )
        
        # Remove non-allowed file types (model weights, datasets, etc.)
        # Build find command to delete files NOT matching allowed extensions
        allowed_pattern = " -o ".join([f'-name "*{ext}"' for ext in ALLOWED_EXTENSIONS])
        cleanup_cmd = f'find /workspace/submission -type f ! \\( {allowed_pattern} \\) -delete 2>/dev/null || true'
        workspace.execute_command(cleanup_cmd)
        
        # Also explicitly remove common large file types
        large_file_extensions = [
            "*.pt", "*.pth", "*.ckpt", "*.safetensors", "*.bin", "*.h5", "*.hdf5",
            "*.pkl", "*.pickle", "*.npy", "*.npz", "*.parquet", "*.arrow",
            "*.tar", "*.tar.gz", "*.tgz", "*.zip", "*.gz", "*.bz2",
            "*.png", "*.jpg", "*.jpeg", "*.gif", "*.bmp", "*.tiff",
            "*.mp3", "*.mp4", "*.wav", "*.avi", "*.mov",
            "*.so", "*.o", "*.a", "*.dylib", "*.dll", "*.exe",
            "*.pyc", "*.pyo"
        ]
        for ext in large_file_extensions:
            workspace.execute_command(
                f"find /workspace/submission -type f -name '{ext}' -delete 2>/dev/null || true"
            )
        
        # Remove empty directories
        workspace.execute_command(
            "find /workspace/submission -type d -empty -delete 2>/dev/null || true"
        )
        
        # List files in submission directory after cleanup
        result = workspace.execute_command("find /workspace/submission -type f 2>/dev/null || true")
        if result.exit_code == 0:
            files = [f for f in result.stdout.strip().split("\n") if f]
            logger.info(f"Found {len(files)} files in submission directory after cleanup")
            
            if not files:
                logger.warning("No files found in submission directory after cleanup")
                return

            # Create tar archive in the container (should be fast now with only source files)
            archive_path = f"{TRANSFER_DIR}/submission.tar.gz"
            archive_command = f"cd /workspace/submission && tar -czf {archive_path} . 2>&1"
            archive_result = workspace.execute_command(archive_command, timeout=300.0)
            if archive_result.exit_code != 0:
                logger.error(f"Failed to create submission archive: {archive_result.stderr}")
                return
            logger.info("Created submission archive in container")

            # Download the tar archive from container to host
            local_archive_path = os.path.join(task_submission_dir, "submission.tar.gz")
            if not download_remote_file(workspace, archive_path, local_archive_path):
                logger.error("Failed to download submission archive via workspace APIs")
                return
            logger.info(f"Downloaded submission archive to {local_archive_path}")

            # Extract the tar archive locally
            import tarfile
            try:
                with tarfile.open(local_archive_path, "r:gz") as tar:
                    tar.extractall(path=task_submission_dir)
                logger.info(f"Extracted submission files to {task_submission_dir}")
                
                # Remove the tar archive after extraction
                os.remove(local_archive_path)
                
                # List extracted files
                extracted_files = []
                for root, dirs, filenames in os.walk(task_submission_dir):
                    for filename in filenames:
                        rel_path = os.path.relpath(os.path.join(root, filename), task_submission_dir)
                        extracted_files.append(rel_path)
                logger.info(f"Extracted {len(extracted_files)} files: {extracted_files[:10]}{'...' if len(extracted_files) > 10 else ''}")
                
            except tarfile.TarError as e:
                logger.error(f"Failed to extract submission archive: {e}")
                
    except Exception as e:
        logger.warning(f"Failed to extract submission: {e}")


def main() -> None:
    """Main entry point for PaperBench inference."""
    parser = get_parser(add_llm_config=True)
    parser.add_argument(
        "--task-name",
        type=str,
        required=True,
        help="Task name(s) - single paper ID or comma-separated list (e.g., 'pinn,rice,other')",
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
    parser.add_argument(
        "--paper-cache-dir",
        type=str,
        default=None,
        help="Directory containing pre-downloaded paper files (avoids Git LFS issues). "
             "Structure should be: <cache_dir>/<task_name>/paper.md, rubric.json, etc.",
    )

    args = parser.parse_args()

    # Load LLM config first (needed for output directory construction)
    llm_config_path = args.llm_config_path
    if not os.path.isfile(llm_config_path):
        raise ValueError(f"LLM config file {llm_config_path} does not exist")
    with open(llm_config_path, "r") as f:
        llm_config = f.read()
    llm = LLM.model_validate_json(llm_config)
    
    # Configure API call logging
    # Logs are written inside the Docker container and copied back after inference
    container_log_dir = "/workspace/logs/completions"  # Path inside Docker
    
    if args.log_completions:
        # Enable logging inside the Docker container
        llm.log_completions = True
        llm.log_completions_folder = container_log_dir
        logger.info(f"✅ API call logging enabled (inside container: {container_log_dir})")
    else:
        logger.info("ℹ️  API call logging disabled (use --log-completions to enable)")
    
    logger.info("Using LLM config: %s", llm.model_dump_json(indent=2))

    # Parse task names (comma-separated)
    task_names = [t.strip() for t in args.task_name.split(",") if t.strip()]
    if not task_names:
        raise ValueError("No valid task names provided")
    
    logger.info(f"Will run inference for {len(task_names)} task(s): {task_names}")

    # Resolve instructions path (do this once, outside the loop)
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

    # Track results for all tasks
    all_results = {}
    
    # Run inference for each task sequentially
    for task_idx, task_name in enumerate(task_names, 1):
        logger.info(f"\n{'='*60}")
        logger.info(f"Starting task {task_idx}/{len(task_names)}: {task_name}")
        logger.info(f"{'='*60}")
        
        # Construct output directory for this task
        dataset_description = f"paperbench-{task_name}"
        task_output_dir = construct_eval_output_dir(
            base_dir=args.output_dir,
            dataset_name=dataset_description,
            model_name=llm.model,
            max_iterations=args.max_iterations,
            eval_note=args.note,
        )
        
        # Configure log directory for this task
        task_host_log_dir = args.log_completions_dir
        if task_host_log_dir is None:
            task_host_log_dir = os.path.join(task_output_dir, "logs", "completions")
        
        # Create workspace for this task
        workspace = get_config_for_task(
            task_name=task_name,
            base_image=args.base_image,
            enable_gpu=args.enable_gpu,
        )

        try:
            # Run inference using workspace context manager (synchronous)
            with workspace:
                # Create log directory inside container if logging is enabled
                if args.log_completions:
                    workspace.execute_command(f"mkdir -p {container_log_dir}")
                    logger.info(f"Created log directory in container: {container_log_dir}")
        
                # Initialize task environment
                init_task_environment(
                    workspace, task_name, instructions_path,
                    paper_cache_dir=args.paper_cache_dir
                )

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
                    task_name=task_name,
                    instructions_path="/workspace/instructions.md",
                    rubric_path=rubric_path,
                    max_iterations_per_agent=args.max_iterations_per_agent,
                ))

                # Extract submission
                extract_submission(workspace, args.submission_dir, task_name)
                
                # Extract completion logs from container if logging was enabled
                if args.log_completions:
                    extract_logs_from_container(workspace, task_host_log_dir, container_log_dir)

                # Save results
                results_file = os.path.join(task_output_dir, "results.json")
                os.makedirs(task_output_dir, exist_ok=True)
                with open(results_file, "w") as f:
                    json.dump(result, f, indent=2)

                logger.info(f"Results saved to: {results_file}")
                
                # Log completion log location if enabled
                if args.log_completions:
                    logger.info(f"📝 API call logs saved to: {task_host_log_dir}")
                    logger.info(f"   Each API request/response is saved as a separate JSON file")
                
                # Store result for this task
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
            # Continue with next task
            continue
        
        logger.info(f"Completed task {task_idx}/{len(task_names)}: {task_name}")

    # Summary
    logger.info(f"\n{'='*60}")
    logger.info("All tasks completed!")
    logger.info(f"{'='*60}")
    for task_name, task_result in all_results.items():
        status = task_result.get('status', 'unknown')
        if status == 'success':
            logger.info(f"  ✅ {task_name}: {status} -> {task_result.get('output_dir', 'N/A')}")
        else:
            logger.info(f"  ❌ {task_name}: {status} - {task_result.get('error', 'Unknown error')}")


if __name__ == "__main__":
    main()
