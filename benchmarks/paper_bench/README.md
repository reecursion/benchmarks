# PaperBench Multi-Agent Evaluation

Multi-agent system for reproducing research papers using OpenHands and the `leandermaben7/pb-env:1.0.0` Docker image.

## Quick Start

### Prerequisites

1. **Docker Desktop** - Must be running
   ```bash
   docker ps  # Should work without errors
   ```

2. **uv** - Python package manager
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

3. **LLM Config** - Create `.llm_config/gpt-3.5-turbo.json` with your API key

### Run Inference

The system will automatically build the Docker image on first run (takes 5-10 minutes):

**With Critic Agent (Default - Recommended):**
```bash
cd /Users/gayathrigl/Desktop/capstone/benchmarks

uv run python -m benchmarks.paper_bench.run_infer \
    .llm_config/gpt-3.5-turbo.json \
    --output-dir ./outputs \
    --task-name pinn \
    --max-iterations 100 \
    --max-iterations-per-agent 50
```

**Without Critic (Faster, Lower Quality):**
```bash
uv run python -m benchmarks.paper_bench.run_infer \
    .llm_config/gpt-3.5-turbo.json \
    --output-dir ./outputs \
    --task-name pinn \
    --disable-critic
```

### Command Options

- `--task-name`: Paper name (see Available Papers below)
- `--output-dir`: Output directory (default: `./outputs`)
- `--max-iterations`: Total max iterations
- `--max-iterations-per-agent`: Max iterations per agent (default: 50)
- `--enable-critic`: Enable critic agent (default: True)
- `--disable-critic`: Disable critic agent for faster execution
- `--max-refinements`: Max refinement iterations per agent (default: 2)
- `--instructions-path`: Path to instructions file (default: `benchmarks/paper_bench/instructions.md`)
- `--rubric-path`: Optional path to rubric.json
- `--log-completions`: Enable LLM API call logging

## Multi-Agent System

The orchestrator coordinates 8 specialized agents in sequence, with an optional **Critic Agent** for quality assurance:

1. **Infrastructure Agent** - Environment setup, dependencies, GPU access
2. **Model & Dataset Agent** - Load pre-trained models and datasets
3. **Method Implementation Agent** - Implement research methods from paper
4. **Experiment Config Agent** - Configure experiments and hyperparameters
5. **Experiment Execution Agent** - Run experiments and train models
6. **Metrics & Evaluation Agent** - Calculate evaluation metrics
7. **Result Analysis Agent** - Analyze results and validate against paper (can request callbacks)
8. **Reporting Agent** - Generate reports and documentation

### Critic Agent

The **Critic Agent** reviews each agent's output for quality and provides iterative refinement:
- Reviews outputs for completeness, correctness, and quality
- Provides actionable feedback for improvement
- Enables iterative refinement (default: 2 attempts per agent)
- **Enabled by default** - use `--disable-critic` to turn off
- See [CRITIC_AGENT_README.md](CRITIC_AGENT_README.md) for details

### Agent Callbacks

Agents can request callbacks to previous agents if issues are found. For example, if the Result Analysis Agent finds a metric calculation error, it can request the Metrics & Evaluation Agent to re-run.

**Safety limits:**
- Max 3 callbacks per agent
- Max 10 total callbacks

## Available Papers

- `pinn`
- `rice`
- `adaptive-pruning`
- `robust-clip`
- `all-in-one`
- `fre`
- `mechanistic-understanding`
- `bridging-data-gaps`
- `test-time-model-adaptation`
- `sequential-neural-score-estimation`
- `what-will-my-model-forget`
- `stay-on-topic-with-classifier-free-guidance`
- `sample-specific-masks`
- `sapg`
- `lca-on-the-line`
- `stochastic-interpolants`
- `bbox`
- `lbcs`
- `bam`
- `ftrl`

## Output

Results are saved to:
- `outputs/submissions/<paper-name>/` - Submission files (reproduce.sh, README.md, code)
- `outputs/results/paperbench-<paper-name>/results.json` - Results and agent outputs

## Architecture

- `run_infer.py` - Main inference script with multi-agent orchestrator
- `orchestrator.py` - Multi-agent orchestrator with callback support
- `agents/` - Specialized agent implementations
  - `agent_base.py` - Base agent class
  - `specialized_agents.py` - 8 specialized agents
- `instructions.md` - Task instructions for agents

## How It Works

1. **Docker Image**: Automatically builds agent-server image from `leandermaben7/pb-env:1.0.0` (first run only, 5-10 min)
2. **Task Setup**: Clones paper files from `openai/frontier-evals` repository
3. **Multi-Agent Workflow**: 8 agents work sequentially, sharing state via `/workspace/shared_state/shared_state.json`
4. **Dynamic Callbacks**: Agents can request previous agents to re-run if issues are found
5. **Submission**: Final submission with `reproduce.sh` and documentation saved to `outputs/`

## References

- [PaperBench Paper](https://arxiv.org/abs/2504.01848)
- [PaperBench Repository](https://github.com/openai/frontier-evals/tree/main/project/paperbench)
