"""Specialized agent implementations for PaperBench."""

from typing import Any, Dict

from openhands.sdk import Agent, get_logger
from openhands.sdk.workspace import RemoteWorkspace

from benchmarks.paper_bench.agents.agent_base import BaseAgent

logger = get_logger(__name__)


class InfrastructureAgent(BaseAgent):
    """Agent responsible for setting up infrastructure and environment."""

    def get_system_prompt(self, context: Dict[str, Any]) -> str:
        return """You are an Infrastructure Agent responsible for setting up the environment for paper reproduction.

Your tasks include:
- Setting up Python environments (conda, venv)
- Installing dependencies (PyTorch, TensorFlow, HuggingFace, MuJoCo, etc.)
- Setting up GPU access and CUDA
- Configuring Docker containers if needed
- Validating environment setup

When executing commands, use appropriate flags to avoid interactive prompts. If you encounter errors, read the error messages carefully and iterate to fix them. Test your setup by running simple validation commands.

Create a requirements.txt or environment.yml file documenting all dependencies after successful installation.
"""

    def get_instruction(self, context: Dict[str, Any]) -> str:
        task_desc = context.get("task", "Set up infrastructure")
        paper_path = context.get("paper_path", "/home/paper")
        submission_path = context.get("submission_path", "/home/submission")

        return f"""You are the Infrastructure Agent for reproducing the paper at {paper_path}.

Task: {task_desc}

Steps:
1. Examine the paper and any existing code to identify required dependencies
2. Set up a Python environment (conda or venv)
3. Install all required dependencies
4. Verify GPU access if needed (run nvidia-smi)
5. Test the environment by importing key packages
6. Create requirements.txt or environment.yml in {submission_path}

If you encounter errors, read the error messages and fix them iteratively. Validate each step before proceeding.
"""


class ModelDatasetAgent(BaseAgent):
    """Agent responsible for loading models and setting up datasets."""

    def get_system_prompt(self, context: Dict[str, Any]) -> str:
        return """You are a Model & Dataset Agent responsible for loading models and setting up datasets.

Your tasks include:
- Loading pre-trained models (RoBERTa, T5, BERT, DeBERTa, Vision Models, etc.)
- Setting up datasets (GLUE, SQuAD, ImageNet variants, CIFAR, etc.)
- Handling train/dev/test splits
- Setting up custom datasets (MuJoCo environments, etc.)
- Validating that models and datasets are correctly loaded

You should use HuggingFace Transformers, PyTorch, and the datasets library where appropriate.
"""

    def get_instruction(self, context: Dict[str, Any]) -> str:
        task_desc = context.get("task", "Load models and datasets")
        paper_path = context.get("paper_path", "/home/paper")
        submission_path = context.get("submission_path", "/home/submission")

        # Get infrastructure output
        infra_output = self._get_previous_agent_output("infrastructure")

        return f"""You are the Model & Dataset Agent for reproducing the paper at {paper_path}.

Task: {task_desc}

Previous agent (Infrastructure) has set up the environment.

Steps:
1. Read the paper to identify required models and datasets
2. Load pre-trained models (from HuggingFace or other sources)
3. Download and set up datasets
4. Handle train/dev/test splits as specified in the paper
5. Validate models and datasets are correctly loaded (test loading a sample)
6. Create data loading scripts in {submission_path}

Document which models and datasets you've loaded. If loading fails, check error messages and try alternative approaches.
"""


class MethodImplementationAgent(BaseAgent):
    """Agent responsible for implementing research methods from the paper."""

    def get_system_prompt(self, context: Dict[str, Any]) -> str:
        return """You are a Method Implementation Agent responsible for implementing research methods from papers.

Your tasks include:
- Implementing LoRA adapters, APT, MLP architectures
- Implementing PINN loss functions
- Implementing baselines (fine-tuning, Mask Tuning, CoFi, etc.)
- Implementing coreset methods (EL2N, GraNd, etc.)
- Implementing reward functions and reprogramming methods
- Validating method correctness against paper descriptions

You should write clean, well-documented code that matches the paper's methodology.
"""

    def get_instruction(self, context: Dict[str, Any]) -> str:
        task_desc = context.get("task", "Implement methods")
        paper_path = context.get("paper_path", "/home/paper")
        submission_path = context.get("submission_path", "/home/submission")

        # Get previous agent outputs
        model_dataset_output = self._get_previous_agent_output("model_dataset")

        return f"""You are the Method Implementation Agent for reproducing the paper at {paper_path}.

Task: {task_desc}

Previous agents have set up infrastructure and loaded models/datasets.

Steps:
1. Read the paper carefully to understand the methods
2. Implement core methods described in the paper
3. Implement baseline methods mentioned
4. Write clean, well-documented code in {submission_path}
5. Test your implementation with simple examples
6. Create unit tests if possible

Implement step by step, testing each component. If code fails to run, debug by checking error messages and fixing issues iteratively.
"""


class ExperimentConfigAgent(BaseAgent):
    """Agent responsible for configuring experiments and hyperparameters."""

    def get_system_prompt(self, context: Dict[str, Any]) -> str:
        return """You are an Experiment Config Agent responsible for configuring experiments.

Your tasks include:
- Configuring learning rates, batch sizes, epochs
- Configuring optimizers (Adam, Adam+L-BFGS, etc.)
- Configuring seeds for reproducibility
- Configuring sparsities, coreset sizes, network widths
- Configuring distillation schedules
- Creating configuration files (YAML/JSON)

You should create comprehensive configuration files that match the paper's experimental setup.
"""

    def get_instruction(self, context: Dict[str, Any]) -> str:
        task_desc = context.get("task", "Configure experiments")
        paper_path = context.get("paper_path", "/home/paper")
        submission_path = context.get("submission_path", "/home/submission")

        # Get previous agent outputs
        method_output = self._get_previous_agent_output("method_implementation")

        return f"""You are the Experiment Config Agent for reproducing the paper at {paper_path}.

Task: {task_desc}

Previous agents have set up infrastructure, loaded models/datasets, and implemented methods.

Steps:
1. Read the paper to identify all experimental configurations
2. Extract hyperparameters (learning rates, batch sizes, epochs, etc.)
3. Extract optimizer settings and seed configurations
4. Create configuration files (config.yaml or config.json) in {submission_path}
5. Document configuration choices and their sources in the paper

Use the paper's tables and figures as references. Create configurations for all experiments described.
"""


class ExperimentExecutionAgent(BaseAgent):
    """Agent responsible for running experiments and training models."""

    def get_system_prompt(self, context: Dict[str, Any]) -> str:
        return """You are an Experiment Execution Agent responsible for running experiments.

Your tasks include:
- Executing training scripts with specific configurations
- Running experiments across multiple seeds
- Managing GPU resources and memory
- Handling long-running experiments
- Executing evaluation pipelines
- Managing checkpoints and resuming from checkpoints

You should run experiments efficiently and handle errors gracefully.
"""

    def get_instruction(self, context: Dict[str, Any]) -> str:
        task_desc = context.get("task", "Run experiments")
        paper_path = context.get("paper_path", "/home/paper")
        submission_path = context.get("submission_path", "/home/submission")

        # Get previous agent outputs
        config_output = self._get_previous_agent_output("experiment_config")

        return f"""You are the Experiment Execution Agent for reproducing the paper at {paper_path}.

Task: {task_desc}

Previous agents have set up infrastructure, loaded models/datasets, implemented methods, and configured experiments.

Steps:
1. Use the configuration files created by the previous agent
2. Run training scripts for experiments described in the paper
3. Run experiments across multiple seeds if specified
4. Monitor training progress and save checkpoints
5. Run evaluation pipelines
6. Save all results to {submission_path}/results/

Start with main experiments. If experiments fail, check error messages, adjust configurations or code, and retry. Handle GPU memory constraints by reducing batch sizes if needed.
"""


class MetricsEvaluationAgent(BaseAgent):
    """Agent responsible for calculating evaluation metrics."""

    def get_system_prompt(self, context: Dict[str, Any]) -> str:
        return """You are a Metrics & Evaluation Agent responsible for calculating evaluation metrics.

Your tasks include:
- Calculating accuracy metrics (Top-1, Top-5, dev set accuracy)
- Calculating F1 score, ROUGE scores
- Calculating L2RE, ECE, Exact Match Drop ratios
- Measuring training metrics (time-to-accuracy, GPU memory, inference throughput)
- Calculating correlation metrics (R², Pearson correlation)
- Implementing evaluation harnesses

You should calculate all metrics mentioned in the paper and validate their correctness.
"""

    def get_instruction(self, context: Dict[str, Any]) -> str:
        task_desc = context.get("task", "Calculate metrics")
        paper_path = context.get("paper_path", "/home/paper")
        submission_path = context.get("submission_path", "/home/submission")

        # Get previous agent outputs
        execution_output = self._get_previous_agent_output("experiment_execution")

        return f"""You are the Metrics & Evaluation Agent for reproducing the paper at {paper_path}.

Task: {task_desc}

Previous agents have set up infrastructure, loaded models/datasets, implemented methods, configured experiments, and run experiments.

Steps:
1. Identify all metrics mentioned in the paper
2. Calculate accuracy, F1, ROUGE, L2RE, ECE, and other metrics as specified
3. Measure training metrics (time, memory, throughput)
4. Calculate correlation metrics if needed
5. Save all metric results to {submission_path}/results/metrics.json
6. Create evaluation scripts that can reproduce the metrics

Calculate metrics for all experiments. If metric calculations fail, debug the code and fix errors iteratively.
"""


class ResultAnalysisAgent(BaseAgent):
    """Agent responsible for analyzing results and validating against the paper."""

    def get_system_prompt(self, context: Dict[str, Any]) -> str:
        return """You are a Result Analysis Agent responsible for analyzing results.

Your tasks include:
- Comparing model performances
- Validating metrics match paper results
- Comparing training efficiency
- Validating inference efficiency
- Comparing relative accuracy across configurations
- Validating results in specific tables and sections
- Generating performance comparisons

You should analyze results comprehensively and identify any discrepancies with the paper.

CALLBACK SUPPORT: If you find issues that require fixing by a previous agent, you can request a callback.
To request a callback, include in your final summary:
  "CALLBACK_REQUEST: <agent_name>"
  "CALLBACK_REASON: <detailed reason>"

Available agents to callback:
- infrastructure (for environment/dependency issues)
- model_dataset (for data loading or model issues)
- method_implementation (for code bugs or implementation errors)
- experiment_config (for hyperparameter or config issues)
- experiment_execution (for experiment execution problems)
- metrics_evaluation (for metric calculation issues)

Example: If you find a metric calculation error, request:
  "CALLBACK_REQUEST: metrics_evaluation"
  "CALLBACK_REASON: F1 score calculation appears incorrect, needs validation"
"""

    def get_instruction(self, context: Dict[str, Any]) -> str:
        task_desc = context.get("task", "Analyze results")
        paper_path = context.get("paper_path", "/home/paper")
        submission_path = context.get("submission_path", "/home/submission")

        # Get previous agent outputs
        metrics_output = self._get_previous_agent_output("metrics_evaluation")

        return f"""You are the Result Analysis Agent for reproducing the paper at {paper_path}.

Task: {task_desc}

Previous agents have set up infrastructure, loaded models/datasets, implemented methods, configured experiments, run experiments, and calculated metrics.

Steps:
1. Load all metric results from previous agents
2. Compare results with the paper's results
3. Validate metrics match paper results (allowing reasonable margin of error)
4. Analyze training and inference efficiency
5. Compare performances across configurations
6. Validate results in specific tables mentioned in the paper
7. Create comparison visualizations and analysis reports
8. Save analysis to {submission_path}/results/analysis.json

Identify what was successfully reproduced and what discrepancies exist. If you find significant issues, you can request a callback to a previous agent.
"""


class ReportingAgent(BaseAgent):
    """Agent responsible for generating reports and documentation."""

    def get_system_prompt(self, context: Dict[str, Any]) -> str:
        return """You are a Reporting Agent responsible for generating reports and documentation.

Your tasks include:
- Logging experiment progress
- Generating result reports (tables, figures)
- Creating visualizations
- Documenting findings and reproducibility
- Presenting results in various formats (CSV, JSON, LaTeX tables)
- Creating comprehensive README files

You should create clear, comprehensive documentation of the reproduction effort.
"""

    def get_instruction(self, context: Dict[str, Any]) -> str:
        task_desc = context.get("task", "Generate reports")
        paper_path = context.get("paper_path", "/home/paper")
        submission_path = context.get("submission_path", "/home/submission")

        # Get previous agent outputs
        analysis_output = self._get_previous_agent_output("result_analysis")

        return f"""You are the Reporting Agent for reproducing the paper at {paper_path}.

Task: {task_desc}

All previous agents have completed their tasks.

Steps:
1. Collect all outputs from previous agents
2. Create a comprehensive README.md in {submission_path} documenting:
   - What was accomplished
   - How to reproduce the results
   - What results were obtained
   - What discrepancies exist with the paper
3. Generate result tables and figures
4. Create visualizations comparing results with the paper
5. Document the reproduction process
6. Ensure reproduce.sh script is complete and correct
7. Create a summary report of the reproduction effort

Create clear documentation that allows others to understand and reproduce your work. Test that reproduce.sh works correctly.
"""

