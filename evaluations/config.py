"""
Configuration for DeepEval Evaluation System

This module configures:
- Azure OpenAI GPT-4 as judge LLM for evaluations
- DeepEval settings
- Langfuse integration
"""

import os
from typing import Optional
from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI

load_dotenv()


# ============================================================================
# Evaluation Model Configuration
# ============================================================================

def get_evaluation_model(temperature: Optional[float] = None) -> AzureChatOpenAI:
    """
    Get Azure OpenAI GPT-4 model configured for evaluation.

    Uses the same Azure OpenAI configuration as the main agent but with
    temperature=0.0 for consistency in evaluation.

    Args:
        temperature: Optional temperature override (default: 0.0 for evaluations)

    Returns:
        AzureChatOpenAI: Configured evaluation model
    """
    eval_temperature = temperature if temperature is not None else float(
        os.getenv("EVALUATION_TEMPERATURE", "0.0")
    )

    return AzureChatOpenAI(
        azure_deployment=os.getenv("EVALUATION_MODEL", os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4o")),
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
        temperature=eval_temperature,
        model_name="gpt-4o"  # For DeepEval compatibility
    )


# ============================================================================
# DeepEval Configuration
# ============================================================================

# Disable DeepEval telemetry if configured
if os.getenv("DEEPEVAL_TELEMETRY_OPT_OUT", "YES").upper() in ("YES", "TRUE", "1"):
    os.environ["DEEPEVAL_TELEMETRY_OPT_OUT"] = "YES"

# DeepEval settings
DEEPEVAL_CONFIG = {
    "use_cache": True,  # Cache evaluation results
    "verbose_mode": False,  # Disable verbose output
    "ignore_errors": False,  # Fail fast on errors
}


# ============================================================================
# Evaluation Settings
# ============================================================================

EVALUATION_SETTINGS = {
    # Scoring thresholds for assertions
    "task_completion_threshold": 0.7,
    "tool_correctness_threshold": 0.8,
    "argument_correctness_threshold": 0.7,  # New: threshold for tool argument validation
    "step_efficiency_threshold": 0.6,
    "plan_adherence_threshold": 0.7,
    "plan_quality_threshold": 0.6,

    # Retry settings for LLM failures
    "max_retries": 3,
    "retry_delay": 1.0,  # seconds

    # Timeout settings
    "agent_timeout": 120.0,  # seconds per agent invocation
    "evaluation_timeout": 30.0,  # seconds per metric evaluation

    # Parallel execution
    "enable_parallel": False,  # Set to True for parallel test case execution
    "max_workers": 4,  # Number of parallel workers
}


# ============================================================================
# Dataset Paths
# ============================================================================

import os.path as osp

# Project root directory (parent of evaluations/)
PROJECT_ROOT = osp.dirname(osp.dirname(osp.abspath(__file__)))

DATASET_PATHS = {
    "main": osp.join(PROJECT_ROOT, "datasets", "evaluation_dataset.json"),
    "small": osp.join(PROJECT_ROOT, "datasets", "evaluation_dataset_small.json"),
}

RESULTS_DIR = osp.join(PROJECT_ROOT, "results")


# ============================================================================
# Langfuse Configuration
# ============================================================================

def is_langfuse_configured() -> bool:
    """Check if Langfuse is configured via environment variables."""
    return all([
        os.getenv("LANGFUSE_PUBLIC_KEY"),
        os.getenv("LANGFUSE_SECRET_KEY")
    ])


LANGFUSE_SETTINGS = {
    "enabled": is_langfuse_configured(),
    "dataset_name": "agent_evaluation_dataset",
    "trace_tags": ["evaluation"],  # Base tags for all evaluation traces
    "version": "1.0.0"
}


# ============================================================================
# Metric Configuration
# ============================================================================

# G-Eval scoring scale: 1-10, will be normalized to 0.0-1.0
GEVAL_SCALE = (1, 10)

# Metric weights for aggregate scoring (sum to 1.0)
METRIC_WEIGHTS = {
    "task_completion": 0.20,
    "tool_correctness": 0.20,
    "argument_correctness": 0.20,  # New: weight for tool argument validation
    "step_efficiency": 0.15,
    "plan_adherence": 0.15,
    "plan_quality": 0.10,
}


# ============================================================================
# Helper Functions
# ============================================================================

def validate_config() -> bool:
    """
    Validate that all required environment variables are set.

    Returns:
        bool: True if configuration is valid, False otherwise
    """
    required_vars = [
        "AZURE_OPENAI_ENDPOINT",
        "AZURE_OPENAI_API_KEY",
        "AZURE_OPENAI_API_VERSION",
    ]

    missing_vars = [var for var in required_vars if not os.getenv(var)]

    if missing_vars:
        print(f"Error: Missing required environment variables: {', '.join(missing_vars)}")
        return False

    return True


def print_config_summary():
    """Print a summary of the evaluation configuration."""
    print("=" * 80)
    print("DeepEval Evaluation System Configuration")
    print("=" * 80)
    print(f"Evaluation Model: {os.getenv('EVALUATION_MODEL', os.getenv('AZURE_OPENAI_DEPLOYMENT', 'gpt-4o'))}")
    print(f"Temperature: {os.getenv('EVALUATION_TEMPERATURE', '0.0')}")
    print(f"Langfuse Enabled: {is_langfuse_configured()}")
    print(f"Main Dataset: {DATASET_PATHS['main']}")
    print(f"Small Dataset: {DATASET_PATHS['small']}")
    print(f"Results Directory: {RESULTS_DIR}")
    print("=" * 80)
    print("\nMetric Thresholds:")
    for metric, threshold in sorted(EVALUATION_SETTINGS.items()):
        if metric.endswith("_threshold"):
            metric_name = metric.replace("_threshold", "").replace("_", " ").title()
            print(f"  {metric_name}: {threshold}")
    print("=" * 80)


if __name__ == "__main__":
    # Validate and print configuration
    if validate_config():
        print_config_summary()
    else:
        print("\n❌ Configuration validation failed!")