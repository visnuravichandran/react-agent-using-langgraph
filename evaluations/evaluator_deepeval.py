"""
DeepEval Official Metrics Evaluation Runner

This module uses DeepEval's out-of-the-box metrics with evals_iterator:
- TaskCompletionMetric
- ToolCorrectnessMetric
- StepEfficiencyMetric
- PlanAdherenceMetric
- PlanQualityMetric
"""

import json
import uuid
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional

from deepeval import evaluate
from deepeval.test_case import LLMTestCase, ToolCall
from deepeval.dataset import EvaluationDataset as DeepEvalDataset, Golden
from deepeval.metrics import (
    TaskCompletionMetric,
    ToolCorrectnessMetric,
    StepEfficiencyMetric,
    PlanAdherenceMetric,
    PlanQualityMetric,
)

# Import local modules
from evaluations.config import (
    DATASET_PATHS,
    RESULTS_DIR,
    EVALUATION_SETTINGS,
    get_evaluation_model,
)
from evaluations.dataset_generator import EvaluationDataset
from evaluations.agent_wrapper import create_evaluation_agent
from evaluations.trace_extractor import extract_execution_trace
from evaluations.model_wrapper import AzureOpenAIModel


# ============================================================================
# Metric Initialization
# ============================================================================

def create_deepeval_metrics(
    model: Optional[AzureOpenAIModel] = None,
    verbose: bool = False
) -> Dict[str, Any]:
    """
    Create DeepEval's official metrics with configured thresholds using Azure OpenAI.

    Args:
        model: Optional AzureOpenAIModel wrapper (if None, creates one)
        verbose: Whether to enable verbose mode

    Returns:
        Dict of metric name to metric instance
    """
    # Initialize evaluation model wrapper if not provided
    # This ensures all metrics use Azure OpenAI for evaluation
    if model is None:
        azure_model = get_evaluation_model()
        model = AzureOpenAIModel(azure_model)

    # Pass the AzureOpenAIModel object (implements DeepEvalBaseLLM) to ensure
    # all LLM-based metrics use Azure OpenAI instead of default OpenAI
    metrics = {
        "task_completion": TaskCompletionMetric(
            threshold=EVALUATION_SETTINGS["task_completion_threshold"],
            model=model,  # Pass Azure model object, not string
            include_reason=True,
            strict_mode=False,
            verbose_mode=verbose
        ),
        "tool_correctness": ToolCorrectnessMetric(
            threshold=EVALUATION_SETTINGS["tool_correctness_threshold"],
            model=model,  # Pass Azure model object
            include_reason=True,
            strict_mode=False,
            should_consider_ordering=True,
            should_exact_match=False,
            verbose_mode=verbose
        ),
        "step_efficiency": StepEfficiencyMetric(
            threshold=EVALUATION_SETTINGS["step_efficiency_threshold"],
            model=model,  # Pass Azure model object, not string
            include_reason=True,
            strict_mode=False,
            verbose_mode=verbose
        ),
        "plan_adherence": PlanAdherenceMetric(
            threshold=EVALUATION_SETTINGS["plan_adherence_threshold"],
            model=model,  # Pass Azure model object, not string
            include_reason=True,
            strict_mode=False,
            verbose_mode=verbose
        ),
        "plan_quality": PlanQualityMetric(
            threshold=EVALUATION_SETTINGS["plan_quality_threshold"],
            model=model,  # Pass Azure model object, not string
            include_reason=True,
            strict_mode=False,
            verbose_mode=verbose
        ),
    }

    return metrics


# ============================================================================
# Dataset Conversion
# ============================================================================

def convert_to_deepeval_dataset(
    dataset_path: str,
    agent_wrapper: Any
) -> DeepEvalDataset:
    """
    Convert evaluation dataset to DeepEval's Golden format.

    Args:
        dataset_path: Path to JSON dataset file
        agent_wrapper: Agent wrapper for running test cases

    Returns:
        DeepEvalDataset with Golden test cases
    """
    # Load custom dataset
    custom_dataset = EvaluationDataset.load_from_file(dataset_path)

    # Convert to Golden format
    goldens = []
    for test_case in custom_dataset.test_cases:
        # For trace-based metrics, we only need input and expected_output
        # The @observe decorator will capture the trace automatically
        golden = Golden(
            input=test_case.query,
            expected_output=test_case.expected_output,
            # Store additional metadata for tool correctness metric
            additional_metadata={
                "test_case_id": test_case.id,
                "category": test_case.category,
                "expected_tools": test_case.expected_tools,
            }
        )
        goldens.append(golden)

    return DeepEvalDataset(
        goldens=goldens,
        name=Path(dataset_path).stem
    )


# ============================================================================
# Evaluation Runner with DeepEval's evals_iterator
# ============================================================================

class DeepEvalRunner:
    """Evaluation runner using DeepEval's official metrics and evals_iterator."""

    def __init__(
        self,
        dataset_path: Optional[str] = None,
        enable_langfuse: bool = False,
        model_name: str = "gpt-4o",
        verbose: bool = True
    ):
        """
        Initialize the DeepEval runner.

        Args:
            dataset_path: Path to dataset JSON file
            enable_langfuse: Whether to enable Langfuse tracing
            model_name: Model name for agent
            verbose: Whether to print verbose output
        """
        self.dataset_path = dataset_path or DATASET_PATHS["main"]
        self.enable_langfuse = enable_langfuse
        self.model_name = model_name
        self.verbose = verbose

        # Initialize agent wrapper
        self.agent_wrapper = create_evaluation_agent(
            model_name=model_name,
            enable_langfuse=enable_langfuse,
            user_id="evaluation_system"
        )

        # Initialize metrics
        self.metrics = create_deepeval_metrics(verbose=verbose)

        # Results storage
        self.run_id = str(uuid.uuid4())[:8]
        self.start_time = None
        self.end_time = None

    def run_evaluation_with_trace_metrics(self) -> Dict[str, Any]:
        """
        Run evaluation using DeepEval's trace-based metrics with evals_iterator.

        This method uses the @observe decorator on invoke_for_deepeval to enable
        trace-based metrics (TaskCompletion, StepEfficiency, PlanAdherence, PlanQuality).

        Returns:
            Dict with evaluation results
        """
        self.start_time = datetime.now()

        if self.verbose:
            print("="*80)
            print(f"Starting DeepEval Evaluation Run: {self.run_id}")
            print(f"Dataset: {self.dataset_path}")
            print("="*80)

        # Convert dataset to DeepEval format
        dataset = convert_to_deepeval_dataset(self.dataset_path, self.agent_wrapper)

        if self.verbose:
            print(f"Test Cases: {len(dataset.goldens)}")
            print("="*80)

        # Select trace-based metrics only for evals_iterator
        trace_metrics = [
            self.metrics["task_completion"],
            self.metrics["step_efficiency"],
            self.metrics["plan_adherence"],
            self.metrics["plan_quality"],
        ]

        # Run evaluation with evals_iterator
        # This will automatically call invoke_for_deepeval with @observe decorator
        results = []
        for i, golden in enumerate(dataset.evals_iterator(metrics=trace_metrics), 1):
            if self.verbose:
                print(f"\n[{i}/{len(dataset.goldens)}] Evaluating: {golden.additional_metadata.get('test_case_id', 'unknown')}")

            try:
                # Call agent with @observe decorator
                actual_output = self.agent_wrapper.invoke_for_deepeval(golden.input)

                # The metrics will be automatically evaluated by DeepEval
                # Results are accessible via golden.test_case after evaluation

                if self.verbose:
                    print(f"  Output: {actual_output[:100]}...")

                results.append({
                    "test_case_id": golden.additional_metadata.get("test_case_id"),
                    "category": golden.additional_metadata.get("category"),
                    "input": golden.input,
                    "actual_output": actual_output,
                    "expected_output": golden.expected_output,
                })

            except Exception as e:
                if self.verbose:
                    print(f"  ❌ Error: {str(e)}")
                results.append({
                    "test_case_id": golden.additional_metadata.get("test_case_id"),
                    "error": str(e)
                })

        self.end_time = datetime.now()

        return {
            "run_id": self.run_id,
            "dataset": Path(self.dataset_path).name,
            "timestamp": self.start_time.isoformat(),
            "results": results,
            "total_test_cases": len(results),
        }

    def run_evaluation_with_tool_correctness(self) -> Dict[str, Any]:
        """
        Run evaluation using ToolCorrectnessMetric separately (non-trace-based).

        This metric doesn't require @observe decorator and works with standalone
        LLMTestCase objects.

        Returns:
            Dict with tool correctness evaluation results
        """
        if self.verbose:
            print("\n" + "="*80)
            print("Running Tool Correctness Evaluation")
            print("="*80)

        # Load dataset
        custom_dataset = EvaluationDataset.load_from_file(self.dataset_path)
        tool_metric = self.metrics["tool_correctness"]

        results = []
        for i, test_case in enumerate(custom_dataset.test_cases, 1):
            if self.verbose:
                print(f"\n[{i}/{len(custom_dataset.test_cases)}] Evaluating: {test_case.id}")

            try:
                # Run agent to get actual tools called
                agent_result, trace = self.agent_wrapper.invoke_with_trace(test_case.query)

                # Create LLMTestCase for ToolCorrectnessMetric
                # Note: ToolCorrectnessMetric expects tools_called and expected_tools
                from deepeval.models.tool_call import ToolCall

                # Convert tool names to ToolCall objects
                tools_called = [ToolCall(name=tool_name) for tool_name in trace.get_tool_names()]
                expected_tools = [ToolCall(name=tool_name) for tool_name in test_case.expected_tools]

                llm_test_case = LLMTestCase(
                    input=test_case.query,
                    actual_output=trace.final_answer,
                    tools_called=tools_called,
                    expected_tools=expected_tools,
                )

                # Measure metric
                score = tool_metric.measure(llm_test_case)

                results.append({
                    "test_case_id": test_case.id,
                    "category": test_case.category,
                    "expected_tools": test_case.expected_tools,
                    "actual_tools": trace.get_tool_names(),
                    "score": score,
                    "passed": tool_metric.is_successful(),
                    "reason": getattr(tool_metric, "reason", ""),
                })

                if self.verbose:
                    print(f"  Score: {score:.2f} {'✅' if tool_metric.is_successful() else '❌'}")
                    print(f"  Expected: {test_case.expected_tools}")
                    print(f"  Actual: {trace.get_tool_names()}")

            except Exception as e:
                if self.verbose:
                    print(f"  ❌ Error: {str(e)}")
                results.append({
                    "test_case_id": test_case.id,
                    "error": str(e)
                })

        return {
            "run_id": self.run_id,
            "metric": "tool_correctness",
            "results": results,
        }

    def run_full_evaluation(self) -> Dict[str, Any]:
        """
        Run complete evaluation with all 5 DeepEval metrics.

        Returns:
            Dict with comprehensive evaluation results
        """
        if self.verbose:
            print("\n" + "="*80)
            print("Starting Full DeepEval Evaluation (All 5 Metrics)")
            print("="*80)

        # Run trace-based metrics (TaskCompletion, StepEfficiency, PlanAdherence, PlanQuality)
        trace_results = self.run_evaluation_with_trace_metrics()

        # Run tool correctness metric separately
        tool_results = self.run_evaluation_with_tool_correctness()

        # Combine results
        combined_results = {
            "run_id": self.run_id,
            "dataset": Path(self.dataset_path).name,
            "timestamp": self.start_time.isoformat() if self.start_time else datetime.now().isoformat(),
            "trace_based_metrics": trace_results,
            "tool_correctness_metric": tool_results,
        }

        if self.verbose:
            print("\n" + "="*80)
            print("Evaluation Complete!")
            print("="*80)

        return combined_results

    def save_results(self, results: Dict[str, Any], output_dir: Optional[str] = None):
        """Save evaluation results to JSON file."""
        output_dir = Path(output_dir or RESULTS_DIR)
        output_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        json_path = output_dir / f"deepeval_run_{self.run_id}_{timestamp}.json"

        with open(json_path, "w") as f:
            json.dump(results, f, indent=2)

        if self.verbose:
            print(f"\n✅ Saved results to: {json_path}")


# ============================================================================
# Helper Functions
# ============================================================================

def run_deepeval_evaluation(
    dataset: str = "main",
    enable_langfuse: bool = False,
    verbose: bool = True,
    save_results: bool = True
) -> Dict[str, Any]:
    """
    Run evaluation with DeepEval's official metrics.

    Args:
        dataset: "main" or "small" or path to dataset
        enable_langfuse: Whether to enable Langfuse tracing
        verbose: Whether to print verbose output
        save_results: Whether to save results to files

    Returns:
        Dict with evaluation results
    """
    # Resolve dataset path
    if dataset in DATASET_PATHS:
        dataset_path = DATASET_PATHS[dataset]
    else:
        dataset_path = dataset

    # Create runner
    runner = DeepEvalRunner(
        dataset_path=dataset_path,
        enable_langfuse=enable_langfuse,
        verbose=verbose
    )

    # Run full evaluation
    results = runner.run_full_evaluation()

    # Save results
    if save_results:
        runner.save_results(results)

    return results


# ============================================================================
# CLI
# ============================================================================

def main():
    """CLI entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Run evaluation with DeepEval official metrics")
    parser.add_argument("--dataset", default="small", help="Dataset to use: main, small, or path")
    parser.add_argument("--langfuse", action="store_true", help="Enable Langfuse tracing")
    parser.add_argument("--no-save", action="store_true", help="Don't save results to files")
    parser.add_argument("--quiet", action="store_true", help="Minimal output")

    args = parser.parse_args()

    run_deepeval_evaluation(
        dataset=args.dataset,
        enable_langfuse=args.langfuse,
        verbose=not args.quiet,
        save_results=not args.no_save
    )


if __name__ == "__main__":
    main()
