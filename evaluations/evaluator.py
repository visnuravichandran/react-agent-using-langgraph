"""
Main Evaluation Runner

Orchestrates the evaluation of the LangGraph ReAct agent using DeepEval's official metrics.
"""

import json
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field

from deepeval.test_case import LLMTestCase, ToolCall
from deepeval.metrics import ToolCorrectnessMetric

# Import local modules
from evaluations.config import (
    DATASET_PATHS,
    RESULTS_DIR,
    EVALUATION_SETTINGS,
    METRIC_WEIGHTS,
    get_evaluation_model,
)
from evaluations.dataset_generator import EvaluationDataset, TestCase
from evaluations.agent_wrapper import create_evaluation_agent
from evaluations.trace_extractor import (
    ExecutionTrace,
    trace_to_deepeval_context,
)
from evaluations.metrics.task_completion import AzureOpenAIModel, create_task_completion_metric
from evaluations.metrics.step_efficiency import create_step_efficiency_metric
from evaluations.metrics.plan_adherence import create_plan_adherence_metric
from evaluations.metrics.plan_quality import create_plan_quality_metric


# ============================================================================
# Data Structures
# ============================================================================

@dataclass
class TestResult:
    """Result for a single test case evaluation."""

    test_case_id: str
    category: str
    query: str
    expected_output: str
    actual_output: str
    expected_tools: List[str]
    actual_tools: List[str]

    # Metric scores
    task_completion_score: float = 0.0
    tool_correctness_score: float = 0.0
    step_efficiency_score: float = 0.0
    plan_adherence_score: float = 0.0
    plan_quality_score: float = 0.0

    # Metric reasons
    task_completion_reason: str = ""
    tool_correctness_reason: str = ""
    step_efficiency_reason: str = ""
    plan_adherence_reason: str = ""
    plan_quality_reason: str = ""

    # Metric success flags
    task_completion_passed: bool = False
    tool_correctness_passed: bool = False
    step_efficiency_passed: bool = False
    plan_adherence_passed: bool = False
    plan_quality_passed: bool = False

    # Aggregate metrics
    overall_score: float = 0.0
    all_passed: bool = False

    # Execution metadata
    execution_time: float = 0.0
    total_steps: int = 0
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "test_case_id": self.test_case_id,
            "category": self.category,
            "query": self.query,
            "expected_output": self.expected_output[:200] + "..." if len(self.expected_output) > 200 else self.expected_output,
            "actual_output": self.actual_output[:200] + "..." if len(self.actual_output) > 200 else self.actual_output,
            "expected_tools": self.expected_tools,
            "actual_tools": self.actual_tools,
            "scores": {
                "task_completion": self.task_completion_score,
                "tool_correctness": self.tool_correctness_score,
                "step_efficiency": self.step_efficiency_score,
                "plan_adherence": self.plan_adherence_score,
                "plan_quality": self.plan_quality_score,
                "overall": self.overall_score,
            },
            "passed": {
                "task_completion": self.task_completion_passed,
                "tool_correctness": self.tool_correctness_passed,
                "step_efficiency": self.step_efficiency_passed,
                "plan_adherence": self.plan_adherence_passed,
                "plan_quality": self.plan_quality_passed,
                "all": self.all_passed,
            },
            "execution": {
                "time": self.execution_time,
                "steps": self.total_steps,
                "error": self.error,
            },
        }


@dataclass
class EvaluationRunSummary:
    """Summary statistics for an evaluation run."""

    run_id: str
    dataset_name: str
    timestamp: str
    total_test_cases: int

    # Aggregate scores
    mean_task_completion: float = 0.0
    mean_tool_correctness: float = 0.0
    mean_step_efficiency: float = 0.0
    mean_plan_adherence: float = 0.0
    mean_plan_quality: float = 0.0
    mean_overall: float = 0.0

    # Pass rates
    task_completion_pass_rate: float = 0.0
    tool_correctness_pass_rate: float = 0.0
    step_efficiency_pass_rate: float = 0.0
    plan_adherence_pass_rate: float = 0.0
    plan_quality_pass_rate: float = 0.0
    all_passed_rate: float = 0.0

    # Category breakdown
    category_scores: Dict[str, Dict[str, float]] = field(default_factory=dict)

    # Execution stats
    total_execution_time: float = 0.0
    mean_execution_time: float = 0.0
    error_count: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "run_id": self.run_id,
            "dataset": self.dataset_name,
            "timestamp": self.timestamp,
            "total_test_cases": self.total_test_cases,
            "mean_scores": {
                "task_completion": self.mean_task_completion,
                "tool_correctness": self.mean_tool_correctness,
                "step_efficiency": self.mean_step_efficiency,
                "plan_adherence": self.mean_plan_adherence,
                "plan_quality": self.mean_plan_quality,
                "overall": self.mean_overall,
            },
            "pass_rates": {
                "task_completion": self.task_completion_pass_rate,
                "tool_correctness": self.tool_correctness_pass_rate,
                "step_efficiency": self.step_efficiency_pass_rate,
                "plan_adherence": self.plan_adherence_pass_rate,
                "plan_quality": self.plan_quality_pass_rate,
                "all_passed": self.all_passed_rate,
            },
            "category_breakdown": self.category_scores,
            "execution_stats": {
                "total_time": self.total_execution_time,
                "mean_time": self.mean_execution_time,
                "error_count": self.error_count,
            },
        }


# ============================================================================
# Evaluation Runner
# ============================================================================

class EvaluationRunner:
    """Main evaluation runner for agent testing."""

    def __init__(
        self,
        dataset_path: Optional[str] = None,
        enable_langfuse: bool = False,
        model_name: str = "gpt-4o",
        verbose: bool = True
    ):
        """
        Initialize the evaluation runner.

        Args:
            dataset_path: Path to dataset JSON file (default: main dataset)
            enable_langfuse: Whether to enable Langfuse tracing
            model_name: Model name for agent
            verbose: Whether to print verbose output
        """
        self.dataset_path = dataset_path or DATASET_PATHS["main"]
        self.enable_langfuse = enable_langfuse
        self.model_name = model_name
        self.verbose = verbose

        # Load dataset
        self.dataset = EvaluationDataset.load_from_file(self.dataset_path)

        # Initialize agent wrapper
        self.agent_wrapper = create_evaluation_agent(
            model_name=model_name,
            enable_langfuse=enable_langfuse,
            user_id="evaluation_system"
        )

        # Initialize evaluation model for metrics
        # We use AzureOpenAIModel wrapper to ensure all metrics use Azure OpenAI
        azure_model = get_evaluation_model()
        azure_deepeval_model = AzureOpenAIModel(azure_model)

        # HYBRID APPROACH:
        # - ToolCorrectnessMetric: Use DeepEval's official metric (works with LLMTestCase)
        # - Other metrics: Use custom implementations (DeepEval's trace-based metrics
        #   require @observe decorator + evals_iterator, which doesn't fit our evaluation pattern)
        self.metrics = {
            "task_completion": create_task_completion_metric(
                threshold=EVALUATION_SETTINGS["task_completion_threshold"]
            ),
            "tool_correctness": ToolCorrectnessMetric(
                threshold=EVALUATION_SETTINGS["tool_correctness_threshold"],
                model=azure_deepeval_model,  # Pass Azure model object
                include_reason=True,
                strict_mode=False,
                should_consider_ordering=True,
                should_exact_match=False,
                verbose_mode=False
            ),
            "step_efficiency": create_step_efficiency_metric(
                threshold=EVALUATION_SETTINGS["step_efficiency_threshold"]
            ),
            "plan_adherence": create_plan_adherence_metric(
                threshold=EVALUATION_SETTINGS["plan_adherence_threshold"]
            ),
            "plan_quality": create_plan_quality_metric(
                threshold=EVALUATION_SETTINGS["plan_quality_threshold"]
            ),
        }

        # Results storage
        self.results: List[TestResult] = []
        self.run_id = str(uuid.uuid4())[:8]
        self.start_time = None
        self.end_time = None

    def run_evaluation(self) -> EvaluationRunSummary:
        """
        Run evaluation on all test cases in the dataset.

        Returns:
            EvaluationRunSummary: Summary of evaluation results
        """
        self.start_time = datetime.now()

        if self.verbose:
            print("="*80)
            print(f"Starting Evaluation Run: {self.run_id}")
            print(f"Dataset: {self.dataset_path}")
            print(f"Test Cases: {len(self.dataset.test_cases)}")
            print("="*80)

        # Run evaluation on each test case
        for i, test_case in enumerate(self.dataset.test_cases, 1):
            if self.verbose:
                print(f"\n[{i}/{len(self.dataset.test_cases)}] Evaluating: {test_case.id} ({test_case.category})")

            try:
                result = self._evaluate_test_case(test_case)
                self.results.append(result)

                if self.verbose:
                    self._print_test_result(result)

            except Exception as e:
                if self.verbose:
                    print(f"  ❌ Error: {str(e)}")

                # Create error result
                error_result = TestResult(
                    test_case_id=test_case.id,
                    category=test_case.category,
                    query=test_case.query,
                    expected_output=test_case.expected_output,
                    actual_output="",
                    expected_tools=test_case.expected_tools,
                    actual_tools=[],
                    error=str(e)
                )
                self.results.append(error_result)

        self.end_time = datetime.now()

        # Calculate summary
        summary = self._calculate_summary()

        if self.verbose:
            self._print_summary(summary)

        return summary

    def _evaluate_test_case(self, test_case: TestCase) -> TestResult:
        """Evaluate a single test case with hybrid metrics (DeepEval + Custom)."""
        # Run agent
        agent_result, trace = self.agent_wrapper.invoke_with_trace(test_case.query)

        # Create DeepEval test case with retrieval_context for custom metrics
        retrieval_context = trace_to_deepeval_context(trace)

        # Convert tools to ToolCall objects for DeepEval's ToolCorrectnessMetric
        tools_called = [ToolCall(name=tool_name) for tool_name in trace.get_tool_names()]
        expected_tools = [ToolCall(name=tool_name) for tool_name in test_case.expected_tools]

        # Create test case with both retrieval_context and tool information
        deepeval_test_case = LLMTestCase(
            input=test_case.query,
            actual_output=trace.final_answer,
            expected_output=test_case.expected_output,
            retrieval_context=retrieval_context,
            tools_called=tools_called,
            expected_tools=expected_tools,
        )

        # Run metrics
        scores = {}
        reasons = {}
        passed = {}

        for metric_name, metric in self.metrics.items():
            try:
                if self.verbose:
                    print(f"  Evaluating {metric_name}...", end=" ")

                # Call measure() to evaluate
                metric.measure(deepeval_test_case)

                # Get score from metric.score attribute (DeepEval pattern)
                # Some metrics return score from measure(), others store it in metric.score
                score = getattr(metric, "score", None)

                # Ensure score is a valid float (not None)
                if score is None:
                    if self.verbose:
                        print(f"returned None, using 0.0")
                    score = 0.0
                else:
                    if self.verbose:
                        print(f"✅ {score:.2f}")

                scores[metric_name] = float(score)
                reasons[metric_name] = getattr(metric, "reason", "")
                passed[metric_name] = metric.is_successful()

            except Exception as e:
                # If metric fails, log error and assign 0.0 score
                scores[metric_name] = 0.0
                reasons[metric_name] = f"Metric evaluation failed: {str(e)}"
                passed[metric_name] = False

                if self.verbose:
                    print(f"❌ {str(e)[:100]}")

        # Calculate overall score (weighted average)
        # Ensure all scores are valid floats before multiplication
        overall_score = sum(
            float(scores.get(metric, 0.0)) * float(METRIC_WEIGHTS.get(metric, 0.0))
            for metric in scores.keys()
        )

        # Create result
        result = TestResult(
            test_case_id=test_case.id,
            category=test_case.category,
            query=test_case.query,
            expected_output=test_case.expected_output,
            actual_output=trace.final_answer,
            expected_tools=test_case.expected_tools,
            actual_tools=trace.get_tool_names(),

            task_completion_score=scores["task_completion"],
            tool_correctness_score=scores["tool_correctness"],
            step_efficiency_score=scores["step_efficiency"],
            plan_adherence_score=scores["plan_adherence"],
            plan_quality_score=scores["plan_quality"],

            task_completion_reason=reasons["task_completion"],
            tool_correctness_reason=reasons["tool_correctness"],
            step_efficiency_reason=reasons["step_efficiency"],
            plan_adherence_reason=reasons["plan_adherence"],
            plan_quality_reason=reasons["plan_quality"],

            task_completion_passed=passed["task_completion"],
            tool_correctness_passed=passed["tool_correctness"],
            step_efficiency_passed=passed["step_efficiency"],
            plan_adherence_passed=passed["plan_adherence"],
            plan_quality_passed=passed["plan_quality"],

            overall_score=overall_score,
            all_passed=all(passed.values()),

            execution_time=trace.execution_time,
            total_steps=trace.total_steps,
        )

        return result

    def _calculate_summary(self) -> EvaluationRunSummary:
        """Calculate summary statistics."""
        # Filter out error results
        valid_results = [r for r in self.results if r.error is None]

        if not valid_results:
            return EvaluationRunSummary(
                run_id=self.run_id,
                dataset_name=Path(self.dataset_path).name,
                timestamp=self.start_time.isoformat() if self.start_time else "",
                total_test_cases=len(self.results),
                error_count=len(self.results),
            )

        n = len(valid_results)

        # Calculate mean scores
        summary = EvaluationRunSummary(
            run_id=self.run_id,
            dataset_name=Path(self.dataset_path).name,
            timestamp=self.start_time.isoformat() if self.start_time else "",
            total_test_cases=len(self.results),

            mean_task_completion=sum(r.task_completion_score for r in valid_results) / n,
            mean_tool_correctness=sum(r.tool_correctness_score for r in valid_results) / n,
            mean_step_efficiency=sum(r.step_efficiency_score for r in valid_results) / n,
            mean_plan_adherence=sum(r.plan_adherence_score for r in valid_results) / n,
            mean_plan_quality=sum(r.plan_quality_score for r in valid_results) / n,
            mean_overall=sum(r.overall_score for r in valid_results) / n,

            task_completion_pass_rate=sum(r.task_completion_passed for r in valid_results) / n,
            tool_correctness_pass_rate=sum(r.tool_correctness_passed for r in valid_results) / n,
            step_efficiency_pass_rate=sum(r.step_efficiency_passed for r in valid_results) / n,
            plan_adherence_pass_rate=sum(r.plan_adherence_passed for r in valid_results) / n,
            plan_quality_pass_rate=sum(r.plan_quality_passed for r in valid_results) / n,
            all_passed_rate=sum(r.all_passed for r in valid_results) / n,

            total_execution_time=sum(r.execution_time for r in valid_results),
            mean_execution_time=sum(r.execution_time for r in valid_results) / n,
            error_count=len(self.results) - len(valid_results),
        )

        # Calculate category breakdown
        categories = set(r.category for r in valid_results)
        for category in categories:
            cat_results = [r for r in valid_results if r.category == category]
            cat_n = len(cat_results)

            summary.category_scores[category] = {
                "count": cat_n,
                "task_completion": sum(r.task_completion_score for r in cat_results) / cat_n,
                "tool_correctness": sum(r.tool_correctness_score for r in cat_results) / cat_n,
                "step_efficiency": sum(r.step_efficiency_score for r in cat_results) / cat_n,
                "plan_adherence": sum(r.plan_adherence_score for r in cat_results) / cat_n,
                "plan_quality": sum(r.plan_quality_score for r in cat_results) / cat_n,
                "overall": sum(r.overall_score for r in cat_results) / cat_n,
            }

        return summary

    def _print_test_result(self, result: TestResult):
        """Print result for a single test case."""
        if result.error:
            print(f"  ❌ Error: {result.error}")
            return

        print(f"  Overall: {result.overall_score:.2f} {'✅' if result.all_passed else '❌'}")
        print(f"    Task Completion: {result.task_completion_score:.2f} {'✅' if result.task_completion_passed else '❌'}")
        print(f"    Tool Correctness: {result.tool_correctness_score:.2f} {'✅' if result.tool_correctness_passed else '❌'}")
        print(f"    Step Efficiency: {result.step_efficiency_score:.2f} {'✅' if result.step_efficiency_passed else '❌'}")
        print(f"    Plan Adherence: {result.plan_adherence_score:.2f} {'✅' if result.plan_adherence_passed else '❌'}")
        print(f"    Plan Quality: {result.plan_quality_score:.2f} {'✅' if result.plan_quality_passed else '❌'}")
        print(f"  Tools: {result.actual_tools} (expected: {result.expected_tools})")

    def _print_summary(self, summary: EvaluationRunSummary):
        """Print evaluation summary."""
        print("\n" + "="*80)
        print("EVALUATION SUMMARY")
        print("="*80)
        print(f"\nRun ID: {summary.run_id}")
        print(f"Dataset: {summary.dataset_name}")
        print(f"Test Cases: {summary.total_test_cases}")
        print(f"Errors: {summary.error_count}")
        print(f"Total Time: {summary.total_execution_time:.2f}s")
        print(f"Mean Time per Case: {summary.mean_execution_time:.2f}s")

        print("\n" + "-"*80)
        print("MEAN SCORES")
        print("-"*80)
        print(f"Overall:          {summary.mean_overall:.3f}")
        print(f"Task Completion:  {summary.mean_task_completion:.3f}")
        print(f"Tool Correctness: {summary.mean_tool_correctness:.3f}")
        print(f"Step Efficiency:  {summary.mean_step_efficiency:.3f}")
        print(f"Plan Adherence:   {summary.mean_plan_adherence:.3f}")
        print(f"Plan Quality:     {summary.mean_plan_quality:.3f}")

        print("\n" + "-"*80)
        print("PASS RATES")
        print("-"*80)
        print(f"All Passed:       {summary.all_passed_rate*100:.1f}%")
        print(f"Task Completion:  {summary.task_completion_pass_rate*100:.1f}%")
        print(f"Tool Correctness: {summary.tool_correctness_pass_rate*100:.1f}%")
        print(f"Step Efficiency:  {summary.step_efficiency_pass_rate*100:.1f}%")
        print(f"Plan Adherence:   {summary.plan_adherence_pass_rate*100:.1f}%")
        print(f"Plan Quality:     {summary.plan_quality_pass_rate*100:.1f}%")

        if summary.category_scores:
            print("\n" + "-"*80)
            print("CATEGORY BREAKDOWN")
            print("-"*80)
            for category, scores in sorted(summary.category_scores.items()):
                print(f"\n{category} (n={scores['count']}):")
                print(f"  Overall: {scores['overall']:.3f}")
                print(f"  Task Completion: {scores['task_completion']:.3f}")
                print(f"  Tool Correctness: {scores['tool_correctness']:.3f}")

        print("\n" + "="*80)

    def save_results(self, output_dir: Optional[str] = None):
        """Save results to JSON and CSV files."""
        output_dir = Path(output_dir or RESULTS_DIR)
        output_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Save detailed JSON
        json_path = output_dir / f"eval_run_{self.run_id}_{timestamp}.json"
        with open(json_path, "w") as f:
            json.dump(
                {
                    "run_id": self.run_id,
                    "timestamp": timestamp,
                    "dataset": self.dataset_path,
                    "results": [r.to_dict() for r in self.results],
                    "summary": self._calculate_summary().to_dict(),
                },
                f,
                indent=2
            )

        if self.verbose:
            print(f"\n✅ Saved detailed results to: {json_path}")

        # Save summary CSV
        csv_path = output_dir / f"eval_run_{self.run_id}_{timestamp}.csv"
        self._save_csv(csv_path)

        if self.verbose:
            print(f"✅ Saved summary CSV to: {csv_path}")

    def _save_csv(self, csv_path: Path):
        """Save results to CSV format."""
        import csv

        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)

            # Header
            writer.writerow([
                "test_id", "category", "query",
                "task_completion", "tool_correctness", "step_efficiency",
                "plan_adherence", "plan_quality", "overall",
                "all_passed", "expected_tools", "actual_tools",
                "execution_time", "error"
            ])

            # Data
            for r in self.results:
                writer.writerow([
                    r.test_case_id,
                    r.category,
                    r.query[:100],
                    f"{r.task_completion_score:.3f}",
                    f"{r.tool_correctness_score:.3f}",
                    f"{r.step_efficiency_score:.3f}",
                    f"{r.plan_adherence_score:.3f}",
                    f"{r.plan_quality_score:.3f}",
                    f"{r.overall_score:.3f}",
                    "yes" if r.all_passed else "no",
                    ",".join(r.expected_tools),
                    ",".join(r.actual_tools),
                    f"{r.execution_time:.2f}",
                    r.error or ""
                ])


# ============================================================================
# Helper Functions
# ============================================================================

def run_evaluation(
    dataset: str = "main",
    enable_langfuse: bool = False,
    verbose: bool = True,
    save_results: bool = True
) -> EvaluationRunSummary:
    """
    Run evaluation with default settings.

    Args:
        dataset: "main" or "small" or path to dataset
        enable_langfuse: Whether to enable Langfuse tracing
        verbose: Whether to print verbose output
        save_results: Whether to save results to files

    Returns:
        EvaluationRunSummary: Summary of results
    """
    # Resolve dataset path
    if dataset in DATASET_PATHS:
        dataset_path = DATASET_PATHS[dataset]
    else:
        dataset_path = dataset

    # Create runner
    runner = EvaluationRunner(
        dataset_path=dataset_path,
        enable_langfuse=enable_langfuse,
        verbose=verbose
    )

    # Run evaluation
    summary = runner.run_evaluation()

    # Save results
    if save_results:
        runner.save_results()

    return summary


# ============================================================================
# CLI
# ============================================================================

def main():
    """CLI entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Run agent evaluation")
    parser.add_argument("--dataset", default="small", help="Dataset to use: main, small, or path")
    parser.add_argument("--langfuse", action="store_true", help="Enable Langfuse tracing")
    parser.add_argument("--no-save", action="store_true", help="Don't save results to files")
    parser.add_argument("--quiet", action="store_true", help="Minimal output")

    args = parser.parse_args()

    run_evaluation(
        dataset=args.dataset,
        enable_langfuse=args.langfuse,
        verbose=not args.quiet,
        save_results=not args.no_save
    )


if __name__ == "__main__":
    main()
