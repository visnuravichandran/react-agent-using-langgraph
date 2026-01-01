"""
Langfuse Reporter for Evaluation Results

Reports evaluation results to Langfuse for centralized tracking and visualization.
"""

import os
from typing import List, Dict, Any, Optional
from datetime import datetime

from evaluations.config import LANGFUSE_SETTINGS, is_langfuse_configured
from evaluations.dataset_generator import EvaluationDataset, TestCase
from evaluations.evaluator import TestResult, EvaluationRunSummary


# ============================================================================
# Langfuse Client Wrapper
# ============================================================================

class LangfuseReporter:
    """Reports evaluation results to Langfuse."""

    def __init__(self, enabled: Optional[bool] = None):
        """
        Initialize Langfuse reporter.

        Args:
            enabled: Whether to enable Langfuse reporting (default: auto-detect from env)
        """
        self.enabled = enabled if enabled is not None else is_langfuse_configured()

        if self.enabled:
            try:
                from langfuse import Langfuse
                self.client = Langfuse(
                    public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
                    secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
                    host=os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com"),
                    tracing_enabled=True  # v3.x uses tracing_enabled instead of enabled
                )
                print("✅ Langfuse reporter initialized")
            except Exception as e:
                print(f"⚠️  Failed to initialize Langfuse: {str(e)}")
                self.enabled = False
                self.client = None
        else:
            self.client = None
            print("ℹ️  Langfuse reporting disabled (credentials not found)")

    def upload_dataset(self, dataset: EvaluationDataset, dataset_name: Optional[str] = None):
        """
        Upload test dataset to Langfuse.

        Args:
            dataset: EvaluationDataset to upload
            dataset_name: Optional name for the dataset (default: from config)
        """
        if not self.enabled or not self.client:
            return

        dataset_name = dataset_name or LANGFUSE_SETTINGS["dataset_name"]

        print(f"\n📤 Uploading dataset to Langfuse: {dataset_name}")

        try:
            # First, try to create the dataset
            print(f"Creating dataset '{dataset_name}'...")
            self.client.create_dataset(
                name=dataset_name,
                description=f"Agent evaluation dataset with {len(dataset.test_cases)} test cases",
                metadata={
                    "version": dataset.version or "1.0",
                    "created_at": datetime.now().isoformat(),
                    "num_test_cases": len(dataset.test_cases),
                }
            )
            print(f"✅ Created dataset: {dataset_name}")

            # Now add items to the dataset
            print(f"Adding {len(dataset.test_cases)} test cases...")
            for i, test_case in enumerate(dataset.test_cases, 1):
                if i % 10 == 0:
                    print(f"  Progress: {i}/{len(dataset.test_cases)} test cases added")

                self.client.create_dataset_item(
                    dataset_name=dataset_name,
                    input=test_case.query,
                    expected_output=test_case.expected_output,
                    metadata={
                        "test_case_id": test_case.id,
                        "category": test_case.category,
                        "expected_tools": test_case.expected_tools,
                        "expected_tool_order": test_case.expected_tool_order,
                        "expected_num_steps": test_case.expected_num_steps,
                        "difficulty": test_case.difficulty,
                        "reasoning": test_case.reasoning,
                    }
                )

            self.client.flush()
            print(f"✅ Uploaded {len(dataset.test_cases)} test cases to Langfuse")

        except Exception as e:
            error_msg = str(e)

            # Check if it's a "dataset already exists" error (resource conflict)
            if "409" in error_msg or "already exists" in error_msg.lower() or "conflict" in error_msg.lower():
                print(f"ℹ️  Dataset '{dataset_name}' already exists. Items will be added to existing dataset.")

                # Try to add items to the existing dataset
                try:
                    print(f"Adding {len(dataset.test_cases)} test cases to existing dataset...")
                    for i, test_case in enumerate(dataset.test_cases, 1):
                        if i % 10 == 0:
                            print(f"  Progress: {i}/{len(dataset.test_cases)} test cases added")

                        self.client.create_dataset_item(
                            dataset_name=dataset_name,
                            input=test_case.query,
                            expected_output=test_case.expected_output,
                            metadata={
                                "test_case_id": test_case.id,
                                "category": test_case.category,
                                "expected_tools": test_case.expected_tools,
                                "expected_tool_order": test_case.expected_tool_order,
                                "expected_num_steps": test_case.expected_num_steps,
                                "difficulty": test_case.difficulty,
                                "reasoning": test_case.reasoning,
                            }
                        )

                    self.client.flush()
                    print(f"✅ Uploaded {len(dataset.test_cases)} test cases to existing dataset")
                except Exception as item_error:
                    print(f"❌ Failed to add items to existing dataset: {str(item_error)}")
                    raise
            else:
                # For other errors, provide more context
                print(f"❌ Failed to upload dataset to Langfuse: {error_msg}")
                print(f"ℹ️  Dataset name: {dataset_name}")
                print(f"ℹ️  Langfuse host: {os.getenv('LANGFUSE_HOST', 'https://cloud.langfuse.com')}")
                raise

    def report_evaluation_run(
        self,
        run_id: str,
        results: List[TestResult],
        summary: EvaluationRunSummary
    ):
        """
        Report evaluation run results to Langfuse.

        Args:
            run_id: Evaluation run ID
            results: List of test results
            summary: Evaluation run summary
        """
        if not self.enabled or not self.client:
            return

        print(f"\n📤 Reporting evaluation run to Langfuse: {run_id}")

        try:
            # Report each test case as a trace
            for result in results:
                self._report_test_result(run_id, result)

            # Report summary as a separate trace
            self._report_summary(run_id, summary)

            self.client.flush()
            print(f"✅ Reported {len(results)} test results to Langfuse")

        except Exception as e:
            print(f"❌ Failed to report to Langfuse: {str(e)}")

    def _report_test_result(self, run_id: str, result: TestResult):
        """Report a single test result as a Langfuse trace."""
        if not self.enabled or not self.client:
            return

        # Create trace for this test case
        trace_tags = LANGFUSE_SETTINGS["trace_tags"].copy()
        trace_tags.extend([
            f"run_{run_id}",
            f"category:{result.category}",
            "evaluation"
        ])

        # In Langfuse v3.x, we use start_span() to create a trace
        # start_span without a trace_context creates a new root span (trace)
        span = self.client.start_span(
            name=f"eval_{result.test_case_id}",
            input=result.query,
            output=result.actual_output,
            metadata={
                "test_case_id": result.test_case_id,
                "category": result.category,
                "expected_tools": result.expected_tools,
                "actual_tools": result.actual_tools,
                "execution_time": result.execution_time,
                "total_steps": result.total_steps,
                "run_id": run_id,
                "tags": trace_tags,  # Tags in metadata for v3
            }
        )

        # Get the trace ID from the span
        trace_id = span._state_client.trace_id if hasattr(span, '_state_client') else None

        # Attach metric scores using the span's score method
        self._attach_scores_to_span(span, result)

        # End the span
        span.end()

    def _attach_scores_to_span(self, span, result: TestResult):
        """Attach metric scores to a span using the v3.x API."""
        if not self.enabled or not span:
            return

        # In Langfuse v3, we use span.score() instead of client.create_score()
        # Task Completion
        span.score(
            name="task_completion",
            value=result.task_completion_score,
            comment=result.task_completion_reason[:500] if result.task_completion_reason else None
        )

        # Tool Correctness
        span.score(
            name="tool_correctness",
            value=result.tool_correctness_score,
            comment=result.tool_correctness_reason[:500] if result.tool_correctness_reason else None
        )

        # Step Efficiency
        span.score(
            name="step_efficiency",
            value=result.step_efficiency_score,
            comment=result.step_efficiency_reason[:500] if result.step_efficiency_reason else None
        )

        # Plan Adherence
        span.score(
            name="plan_adherence",
            value=result.plan_adherence_score,
            comment=result.plan_adherence_reason[:500] if result.plan_adherence_reason else None
        )

        # Plan Quality
        span.score(
            name="plan_quality",
            value=result.plan_quality_score,
            comment=result.plan_quality_reason[:500] if result.plan_quality_reason else None
        )

        # Overall Score
        span.score(
            name="overall",
            value=result.overall_score,
            comment=f"Weighted average. All passed: {result.all_passed}"
        )

    def _report_summary(self, run_id: str, summary: EvaluationRunSummary):
        """Report evaluation summary as a Langfuse trace."""
        if not self.enabled or not self.client:
            return

        trace_tags = LANGFUSE_SETTINGS["trace_tags"].copy()
        trace_tags.extend([
            f"run_{run_id}",
            "evaluation_summary"
        ])

        # In Langfuse v3.x, we use start_span() to create a trace
        span = self.client.start_span(
            name=f"eval_summary_{run_id}",
            input=f"Evaluation run on {summary.dataset_name}",
            output=self._format_summary(summary),
            metadata={
                **summary.to_dict(),
                "tags": trace_tags,  # Tags in metadata for v3
            }
        )

        # End the span
        span.end()

    def _format_summary(self, summary: EvaluationRunSummary) -> str:
        """Format summary as readable text."""
        lines = [
            f"Evaluation Run Summary",
            f"",
            f"Dataset: {summary.dataset_name}",
            f"Test Cases: {summary.total_test_cases}",
            f"Errors: {summary.error_count}",
            f"",
            f"Mean Scores:",
            f"  Overall: {summary.mean_overall:.3f}",
            f"  Task Completion: {summary.mean_task_completion:.3f}",
            f"  Tool Correctness: {summary.mean_tool_correctness:.3f}",
            f"  Step Efficiency: {summary.mean_step_efficiency:.3f}",
            f"  Plan Adherence: {summary.mean_plan_adherence:.3f}",
            f"  Plan Quality: {summary.mean_plan_quality:.3f}",
            f"",
            f"Pass Rates:",
            f"  All Passed: {summary.all_passed_rate*100:.1f}%",
            f"  Task Completion: {summary.task_completion_pass_rate*100:.1f}%",
            f"  Tool Correctness: {summary.tool_correctness_pass_rate*100:.1f}%",
            f"  Step Efficiency: {summary.step_efficiency_pass_rate*100:.1f}%",
            f"  Plan Adherence: {summary.plan_adherence_pass_rate*100:.1f}%",
            f"  Plan Quality: {summary.plan_quality_pass_rate*100:.1f}%",
        ]

        return "\n".join(lines)


# ============================================================================
# Helper Functions
# ============================================================================

def create_reporter(enabled: Optional[bool] = None) -> LangfuseReporter:
    """
    Create a Langfuse reporter.

    Args:
        enabled: Whether to enable reporting (default: auto-detect from env)

    Returns:
        LangfuseReporter: Configured reporter
    """
    return LangfuseReporter(enabled=enabled)


def upload_dataset_to_langfuse(
    dataset_path: str,
    dataset_name: Optional[str] = None
):
    """
    Upload a test dataset to Langfuse.

    Args:
        dataset_path: Path to dataset JSON file
        dataset_name: Optional name for the dataset
    """
    reporter = create_reporter()

    if not reporter.enabled:
        print("⚠️  Langfuse not configured. Skipping upload.")
        return

    dataset = EvaluationDataset.load_from_file(dataset_path)
    dataset.version = LANGFUSE_SETTINGS["version"]
    dataset_name = LANGFUSE_SETTINGS["dataset_name"]
    reporter.upload_dataset(dataset, dataset_name)


# ============================================================================
# CLI
# ============================================================================

def main():
    """CLI entry point."""
    import argparse
    from evaluations.config import DATASET_PATHS

    parser = argparse.ArgumentParser(description="Upload dataset to Langfuse")
    parser.add_argument("--dataset", default="main", help="Dataset to upload: main, small, or path")
    parser.add_argument("--name", help="Dataset name in Langfuse")

    args = parser.parse_args()

    # Resolve dataset path
    if args.dataset in DATASET_PATHS:
        dataset_path = DATASET_PATHS[args.dataset]
    else:
        dataset_path = args.dataset

    upload_dataset_to_langfuse(dataset_path, args.name)


if __name__ == "__main__":
    main()
