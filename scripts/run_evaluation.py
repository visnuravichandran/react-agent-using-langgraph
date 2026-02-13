#!/usr/bin/env python
"""
Evaluation CLI Script

Main entry point for running agent evaluations from the command line.
"""

import sys
import os
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
from evaluations.evaluator import EvaluationRunner
from evaluations.langfuse_reporter import LangfuseReporter
from evaluations.config import (
    DATASET_PATHS,
    validate_config,
    print_config_summary,
    EVALUATION_SETTINGS
)


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Run DeepEval evaluation on LangGraph ReAct agent",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run evaluation on small dataset (quick test)
  python scripts/run_evaluation.py --dataset small

  # Run evaluation on full dataset with Langfuse reporting
  python scripts/run_evaluation.py --dataset main --langfuse

  # Run with custom dataset and save to specific directory
  python scripts/run_evaluation.py --dataset /path/to/dataset.json --output /path/to/results

  # Run in quiet mode (minimal output)
  python scripts/run_evaluation.py --dataset small --quiet

  # Validate configuration only
  python scripts/run_evaluation.py --check-config
        """
    )

    # Dataset options
    parser.add_argument(
        "--dataset",
        default="small",
        help="Dataset to use: 'main', 'small', or path to custom dataset JSON (default: small)"
    )

    # Output options
    parser.add_argument(
        "--output",
        default=None,
        help="Output directory for results (default: results/)"
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Don't save results to files"
    )

    # Langfuse options
    parser.add_argument(
        "--langfuse",
        action="store_true",
        help="Enable Langfuse reporting (requires LANGFUSE_* env vars)"
    )
    parser.add_argument(
        "--upload-dataset",
        action="store_true",
        help="Upload dataset to Langfuse before evaluation"
    )

    # Model options
    parser.add_argument(
        "--model",
        default="gpt-4o",
        help="Azure OpenAI model deployment name (default: gpt-4o)"
    )

    # Display options
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Minimal output (only summary)"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Verbose output (detailed per-test results)"
    )

    # Configuration options
    parser.add_argument(
        "--check-config",
        action="store_true",
        help="Validate configuration and exit"
    )
    parser.add_argument(
        "--show-config",
        action="store_true",
        help="Show configuration summary and exit"
    )

    # Metric threshold overrides
    parser.add_argument(
        "--threshold-task-completion",
        type=float,
        help=f"Task completion threshold (default: {EVALUATION_SETTINGS['task_completion_threshold']})"
    )
    parser.add_argument(
        "--threshold-tool-correctness",
        type=float,
        help=f"Tool correctness threshold (default: {EVALUATION_SETTINGS['tool_correctness_threshold']})"
    )
    parser.add_argument(
        "--threshold-argument-correctness",
        type=float,
        help=f"Argument correctness threshold (default: {EVALUATION_SETTINGS['argument_correctness_threshold']})"
    )

    args = parser.parse_args()

    # Handle configuration commands
    if args.check_config:
        print("Checking configuration...")
        if validate_config():
            print("✅ Configuration is valid")
            return 0
        else:
            print("❌ Configuration is invalid")
            return 1

    if args.show_config:
        print_config_summary()
        return 0

    # Validate configuration before running
    if not validate_config():
        print("\n❌ Configuration validation failed!")
        print("Please check your environment variables.")
        print("See evaluations/ENVIRONMENT_VARIABLES.md for details.")
        return 1

    # Resolve dataset path
    if args.dataset in DATASET_PATHS:
        dataset_path = DATASET_PATHS[args.dataset]
        dataset_name = args.dataset
    else:
        dataset_path = args.dataset
        dataset_name = Path(args.dataset).stem

    # Check if dataset exists
    if not Path(dataset_path).exists():
        print(f"❌ Dataset not found: {dataset_path}")
        return 1

    # Determine verbosity
    if args.quiet:
        verbose = False
    elif args.verbose:
        verbose = True
    else:
        verbose = True  # Default to verbose

    print("\n" + "="*80)
    print("AGENT EVALUATION RUNNER")
    print("="*80)
    print(f"Dataset: {dataset_name} ({dataset_path})")
    print(f"Model: {args.model}")
    print(f"Langfuse: {'enabled' if args.langfuse else 'disabled'}")
    print(f"Output: {'disabled' if args.no_save else args.output or 'results/'}")
    print("="*80)

    # Upload dataset to Langfuse if requested
    if args.upload_dataset and args.langfuse:
        from evaluations.dataset_generator import EvaluationDataset

        print("\n📤 Uploading dataset to Langfuse...")
        reporter = LangfuseReporter(enabled=True)
        dataset = EvaluationDataset.load_from_file(dataset_path)
        reporter.upload_dataset(dataset)

    # Apply threshold overrides if provided
    if args.threshold_task_completion is not None:
        EVALUATION_SETTINGS["task_completion_threshold"] = args.threshold_task_completion
    if args.threshold_tool_correctness is not None:
        EVALUATION_SETTINGS["tool_correctness_threshold"] = args.threshold_tool_correctness
    if args.threshold_argument_correctness is not None:
        EVALUATION_SETTINGS["argument_correctness_threshold"] = args.threshold_argument_correctness

    # Create evaluation runner
    runner = EvaluationRunner(
        dataset_path=dataset_path,
        enable_langfuse=args.langfuse,
        model_name=args.model,
        verbose=verbose
    )

    # Run evaluation
    try:
        summary = runner.run_evaluation()

        # Save results
        if not args.no_save:
            runner.save_results(output_dir=args.output)

        # Report to Langfuse
        if args.langfuse:
            reporter = LangfuseReporter(enabled=True)
            reporter.report_evaluation_run(
                run_id=runner.run_id,
                results=runner.results,
                summary=summary
            )

        # Print final summary
        print("\n" + "="*80)
        print("✅ EVALUATION COMPLETE")
        print("="*80)
        print(f"Run ID: {runner.run_id}")
        print(f"Overall Score: {summary.mean_overall:.3f}")
        print(f"Pass Rate: {summary.all_passed_rate*100:.1f}%")
        print(f"Test Cases: {summary.total_test_cases}")
        print(f"Errors: {summary.error_count}")

        if not args.no_save:
            print(f"\n📁 Results saved to: {args.output or 'results/'}")

        if args.langfuse:
            print(f"📊 View results in Langfuse: {os.getenv('LANGFUSE_HOST', 'https://cloud.langfuse.com')}")

        print("="*80)

        # Return exit code based on pass rate
        if summary.all_passed_rate >= 0.8:  # 80% pass rate
            return 0
        else:
            return 1

    except KeyboardInterrupt:
        print("\n\n⚠️  Evaluation interrupted by user")
        return 1
    except Exception as e:
        print(f"\n❌ Evaluation failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
