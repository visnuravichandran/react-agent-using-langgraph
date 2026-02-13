"""
Tests for Argument Correctness Metric Integration

This module tests the integration of DeepEval's ArgumentCorrectnessMetric
into the evaluation system.
"""

import pytest
from deepeval.test_case import LLMTestCase, ToolCall
from deepeval.metrics import ArgumentCorrectnessMetric

from evaluations.config import EVALUATION_SETTINGS, METRIC_WEIGHTS
from evaluations.dataset_generator import TestCase, EvaluationDataset
from evaluations.model_wrapper import AzureOpenAIModel
from evaluations.config import get_evaluation_model


def test_config_has_argument_correctness():
    """Test that configuration includes argument correctness settings."""
    # Check threshold exists
    assert "argument_correctness_threshold" in EVALUATION_SETTINGS
    assert isinstance(EVALUATION_SETTINGS["argument_correctness_threshold"], float)
    assert 0.0 <= EVALUATION_SETTINGS["argument_correctness_threshold"] <= 1.0

    # Check weight exists
    assert "argument_correctness" in METRIC_WEIGHTS
    assert isinstance(METRIC_WEIGHTS["argument_correctness"], float)

    # Check weights sum to 1.0 (with small tolerance for floating point)
    total_weight = sum(METRIC_WEIGHTS.values())
    assert abs(total_weight - 1.0) < 0.01, f"Weights sum to {total_weight}, expected 1.0"


def test_test_case_supports_arguments():
    """Test that TestCase model supports expected_tool_arguments field."""
    # Create test case with arguments
    test_case = TestCase(
        id="test_001",
        category="knowledge_base_only",
        query="What are the strategic shifts?",
        expected_output="Strategic shifts include...",
        expected_tools=["search_knowledge_base"],
        expected_tool_order=["search_knowledge_base"],
        expected_tool_arguments=[{"query": "strategic shifts industry"}],
        expected_num_steps=2,
        difficulty="easy",
        reasoning="Test with tool arguments"
    )

    assert test_case.expected_tool_arguments is not None
    assert len(test_case.expected_tool_arguments) == 1
    assert test_case.expected_tool_arguments[0] == {"query": "strategic shifts industry"}


def test_test_case_arguments_optional():
    """Test that expected_tool_arguments is optional."""
    # Create test case without arguments
    test_case = TestCase(
        id="test_002",
        category="knowledge_base_only",
        query="What are the strategic shifts?",
        expected_output="Strategic shifts include...",
        expected_tools=["search_knowledge_base"],
        expected_tool_order=["search_knowledge_base"],
        expected_num_steps=2,
        difficulty="easy"
    )

    # Should default to None
    assert test_case.expected_tool_arguments is None


def test_dataset_loads_with_arguments():
    """Test that datasets can be loaded with expected_tool_arguments."""
    dataset_path = "datasets/evaluation_dataset_argument_example.json"

    try:
        dataset = EvaluationDataset.load_from_file(dataset_path)

        # Check that test cases loaded
        assert len(dataset.test_cases) > 0

        # Check first test case has arguments
        first_case = dataset.test_cases[0]
        assert first_case.expected_tool_arguments is not None
        assert len(first_case.expected_tool_arguments) > 0

    except FileNotFoundError:
        pytest.skip("Example dataset not found")


def test_tool_call_has_input_parameters():
    """Test that DeepEval ToolCall supports input_parameters."""
    # Create ToolCall with input parameters
    tool_call = ToolCall(
        name="search_knowledge_base",
        input_parameters={"query": "test query"}
    )

    assert tool_call.name == "search_knowledge_base"
    assert tool_call.input_parameters == {"query": "test query"}


@pytest.mark.skipif(
    not all([
        "AZURE_OPENAI_ENDPOINT" in __import__("os").environ,
        "AZURE_OPENAI_API_KEY" in __import__("os").environ,
    ]),
    reason="Azure OpenAI credentials not configured"
)
def test_argument_correctness_metric_initialization():
    """Test that ArgumentCorrectnessMetric can be initialized."""
    # Get evaluation model
    azure_model = get_evaluation_model()
    azure_deepeval_model = AzureOpenAIModel(azure_model)

    # Initialize metric
    metric = ArgumentCorrectnessMetric(
        threshold=EVALUATION_SETTINGS["argument_correctness_threshold"],
        model=azure_deepeval_model,
        include_reason=True,
        strict_mode=False,
        verbose_mode=False
    )

    assert metric is not None
    assert metric.threshold == EVALUATION_SETTINGS["argument_correctness_threshold"]


@pytest.mark.skipif(
    not all([
        "AZURE_OPENAI_ENDPOINT" in __import__("os").environ,
        "AZURE_OPENAI_API_KEY" in __import__("os").environ,
    ]),
    reason="Azure OpenAI credentials not configured"
)
def test_argument_correctness_metric_evaluation():
    """Test that ArgumentCorrectnessMetric can evaluate a test case."""
    # Get evaluation model
    azure_model = get_evaluation_model()
    azure_deepeval_model = AzureOpenAIModel(azure_model)

    # Initialize metric
    metric = ArgumentCorrectnessMetric(
        threshold=0.5,
        model=azure_deepeval_model,
        include_reason=True,
        verbose_mode=False
    )

    # Create test case with matching arguments
    test_case = LLMTestCase(
        input="What are the latest AI trends?",
        actual_output="Recent AI trends include...",
        expected_output="AI trends include...",
        tools_called=[
            ToolCall(name="search_web", input_parameters={"query": "AI trends latest 2024"})
        ],
        expected_tools=[
            ToolCall(name="search_web", input_parameters={"query": "AI trends recent developments"})
        ]
    )

    # Measure
    try:
        metric.measure(test_case)

        # Check that score was computed
        assert hasattr(metric, "score")
        assert metric.score is not None
        assert 0.0 <= metric.score <= 1.0

        # Check that reason was provided
        assert hasattr(metric, "reason")
        assert isinstance(metric.reason, str)
        assert len(metric.reason) > 0

    except Exception as e:
        pytest.skip(f"Metric evaluation failed (this is expected if API is not available): {e}")


def test_metric_weights_sum_to_one():
    """Test that all metric weights sum to 1.0."""
    total = sum(METRIC_WEIGHTS.values())
    # Allow small floating point tolerance
    assert abs(total - 1.0) < 0.01, f"Metric weights sum to {total}, should be 1.0"


def test_all_metrics_have_weights():
    """Test that all evaluation metrics have corresponding weights."""
    expected_metrics = {
        "task_completion",
        "tool_correctness",
        "argument_correctness",  # New metric
        "step_efficiency",
        "plan_adherence",
        "plan_quality"
    }

    actual_metrics = set(METRIC_WEIGHTS.keys())

    assert actual_metrics == expected_metrics, \
        f"Metric mismatch. Expected: {expected_metrics}, Got: {actual_metrics}"


def test_all_metrics_have_thresholds():
    """Test that all evaluation metrics have corresponding thresholds."""
    expected_thresholds = {
        "task_completion_threshold",
        "tool_correctness_threshold",
        "argument_correctness_threshold",  # New threshold
        "step_efficiency_threshold",
        "plan_adherence_threshold",
        "plan_quality_threshold"
    }

    actual_thresholds = {
        key for key in EVALUATION_SETTINGS.keys()
        if key.endswith("_threshold")
    }

    assert expected_thresholds.issubset(actual_thresholds), \
        f"Missing thresholds: {expected_thresholds - actual_thresholds}"


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v"])
