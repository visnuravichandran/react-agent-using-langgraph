"""
Pytest Tests for Evaluation System

Tests for dataset loading, trace extraction, metrics, and evaluator.
"""

import pytest
import json
from pathlib import Path
from datetime import datetime

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

# Import evaluation modules
from evaluations.dataset_generator import (
    EvaluationDataset,
    TestCase,
    generate_small_dataset,
    validate_dataset
)
from evaluations.trace_extractor import (
    ExecutionTrace,
    ToolCall,
    ReasoningStep,
    extract_execution_trace,
    trace_to_deepeval_context,
    compare_tool_calls,
    validate_trace
)
from evaluations.config import DATASET_PATHS, validate_config


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def sample_test_case():
    """Create a sample test case."""
    return TestCase(
        id="test_001",
        category="knowledge_base_only",
        query="What are our company policies?",
        expected_output="Company policies include remote work, security, and benefits.",
        expected_tools=["search_knowledge_base"],
        expected_tool_order=["search_knowledge_base"],
        expected_num_steps=2,
        difficulty="easy",
        reasoning="Knowledge base query for internal information."
    )


@pytest.fixture
def sample_agent_result():
    """Create a sample agent result with messages."""
    messages = [
        HumanMessage(content="What are our company policies?"),
        AIMessage(
            content="I'll search the knowledge base for company policies.",
            tool_calls=[
                {
                    "name": "search_knowledge_base",
                    "args": {"query": "company policies"},
                    "id": "call_123"
                }
            ]
        ),
        ToolMessage(
            content="Company policies include: 1. Remote work guidelines 2. Data security protocols 3. Employee benefits",
            tool_call_id="call_123"
        ),
        AIMessage(
            content="Based on the knowledge base, our company policies include remote work guidelines, data security protocols, and employee benefits."
        )
    ]

    return {
        "messages": messages,
        "tool_calls_count": 1,
        "last_tool_used": "search_knowledge_base"
    }


# ============================================================================
# Dataset Tests
# ============================================================================

def test_dataset_loading():
    """Test loading the evaluation dataset."""
    dataset_path = DATASET_PATHS["small"]
    dataset = EvaluationDataset.load_from_file(dataset_path)

    assert len(dataset.test_cases) > 0
    assert all(isinstance(tc, TestCase) for tc in dataset.test_cases)


def test_dataset_validation():
    """Test dataset validation."""
    dataset_path = DATASET_PATHS["small"]
    assert validate_dataset(dataset_path)


def test_test_case_creation(sample_test_case):
    """Test creating a test case."""
    assert sample_test_case.id == "test_001"
    assert sample_test_case.category == "knowledge_base_only"
    assert "search_knowledge_base" in sample_test_case.expected_tools


def test_dataset_category_distribution():
    """Test dataset category distribution."""
    dataset = generate_small_dataset()
    distribution = dataset.get_category_distribution()

    assert len(distribution) > 0
    assert all(count > 0 for count in distribution.values())


def test_dataset_difficulty_distribution():
    """Test dataset difficulty distribution."""
    dataset = generate_small_dataset()
    distribution = dataset.get_difficulty_distribution()

    assert len(distribution) > 0
    assert all(diff in ["easy", "medium", "hard"] for diff in distribution.keys())


# ============================================================================
# Trace Extraction Tests
# ============================================================================

def test_extract_execution_trace(sample_agent_result):
    """Test extracting execution trace from agent result."""
    start_time = datetime.now()
    end_time = datetime.now()

    trace = extract_execution_trace(sample_agent_result, start_time, end_time)

    assert isinstance(trace, ExecutionTrace)
    assert trace.query == "What are our company policies?"
    assert len(trace.tool_calls) > 0
    assert trace.tool_calls[0].name == "search_knowledge_base"
    assert trace.final_answer != ""


def test_tool_call_extraction(sample_agent_result):
    """Test extracting tool calls from trace."""
    trace = extract_execution_trace(sample_agent_result)

    assert len(trace.tool_calls) == 1
    assert trace.tool_calls[0].name == "search_knowledge_base"
    assert trace.tool_calls[0].input == {"query": "company policies"}
    assert "Company policies" in trace.tool_calls[0].output


def test_trace_to_deepeval_context(sample_agent_result):
    """Test converting trace to DeepEval context."""
    trace = extract_execution_trace(sample_agent_result)
    context = trace_to_deepeval_context(trace)

    assert isinstance(context, list)
    assert len(context) > 0
    assert any("search_knowledge_base" in item for item in context)


def test_validate_trace(sample_agent_result):
    """Test trace validation."""
    trace = extract_execution_trace(sample_agent_result)
    assert validate_trace(trace)


def test_compare_tool_calls_exact_match():
    """Test comparing tool calls with exact match."""
    actual = ["search_knowledge_base"]
    expected = ["search_knowledge_base"]

    score = compare_tool_calls(actual, expected, check_order=True)
    assert score == 1.0


def test_compare_tool_calls_wrong_order():
    """Test comparing tool calls with wrong order."""
    actual = ["search_web", "search_knowledge_base"]
    expected = ["search_knowledge_base", "search_web"]

    score = compare_tool_calls(actual, expected, check_order=True)
    assert score == 0.0  # Wrong order


def test_compare_tool_calls_missing_tool():
    """Test comparing tool calls with missing tool."""
    actual = ["search_knowledge_base"]
    expected = ["search_knowledge_base", "search_web"]

    score = compare_tool_calls(actual, expected, check_order=False)
    assert score < 1.0  # Partial match


# ============================================================================
# Metric Tests (Require API calls - skipped by default)
# ============================================================================

@pytest.mark.skip(reason="Requires Azure OpenAI API access and is slow")
def test_tool_correctness_metric_perfect_match():
    """Test tool correctness metric with perfect match."""
    from deepeval.test_case import LLMTestCase, ToolCall
    from deepeval.metrics import ToolCorrectnessMetric
    from evaluations.model_wrapper import AzureOpenAIModel
    from evaluations.config import get_evaluation_model

    # Initialize Azure OpenAI model
    azure_model = get_evaluation_model()
    azure_deepeval_model = AzureOpenAIModel(azure_model)

    test_case = LLMTestCase(
        input="What are our policies?",
        actual_output="Our policies include...",
        expected_output="Policies...",
        tools_called=[ToolCall(name="search_knowledge_base")],
        expected_tools=[ToolCall(name="search_knowledge_base")]
    )

    metric = ToolCorrectnessMetric(threshold=0.8, model=azure_deepeval_model)
    metric.measure(test_case)

    assert metric.score == 1.0
    assert metric.is_successful()


@pytest.mark.skip(reason="Requires Azure OpenAI API access and is slow")
def test_tool_correctness_metric_mismatch():
    """Test tool correctness metric with tool mismatch."""
    from deepeval.test_case import LLMTestCase, ToolCall
    from deepeval.metrics import ToolCorrectnessMetric
    from evaluations.model_wrapper import AzureOpenAIModel
    from evaluations.config import get_evaluation_model

    # Initialize Azure OpenAI model
    azure_model = get_evaluation_model()
    azure_deepeval_model = AzureOpenAIModel(azure_model)

    test_case = LLMTestCase(
        input="What are our policies?",
        actual_output="Our policies include...",
        expected_output="Policies...",
        tools_called=[ToolCall(name="search_web")],
        expected_tools=[ToolCall(name="search_knowledge_base")]
    )

    metric = ToolCorrectnessMetric(threshold=0.8, model=azure_deepeval_model)
    metric.measure(test_case)

    assert metric.score < 0.8
    assert not metric.is_successful()


# ============================================================================
# Configuration Tests
# ============================================================================

def test_config_validation():
    """Test configuration validation."""
    # This will check if required env vars are set
    # May fail if .env is not properly configured
    is_valid = validate_config()
    # Don't assert here as it depends on env setup
    # Just ensure function runs without error
    assert isinstance(is_valid, bool)


def test_dataset_paths_exist():
    """Test that dataset paths are configured."""
    from evaluations.config import DATASET_PATHS

    assert "main" in DATASET_PATHS
    assert "small" in DATASET_PATHS


# ============================================================================
# Integration Tests (Commented - require actual API calls)
# ============================================================================

# @pytest.mark.skip(reason="Requires API access and is slow")
# @pytest.mark.asyncio
# async def test_agent_wrapper_with_trace():
#     """Test agent wrapper with trace capture."""
#     from evaluations.agent_wrapper import create_evaluation_agent
#
#     agent = create_evaluation_agent(enable_langfuse=False)
#     result, trace = agent.invoke_with_trace("What are our company policies?")
#
#     assert trace.query == "What are our company policies?"
#     assert trace.final_answer != ""
#     assert len(trace.tool_calls) >= 0


# @pytest.mark.skip(reason="Requires API access and is slow")
# def test_evaluation_runner_on_single_case():
#     """Test evaluation runner on a single test case."""
#     from evaluations.evaluator import EvaluationRunner
#     from evaluations.config import DATASET_PATHS
#
#     runner = EvaluationRunner(
#         dataset_path=DATASET_PATHS["small"],
#         enable_langfuse=False,
#         verbose=False
#     )
#
#     # Run on first test case only
#     test_case = runner.dataset.test_cases[0]
#     result = runner._evaluate_test_case(test_case)
#
#     assert result.test_case_id == test_case.id
#     assert result.overall_score >= 0.0
#     assert result.overall_score <= 1.0


# ============================================================================
# Helper Function Tests
# ============================================================================

def test_trace_to_dict(sample_agent_result):
    """Test converting trace to dictionary."""
    trace = extract_execution_trace(sample_agent_result)
    trace_dict = trace.to_dict()

    assert isinstance(trace_dict, dict)
    assert "query" in trace_dict
    assert "tool_calls" in trace_dict
    assert "total_steps" in trace_dict


def test_trace_str_representation(sample_agent_result):
    """Test string representation of trace."""
    trace = extract_execution_trace(sample_agent_result)
    trace_str = str(trace)

    assert "ExecutionTrace" in trace_str
    assert "search_knowledge_base" in trace_str


# ============================================================================
# Run Tests
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
