"""
Plan Adherence Metric

Evaluates adherence to the documented tool routing strategy using G-Eval.
Measures whether the agent followed the "KB first" principle and tool selection rules.
"""

from typing import Optional
from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCaseParams

from evaluations.config import get_evaluation_model
from evaluations.metrics.task_completion import AzureOpenAIModel


# ============================================================================
# Plan Adherence Metric
# ============================================================================

class PlanAdherenceMetric(GEval):
    """
    Evaluates adherence to the agent's documented tool routing strategy.

    System Strategy Rules:
    1. ALWAYS check search_knowledge_base FIRST unless query explicitly requires real-time info
    2. Use search_web ONLY for "latest", "current", "today", "breaking news" queries
    3. Use BOTH tools when comparison or combined context is needed

    G-Eval Criteria:
    - Did agent follow "KB first" principle?
    - Was web search only used when appropriate (real-time queries)?
    - Were tool selection decisions aligned with system prompt strategy?
    - Did agent respect the tool routing hierarchy?

    Scoring (1-10 scale, normalized to 0.0-1.0):
    - 10: Perfect adherence to all rules
    - 7-9: Mostly adheres, minor deviations
    - 4-6: Some adherence, significant deviations
    - 1-3: Poor adherence, strategy not followed
    """

    def __init__(
        self,
        threshold: float = 0.7,
        model: Optional[AzureOpenAIModel] = None,
        strict_mode: bool = False
    ):
        """
        Initialize Plan Adherence Metric.

        Args:
            threshold: Minimum score to pass (default: 0.7)
            model: Optional DeepEval model for evaluation
            strict_mode: Whether to use strict evaluation
        """
        # Initialize evaluation model
        if model is None:
            azure_model = get_evaluation_model()
            model = AzureOpenAIModel(azure_model)

        # Define evaluation steps
        evaluation_steps = [
            "1. Identify the query type: Does it ask for current/real-time information or could it be answered by existing documents?",
            "2. Check if search_knowledge_base was called FIRST when it should have been (Rule 1).",
            "3. Verify if search_web was ONLY used for queries explicitly needing real-time info (Rule 2).",
            "4. Confirm if BOTH tools were used appropriately for comparison/combined queries (Rule 3).",
            "5. Assess overall alignment with the documented tool routing strategy.",
        ]

        # Build evaluation criteria text
        criteria = self._build_criteria(strict_mode)

        # Initialize GEval parent class
        super().__init__(
            name="Plan Adherence",
            criteria=criteria,
            evaluation_steps=evaluation_steps,
            evaluation_params=[
                LLMTestCaseParams.INPUT,
                LLMTestCaseParams.ACTUAL_OUTPUT,
                LLMTestCaseParams.RETRIEVAL_CONTEXT
            ],
            model=model,
            threshold=threshold,
            strict_mode=strict_mode
        )

    def _build_criteria(self, strict_mode: bool) -> str:
        """Build the evaluation criteria text."""
        base_criteria = """Evaluate adherence to the agent's documented tool routing strategy.

## Tool Routing Strategy Rules

**Rule 1: Knowledge Base First**
- ALWAYS check search_knowledge_base FIRST for most queries
- Default behavior unless query explicitly requires real-time information
- Examples of KB-first queries:
  - "What are our company policies?"
  - "Explain our product strategy"
  - "What are industry trends in X?" (historical/analysis)
  - "Tell me about company Y" (general information)

**Rule 2: Web Search Only When Needed**
- ONLY use search_web for queries explicitly asking for:
  - "latest" / "current" / "today" / "recent" information
  - "breaking news" / "just announced"
  - Real-time data (stock prices, weather, current events)
- Examples of web-search queries:
  - "What's the latest news on AI?"
  - "What are today's market trends?"
  - "What was just announced by company X?"

**Rule 3: Both Tools for Comparison**
- Use BOTH tools when query requires:
  - Comparison of internal vs external information
  - Benchmarking against industry
  - Combining historical and current context
- Examples of combined queries:
  - "Compare our strategy to market trends"
  - "How do our results compare to competitors?"

## Evaluation Guidelines

Perfect adherence (10):
- KB called first for non-real-time queries
- Web only used for explicit real-time queries
- Both tools used appropriately for comparisons
- No unnecessary tool calls

Good adherence (7-9):
- Minor deviations from strategy
- Generally follows the principles
- Tool selection mostly appropriate

Poor adherence (1-3):
- Does not follow "KB first" principle
- Uses web search for queries that should use KB
- Ignores comparison needs

IMPORTANT: Evaluate based on the query intent and the tools called (shown in context)."""

        if strict_mode:
            base_criteria += "\n\nSTRICT MODE: Any deviation from the documented strategy should significantly reduce the score."

        return base_criteria


# ============================================================================
# Helper Functions
# ============================================================================

def create_plan_adherence_metric(
    threshold: float = 0.7,
    strict_mode: bool = False
) -> PlanAdherenceMetric:
    """
    Create a Plan Adherence metric with default configuration.

    Args:
        threshold: Minimum score to pass (default: 0.7)
        strict_mode: Whether to use strict evaluation

    Returns:
        PlanAdherenceMetric: Configured metric
    """
    return PlanAdherenceMetric(
        threshold=threshold,
        strict_mode=strict_mode
    )


# ============================================================================
# Testing
# ============================================================================

def test_plan_adherence_metric():
    """Test the Plan Adherence metric with sample cases."""
    from deepeval.test_case import LLMTestCase

    # Test case 1: Perfect adherence (KB first)
    test_case_1 = LLMTestCase(
        input="What are our company policies?",
        actual_output="Our company policies include...",
        retrieval_context=[
            "Tools called: search_knowledge_base",
            "Tool: search_knowledge_base\nInput: {'query': 'company policies'}\nOutput: Policies...",
        ]
    )

    # Test case 2: Good adherence (web for current info)
    test_case_2 = LLMTestCase(
        input="What are today's breaking AI news?",
        actual_output="Today's AI news includes...",
        retrieval_context=[
            "Tools called: search_web",
            "Tool: search_web\nInput: {'query': 'today AI news'}\nOutput: Latest AI news...",
        ]
    )

    # Test case 3: Poor adherence (should have used KB first)
    test_case_3 = LLMTestCase(
        input="Tell me about footwear industry trends",
        actual_output="Footwear industry trends include...",
        retrieval_context=[
            "Tools called: search_web",
            "Tool: search_web\nInput: {'query': 'footwear trends'}\nOutput: Industry trends...",
        ]
    )

    # Test case 4: Perfect adherence (both tools for comparison)
    test_case_4 = LLMTestCase(
        input="Compare our AI strategy to current market trends",
        actual_output="Our strategy focuses on X, while market shows Y...",
        retrieval_context=[
            "Tools called: search_knowledge_base, search_web",
            "Tool: search_knowledge_base\nInput: {'query': 'our AI strategy'}\nOutput: Our strategy...",
            "Tool: search_web\nInput: {'query': 'current AI market trends'}\nOutput: Market trends...",
        ]
    )

    metric = create_plan_adherence_metric()

    print("Testing Plan Adherence Metric...")
    print("="*80)

    for i, test_case in enumerate([test_case_1, test_case_2, test_case_3, test_case_4], 1):
        try:
            score = metric.measure(test_case)
            print(f"\nTest Case {i}:")
            print(f"  Query: {test_case.input[:60]}...")
            print(f"  Score: {score:.2f}")
            print(f"  Passed: {metric.is_successful()}")
            if hasattr(metric, "reason"):
                print(f"  Reason: {metric.reason}")
        except Exception as e:
            print(f"\nTest Case {i}:")
            print(f"  Error: {str(e)}")
        print("-"*80)


if __name__ == "__main__":
    test_plan_adherence_metric()
