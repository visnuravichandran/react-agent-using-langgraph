"""
Step Efficiency Metric

Evaluates the efficiency of the agent's execution path using G-Eval.
Measures whether the agent reached conclusions with minimal steps and no redundancy.
"""

from typing import Optional
from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCaseParams

from evaluations.config import get_evaluation_model
from evaluations.metrics.task_completion import AzureOpenAIModel


# ============================================================================
# Step Efficiency Metric
# ============================================================================

class StepEfficiencyMetric(GEval):
    """
    Evaluates the efficiency of the agent's execution path.

    G-Eval Criteria:
    - Minimal number of reasoning steps to reach conclusion (1-3 steps ideal)
    - No redundant tool calls
    - Efficient use of tool results (doesn't ignore tool output)
    - Quick convergence to final answer
    - No unnecessary loops or repeated reasoning

    Scoring (1-10 scale, normalized to 0.0-1.0):
    - 10: Optimal path, no waste, minimal steps
    - 7-9: Efficient with minor inefficiencies
    - 4-6: Moderate inefficiency, some redundancy
    - 1-3: Highly inefficient, many unnecessary steps
    """

    def __init__(
        self,
        threshold: float = 0.6,
        model: Optional[AzureOpenAIModel] = None,
        strict_mode: bool = False
    ):
        """
        Initialize Step Efficiency Metric.

        Args:
            threshold: Minimum score to pass (default: 0.6)
            model: Optional DeepEval model for evaluation
            strict_mode: Whether to use strict evaluation
        """
        # Initialize evaluation model
        if model is None:
            azure_model = get_evaluation_model()
            model = AzureOpenAIModel(azure_model)

        # Define evaluation criteria
        evaluation_steps = [
            "1. Analyze the number of reasoning steps taken by the agent. Ideal is 1-3 steps.",
            "2. Check if any tool calls were redundant or unnecessary.",
            "3. Evaluate if tool results were used effectively (not ignored).",
            "4. Assess if the agent converged quickly to the final answer.",
            "5. Identify any unnecessary loops or repeated reasoning.",
        ]

        # Build evaluation criteria text
        criteria = self._build_criteria(strict_mode)

        # Initialize GEval parent class
        super().__init__(
            name="Step Efficiency",
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
        base_criteria = """Evaluate the efficiency of the agent's execution path.

An efficient execution should:
- Reach a conclusion with minimal reasoning steps (1-3 steps is ideal, 4-5 is acceptable)
- Make only necessary tool calls (no redundant searches)
- Use tool results effectively (not ignore retrieved information)
- Converge quickly to the final answer (no excessive back-and-forth)
- Avoid loops or repetitive reasoning patterns

Consider:
- Fewer steps are better, but steps must be sufficient to answer the query
- Each tool call should contribute meaningfully to the answer
- Tool results should be synthesized and used in the final response
- The agent should not call the same tool twice with similar queries
- Reasoning should be direct and purposeful

Scoring Guidelines (1-10):
- 10: Perfect efficiency - minimal steps, all necessary, no waste
- 8-9: Very efficient - minor room for optimization
- 6-7: Reasonably efficient - some inefficiency but acceptable
- 4-5: Moderately inefficient - noticeable redundancy or extra steps
- 2-3: Inefficient - significant waste, unnecessary complexity
- 1: Highly inefficient - major problems, excessive steps

IMPORTANT: Base your score on the execution trace provided in the context."""

        if strict_mode:
            base_criteria += "\n\nSTRICT MODE: Apply stringent standards. Even minor inefficiencies should significantly reduce the score."

        return base_criteria


# ============================================================================
# Helper Functions
# ============================================================================

def create_step_efficiency_metric(
    threshold: float = 0.6,
    strict_mode: bool = False
) -> StepEfficiencyMetric:
    """
    Create a Step Efficiency metric with default configuration.

    Args:
        threshold: Minimum score to pass (default: 0.6)
        strict_mode: Whether to use strict evaluation

    Returns:
        StepEfficiencyMetric: Configured metric
    """
    return StepEfficiencyMetric(
        threshold=threshold,
        strict_mode=strict_mode
    )


# ============================================================================
# Testing
# ============================================================================

def test_step_efficiency_metric():
    """Test the Step Efficiency metric with sample cases."""
    from deepeval.test_case import LLMTestCase

    # Test case 1: Efficient execution
    test_case_1 = LLMTestCase(
        input="What are our company policies?",
        actual_output="Our company policies include remote work guidelines, data security protocols, and employee benefits.",
        retrieval_context=[
            "Tools called: search_knowledge_base",
            "Tool: search_knowledge_base\nInput: {'query': 'company policies'}\nOutput: Company policies include...",
            "Step 1: Retrieved company policies from knowledge base",
            "Total steps: 2, Execution time: 3.50s"
        ]
    )

    # Test case 2: Moderately efficient
    test_case_2 = LLMTestCase(
        input="Compare our strategy to market trends",
        actual_output="Our strategy focuses on X, while market trends show Y.",
        retrieval_context=[
            "Tools called: search_knowledge_base, search_web",
            "Tool: search_knowledge_base\nInput: {'query': 'our strategy'}\nOutput: Our strategy...",
            "Tool: search_web\nInput: {'query': 'market trends'}\nOutput: Market trends...",
            "Step 1: Retrieved internal strategy",
            "Step 2: Retrieved external market trends",
            "Step 3: Compared both sources",
            "Total steps: 3, Execution time: 8.20s"
        ]
    )

    # Test case 3: Inefficient execution
    test_case_3 = LLMTestCase(
        input="What are today's AI news?",
        actual_output="Today's AI news includes...",
        retrieval_context=[
            "Tools called: search_knowledge_base, search_web, search_web",
            "Tool: search_knowledge_base\nInput: {'query': 'AI news'}\nOutput: No recent results",
            "Tool: search_web\nInput: {'query': 'AI news'}\nOutput: AI news...",
            "Tool: search_web\nInput: {'query': 'latest AI news'}\nOutput: Similar AI news...",
            "Step 1: Checked knowledge base (unnecessary for current news)",
            "Step 2: Searched web for AI news",
            "Step 3: Searched web again with similar query (redundant)",
            "Step 4: Synthesized results",
            "Total steps: 4, Execution time: 12.50s"
        ]
    )

    metric = create_step_efficiency_metric()

    print("Testing Step Efficiency Metric...")
    print("="*80)

    for i, test_case in enumerate([test_case_1, test_case_2, test_case_3], 1):
        try:
            score = metric.measure(test_case)
            print(f"\nTest Case {i}:")
            print(f"  Score: {score:.2f}")
            print(f"  Passed: {metric.is_successful()}")
            if hasattr(metric, "reason"):
                print(f"  Reason: {metric.reason}")
        except Exception as e:
            print(f"\nTest Case {i}:")
            print(f"  Error: {str(e)}")
        print("-"*80)


if __name__ == "__main__":
    test_step_efficiency_metric()
