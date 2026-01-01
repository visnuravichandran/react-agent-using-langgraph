"""
Plan Quality Metric

Evaluates the quality of the agent's reasoning and planning using G-Eval.
Measures logical coherence, tool selection justification, and synthesis quality.
"""

from typing import Optional
from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCaseParams

from evaluations.config import get_evaluation_model
from evaluations.metrics.task_completion import AzureOpenAIModel


# ============================================================================
# Plan Quality Metric
# ============================================================================

class PlanQualityMetric(GEval):
    """
    Evaluates the quality of the agent's reasoning and planning.

    G-Eval Criteria:
    - Logical coherence of reasoning steps
    - Appropriate tool selection with clear justification
    - Effective synthesis of tool results into final answer
    - Clear connection between query intent and actions taken
    - Quality of thought process and decision-making

    Scoring (1-10 scale, normalized to 0.0-1.0):
    - 10: Excellent reasoning, clear logic, perfect synthesis
    - 7-9: Good reasoning with minor gaps in logic or synthesis
    - 4-6: Adequate reasoning, some confusion or unclear logic
    - 1-3: Poor reasoning, logic issues, weak synthesis
    """

    def __init__(
        self,
        threshold: float = 0.6,
        model: Optional[AzureOpenAIModel] = None,
        strict_mode: bool = False
    ):
        """
        Initialize Plan Quality Metric.

        Args:
            threshold: Minimum score to pass (default: 0.6)
            model: Optional DeepEval model for evaluation
            strict_mode: Whether to use strict evaluation
        """
        # Initialize evaluation model
        if model is None:
            azure_model = get_evaluation_model()
            model = AzureOpenAIModel(azure_model)

        # Define evaluation steps
        evaluation_steps = [
            "1. Assess the logical coherence of the agent's reasoning steps.",
            "2. Evaluate if tool selections were appropriate and justified based on query intent.",
            "3. Check if tool results were effectively synthesized into the final answer.",
            "4. Verify there is a clear connection between the user's query and the agent's actions.",
            "5. Rate the overall quality of thinking and decision-making process.",
        ]

        # Build evaluation criteria text
        criteria = self._build_criteria(strict_mode)

        # Initialize GEval parent class
        super().__init__(
            name="Plan Quality",
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
        base_criteria = """Evaluate the quality of the agent's reasoning and planning.

## Evaluation Dimensions

### 1. Logical Coherence
- Are reasoning steps logically connected?
- Does the thought process flow naturally?
- Are conclusions supported by evidence?
- Is there clear cause-and-effect in the decision-making?

### 2. Tool Selection Quality
- Were appropriate tools chosen for the query type?
- Is there clear justification for why tools were selected?
- Did the agent demonstrate understanding of query intent?
- Were tool inputs well-formulated?

### 3. Synthesis Quality
- Were tool results effectively incorporated into the answer?
- Is the final response coherent and well-structured?
- Did the agent combine information from multiple sources effectively?
- Does the answer demonstrate understanding of tool outputs?

### 4. Intent Alignment
- Do the agent's actions clearly address the user's query?
- Is there a clear connection between query and tool selections?
- Does the reasoning show understanding of user needs?
- Is the response relevant to what was asked?

### 5. Overall Quality
- Is the thinking process clear and well-organized?
- Does the agent demonstrate good judgment?
- Is the problem-solving approach sound?
- Would a human find this reasoning process sensible?

## Scoring Guidelines (1-10)

**Excellent Quality (9-10)**
- Impeccable logical flow
- Perfect tool selection with clear reasoning
- Masterful synthesis of information
- Crystal-clear intent alignment
- Demonstrates sophisticated understanding

**Good Quality (7-8)**
- Strong logical coherence with minor gaps
- Appropriate tool selection, mostly well-justified
- Effective synthesis with small room for improvement
- Clear intent alignment
- Sound reasoning overall

**Adequate Quality (5-6)**
- Reasonable logic but some unclear connections
- Acceptable tool selection but weak justification
- Basic synthesis, could be more comprehensive
- Intent mostly addressed
- Functional but not exceptional reasoning

**Poor Quality (3-4)**
- Noticeable logic gaps or inconsistencies
- Questionable tool selections or weak justification
- Weak synthesis, tool results not well-integrated
- Intent partially missed
- Reasoning shows confusion or uncertainty

**Very Poor Quality (1-2)**
- Significant logic problems
- Inappropriate tool selections
- Poor or absent synthesis
- Intent not understood
- Reasoning is flawed or incoherent

IMPORTANT: Base your evaluation on the reasoning steps and tool usage shown in the context."""

        if strict_mode:
            base_criteria += "\n\nSTRICT MODE: Apply rigorous standards. Even minor issues in reasoning or synthesis should impact the score."

        return base_criteria


# ============================================================================
# Helper Functions
# ============================================================================

def create_plan_quality_metric(
    threshold: float = 0.6,
    strict_mode: bool = False
) -> PlanQualityMetric:
    """
    Create a Plan Quality metric with default configuration.

    Args:
        threshold: Minimum score to pass (default: 0.6)
        strict_mode: Whether to use strict evaluation

    Returns:
        PlanQualityMetric: Configured metric
    """
    return PlanQualityMetric(
        threshold=threshold,
        strict_mode=strict_mode
    )


# ============================================================================
# Testing
# ============================================================================

def test_plan_quality_metric():
    """Test the Plan Quality metric with sample cases."""
    from deepeval.test_case import LLMTestCase

    # Test case 1: High quality reasoning
    test_case_1 = LLMTestCase(
        input="Compare our AI strategy to current market trends",
        actual_output="Our AI strategy focuses on ethical AI and enterprise solutions, emphasizing responsible deployment and ROI. Current market trends show rapid adoption of generative AI across industries, with emphasis on productivity gains. Our approach aligns with the market's focus on practical applications while maintaining our distinctive emphasis on ethics and governance.",
        retrieval_context=[
            "Tools called: search_knowledge_base, search_web",
            "Step 1: Identified query requires both internal (our strategy) and external (market trends) information",
            "Step 2: Retrieved internal AI strategy from knowledge base, focusing on ethical AI framework",
            "Step 3: Retrieved current market trends from web search, noting emphasis on generative AI adoption",
            "Step 4: Synthesized both sources to provide comparison highlighting alignment and differentiation",
        ]
    )

    # Test case 2: Moderate quality reasoning
    test_case_2 = LLMTestCase(
        input="What are our company policies?",
        actual_output="Company policies cover remote work, security, and benefits.",
        retrieval_context=[
            "Tools called: search_knowledge_base",
            "Step 1: Retrieved company policies from knowledge base",
            "Step 2: Summarized key policy areas",
        ]
    )

    # Test case 3: Poor quality reasoning
    test_case_3 = LLMTestCase(
        input="What are the latest AI regulations?",
        actual_output="There are various AI regulations.",
        retrieval_context=[
            "Tools called: search_knowledge_base, search_web",
            "Step 1: Searched knowledge base (no clear justification why)",
            "Step 2: Searched web",
            "Step 3: Provided vague answer without synthesizing tool results",
        ]
    )

    metric = create_plan_quality_metric()

    print("Testing Plan Quality Metric...")
    print("="*80)

    for i, test_case in enumerate([test_case_1, test_case_2, test_case_3], 1):
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
    test_plan_quality_metric()
