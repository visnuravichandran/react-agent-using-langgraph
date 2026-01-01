"""
Tool Correctness Metric

Evaluates whether the agent selected the correct tool(s) using a hybrid approach:
- Deterministic scoring for exact matches
- LLM-as-judge for borderline cases
"""

from typing import Optional, List
from deepeval.metrics import BaseMetric
from deepeval.test_case import LLMTestCase
from deepeval.models import DeepEvalBaseLLM

from evaluations.config import get_evaluation_model
from evaluations.metrics.task_completion import AzureOpenAIModel


# ============================================================================
# Tool Correctness Metric
# ============================================================================

class ToolCorrectnessMetric(BaseMetric):
    """
    Evaluates whether the agent selected the correct tool(s).

    Evaluation Strategy:
    1. Deterministic scoring:
       - Perfect match (same tools, same order): 1.0
       - Same tools, wrong order: 0.8
       - Missing tools or extra tools: calculated penalty
       - Completely wrong tools: 0.0

    2. LLM-as-judge (for borderline cases):
       - When tools differ but might be justifiable
       - Evaluates if tool selection was reasonable given the query

    3. Penalties:
       - Extra unnecessary tools: -0.2 per extra tool
       - Missing required tools: -0.4 per missing tool
       - Wrong order (KB should be first): -0.2
    """

    def __init__(
        self,
        threshold: float = 0.8,
        model: Optional[DeepEvalBaseLLM] = None,
        check_order: bool = True,
        use_llm_for_ambiguous: bool = True
    ):
        """
        Initialize Tool Correctness Metric.

        Args:
            threshold: Minimum score to pass (default: 0.8)
            model: Optional DeepEval model for evaluation
            check_order: Whether to check tool call order
            use_llm_for_ambiguous: Whether to use LLM for ambiguous cases
        """
        self.threshold = threshold
        self.check_order = check_order
        self.use_llm_for_ambiguous = use_llm_for_ambiguous

        # Initialize evaluation model
        if model is None:
            azure_model = get_evaluation_model()
            self.model = AzureOpenAIModel(azure_model)
        else:
            self.model = model

    @property
    def __name__(self):
        return "Tool Correctness"

    def measure(self, test_case: LLMTestCase) -> float:
        """
        Measure tool correctness for a test case.

        Args:
            test_case: LLMTestCase containing:
                - input: user query
                - actual_output: agent response
                - expected_output: expected response
                - context: dictionary with 'expected_tools' and 'actual_tools'

        Returns:
            float: Score between 0.0 and 1.0
        """
        # Extract expected and actual tools from test case
        expected_tools, actual_tools = self._extract_tools(test_case)

        if expected_tools is None or actual_tools is None:
            self.score = 0.0
            self.reason = "Failed to extract tool information from test case"
            self.success = False
            return 0.0

        # Step 1: Deterministic scoring
        deterministic_score = self._calculate_deterministic_score(
            expected_tools, actual_tools
        )

        # Step 2: Decide if we need LLM evaluation
        needs_llm_eval = self._needs_llm_evaluation(
            expected_tools, actual_tools, deterministic_score
        )

        if needs_llm_eval and self.use_llm_for_ambiguous:
            # Use LLM to evaluate tool selection
            llm_score, llm_reason = self._evaluate_with_llm(
                query=test_case.input,
                expected_tools=expected_tools,
                actual_tools=actual_tools,
                actual_output=test_case.actual_output
            )

            # Combine deterministic and LLM scores (weighted average)
            final_score = (deterministic_score * 0.4) + (llm_score * 0.6)
            reason = f"Deterministic: {deterministic_score:.2f}, LLM: {llm_score:.2f}. {llm_reason}"
        else:
            # Use deterministic score only
            final_score = deterministic_score
            reason = self._get_deterministic_reason(expected_tools, actual_tools)

        # Store results
        self.score = final_score
        self.reason = reason
        self.success = final_score >= self.threshold

        return final_score

    def _extract_tools(self, test_case: LLMTestCase) -> tuple[Optional[List[str]], Optional[List[str]]]:
        """Extract expected and actual tools from test case."""
        import json

        expected_tools = None
        actual_tools = None

        # Try to parse from retrieval_context (primary method)
        if test_case.retrieval_context:
            for item in test_case.retrieval_context:
                if isinstance(item, str):
                    try:
                        # Try to parse as JSON
                        parsed = json.loads(item)
                        if isinstance(parsed, dict) and parsed.get("__tool_metadata__"):
                            if expected_tools is None:
                                expected_tools = parsed.get("expected_tools", [])
                            if actual_tools is None:
                                actual_tools = parsed.get("actual_tools", [])
                            break
                    except (json.JSONDecodeError, ValueError):
                        # Not JSON, skip
                        pass

        # Try to get tools from test_case attributes (legacy support)
        if expected_tools is None and hasattr(test_case, "expected_tools"):
            expected_tools = test_case.expected_tools
        if actual_tools is None and hasattr(test_case, "actual_tools"):
            actual_tools = test_case.actual_tools

        # Try to get tools from context (legacy support)
        if (expected_tools is None or actual_tools is None) and test_case.context:
            if isinstance(test_case.context, dict):
                if expected_tools is None:
                    expected_tools = test_case.context.get("expected_tools", [])
                if actual_tools is None:
                    actual_tools = test_case.context.get("actual_tools", [])

        # Ensure lists
        if expected_tools is None:
            expected_tools = []
        if actual_tools is None:
            actual_tools = []

        return expected_tools, actual_tools

    def _calculate_deterministic_score(
        self,
        expected_tools: List[str],
        actual_tools: List[str]
    ) -> float:
        """Calculate deterministic score based on tool matching."""
        # Handle empty cases
        if not expected_tools and not actual_tools:
            return 1.0  # Both empty, perfect match

        if not expected_tools and actual_tools:
            # Agent called tools when none were expected
            return max(0.0, 1.0 - (0.2 * len(actual_tools)))

        if expected_tools and not actual_tools:
            # Agent didn't call tools when it should have
            return max(0.0, 1.0 - (0.4 * len(expected_tools)))

        # Convert to sets for comparison
        expected_set = set(expected_tools)
        actual_set = set(actual_tools)

        # Perfect match (same tools, same order)
        if expected_tools == actual_tools:
            return 1.0

        # Same tools, different order
        if expected_set == actual_set:
            if self.check_order:
                # Check if KB was called first (required strategy)
                if "search_knowledge_base" in expected_tools and "search_knowledge_base" in actual_tools:
                    if expected_tools.index("search_knowledge_base") == 0 and actual_tools.index("search_knowledge_base") != 0:
                        return 0.6  # Wrong order, KB should be first
                return 0.8  # Same tools but wrong order
            else:
                return 1.0  # Order doesn't matter

        # Calculate Jaccard similarity
        intersection = len(expected_set & actual_set)
        union = len(expected_set | actual_set)
        jaccard_score = intersection / union if union > 0 else 0.0

        # Apply penalties
        extra_tools = actual_set - expected_set
        missing_tools = expected_set - actual_set

        penalty = 0.0
        penalty += len(extra_tools) * 0.2  # -0.2 per extra tool
        penalty += len(missing_tools) * 0.4  # -0.4 per missing tool

        final_score = max(0.0, jaccard_score - penalty)

        return final_score

    def _needs_llm_evaluation(
        self,
        expected_tools: List[str],
        actual_tools: List[str],
        deterministic_score: float
    ) -> bool:
        """Determine if LLM evaluation is needed for ambiguous cases."""
        # Use LLM for scores between 0.3 and 0.8 (borderline cases)
        if 0.3 <= deterministic_score <= 0.8:
            return True

        # Use LLM when there's a mismatch but not completely wrong
        expected_set = set(expected_tools)
        actual_set = set(actual_tools)

        if expected_set != actual_set and len(expected_set & actual_set) > 0:
            return True

        return False

    def _evaluate_with_llm(
        self,
        query: str,
        expected_tools: List[str],
        actual_tools: List[str],
        actual_output: str
    ) -> tuple[float, str]:
        """Use LLM to evaluate tool selection for ambiguous cases."""
        prompt = f"""You are an expert evaluator assessing whether an AI agent selected the correct tools for a given query.

## Available Tools
1. **search_knowledge_base**: Searches internal documents, policies, company information, historical data
2. **search_web**: Performs real-time web search for current events, latest news, external information

## Tool Selection Strategy
The agent should:
- ALWAYS use search_knowledge_base FIRST for most queries (default)
- ONLY use search_web for queries explicitly asking for "latest", "today", "current", "breaking news"
- Use BOTH when comparison or combined context is needed

## User Query
{query}

## Expected Tools
{expected_tools if expected_tools else "No tools (conversational query)"}

## Actual Tools Called
{actual_tools if actual_tools else "No tools called"}

## Agent's Response
{actual_output[:300]}...

## Your Task
Evaluate whether the agent's tool selection was appropriate for this query.

Consider:
1. Did the agent follow the "KB first" strategy?
2. Was web search only used when truly needed (real-time info)?
3. Did the agent successfully answer the query with the tools chosen?
4. Were any tools unnecessarily called?

## Output Format
SCORE: <float between 0.0 and 1.0>
REASON: <brief explanation in 1-2 sentences>

Begin your evaluation:"""

        try:
            response = self.model.generate(prompt)
            score, reason = self._parse_llm_response(response)
            return score, reason
        except Exception as e:
            return 0.5, f"LLM evaluation failed: {str(e)}"

    def _parse_llm_response(self, response: str) -> tuple[float, str]:
        """Parse LLM evaluation response."""
        score = 0.5
        reason = "Failed to parse LLM response"

        try:
            lines = response.strip().split("\n")
            for line in lines:
                line = line.strip()
                if line.startswith("SCORE:"):
                    score_str = line.replace("SCORE:", "").strip()
                    score = float(score_str)
                elif line.startswith("REASON:"):
                    reason = line.replace("REASON:", "").strip()

            score = max(0.0, min(1.0, score))
        except Exception:
            pass

        return score, reason

    def _get_deterministic_reason(
        self,
        expected_tools: List[str],
        actual_tools: List[str]
    ) -> str:
        """Generate reason for deterministic scoring."""
        if expected_tools == actual_tools:
            return "Perfect match: correct tools in correct order"

        if set(expected_tools) == set(actual_tools):
            return f"Same tools but different order. Expected: {expected_tools}, Actual: {actual_tools}"

        expected_set = set(expected_tools)
        actual_set = set(actual_tools)
        extra_tools = actual_set - expected_set
        missing_tools = expected_set - actual_set

        reasons = []
        if extra_tools:
            reasons.append(f"Extra tools called: {list(extra_tools)}")
        if missing_tools:
            reasons.append(f"Missing tools: {list(missing_tools)}")

        if not reasons:
            reasons.append(f"Expected: {expected_tools}, Actual: {actual_tools}")

        return ". ".join(reasons)

    def is_successful(self) -> bool:
        """Check if the metric passed the threshold."""
        return self.success if hasattr(self, "success") else False


# ============================================================================
# Helper Functions
# ============================================================================

def create_tool_correctness_metric(
    threshold: float = 0.8,
    check_order: bool = True
) -> ToolCorrectnessMetric:
    """
    Create a Tool Correctness metric with default configuration.

    Args:
        threshold: Minimum score to pass (default: 0.8)
        check_order: Whether to check tool call order

    Returns:
        ToolCorrectnessMetric: Configured metric
    """
    return ToolCorrectnessMetric(
        threshold=threshold,
        check_order=check_order,
        use_llm_for_ambiguous=True
    )


# ============================================================================
# Testing
# ============================================================================

def test_tool_correctness_metric():
    """Test the Tool Correctness metric with sample cases."""
    from deepeval.test_case import LLMTestCase

    # Test case 1: Perfect match
    test_case_1 = LLMTestCase(
        input="What are our company policies?",
        actual_output="Our company policies include...",
        expected_output="Company policies...",
        context={
            "expected_tools": ["search_knowledge_base"],
            "actual_tools": ["search_knowledge_base"]
        }
    )

    # Test case 2: Missing tool
    test_case_2 = LLMTestCase(
        input="Compare our strategy to current market trends",
        actual_output="Our strategy focuses on...",
        expected_output="Comparison of strategy and trends...",
        context={
            "expected_tools": ["search_knowledge_base", "search_web"],
            "actual_tools": ["search_knowledge_base"]
        }
    )

    # Test case 3: Extra tool
    test_case_3 = LLMTestCase(
        input="What are our Q3 results?",
        actual_output="Q3 results show...",
        expected_output="Q3 financial results...",
        context={
            "expected_tools": ["search_knowledge_base"],
            "actual_tools": ["search_knowledge_base", "search_web"]
        }
    )

    metric = create_tool_correctness_metric()

    print("Testing Tool Correctness Metric...")
    print("="*80)

    for i, test_case in enumerate([test_case_1, test_case_2, test_case_3], 1):
        score = metric.measure(test_case)
        print(f"\nTest Case {i}:")
        print(f"  Score: {score:.2f}")
        print(f"  Passed: {metric.is_successful()}")
        if hasattr(metric, "reason"):
            print(f"  Reason: {metric.reason}")
        print("-"*80)


if __name__ == "__main__":
    test_tool_correctness_metric()
