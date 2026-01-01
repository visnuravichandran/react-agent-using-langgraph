"""
Task Completion Metric

Evaluates whether the agent successfully completed the user's task using an LLM as judge.
"""

from typing import Optional
from deepeval.metrics import BaseMetric
from deepeval.test_case import LLMTestCase, LLMTestCaseParams
from deepeval.models import DeepEvalBaseLLM

from evaluations.config import get_evaluation_model


# ============================================================================
# Azure OpenAI Wrapper for DeepEval
# ============================================================================

class AzureOpenAIModel(DeepEvalBaseLLM):
    """Wrapper to make Azure OpenAI compatible with DeepEval."""

    def __init__(self, model):
        self.model = model

    def load_model(self):
        return self.model

    def generate(self, prompt: str) -> str:
        """Generate response from the model."""
        chat_model = self.load_model()
        response = chat_model.invoke(prompt)
        return response.content

    async def a_generate(self, prompt: str) -> str:
        """Async generate response from the model."""
        chat_model = self.load_model()
        response = await chat_model.ainvoke(prompt)
        return response.content

    def get_model_name(self) -> str:
        return "azure-gpt-4o"


# ============================================================================
# Task Completion Metric
# ============================================================================

class TaskCompletionMetric(BaseMetric):
    """
    Evaluates whether the agent successfully completed the user's task.

    Scoring Criteria:
    - 1.0: Task fully completed, all requirements addressed, accurate information
    - 0.7-0.9: Task mostly completed, minor gaps or missing details
    - 0.4-0.6: Partial completion, significant gaps or inaccuracies
    - 0.0-0.3: Task not completed, wrong information, or completely off-topic

    This metric uses an LLM (Azure GPT-4) as a judge to evaluate task completion.
    """

    def __init__(
        self,
        threshold: float = 0.7,
        model: Optional[DeepEvalBaseLLM] = None,
        include_reason: bool = True,
        strict_mode: bool = False
    ):
        """
        Initialize Task Completion Metric.

        Args:
            threshold: Minimum score to pass (default: 0.7)
            model: Optional DeepEval model for evaluation
            include_reason: Whether to include reasoning in results
            strict_mode: Whether to use strict evaluation (higher standards)
        """
        self.threshold = threshold
        self.include_reason = include_reason
        self.strict_mode = strict_mode

        # Initialize evaluation model
        if model is None:
            azure_model = get_evaluation_model()
            self.model = AzureOpenAIModel(azure_model)
        else:
            self.model = model

    @property
    def __name__(self):
        return "Task Completion"

    def measure(self, test_case: LLMTestCase) -> float:
        """
        Measure task completion for a test case.

        Args:
            test_case: LLMTestCase containing input, actual_output, expected_output

        Returns:
            float: Score between 0.0 and 1.0
        """
        # Build evaluation prompt
        evaluation_prompt = self._build_evaluation_prompt(
            query=test_case.input,
            actual_output=test_case.actual_output,
            expected_output=test_case.expected_output,
            context=test_case.retrieval_context if test_case.retrieval_context else []
        )

        # Get evaluation from LLM
        try:
            evaluation_response = self.model.generate(evaluation_prompt)
            score, reason = self._parse_evaluation_response(evaluation_response)

            # Store score and reason
            self.score = score
            self.reason = reason
            self.success = score >= self.threshold

            return score

        except Exception as e:
            # If evaluation fails, return 0.0 and log the error
            self.score = 0.0
            self.reason = f"Evaluation failed: {str(e)}"
            self.success = False
            return 0.0

    def _build_evaluation_prompt(
        self,
        query: str,
        actual_output: str,
        expected_output: str,
        context: list
    ) -> str:
        """Build the evaluation prompt for the LLM judge."""
        strictness_note = ""
        if self.strict_mode:
            strictness_note = "\nUse STRICT evaluation standards. Minor issues should significantly impact the score."

        prompt = f"""You are an expert evaluator assessing whether an AI agent successfully completed a user's task.

## User Query
{query}

## Agent's Actual Response
{actual_output}

## Expected Response / Success Criteria
{expected_output}

## Additional Context
{chr(10).join(context) if context else "No additional context provided."}

## Your Task
Evaluate whether the agent successfully completed the user's task. Consider:

1. **Completeness**: Did the agent fully address the user's query?
2. **Accuracy**: Is the information provided accurate and relevant?
3. **Relevance**: Is the response on-topic and appropriate for the query?
4. **Comprehensiveness**: Are all aspects of the query covered?{strictness_note}

## Scoring Guidelines
- **0.9-1.0**: Excellent - Task fully completed with comprehensive, accurate information
- **0.7-0.8**: Good - Task completed, minor gaps or missing details
- **0.5-0.6**: Fair - Partial completion, some key information missing
- **0.3-0.4**: Poor - Significant gaps, major information missing
- **0.0-0.2**: Failure - Task not completed, wrong information, or completely off-topic

## Output Format
Provide your evaluation in the following format:
SCORE: <float between 0.0 and 1.0>
REASON: <brief explanation of your scoring in 1-2 sentences>

Begin your evaluation:"""

        return prompt

    def _parse_evaluation_response(self, response: str) -> tuple[float, str]:
        """
        Parse the LLM's evaluation response.

        Args:
            response: Raw response from LLM

        Returns:
            tuple: (score, reason)
        """
        score = 0.0
        reason = "Failed to parse evaluation response"

        try:
            lines = response.strip().split("\n")
            for line in lines:
                line = line.strip()
                if line.startswith("SCORE:"):
                    score_str = line.replace("SCORE:", "").strip()
                    score = float(score_str)
                elif line.startswith("REASON:"):
                    reason = line.replace("REASON:", "").strip()

            # Ensure score is within valid range
            score = max(0.0, min(1.0, score))

        except Exception as e:
            reason = f"Error parsing response: {str(e)}"
            score = 0.0

        return score, reason

    def is_successful(self) -> bool:
        """Check if the metric passed the threshold."""
        return self.success if hasattr(self, "success") else False


# ============================================================================
# Helper Functions
# ============================================================================

def create_task_completion_metric(
    threshold: float = 0.7,
    strict_mode: bool = False
) -> TaskCompletionMetric:
    """
    Create a Task Completion metric with default configuration.

    Args:
        threshold: Minimum score to pass (default: 0.7)
        strict_mode: Whether to use strict evaluation

    Returns:
        TaskCompletionMetric: Configured metric
    """
    return TaskCompletionMetric(
        threshold=threshold,
        strict_mode=strict_mode,
        include_reason=True
    )


# ============================================================================
# Testing
# ============================================================================

def test_task_completion_metric():
    """Test the Task Completion metric with sample cases."""
    from deepeval.test_case import LLMTestCase

    # Test case 1: Good completion
    test_case_1 = LLMTestCase(
        input="What are the strategic shifts in footwear companies?",
        actual_output="Strategic shifts in footwear companies include digital transformation, direct-to-consumer (DTC) models, sustainability initiatives, and supply chain optimization. Companies are investing in e-commerce platforms, reducing reliance on wholesale channels, and implementing eco-friendly manufacturing processes.",
        expected_output="Strategic shifts in the footwear industry include digital transformation, direct-to-consumer models, sustainability initiatives, and supply chain optimization."
    )

    # Test case 2: Partial completion
    test_case_2 = LLMTestCase(
        input="What are the strategic shifts in footwear companies?",
        actual_output="Companies are focusing on digital transformation and online sales.",
        expected_output="Strategic shifts in the footwear industry include digital transformation, direct-to-consumer models, sustainability initiatives, and supply chain optimization."
    )

    # Test case 3: Poor completion
    test_case_3 = LLMTestCase(
        input="What are the strategic shifts in footwear companies?",
        actual_output="I don't have enough information to answer this question.",
        expected_output="Strategic shifts in the footwear industry include digital transformation, direct-to-consumer models, sustainability initiatives, and supply chain optimization."
    )

    metric = create_task_completion_metric()

    print("Testing Task Completion Metric...")
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
    test_task_completion_metric()
