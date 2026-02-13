# Argument Correctness Metric Integration Guide

## Overview

The **ArgumentCorrectnessMetric** from DeepEval has been integrated into the evaluation system to validate that agents not only call the correct tools but also pass appropriate arguments to those tools.

## What is Argument Correctness?

Argument Correctness evaluates whether the input parameters passed to tool calls match the expected arguments. This is crucial for:

- **Query Accuracy**: Ensuring search queries contain relevant keywords
- **Parameter Validation**: Checking that tool parameters are correctly formatted
- **Semantic Matching**: Verifying arguments capture the intent of the user's request
- **Quality Assurance**: Identifying cases where the agent uses tools with inappropriate inputs

## How It Works

### 1. Metric Configuration

The metric is configured in `evaluations/config.py`:

```python
EVALUATION_SETTINGS = {
    "argument_correctness_threshold": 0.7,  # 70% match required to pass
    # ... other thresholds
}

METRIC_WEIGHTS = {
    "argument_correctness": 0.20,  # 20% of overall score
    # ... other weights (sum to 1.0)
}
```

### 2. Dataset Format

Test cases can now include `expected_tool_arguments` to specify the expected input parameters:

```json
{
  "id": "test_001",
  "query": "What are the latest AI trends?",
  "expected_tools": ["search_web"],
  "expected_tool_arguments": [
    {
      "query": "AI trends latest 2024"
    }
  ]
}
```

**Important Notes:**
- `expected_tool_arguments` is optional (defaults to empty dict)
- Array length should match `expected_tools`
- Each element is a dictionary of parameter name → expected value
- Arguments are semantically compared by the LLM evaluator

### 3. Evaluation Process

During evaluation (`evaluations/evaluator.py`):

1. **Extract Actual Arguments**: Tool calls from the trace include `input_parameters`
   ```python
   tools_called = [
       ToolCall(name=tc.name, input_parameters=tc.input)
       for tc in trace.tool_calls
   ]
   ```

2. **Build Expected Arguments**: From the test case definition
   ```python
   expected_tools = [
       ToolCall(name=tool_name, input_parameters=expected_args)
       for tool_name, expected_args in zip(expected_tools, expected_arguments)
   ]
   ```

3. **Measure Correctness**: DeepEval's ArgumentCorrectnessMetric compares actual vs expected
   - Uses LLM to evaluate semantic similarity
   - Returns score 0.0-1.0
   - Passes if score ≥ threshold

### 4. Result Tracking

Results include detailed information about argument correctness:

```python
TestResult(
    argument_correctness_score=0.85,      # How well arguments matched
    argument_correctness_reason="...",    # LLM explanation
    argument_correctness_passed=True,     # Whether threshold was met
)
```

## Usage Examples

### Example 1: Knowledge Base Search

```json
{
  "id": "kb_001",
  "query": "What are our company's remote work policies?",
  "expected_tools": ["search_knowledge_base"],
  "expected_tool_arguments": [
    {
      "query": "remote work policy company"
    }
  ]
}
```

**What's Evaluated:**
- Did the agent search the knowledge base? (Tool Correctness)
- Did the search query include relevant terms like "remote work" and "policy"? (Argument Correctness)

### Example 2: Web Search with Current Events

```json
{
  "id": "web_001",
  "query": "What happened in the tech industry today?",
  "expected_tools": ["search_web"],
  "expected_tool_arguments": [
    {
      "query": "tech industry news today latest"
    }
  ]
}
```

**What's Evaluated:**
- Did the agent use web search instead of knowledge base? (Tool Correctness)
- Did the query include temporal terms like "today" or "latest"? (Argument Correctness)

### Example 3: Combined Tools with Multiple Arguments

```json
{
  "id": "combined_001",
  "query": "Compare our Q4 results with market trends",
  "expected_tools": ["search_knowledge_base", "search_web"],
  "expected_tool_arguments": [
    {
      "query": "Q4 results quarterly earnings"
    },
    {
      "query": "market trends industry 2024"
    }
  ]
}
```

**What's Evaluated:**
- Did the agent call both tools in the right order? (Tool Correctness)
- Did the first query focus on internal Q4 results? (Argument Correctness)
- Did the second query focus on external market trends? (Argument Correctness)

## Running Evaluations

### With Default Settings

```bash
python scripts/run_evaluation.py --dataset small
```

### Override Argument Correctness Threshold

```bash
python scripts/run_evaluation.py \
  --dataset main \
  --threshold-argument-correctness 0.8
```

### Check Metric Configuration

```bash
python scripts/run_evaluation.py --check-config
```

## Interpreting Results

### Console Output

```
[1/10] Evaluating: kb_001 (knowledge_base_only)
  Evaluating task_completion... ✅ 0.90
  Evaluating tool_correctness... ✅ 1.00
  Evaluating argument_correctness... ✅ 0.85
  Evaluating step_efficiency... ✅ 0.75
  Evaluating plan_adherence... ✅ 0.80
  Evaluating plan_quality... ✅ 0.70
  Overall: 0.83 ✅
    Task Completion: 0.90 ✅
    Tool Correctness: 1.00 ✅
    Argument Correctness: 0.85 ✅
    Step Efficiency: 0.75 ✅
    Plan Adherence: 0.80 ✅
    Plan Quality: 0.70 ✅
```

### Summary Statistics

```
MEAN SCORES
--------------------------------------------------------------------------------
Overall:              0.830
Task Completion:      0.900
Tool Correctness:     1.000
Argument Correctness: 0.850  ← New metric
Step Efficiency:      0.750
Plan Adherence:       0.800
Plan Quality:         0.700

PASS RATES
--------------------------------------------------------------------------------
All Passed:           100.0%
Task Completion:      100.0%
Tool Correctness:     100.0%
Argument Correctness: 100.0%  ← New metric
Step Efficiency:      100.0%
Plan Adherence:       100.0%
Plan Quality:         100.0%
```

### CSV Export

Results include `argument_correctness` column:

```csv
test_id,category,query,task_completion,tool_correctness,argument_correctness,...
kb_001,knowledge_base_only,What are...,0.900,1.000,0.850,...
```

## Best Practices

### 1. Start Without Arguments

For initial testing, omit `expected_tool_arguments`:
- Tool Correctness will still be evaluated
- Argument Correctness will receive a default score
- Focus on getting tool selection right first

### 2. Add Arguments Gradually

Once tool selection is stable, add expected arguments:
- Start with key search terms
- Don't be overly prescriptive
- Allow for semantic variations

### 3. Use Flexible Expectations

The LLM evaluator allows semantic matching:
```json
// These would both match well:
"expected": {"query": "remote work policy"}
"actual": {"query": "company remote work policies and guidelines"}
```

### 4. Test Edge Cases

Include test cases that check:
- Missing required parameters
- Incorrect parameter types
- Irrelevant search terms
- Too broad or too narrow queries

### 5. Adjust Threshold as Needed

- **0.5-0.6**: Lenient, allows significant variation
- **0.7**: Default, balanced strictness
- **0.8-0.9**: Strict, requires close alignment
- **1.0**: Exact match (not recommended with LLM evaluation)

## Troubleshooting

### Low Argument Correctness Scores

**Problem**: Agent consistently scores low on argument correctness

**Solutions**:
1. Check if expected arguments are too prescriptive
2. Review agent's system prompt for search query guidance
3. Verify trace extraction is capturing input parameters correctly
4. Consider lowering threshold or adjusting expected arguments

### Metric Fails to Run

**Problem**: ArgumentCorrectnessMetric throws errors

**Solutions**:
1. Ensure DeepEval version is 3.7.7 or higher: `pip show deepeval`
2. Check that both actual and expected ToolCalls have `input_parameters`
3. Verify Azure OpenAI credentials are configured
4. Check logs for specific error messages

### Arguments Not Captured

**Problem**: Actual arguments show as empty dict

**Solutions**:
1. Verify tools are receiving arguments in agent implementation
2. Check trace extraction in `evaluations/trace_extractor.py`
3. Ensure tool calls include `args` or `input` field

## Implementation Details

### Key Files Modified

1. **evaluations/config.py**
   - Added `argument_correctness_threshold`
   - Added `argument_correctness` weight

2. **evaluations/dataset_generator.py**
   - Added `expected_tool_arguments` field to TestCase

3. **evaluations/evaluator.py**
   - Imported `ArgumentCorrectnessMetric`
   - Added metric to evaluation pipeline
   - Updated `TestResult` and `EvaluationRunSummary` data structures
   - Enhanced tool call creation to include `input_parameters`

4. **scripts/run_evaluation.py**
   - Added CLI argument for threshold override

5. **CLAUDE.md**
   - Updated documentation with new metric

### Metric Weights Distribution

```python
METRIC_WEIGHTS = {
    "task_completion": 0.20,       # 20%
    "tool_correctness": 0.20,      # 20%
    "argument_correctness": 0.20,  # 20% ← New
    "step_efficiency": 0.15,       # 15%
    "plan_adherence": 0.15,        # 15%
    "plan_quality": 0.10,          # 10%
}
# Total: 100%
```

The weights were rebalanced to give equal importance to tool selection, argument validation, and task completion.

## Future Enhancements

Potential improvements:
1. Support for argument schemas/types validation
2. Per-tool argument templates
3. Argument importance weighting
4. Custom argument comparison functions
5. Integration with tool documentation

## Support

For issues or questions:
- Check logs in `results/` directory
- Review test case definitions
- Verify DeepEval and Azure OpenAI setup
- Consult DeepEval documentation: https://docs.confident-ai.com/

---

**Last Updated**: 2024-02-13
**DeepEval Version**: 3.7.7+
**Integration Status**: ✅ Complete
