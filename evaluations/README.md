# Agent Evaluation System

Comprehensive evaluation system for the LangGraph ReAct agent using DeepEval 3.7.7 and Langfuse integration.

## Table of Contents
- [Overview](#overview)
- [Quick Start](#quick-start)
- [Metrics](#metrics)
- [Architecture](#architecture)
- [Usage](#usage)
- [Test Datasets](#test-datasets)
- [Langfuse Integration](#langfuse-integration)
- [Customization](#customization)
- [Troubleshooting](#troubleshooting)

## Overview

This evaluation system measures agent performance across 5 key metrics:

1. **Task Completion** - Did the agent complete the user's task?
2. **Tool Correctness** - Did the agent select the right tools?
3. **Step Efficiency** - Was the execution path optimal?
4. **Plan Adherence** - Did the agent follow the documented strategy?
5. **Plan Quality** - Was the reasoning clear and logical?

### Key Features

- ✅ **46 test cases** across 5 categories (KB queries, web search, combined, conversational, edge cases)
- ✅ **5 custom metrics** using DeepEval framework (LLM-as-judge + G-Eval)
- ✅ **Langfuse integration** for centralized results tracking
- ✅ **Azure OpenAI GPT-4** as evaluation judge
- ✅ **CSV/JSON export** for analysis
- ✅ **CLI interface** for easy execution
- ✅ **Pytest tests** for validation

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

This will install `deepeval==3.7.7` along with all other dependencies.

### 2. Configure Environment

Ensure your `.env` file has the required variables:

```bash
# Azure OpenAI (required)
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
AZURE_OPENAI_API_KEY=your-api-key
AZURE_OPENAI_API_VERSION=2024-02-15-preview
AZURE_OPENAI_DEPLOYMENT=gpt-4o

# Evaluation-specific (optional)
EVALUATION_MODEL=gpt-4o
EVALUATION_TEMPERATURE=0.0
DEEPEVAL_TELEMETRY_OPT_OUT=YES

# Langfuse (optional, for results tracking)
LANGFUSE_PUBLIC_KEY=your-public-key
LANGFUSE_SECRET_KEY=your-secret-key
LANGFUSE_HOST=https://cloud.langfuse.com
```

### 3. Run Evaluation

```bash
# Quick test with small dataset (10 test cases)
python scripts/run_evaluation.py --dataset small

# Full evaluation with Langfuse reporting
python scripts/run_evaluation.py --dataset main --langfuse

# Custom dataset
python scripts/run_evaluation.py --dataset /path/to/dataset.json
```

### 4. View Results

Results are saved to `results/` directory:
- `eval_run_{id}_{timestamp}.json` - Detailed results
- `eval_run_{id}_{timestamp}.csv` - Summary CSV

If Langfuse is enabled, view results at: https://cloud.langfuse.com

## Metrics

### 1. Task Completion (Custom LLM-as-Judge)

**What it measures:** Whether the agent successfully completed the user's task.

**Scoring:**
- 1.0 = Task fully completed with accurate information
- 0.7-0.9 = Task mostly completed, minor gaps
- 0.4-0.6 = Partial completion, significant gaps
- 0.0-0.3 = Task not completed or incorrect

**Threshold:** 0.7 (configurable)

**Implementation:** `metrics/task_completion.py`

### 2. Tool Correctness (Hybrid: Deterministic + LLM)

**What it measures:** Whether the agent selected the correct tool(s).

**Scoring:**
- Deterministic matching for exact tool matches
- Penalties for extra tools (-0.2) or missing tools (-0.4)
- LLM judge for borderline cases
- Checks tool order (KB should be first)

**Threshold:** 0.8 (configurable)

**Implementation:** `metrics/tool_correctness.py`

### 3. Step Efficiency (G-Eval)

**What it measures:** Efficiency of the agent's execution path.

**Evaluation Criteria:**
- Minimal reasoning steps (1-3 ideal)
- No redundant tool calls
- Effective use of tool results
- Quick convergence to answer

**Scoring:** 1-10 scale (normalized to 0.0-1.0)

**Threshold:** 0.6 (configurable)

**Implementation:** `metrics/step_efficiency.py`

### 4. Plan Adherence (G-Eval)

**What it measures:** Adherence to documented tool routing strategy.

**Strategy Rules:**
1. Always check `search_knowledge_base` FIRST
2. Use `search_web` only for real-time queries
3. Use BOTH when comparison needed

**Scoring:** 1-10 scale (normalized to 0.0-1.0)

**Threshold:** 0.7 (configurable)

**Implementation:** `metrics/plan_adherence.py`

### 5. Plan Quality (G-Eval)

**What it measures:** Quality of agent's reasoning and planning.

**Evaluation Criteria:**
- Logical coherence of reasoning
- Appropriate tool selection justification
- Effective synthesis of tool results
- Clear connection between query and actions

**Scoring:** 1-10 scale (normalized to 0.0-1.0)

**Threshold:** 0.6 (configurable)

**Implementation:** `metrics/plan_quality.py`

## Architecture

```
Test Dataset → Evaluator → Agent Wrapper → LangGraph Agent
                    ↓                           ↓
             Custom Metrics ← Execution Trace (tools, reasoning, timing)
                    ↓
          DeepEval Results → Langfuse Reporter → Langfuse
```

### Components

**1. Dataset Generator** (`dataset_generator.py`)
- Generates and validates test datasets
- 46 test cases across 5 categories
- Pydantic models for type safety

**2. Agent Wrapper** (`agent_wrapper.py`)
- Wraps existing LangGraph agent
- Captures execution traces
- Provides `invoke_with_trace()` method

**3. Trace Extractor** (`trace_extractor.py`)
- Extracts structured data from agent runs
- Parses tool calls, reasoning steps, timing
- Converts to DeepEval format

**4. Metrics** (`metrics/`)
- 5 custom metrics using DeepEval framework
- Azure OpenAI GPT-4 as judge LLM
- Configurable thresholds

**5. Evaluator** (`evaluator.py`)
- Orchestrates evaluation runs
- Aggregates results
- Exports to JSON/CSV

**6. Langfuse Reporter** (`langfuse_reporter.py`)
- Uploads datasets to Langfuse
- Reports traces and scores
- Enables centralized tracking

## Usage

### Command-Line Interface

```bash
# Basic usage
python scripts/run_evaluation.py --dataset small

# With options
python scripts/run_evaluation.py \
  --dataset main \
  --langfuse \
  --output ./my_results \
  --model gpt-4o

# Upload dataset to Langfuse
python scripts/run_evaluation.py \
  --dataset main \
  --langfuse \
  --upload-dataset

# Quiet mode (minimal output)
python scripts/run_evaluation.py --dataset small --quiet

# Check configuration
python scripts/run_evaluation.py --check-config
```

### Python API

```python
from evaluations.evaluator import run_evaluation

# Run evaluation
summary = run_evaluation(
    dataset="small",
    enable_langfuse=True,
    verbose=True,
    save_results=True
)

print(f"Overall Score: {summary.mean_overall:.3f}")
print(f"Pass Rate: {summary.all_passed_rate*100:.1f}%")
```

### Custom Evaluation

```python
from evaluations.evaluator import EvaluationRunner
from evaluations.config import DATASET_PATHS

# Create runner
runner = EvaluationRunner(
    dataset_path=DATASET_PATHS["small"],
    enable_langfuse=False,
    model_name="gpt-4o",
    verbose=True
)

# Run evaluation
summary = runner.run_evaluation()

# Access results
for result in runner.results:
    print(f"{result.test_case_id}: {result.overall_score:.2f}")

# Save results
runner.save_results(output_dir="custom_results")
```

## Test Datasets

### Main Dataset (`evaluation_dataset.json`)

**46 test cases** across 5 categories:

- **Knowledge Base Only** (32.6%): Company policies, industry analysis, historical data
- **Web Search Only** (21.7%): Current events, latest news, breaking developments
- **Combined Tools** (26.1%): Comparison queries, benchmarking, comprehensive research
- **Conversational** (10.9%): Greetings, follow-ups, capability questions
- **Edge Cases** (8.7%): Ambiguous queries, multi-intent, error scenarios

### Small Dataset (`evaluation_dataset_small.json`)

**10 representative test cases** for quick iteration:
- 3 Knowledge Base queries
- 3 Web Search queries
- 3 Combined queries
- 1 Conversational query

### Dataset Format

```json
{
  "test_cases": [
    {
      "id": "kb_001",
      "category": "knowledge_base_only",
      "query": "What are the strategic shifts in footwear companies?",
      "expected_output": "Strategic shifts include...",
      "expected_tools": ["search_knowledge_base"],
      "expected_tool_order": ["search_knowledge_base"],
      "expected_num_steps": 2,
      "difficulty": "easy",
      "reasoning": "Industry analysis question..."
    }
  ]
}
```

### Creating Custom Datasets

```python
from evaluations.dataset_generator import TestCase, EvaluationDataset

# Create test cases
test_cases = [
    TestCase(
        id="custom_001",
        category="knowledge_base_only",
        query="Your query here",
        expected_output="Expected response",
        expected_tools=["search_knowledge_base"],
        expected_tool_order=["search_knowledge_base"],
        expected_num_steps=2,
        difficulty="medium"
    )
]

# Create dataset
dataset = EvaluationDataset(test_cases=test_cases)
dataset.save_to_file("custom_dataset.json")
```

## Langfuse Integration

### Setup

1. **Get Langfuse Credentials:**
   - Sign up at https://cloud.langfuse.com
   - Get Public Key and Secret Key from project settings

2. **Configure Environment:**
   ```bash
   LANGFUSE_PUBLIC_KEY=pk-lf-...
   LANGFUSE_SECRET_KEY=sk-lf-...
   LANGFUSE_HOST=https://cloud.langfuse.com
   ```

3. **Upload Dataset:**
   ```bash
   python scripts/run_evaluation.py --dataset main --langfuse --upload-dataset
   ```

### Viewing Results

In Langfuse, you'll see:

- **Datasets**: Test cases with expected outputs
- **Traces**: Each test case execution with tags (`evaluation`, `run_{id}`, `category:{type}`)
- **Scores**: All 5 metrics attached to each trace
- **Filtering**: Filter by run_id, category, or metric scores

### Benefits

- **Time-Series Tracking**: Compare evaluation runs over time
- **Aggregation**: View mean scores across categories
- **Debugging**: Click through to see full execution traces
- **Dashboards**: Visualize metric trends

## Customization

### Adjusting Thresholds

```python
from evaluations.config import EVALUATION_SETTINGS

# Modify thresholds
EVALUATION_SETTINGS["task_completion_threshold"] = 0.8
EVALUATION_SETTINGS["tool_correctness_threshold"] = 0.9
```

Or via CLI:
```bash
python scripts/run_evaluation.py \
  --threshold-task-completion 0.8 \
  --threshold-tool-correctness 0.9
```

### Adding New Metrics

1. Create metric class in `metrics/`:
   ```python
   from deepeval.metrics import BaseMetric
   from deepeval.test_case import LLMTestCase

   class MyCustomMetric(BaseMetric):
       @property
       def __name__(self):
           return "My Custom Metric"

       def measure(self, test_case: LLMTestCase) -> float:
           # Your evaluation logic
           return score
   ```

2. Add to `evaluator.py`:
   ```python
   self.metrics["my_metric"] = MyCustomMetric()
   ```

3. Update `METRIC_WEIGHTS` in `config.py`.

### Using Different Judge LLM

To use a different model as judge:

```python
from evaluations.config import get_evaluation_model

# In config.py, modify get_evaluation_model()
def get_evaluation_model(temperature=None):
    return AzureChatOpenAI(
        azure_deployment="your-model",
        ...
    )
```

## Troubleshooting

### Issue: "Configuration validation failed"

**Solution:** Check that all required environment variables are set:
```bash
python scripts/run_evaluation.py --check-config
```

See `ENVIRONMENT_VARIABLES.md` for required variables.

### Issue: "Dataset not found"

**Solution:** Ensure datasets were generated:
```bash
python evaluations/dataset_generator.py
```

### Issue: "Langfuse not configured"

**Solution:** Set Langfuse environment variables:
```bash
export LANGFUSE_PUBLIC_KEY=pk-lf-...
export LANGFUSE_SECRET_KEY=sk-lf-...
```

### Issue: "DeepEval metric failed"

**Solution:** Check Azure OpenAI API access and rate limits. The evaluation judge LLM needs to make API calls.

### Issue: "Evaluation taking too long"

**Solution:**
- Use the small dataset for quick tests: `--dataset small`
- Reduce metrics evaluated (customize in `evaluator.py`)
- Check network connectivity

### Issue: "Scores seem incorrect"

**Solution:**
- Review metric thresholds in `config.py`
- Check test case expectations in dataset
- Enable verbose mode to see metric reasons: `--verbose`

## Running Tests

```bash
# Run all tests
pytest tests/test_evaluations.py -v

# Run specific test
pytest tests/test_evaluations.py::test_dataset_loading -v

# With async support
pytest tests/ -v --asyncio-mode=auto
```

## Performance

**Typical Run Times:**
- Small dataset (10 cases): ~2-3 minutes
- Full dataset (46 cases): ~8-12 minutes

Each test case involves:
- 1 agent invocation
- 5 metric evaluations (5 LLM judge calls)

**Cost Estimate:**
- ~50-100 tokens per metric evaluation
- ~5 metrics × 46 cases = 230 LLM calls
- At GPT-4o prices: ~$0.50-$1.00 per full evaluation run

## Contributing

To add new test cases:

1. Edit `dataset_generator.py`
2. Add test cases to appropriate category
3. Regenerate datasets: `python evaluations/dataset_generator.py`
4. Validate: `python evaluations/dataset_generator.py --validate datasets/evaluation_dataset.json`

## References

- [DeepEval Documentation](https://docs.confident-ai.com/)
- [Langfuse Documentation](https://langfuse.com/docs)
- [LangGraph Documentation](https://langchain-ai.github.io/langgraph/)
- [Azure OpenAI Documentation](https://learn.microsoft.com/en-us/azure/ai-services/openai/)

## Support

For issues or questions:
1. Check [Troubleshooting](#troubleshooting)
2. Review test outputs: `results/eval_run_*.json`
3. Check Langfuse traces (if enabled)
4. Review environment variables: `evaluations/ENVIRONMENT_VARIABLES.md`
