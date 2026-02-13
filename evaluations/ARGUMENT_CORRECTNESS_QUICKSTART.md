# Argument Correctness - Quick Start Guide

## 🚀 TL;DR

The evaluation system now includes **ArgumentCorrectnessMetric** to validate that your agent passes correct arguments to tools.

```bash
# Run evaluation with the new metric (it's already enabled!)
python scripts/run_evaluation.py --dataset small
```

That's it! The metric is automatically included in all evaluations.

## ✅ What's New

**Before:** Only checked if agent called the right tool
```
✅ Agent called search_web ← Tool Correctness
```

**Now:** Also checks if agent passed the right arguments
```
✅ Agent called search_web ← Tool Correctness
✅ Query included "latest", "today" ← Argument Correctness
```

## 📊 See It in Action

Run your first evaluation:

```bash
python scripts/run_evaluation.py --dataset small
```

Output now includes:
```
Evaluating test_001...
  ✅ task_completion... 0.90
  ✅ tool_correctness... 1.00
  ✅ argument_correctness... 0.85  ← NEW!
  ✅ step_efficiency... 0.75
```

## 📝 Adding Expected Arguments (Optional)

Your existing test cases work without changes. To leverage the new metric, add `expected_tool_arguments`:

```json
{
  "id": "test_001",
  "query": "What are the latest AI trends?",
  "expected_tools": ["search_web"],
  "expected_tool_arguments": [
    {"query": "AI trends latest"}
  ]
}
```

## 🎯 What Gets Evaluated

The LLM evaluator checks if actual arguments semantically match expected arguments:

| Expected | Actual | Score | Result |
|----------|--------|-------|--------|
| `{"query": "AI trends"}` | `{"query": "artificial intelligence trends"}` | 0.95 | ✅ PASS |
| `{"query": "AI trends"}` | `{"query": "weather forecast"}` | 0.10 | ❌ FAIL |
| `{"query": "Q4 results"}` | `{"query": "quarterly earnings Q4 2024"}` | 0.88 | ✅ PASS |

## ⚙️ Configuration

Default settings (in `evaluations/config.py`):

```python
# Threshold: score needed to pass (0.0-1.0)
"argument_correctness_threshold": 0.7

# Weight: importance in overall score
"argument_correctness": 0.20  # 20% of total
```

Override from command line:
```bash
python scripts/run_evaluation.py \
  --threshold-argument-correctness 0.8
```

## 📖 Example Test Cases

See `datasets/evaluation_dataset_argument_example.json` for complete examples.

**Simple KB Search:**
```json
{
  "query": "What are our remote work policies?",
  "expected_tools": ["search_knowledge_base"],
  "expected_tool_arguments": [
    {"query": "remote work policy"}
  ]
}
```

**Web Search with Temporal Context:**
```json
{
  "query": "What happened in tech today?",
  "expected_tools": ["search_web"],
  "expected_tool_arguments": [
    {"query": "tech news today latest"}
  ]
}
```

**Multiple Tools:**
```json
{
  "query": "Compare our strategy with market trends",
  "expected_tools": ["search_knowledge_base", "search_web"],
  "expected_tool_arguments": [
    {"query": "strategy internal"},
    {"query": "market trends industry"}
  ]
}
```

## 🧪 Running Tests

Verify the integration:
```bash
pytest tests/test_argument_correctness.py -v
```

All 10 tests should pass ✅

## 📚 Learn More

- **Full Guide**: `evaluations/ARGUMENT_CORRECTNESS_GUIDE.md`
- **Integration Summary**: `INTEGRATION_SUMMARY.md`
- **Main Docs**: `CLAUDE.md`

## 🔧 Common Commands

```bash
# Run evaluation (new metric included automatically)
python scripts/run_evaluation.py --dataset small

# Run with custom threshold
python scripts/run_evaluation.py --dataset main \
  --threshold-argument-correctness 0.8

# Run with Langfuse tracking
python scripts/run_evaluation.py --dataset main --langfuse

# Check configuration
python scripts/run_evaluation.py --check-config

# Run tests
pytest tests/test_argument_correctness.py -v
```

## ✨ Key Takeaways

1. ✅ **Already Enabled**: New metric is active in all evaluations
2. ✅ **Backward Compatible**: Existing test cases work unchanged
3. ✅ **Optional Enhancement**: Add expected_tool_arguments when ready
4. ✅ **Semantic Matching**: LLM evaluates meaning, not exact strings
5. ✅ **Production Ready**: Fully tested and documented

## 🆘 Need Help?

- **Documentation**: `evaluations/ARGUMENT_CORRECTNESS_GUIDE.md`
- **Examples**: `datasets/evaluation_dataset_argument_example.json`
- **Tests**: `tests/test_argument_correctness.py`

---

**Ready to use!** Just run your evaluations as usual - the new metric is already working. 🎉
