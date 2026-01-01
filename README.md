# ReAct Agent with Dual Tool Routing

A LangGraph-based ReAct agent that intelligently routes queries between Azure AI Search (internal knowledge base) and Gemini Web Search (real-time external information).

## Features

- **Intelligent Query Routing**: Automatically determines whether to use internal KB, web search, or both
- **Query Pre-Analysis**: Enhanced version analyzes queries before execution for better tool selection
- **Streaming Support**: Real-time streaming of agent responses and tool executions
- **Conversation Memory**: Persistent conversation threads with checkpointing
- **FastAPI Server**: Production-ready API with health checks and streaming endpoints
- **Langfuse Integration**: Built-in observability for tracing, monitoring, and debugging agent executions

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        User Query                                │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Query Analyzer (Optional)                     │
│         Classifies query → KB / Web / Combined / Chat           │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                      ReAct Agent Loop                            │
│  ┌─────────────┐    ┌──────────────┐    ┌─────────────────┐    │
│  │   Reason    │───▶│  Act (Tools) │───▶│    Observe      │    │
│  └─────────────┘    └──────────────┘    └─────────────────┘    │
│         ▲                                        │              │
│         └────────────────────────────────────────┘              │
└─────────────────────────────────────────────────────────────────┘
                              │
              ┌───────────────┴───────────────┐
              ▼                               ▼
┌──────────────────────┐        ┌──────────────────────────┐
│  search_knowledge_base│        │      search_web          │
│   (Azure AI Search)   │        │  (Gemini Grounded Search)│
└──────────────────────┘        └──────────────────────────┘
```

## Installation

```bash
# Clone and navigate to project
cd langgraph-react-agent

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Copy and configure environment
cp .env.example .env
# Edit .env with your credentials
```

## Configuration

### Environment Variables

Create a `.env` file with the following configuration:

```bash
# Azure OpenAI
AZURE_OPENAI_DEPLOYMENT=gpt-4o
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
AZURE_OPENAI_API_KEY=your-api-key
AZURE_OPENAI_API_VERSION=2024-08-01-preview

# Azure AI Search
AZURE_SEARCH_ENDPOINT=https://your-search.search.windows.net
AZURE_SEARCH_INDEX_NAME=your-index-name
AZURE_SEARCH_API_KEY=your-search-key
AZURE_SEARCH_SEMANTIC_CONFIG=default

# Google Gemini
GOOGLE_API_KEY=your-google-api-key

# Langfuse (Optional - for observability)
LANGFUSE_PUBLIC_KEY=your-public-key
LANGFUSE_SECRET_KEY=your-secret-key
LANGFUSE_HOST=https://cloud.langfuse.com  # Or your self-hosted instance
```

### Azure AI Search Setup

Ensure your Azure AI Search index has:
- Fields: `content`, `title`, `source` (adjust in code if different)
- Semantic configuration named `default` (or update `AZURE_SEARCH_SEMANTIC_CONFIG`)

### Google Gemini Setup

1. Get API key from [Google AI Studio](https://aistudio.google.com/)
2. Ensure the `gemini-2.0-flash` model is available in your region

### Langfuse Setup (Optional)

Langfuse provides observability and analytics for your LLM application:

1. **Sign up for Langfuse**:
   - Cloud: Visit [cloud.langfuse.com](https://cloud.langfuse.com) and create an account
   - Self-hosted: Follow the [self-hosting guide](https://langfuse.com/docs/deployment/self-host)

2. **Get API Keys**:
   - Navigate to your project settings
   - Copy your Public Key and Secret Key
   - Add them to your `.env` file

3. **Benefits**:
   - Track all LLM calls and tool executions
   - Monitor token usage and costs
   - Debug agent behavior with detailed traces
   - Analyze conversation patterns and user interactions
   - Track performance metrics across sessions

## Usage

### Basic Usage

```python
from agent import run_agent

# Simple query
result = run_agent("What is our company's vacation policy?")
print(result["response"])

# With conversation persistence
result = run_agent(
    "What are the latest AI regulations in the EU?",
    thread_id="user-123"
)
```

### Enhanced Agent with Query Analysis

```python
from agent_enhanced import query_agent

result = query_agent(
    "How does our ML pipeline compare to industry best practices?"
)

print(f"Query Type: {result['query_analysis']['query_type']}")
print(f"Tools Used: {result['tools_used']}")
print(f"Response: {result['response']}")
```

### Streaming Responses

```python
import asyncio
from agent import run_agent_stream

async def main():
    async for chunk in run_agent_stream("What's trending in AI today?"):
        if chunk["type"] == "token":
            print(chunk["content"], end="", flush=True)
        elif chunk["type"] == "tool_start":
            print(f"\n🔧 Using tool: {chunk['tool']}")

asyncio.run(main())
```

### FastAPI Server

```bash
# Start server
python src/server.py

# Or with uvicorn for development
uvicorn src.server:app --reload --port 8000
```

#### API Endpoints

```bash
# Health check
curl http://localhost:8000/health

# Non-streaming chat
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "What is our refund policy?"}'

# With user tracking for Langfuse
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{
    "message": "What is our refund policy?",
    "thread_id": "conversation-123",
    "user_id": "john@example.com"
  }'

# Streaming chat
curl -X POST http://localhost:8000/chat/stream \
  -H "Content-Type: application/json" \
  -d '{"message": "Latest news on AI safety"}'

# Get conversation history
curl http://localhost:8000/threads/{thread_id}/history
```

## Query Routing Logic

The agent uses the following heuristics to route queries:

| Query Type | Route To | Examples |
|------------|----------|----------|
| Internal | Knowledge Base | "What is our PTO policy?", "Find the Q3 report" |
| External | Web Search | "Latest AI news", "Current stock price of AAPL" |
| Combined | Both | "How does our pricing compare to competitors?" |
| Chat | No tools | "Hello", "What can you do?" |

## Customization

### Using Claude as the Main LLM

```python
from agent import create_agent_with_claude

agent = create_agent_with_claude()
```

### Custom Tool Definitions

Modify the `@tool` decorated functions in `agent.py` to customize:
- Azure Search fields and semantic config
- Gemini model and search parameters
- Response formatting

### Production Checkpointing

Replace `MemorySaver` with a persistent checkpointer:

```python
from langgraph.checkpoint.postgres import PostgresSaver

# PostgreSQL
connection_string = "postgresql://user:pass@localhost/db"
checkpointer = PostgresSaver.from_conn_string(connection_string)

# Or Redis
from langgraph.checkpoint.redis import RedisSaver
checkpointer = RedisSaver.from_url("redis://localhost:6379")
```

## Project Structure

```
langgraph-react-agent/
├── src/
│   ├── agent.py           # Core ReAct agent implementation
│   ├── agent_enhanced.py  # Enhanced agent with query analysis
│   └── server.py          # FastAPI server
├── requirements.txt
├── .env.example
└── README.md
```

## Testing

```python
# Run example queries
python src/agent.py

# Test enhanced agent
python src/agent_enhanced.py
```

## Troubleshooting

### "Azure Search semantic configuration not found"
- Ensure semantic search is enabled on your index
- Check `AZURE_SEARCH_SEMANTIC_CONFIG` matches your config name
- The agent falls back to keyword search if semantic fails

### "Google API quota exceeded"
- Check your Gemini API quota at Google AI Studio
- Consider implementing rate limiting
- Cache common web search queries

### "Tool calls not executing"
- Verify environment variables are set correctly
- Check LangSmith traces if enabled
- Ensure the LLM supports tool calling

## Observability

### Langfuse Tracing

Langfuse is automatically enabled when you configure the environment variables. All agent runs will be traced:

```python
from agent import run_agent

# Langfuse will automatically track this execution
result = run_agent(
    "What are the latest AI developments?",
    thread_id="user-123",
    user_id="john@example.com"  # Optional: track user
)
```

**Disabling Langfuse** for specific calls:
```python
result = run_agent(
    "What is our vacation policy?",
    enable_langfuse=False  # Disable Langfuse for this call
)
```

**View traces** at your Langfuse dashboard:
- Cloud: [cloud.langfuse.com](https://cloud.langfuse.com)
- Self-hosted: Your configured LANGFUSE_HOST

### LangSmith Tracing

You can also use LangSmith for debugging:

```bash
export LANGCHAIN_TRACING_V2=true
export LANGCHAIN_API_KEY=your-key
export LANGCHAIN_PROJECT=react-agent-dual-tools
```

View traces at [smith.langchain.com](https://smith.langchain.com)

## Evaluation

This project includes a comprehensive evaluation system using DeepEval 3.7.7 to measure agent performance across 5 key metrics.

### Quick Start

```bash
# Install evaluation dependencies (included in requirements.txt)
pip install deepeval==3.7.7

# Run evaluation on small dataset (10 test cases, ~2-3 minutes)
python scripts/run_evaluation.py --dataset small

# Run full evaluation with Langfuse reporting (46 test cases, ~8-12 minutes)
python scripts/run_evaluation.py --dataset main --langfuse
```

### Metrics

The evaluation system measures:

1. **Task Completion** (threshold: 0.7) - Did the agent complete the user's task?
2. **Tool Correctness** (threshold: 0.8) - Were the correct tools selected?
3. **Step Efficiency** (threshold: 0.6) - Was the execution path optimal?
4. **Plan Adherence** (threshold: 0.7) - Did the agent follow the documented strategy?
5. **Plan Quality** (threshold: 0.6) - Was the reasoning clear and logical?

### Test Datasets

- **Small Dataset**: 10 representative test cases for quick iteration
- **Main Dataset**: 46 test cases across 5 categories (KB queries, web search, combined, conversational, edge cases)

### Results

Evaluation results are saved to `results/` directory as JSON and CSV files. If Langfuse is enabled, results are also sent to Langfuse for centralized tracking and visualization.

### CLI Options

```bash
# Custom dataset
python scripts/run_evaluation.py --dataset /path/to/dataset.json

# Specify output directory
python scripts/run_evaluation.py --dataset small --output ./my_results

# Upload dataset to Langfuse first
python scripts/run_evaluation.py --dataset main --langfuse --upload-dataset

# Check configuration
python scripts/run_evaluation.py --check-config
```

### Documentation

For comprehensive documentation on the evaluation system, see [evaluations/README.md](./evaluations/README.md)
