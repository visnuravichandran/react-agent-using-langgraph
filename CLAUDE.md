# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

This is a LangGraph-based ReAct agent that intelligently routes queries between Azure AI Search (internal knowledge base) and Gemini Web Search (real-time external information). The agent uses tool calling to decide which data source to query based on the user's question.

## Commands

### Environment Setup
```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Configure environment variables
cp .env.example .env
# Edit .env with Azure OpenAI, Azure AI Search, Google API, and optional Langfuse credentials
```

### Running the Agent
```bash
# Run basic agent (modify test queries in agent.py main block)
python agent.py

# Run enhanced agent with query pre-analysis
python agent_enhanced.py

# Run MCP demo (includes time/date tools)
python simple_mcp_demo.py

# Start FastAPI server
python server.py
# Or with auto-reload: uvicorn server:app --reload --port 8000
```

### Testing
```bash
# Run tests
pytest

# Test with async support
pytest -v tests/ --asyncio-mode=auto
```

### Evaluation
```bash
# Run evaluation on small dataset (10 test cases, ~2-3 minutes)
python scripts/run_evaluation.py --dataset small

# Run full evaluation with Langfuse reporting (46 test cases, ~8-12 minutes)
python scripts/run_evaluation.py --dataset main --langfuse

# Upload dataset to Langfuse
python scripts/run_evaluation.py --dataset main --langfuse --upload-dataset

# Custom dataset or output directory
python scripts/run_evaluation.py --dataset /path/to/dataset.json --output ./results

# Check evaluation configuration
python scripts/run_evaluation.py --check-config

# Run evaluation tests
pytest tests/test_evaluations.py -v
```

**Evaluation Metrics:**
- **Task Completion**: Did the agent complete the task?
- **Tool Correctness**: Were the right tools selected?
- **Step Efficiency**: Was execution optimal?
- **Plan Adherence**: Did agent follow the strategy?
- **Plan Quality**: Was reasoning clear?

See [evaluations/README.md](evaluations/README.md) for comprehensive documentation.

## Architecture

### Core Components

**ReAct Agent Loop**: The agent follows the Reason-Act-Observe pattern:
1. **Reason** - LLM analyzes the query and decides which tool(s) to call
2. **Act** - Execute the selected tool(s) (search_knowledge_base, search_web)
3. **Observe** - LLM receives tool results and formulates response
4. Loop continues until the agent decides to end

**LangGraph Workflow**: Implemented as a StateGraph with three nodes:
- `agent` node: LLM reasoning and tool selection (agent.py:271-288)
- `tools` node: Tool execution with usage tracking (agent.py:301-316)
- Conditional edges route between nodes based on whether tool calls are needed (agent.py:290-299)

**State Management**: Uses `AgentState` (agent.py:183-186) extending `MessagesState` with:
- `messages`: Full conversation history
- `tool_calls_count`: Tracks number of tool invocations
- `last_tool_used`: Records which tools were called

**Conversation Memory**: Uses `MemorySaver` checkpointer for in-memory conversation persistence. For production, replace with PostgreSQL or Redis checkpointer (see README.md:242-251).

### Tool Routing Strategy

The agent's tool selection is **guided by explicit system prompts** (agent.py:193-237) that establish a clear hierarchy:

**Priority Order**:
1. **Always start with `search_knowledge_base`** - Default for most queries
2. **Use `search_web`** only when explicitly needed (current events, breaking news, "latest", "today")
3. **Use both** when comparison or combined information is required

This is a **deliberate design choice** because internal knowledge base contains curated, organization-specific insights that should be prioritized over general web search.

### Tools

**search_knowledge_base** (agent.py:78-126):
- Searches Azure AI Search using hybrid search (semantic + keyword)
- Returns top 5 results with title and content
- Uses semantic configuration for better relevance
- **Primary tool** - should be used first for most queries

**search_web** (agent.py:130-176):
- Uses Gemini 2.5 Flash with grounded search
- Returns web search results with grounding metadata
- **Secondary tool** - only for real-time/external information

### Enhanced Agent Variant

`agent_enhanced.py` adds **query pre-analysis** before tool execution:
- Classifies queries into 4 types: KNOWLEDGE_BASE, WEB_SEARCH, COMBINED, CONVERSATIONAL
- Uses structured output (Pydantic models) for classification
- Provides query reformulation and tool recommendations
- Helps improve tool routing decisions

### MCP Integration

**Model Context Protocol (MCP)** support allows extending the agent with external tools:

**Simple MCP Style** (`simple_mcp_demo.py`):
- Add new tools using `@tool` decorator
- Combine with existing tools in a list
- LLM automatically routes based on tool descriptions

**Full MCP Server** (`time_mcp_server.py`, `mcp_integration.py`):
- Run external MCP servers that communicate via stdio
- Convert MCP tools to LangChain format
- Enable integration with official MCP servers

**Adding New MCP Tools**: See MCP_INTEGRATION_GUIDE.md for comprehensive examples and patterns.

### FastAPI Server

`server.py` provides production-ready HTTP endpoints:
- `/health` - Health check
- `/chat` - Non-streaming chat with conversation persistence
- `/chat/stream` - Server-Sent Events (SSE) streaming
- `/threads/{thread_id}/history` - Retrieve conversation history

All endpoints support:
- Thread-based conversation memory via `thread_id`
- User tracking via `user_id` (for Langfuse)
- Automatic Langfuse tracing when configured

### Observability

**Langfuse Integration** (agent.py:34-70):
- Automatically enabled when environment variables are set
- Tracks all LLM calls, tool executions, and conversation flows
- Provides token usage, cost tracking, and performance metrics
- Can be disabled per-call with `enable_langfuse=False`

**LangSmith**: Also supported via environment variables for debugging traces.

## Key Implementation Patterns

### Tool Definition Pattern
All tools use `@tool` decorator with detailed docstrings that guide LLM behavior:
```python
@tool
def your_tool(param: str) -> str:
    """
    Clear description of what the tool does.

    Use this tool when:
    - Condition 1
    - Condition 2

    Args:
        param: Parameter description

    Returns:
        Return value description
    """
```

The docstring is **critical** - the LLM uses it to decide when to call the tool.

### Agent Creation Pattern
The `create_agent()` function (agent.py:240-344) follows this structure:
1. Initialize LLM (Azure OpenAI or Anthropic Claude)
2. Bind tools to LLM
3. Create ToolNode for execution
4. Define agent_node, should_continue, tool_execution_node
5. Build StateGraph with conditional routing
6. Compile with checkpointer for memory

### Streaming Pattern
Use `astream_events()` with version="v2" for streaming (agent.py:490-494):
- Stream tokens as they're generated (`on_chat_model_stream`)
- Track tool start/end events
- Handle async iteration properly

## Environment Variables

Required:
- `AZURE_OPENAI_DEPLOYMENT`, `AZURE_OPENAI_ENDPOINT`, `AZURE_OPENAI_API_KEY`, `AZURE_OPENAI_API_VERSION`
- `AZURE_SEARCH_ENDPOINT`, `AZURE_SEARCH_INDEX_NAME`, `AZURE_SEARCH_API_KEY`
- `AZURE_SEARCH_SEMANTIC_CONFIG` (default: "alpha-sense-semantic-config")
- `GOOGLE_API_KEY` (for Gemini web search)

Optional (for observability):
- `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY`, `LANGFUSE_HOST`

## Common Customizations

### Switch to Claude as Main LLM
Use `create_agent_with_claude()` (agent.py:351-394) instead of `create_agent()`. Requires `ANTHROPIC_API_KEY`.

### Modify Tool Behavior
- Azure Search: Adjust fields, semantic config, top_k in `search_knowledge_base`
- Gemini Search: Change model, temperature, or search parameters in `search_web`

### Add New Tools
1. Define tool with `@tool` decorator and clear docstring
2. Add to tools list when binding to LLM
3. Update SYSTEM_PROMPT to guide tool usage
4. Test routing behavior with example queries

### Change Checkpointer for Production
Replace `MemorySaver()` with persistent storage:
- PostgreSQL: `from langgraph.checkpoint.postgres import PostgresSaver`
- Redis: `from langgraph.checkpoint.redis import RedisSaver`

## Important Notes

- **Tool routing is prompt-driven**: The system prompt heavily influences which tools the LLM selects
- **Default to knowledge base**: The agent is designed to prefer internal data over web search
- **Semantic search requires configuration**: Ensure Azure Search index has semantic configuration enabled
- **Async complexity**: Keep custom tools synchronous when possible to avoid thread pool issues
- **Langfuse flush**: Always call `langfuse_handler.client.flush()` after agent runs to ensure traces are sent
