# MCP Integration Guide

## Overview

This guide explains how Model Context Protocol (MCP) tools are integrated into your LangGraph agent. MCP allows you to extend your agent with external tools and capabilities.

## What We Built

We've created a complete MCP integration that adds **time and date tools** to your existing agent, demonstrating how to:

1. Create MCP-style tools
2. Integrate them with LangChain/LangGraph
3. Use them alongside your existing search tools

## Files Created

### 1. `simple_mcp_demo.py` (✅ Working Demo)

The simplest working example showing MCP-style tool integration.

**What it does:**
- Adds 2 MCP-style tools (`get_current_time`, `get_current_date`) to your agent
- Shows how to integrate new tools alongside existing ones
- Demonstrates the complete agent workflow

**Tools available:**
- `search_knowledge_base` - Your Azure AI Search (existing)
- `search_web` - Gemini web search (existing)
- `get_current_time` - Get time in any timezone (NEW MCP-style)
- `get_current_date` - Get date in any timezone (NEW MCP-style)

**Run it:**
```bash
python simple_mcp_demo.py
```

### 2. `time_mcp_server.py` (MCP Server)

A standalone MCP server that provides time/date tools using the official MCP protocol.

**What it does:**
- Implements a proper MCP server using the `mcp` SDK
- Provides 3 tools: `get_current_time`, `get_current_date`, `convert_timezone`
- Runs as a separate process and communicates via stdio

**Note:** This demonstrates the full MCP protocol but requires more complex async handling.

### 3. `mcp_integration.py` (MCP Client Utilities)

Utilities for connecting to MCP servers and converting their tools to LangChain format.

**Key components:**
- `MCPClientManager` - Manages connection to MCP servers
- `create_langchain_tool_from_mcp()` - Converts MCP tools to LangChain tools
- `get_mcp_tools_as_langchain()` - Full workflow helper

### 4. `agent_with_mcp.py` (Advanced Integration)

Shows how to connect to a real MCP server and use its tools.

**Note:** This has async complexity that's being refined.

---

## How MCP Integration Works

### Concept Overview

```
┌─────────────────────────────────────────────────────────┐
│                    Your Agent                           │
│  ┌────────────────────────────────────────────────┐    │
│  │  LLM (Azure OpenAI GPT-4)                      │    │
│  └────────────────────────────────────────────────┘    │
│                         │                               │
│                         ├── Decides which tool to use   │
│                         ↓                               │
│  ┌────────────────────────────────────────────────┐    │
│  │  Tool Router                                   │    │
│  └────────────────────────────────────────────────┘    │
│          │              │              │                │
│          ↓              ↓              ↓                │
│  ┌──────────┐   ┌──────────┐   ┌──────────────┐       │
│  │  Azure   │   │  Gemini  │   │  MCP Tools   │       │
│  │  Search  │   │  Search  │   │  (Time/Date) │       │
│  └──────────┘   └──────────┘   └──────────────┘       │
└─────────────────────────────────────────────────────────┘
```

### Step-by-Step Workflow

#### 1. **Define MCP-Style Tools**

MCP tools are just regular Python functions decorated with `@tool`:

```python
from langchain_core.tools import tool

@tool
def get_current_time(timezone: str = "UTC") -> str:
    """
    Get the current time in a specific timezone.

    Args:
        timezone: The timezone name (e.g., 'Asia/Tokyo')

    Returns:
        Current time in the specified timezone
    """
    # Your implementation here
    return f"Current time in {timezone}: ..."
```

**Key points:**
- Clear docstring (LLM uses this to understand when to use the tool)
- Type hints for parameters
- Simple return type (usually `str`)

#### 2. **Combine with Existing Tools**

Add MCP tools to your existing tool list:

```python
from agent import search_knowledge_base, search_web

all_tools = [
    search_knowledge_base,  # Existing tool
    search_web,             # Existing tool
    get_current_time,       # NEW MCP tool
    get_current_date,       # NEW MCP tool
]
```

#### 3. **Bind to LLM**

```python
llm_with_tools = llm.bind_tools(all_tools)
```

This tells the LLM about all available tools and their schemas.

#### 4. **Create Tool Node**

```python
from langgraph.prebuilt import ToolNode

tool_node = ToolNode(all_tools)
```

This executes tool calls during agent runtime.

#### 5. **Update System Prompt**

Tell the LLM about the new tools:

```python
SYSTEM_PROMPT = """
You have access to:
1. search_knowledge_base - Internal documents
2. search_web - Web search
3. get_current_time - Get time in any timezone
4. get_current_date - Get date in any timezone

Examples:
- "What time in Tokyo?" → use get_current_time
- "Today's date?" → use get_current_date
"""
```

#### 6. **Build the Graph**

The agent graph remains the same:

```python
workflow = StateGraph(AgentState)
workflow.add_node("agent", agent_node)       # LLM decides
workflow.add_node("tools", tool_node)         # Execute tools
workflow.add_edge(START, "agent")
workflow.add_conditional_edges("agent", should_continue,
    {"tools": "tools", "end": END})
workflow.add_edge("tools", "agent")           # Loop back

agent = workflow.compile(checkpointer=MemorySaver())
```

---

## Testing the Integration

### Test 1: Time Query

```bash
python simple_mcp_demo.py
```

**Query:** "What time is it in Tokyo right now?"

**Expected behavior:**
1. Agent analyzes query
2. Realizes it needs timezone information
3. Calls `get_current_time(timezone="Asia/Tokyo")`
4. Returns formatted time

**Output:**
```
Tools used: get_current_time
Response: The current time in Tokyo (JST) is 23:20:49.
```

### Test 2: Date Query

**Query:** "What's today's date in New York?"

**Expected behavior:**
1. Agent identifies date request
2. Calls `get_current_date(timezone="America/New_York")`
3. Returns formatted date

**Output:**
```
Tools used: get_current_date
Response: Today's date in New York is Tuesday, December 30, 2025.
```

### Test 3: Search Query

**Query:** "Explain about strategic shifts in footwear companies"

**Expected behavior:**
1. Agent recognizes this as a research query
2. Calls `search_knowledge_base(query="...")`
3. Returns information from Azure Search

---

## Key Concepts Explained

### 1. **MCP vs Regular Tools**

| Aspect | Regular LangChain Tool | MCP Tool |
|--------|----------------------|----------|
| Definition | Python function with `@tool` | Same, but follows MCP protocol |
| Communication | Direct function call | Can be local or remote (via stdio/HTTP) |
| Use case | Simple integrations | Complex external services |

**In our demo:** We use the MCP *style* (simple functions) for clarity.

### 2. **Tool Routing**

The LLM automatically decides which tool to use based on:
- Tool name and description
- Query intent
- System prompt guidance

**Example decision process:**
```
User: "What time is it in Paris?"

LLM thinks:
- Not a research query → skip search_knowledge_base
- Not about current events → skip search_web
- Asks about current time → use get_current_time
- Needs timezone parameter → Paris is "Europe/Paris"

LLM calls: get_current_time(timezone="Europe/Paris")
```

### 3. **Tool Execution Flow**

```
User Query
    ↓
Agent Node (LLM decides which tool)
    ↓
Tool Node (executes the chosen tool)
    ↓
Agent Node (LLM sees result, formulates response)
    ↓
Final Response
```

---

## Adding Your Own MCP Tools

### Option 1: Simple Function (Recommended)

```python
@tool
def your_custom_tool(param1: str, param2: int = 10) -> str:
    """
    Brief description of what this tool does.

    Use this tool when:
    - Condition 1
    - Condition 2

    Args:
        param1: Description of param1
        param2: Description of param2 (default: 10)

    Returns:
        Description of return value
    """
    # Your logic here
    result = f"Processed {param1} with {param2}"
    return result

# Add to agent
all_tools = [
    search_knowledge_base,
    search_web,
    your_custom_tool,  # Your new tool!
]
```

### Option 2: Connect to External MCP Server

Use `mcp_integration.py` utilities:

```python
from mcp_integration import get_mcp_tools_as_langchain

# Connect to MCP server
client, mcp_tools = await get_mcp_tools_as_langchain(
    command="python",
    args=["your_mcp_server.py"]
)

# Combine with existing tools
all_tools = [
    search_knowledge_base,
    search_web,
    *mcp_tools  # All tools from MCP server
]
```

---

## Common MCP Tool Examples

### Weather Tool

```python
@tool
def get_weather(location: str) -> str:
    """Get current weather for a location."""
    # Call weather API
    return f"Weather in {location}: ..."
```

### Calculator Tool

```python
@tool
def calculate(expression: str) -> str:
    """Evaluate a mathematical expression safely."""
    # Use eval with safety checks
    return str(eval(expression))
```

### Database Query Tool

```python
@tool
def query_database(sql: str) -> str:
    """Execute a read-only SQL query."""
    # Connect to DB and execute
    return "Query results: ..."
```

---

## Troubleshooting

### Issue: Tool not being called

**Solution:** Improve tool description and system prompt

```python
@tool
def my_tool(query: str) -> str:
    """
    BEFORE: A tool that does something.

    AFTER: Search internal knowledge base for company information.
    Use this tool for questions about products, policies, or procedures.
    Do NOT use for current events or real-time data.
    """
```

### Issue: Wrong tool being called

**Solution:** Add explicit guidance in system prompt

```python
SYSTEM_PROMPT = """
For time/date questions → use get_current_time or get_current_date
For research queries → use search_knowledge_base FIRST
For current events → use search_web
"""
```

### Issue: Async errors

**Solution:** Keep tools synchronous (like in `simple_mcp_demo.py`)

```python
# ❌ Async (causes issues in thread pools)
@tool
async def my_tool(query: str) -> str:
    return await some_async_call()

# ✅ Sync (works reliably)
@tool
def my_tool(query: str) -> str:
    return some_sync_call()
```

---

## Next Steps

1. **Experiment with simple_mcp_demo.py**
   - Modify the time tools
   - Add your own custom tools

2. **Create domain-specific tools**
   - Database queries
   - API integrations
   - File operations

3. **Test tool routing**
   - Try queries that should use different tools
   - Refine system prompts

4. **Connect to real MCP servers**
   - Use official MCP servers (when needed)
   - Build your own MCP servers

---

## Summary

**What you have now:**
- ✅ Working agent with 4 tools (2 search + 2 MCP time tools)
- ✅ Clear pattern for adding new tools
- ✅ Example code to extend

**The MCP integration pattern:**
1. Define tools with `@tool` decorator
2. Add to tool list
3. Update system prompt
4. Let the LLM route automatically

**Key files:**
- `simple_mcp_demo.py` - Start here!
- `mcp_integration.py` - Advanced MCP client utilities
- `time_mcp_server.py` - Example MCP server

The beauty of this approach is that **adding new capabilities is as simple as defining a new Python function!**
