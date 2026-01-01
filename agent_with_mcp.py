"""
Agent with MCP Tool Integration - Complete Example with Langfuse Observability

This demonstrates the full workflow of integrating MCP tools with your LangGraph agent:
1. Start an MCP server (time_mcp_server.py)
2. Connect to it using the MCP client
3. Convert MCP tools to LangChain tools
4. Add them to your agent alongside existing tools
5. Track everything with Langfuse for observability

## Langfuse Integration

Langfuse is ENABLED by default and tracks:
- ✓ All LLM calls (prompts, completions, tokens)
- ✓ Tool calls (which tools, inputs, outputs)
- ✓ Agent reasoning steps
- ✓ User sessions and traces
- ✓ Latency and performance metrics

### What Gets Tracked:

**Metadata:**
- session_id: Thread ID for conversation grouping
- user_id: User identifier (default: "anonymous")
- trace_name: Custom trace name for this query
- query: The original user query
- agent_type: "mcp_integrated_agent"
- tools_available: List of all tools the agent can use

**Traces Include:**
- LLM reasoning (system prompt, user message, AI response)
- Tool selections and executions
- MCP server communication
- Response generation

### Setup:

Requires these environment variables in .env:
- LANGFUSE_PUBLIC_KEY=pk-lf-...
- LANGFUSE_SECRET_KEY=sk-lf-...
- LANGFUSE_HOST=https://observability.slatepathcapital.ai

### Viewing Traces:

After running queries, view traces at your Langfuse dashboard:
https://observability.slatepathcapital.ai

No external API keys required for MCP tools (time/date)!
"""

import os
import asyncio
from typing import Optional
from dotenv import load_dotenv

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import AzureChatOpenAI
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver

# Import existing tools and utilities
from agent import (
    search_knowledge_base,
    search_web,
    AgentState,
    get_langfuse_handler
)

# Import MCP integration utilities
from mcp_integration import get_mcp_tools_as_langchain

load_dotenv()


# ============================================================================
# Enhanced System Prompt with MCP Tools
# ============================================================================

ENHANCED_SYSTEM_PROMPT = """You are an intelligent research assistant with access to multiple powerful tools:

**Search Tools:**
1. **search_knowledge_base** (PRIMARY): Internal Azure AI Search with organization data
2. **search_web** (SECONDARY): Gemini grounded search for current information

**MCP Tools (Time & Date):**
3. **get_current_time**: Get current time in any timezone
4. **get_current_date**: Get current date in any timezone
5. **convert_timezone**: Convert time between timezones

## Tool Selection Strategy

**For research queries:**
- ALWAYS check search_knowledge_base FIRST
- Use search_web only for current/real-time information

**For time/date queries:**
- Use get_current_time for "what time is it" questions
- Use get_current_date for "what's the date" questions
- Use convert_timezone for timezone conversions

**Examples:**
- "What time is it in Tokyo?" → use get_current_time with timezone="Asia/Tokyo"
- "What's today's date?" → use get_current_date
- "Convert 2 PM EST to PST" → use convert_timezone
- "Strategic shifts in footwear companies?" → use search_knowledge_base

Think step-by-step about which tool(s) to use before responding."""


# ============================================================================
# Create Agent with MCP Tools
# ============================================================================

async def create_agent_with_mcp(
    model_name: str = "gpt-4o",
    langfuse_handler: Optional = None
):
    """
    Create agent with both existing tools and MCP tools.

    This demonstrates the complete integration workflow:
    1. Connect to MCP server
    2. Get MCP tools as LangChain tools
    3. Combine with existing tools
    4. Create unified agent
    """

    print("\n" + "=" * 60)
    print("Setting up Agent with MCP Integration")
    print("=" * 60)

    # Initialize LLM
    llm = AzureChatOpenAI(
        azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT", model_name),
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-08-01-preview"),
        temperature=1
    )

    # ========================================================================
    # STEP 1: Connect to MCP server and get its tools
    # ========================================================================
    print("\n[1/4] Connecting to MCP Time Server...")

    mcp_client, mcp_tools, event_loop = await get_mcp_tools_as_langchain(
        command="python",
        args=["time_mcp_server.py"]
    )

    print(f"✓ Successfully connected to MCP server")
    print(f"✓ Retrieved {len(mcp_tools)} MCP tools")
    print(f"✓ Event loop captured for thread-safe async calls")

    # ========================================================================
    # STEP 2: Combine MCP tools with existing tools
    # ========================================================================
    print("\n[2/4] Combining tools...")

    all_tools = [
        # Existing search tools
        search_knowledge_base,
        search_web,
        # MCP tools (time & date)
        *mcp_tools
    ]

    print(f"✓ Total tools available: {len(all_tools)}")
    print(f"  - Search tools: 2")
    print(f"  - MCP tools: {len(mcp_tools)}")

    # ========================================================================
    # STEP 3: Bind tools to LLM
    # ========================================================================
    print("\n[3/4] Binding tools to LLM...")

    llm_with_tools = llm.bind_tools(all_tools)
    tool_node = ToolNode(all_tools)

    print(f"✓ Tools bound to {model_name}")

    # ========================================================================
    # STEP 4: Create agent graph
    # ========================================================================
    print("\n[4/4] Creating agent graph...")

    def agent_node(state: AgentState) -> AgentState:
        messages = state["messages"]
        if not any(isinstance(m, SystemMessage) for m in messages):
            messages = [SystemMessage(content=ENHANCED_SYSTEM_PROMPT)] + messages

        if langfuse_handler:
            response = llm_with_tools.invoke(messages, config={"callbacks": [langfuse_handler]})
        else:
            response = llm_with_tools.invoke(messages)

        return {
            "messages": [response],
            "tool_calls_count": state.get("tool_calls_count", 0)
        }

    def should_continue(state: AgentState):
        last_message = state["messages"][-1]
        if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
            return "tools"
        return "end"

    def tool_execution_node(state: AgentState) -> AgentState:
        messages = state["messages"]
        last_message = messages[-1]
        tool_names = [tc["name"] for tc in last_message.tool_calls] if hasattr(last_message, 'tool_calls') and last_message.tool_calls else []

        tool_results = tool_node.invoke(state)

        return {
            "messages": tool_results["messages"],
            "tool_calls_count": state.get("tool_calls_count", 0) + len(tool_names),
            "last_tool_used": ", ".join(tool_names)
        }

    # Build graph
    workflow = StateGraph(AgentState)
    workflow.add_node("agent", agent_node)
    workflow.add_node("tools", tool_execution_node)
    workflow.add_edge(START, "agent")
    workflow.add_conditional_edges("agent", should_continue, {"tools": "tools", "end": END})
    workflow.add_edge("tools", "agent")

    agent = workflow.compile(checkpointer=MemorySaver())

    print("✓ Agent graph created successfully")
    print("=" * 60)

    return agent, mcp_client, event_loop


# ============================================================================
# Run Agent with MCP Tools
# ============================================================================

async def run_agent_with_mcp(
    query: str,
    thread_id: str = "mcp_demo",
    user_id: Optional[str] = None,
    trace_name: Optional[str] = None,
    enable_langfuse: bool = True
):
    """
    Run a query through the agent with MCP tools.

    IMPORTANT: This runs the agent in an async context to keep the event loop
    alive for MCP tool calls that need to communicate with the MCP server.

    Args:
        query: User's question
        thread_id: Conversation thread ID for memory persistence
        user_id: Optional user ID for Langfuse tracking
        trace_name: Optional custom trace name for Langfuse
        enable_langfuse: Whether to enable Langfuse tracing (default: True)

    Returns:
        Dict with response, tool_calls_count, last_tool_used, and messages
    """

    # ========================================================================
    # Langfuse Setup
    # ========================================================================
    langfuse_handler = None
    if enable_langfuse:
        langfuse_handler = get_langfuse_handler()
        if langfuse_handler:
            print("✓ Langfuse tracing enabled")
        else:
            print("⚠ Langfuse not configured (missing API keys)")

    # Create agent with MCP tools
    agent, mcp_client, event_loop = await create_agent_with_mcp(langfuse_handler=langfuse_handler)

    try:
        # ====================================================================
        # Configure Agent Run
        # ====================================================================
        config = {"configurable": {"thread_id": thread_id}}

        # Add Langfuse callbacks and metadata
        if langfuse_handler:
            config["callbacks"] = [langfuse_handler]
            config["metadata"] = {
                "session_id": thread_id,
                "user_id": user_id or "anonymous",
                "trace_name": trace_name or "mcp_agent_query",
                "query": query,
                "agent_type": "mcp_integrated_agent",
                "tools_available": ["search_knowledge_base", "search_web", "get_current_time", "get_current_date", "convert_timezone"]
            }

        # ====================================================================
        # Execute Agent
        # ====================================================================
        # Run the sync agent.invoke() in a thread pool while keeping the async loop alive
    # This allows MCP tool calls to use asyncio.run_coroutine_threadsafe
        result = await asyncio.to_thread(
            agent.invoke,
            {"messages": [HumanMessage(content=query)]},
            config
        )

        # ====================================================================
        # Flush Langfuse Data
        # ====================================================================
        if langfuse_handler:
            langfuse_handler.client.flush()
            print("✓ Langfuse trace flushed")

        final_message = result["messages"][-1]
        return {
            "response": final_message.content,
            "tool_calls_count": result.get("tool_calls_count", 0),
            "last_tool_used": result.get("last_tool_used", ""),
            "messages": result["messages"]
        }

    finally:
        # IMPORTANT: Disconnect from MCP server when done
        await mcp_client.disconnect()


# ============================================================================
# Demo
# ============================================================================

async def main():
    print("\n" + "=" * 80)
    print("Agent with MCP Tool Integration - Demo")
    print("=" * 80)
    print("\n📊 Langfuse tracing is ENABLED by default")
    print("   View traces at: https://observability.slatepathcapital.ai")
    print("=" * 80)

    # Test queries with different trace configurations
    test_cases = [
        {
            "query": "What time is it in Tokyo?",
            "user_id": "demo_user_1",
            "trace_name": "timezone_query_tokyo"
        },
        # {
        #     "query": "What's today's date in New York?",
        #     "user_id": "demo_user_1",
        #     "trace_name": "date_query_newyork"
        # },
        # {
        #     "query": "Convert 2 PM from America/New_York to Asia/Tokyo timezone",
        #     "user_id": "demo_user_2",
        #     "trace_name": "timezone_conversion"
        # },
        # Uncomment to test search tools:
        {
            "query": "Explain about the strategic shifts in footwear and apparel companies?",
            "user_id": "demo_user_2",
            "trace_name": "knowledge_base_query"
        },
        {
            "query": "Explain about the strategic shifts in footwear and apparel companies and what are the latest developments in AI regulation in 2024?",
            "user_id": "demo_user_3",
            "trace_name": "knowledge_base_query"
        },
    ]

    for idx, test_case in enumerate(test_cases, 1):
        print(f"\n{'='*80}")
        print(f"[Query {idx}/{len(test_cases)}] {test_case['query']}")
        print(f"User: {test_case['user_id']} | Trace: {test_case['trace_name']}")
        print('='*80)

        result = await run_agent_with_mcp(
            query=test_case["query"],
            thread_id=f"demo_thread_{idx}",
            user_id=test_case["user_id"],
            trace_name=test_case["trace_name"],
            enable_langfuse=True  # Explicitly enable Langfuse
        )

        print(f"\nResponse:\n{result['response']}")
        print(f"\nTools used: {result['last_tool_used']}")
        print(f"Total tool calls: {result['tool_calls_count']}")
        print("-" * 80)


if __name__ == "__main__":
    asyncio.run(main())
