"""
Simple MCP Integration Demo

This is a simplified example showing how MCP tools work.
We'll demonstrate the concept without the full async complexity.
"""

import os
from dotenv import load_dotenv
from datetime import datetime
import pytz

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool
from langchain_openai import AzureChatOpenAI
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver

# Import existing tools
from agent import search_knowledge_base, search_web, AgentState, get_langfuse_handler

load_dotenv()


# ============================================================================
# Simplified MCP-Style Tools (Synchronous Wrappers)
# ============================================================================

@tool
def get_current_time(timezone: str = "UTC") -> str:
    """
    Get the current time in a specific timezone.

    This is an MCP-style tool that demonstrates how external functionality
    can be integrated into your agent.

    Args:
        timezone: The timezone name (e.g., 'America/New_York', 'Asia/Tokyo'). Defaults to 'UTC'.

    Returns:
        Current time in the specified timezone
    """
    try:
        tz = pytz.timezone(timezone)
        current_time = datetime.now(tz)
        formatted_time = current_time.strftime("%H:%M:%S %Z")
        return f"Current time in {timezone}: {formatted_time}"
    except Exception as e:
        return f"Error: Invalid timezone '{timezone}'. {str(e)}"


@tool
def get_current_date(timezone: str = "UTC") -> str:
    """
    Get the current date in a specific timezone.

    This is an MCP-style tool for getting date information.

    Args:
        timezone: The timezone name. Defaults to 'UTC'.

    Returns:
        Current date in the specified timezone
    """
    try:
        tz = pytz.timezone(timezone)
        current_date = datetime.now(tz)
        formatted_date = current_date.strftime("%Y-%m-%d")
        day_name = current_date.strftime("%A")
        return f"Current date in {timezone}: {formatted_date} ({day_name})"
    except Exception as e:
        return f"Error: Invalid timezone '{timezone}'. {str(e)}"


# ============================================================================
# Enhanced System Prompt
# ============================================================================

ENHANCED_SYSTEM_PROMPT = """You are an intelligent research assistant with access to multiple tools:

**Search Tools:**
1. **search_knowledge_base** (PRIMARY): Internal Azure AI Search
2. **search_web** (SECONDARY): Gemini search for current information

**MCP-Style Time Tools:**
3. **get_current_time**: Get current time in any timezone
4. **get_current_date**: Get current date in any timezone

## Tool Selection:
- For research → use search_knowledge_base first
- For current/real-time info → use search_web
- For time/date questions → use get_current_time or get_current_date

Examples:
- "What time is it in Tokyo?" → get_current_time(timezone="Asia/Tokyo")
- "What's the date today?" → get_current_date()
- "Strategic shifts in footwear?" → search_knowledge_base

Think step-by-step before choosing tools."""


# ============================================================================
# Create Enhanced Agent
# ============================================================================

def create_simple_enhanced_agent(langfuse_handler=None):
    """Create agent with MCP-style tools."""

    # Initialize LLM
    llm = AzureChatOpenAI(
        azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4o"),
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-08-01-preview"),
        temperature=1
    )

    # All tools: 2 search + 2 MCP-style time tools
    all_tools = [
        search_knowledge_base,
        search_web,
        get_current_time,  # MCP-style tool 1
        get_current_date,  # MCP-style tool 2
    ]

    print(f"\n✓ Agent created with {len(all_tools)} tools:")
    print(f"  - search_knowledge_base")
    print(f"  - search_web")
    print(f"  - get_current_time (MCP-style)")
    print(f"  - get_current_date (MCP-style)")

    llm_with_tools = llm.bind_tools(all_tools)
    tool_node = ToolNode(all_tools)

    # Agent logic
    def agent_node(state: AgentState) -> AgentState:
        messages = state["messages"]
        if not any(isinstance(m, SystemMessage) for m in messages):
            messages = [SystemMessage(content=ENHANCED_SYSTEM_PROMPT)] + messages

        if langfuse_handler:
            response = llm_with_tools.invoke(messages, config={"callbacks": [langfuse_handler]})
        else:
            response = llm_with_tools.invoke(messages)

        return {"messages": [response], "tool_calls_count": state.get("tool_calls_count", 0)}

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

    return workflow.compile(checkpointer=MemorySaver())


# ============================================================================
# Run Agent
# ============================================================================

def run_simple_enhanced_agent(query: str, enable_langfuse: bool = False):
    """Run a query through the enhanced agent."""

    langfuse_handler = None
    if enable_langfuse:
        langfuse_handler = get_langfuse_handler()

    agent = create_simple_enhanced_agent(langfuse_handler)

    config = {"configurable": {"thread_id": "demo"}}
    if langfuse_handler:
        config["callbacks"] = [langfuse_handler]
        config["metadata"] = {"trace_name": "mcp_demo"}

    result = agent.invoke(
        {"messages": [HumanMessage(content=query)]},
        config=config
    )

    if langfuse_handler:
        langfuse_handler.client.flush()

    final_message = result["messages"][-1]
    return {
        "response": final_message.content,
        "tool_calls_count": result.get("tool_calls_count", 0),
        "last_tool_used": result.get("last_tool_used", ""),
    }


if __name__ == "__main__":
    print("=" * 80)
    print("Simple MCP-Style Tool Integration Demo")
    print("=" * 80)

    test_queries = [
        "What time is it in Tokyo right now?",
        "What's today's date in New York?",
        # "Explain about the strategic shifts in footwear and apparel companies?",
    ]

    for query in test_queries:
        print(f"\n{'='*80}")
        print(f"Query: {query}")
        print('='*80)

        result = run_simple_enhanced_agent(query)

        print(f"\nResponse:\n{result['response']}")
        print(f"\nTools used: {result['last_tool_used']}")
        print(f"Total tool calls: {result['tool_calls_count']}")
