"""
ReAct Agent with Dual Tool Routing
- Azure AI Search for knowledge base queries
- Gemini for web search queries
- Intelligent routing for combination queries
"""

import os
from typing import Annotated, Literal, TypedDict, Optional
from dotenv import load_dotenv

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langchain_openai import AzureChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver

from langfuse.langchain import CallbackHandler

from azure.search.documents import SearchClient
from azure.core.credentials import AzureKeyCredential
from google import genai
from google.genai.types import Tool as GeminiTool, GoogleSearch

load_dotenv()


# ============================================================================
# Langfuse Configuration
# ============================================================================

def get_langfuse_handler(
    session_id: Optional[str] = None,
    user_id: Optional[str] = None,
    trace_name: Optional[str] = None
) -> Optional[CallbackHandler]:
    """
    Create and return a Langfuse callback handler if configured.

    Note: Langfuse credentials should be set via environment variables:
    - LANGFUSE_PUBLIC_KEY
    - LANGFUSE_SECRET_KEY
    - LANGFUSE_HOST (optional, defaults to https://cloud.langfuse.com)

    Args:
        session_id: Optional session ID for grouping traces (set via metadata)
        user_id: Optional user ID for tracking (set via metadata)
        trace_name: Optional name for the trace (set via metadata)

    Returns:
        CallbackHandler if Langfuse is configured, None otherwise
    """
    # Check if Langfuse is configured
    if not all([
        os.getenv("LANGFUSE_PUBLIC_KEY"),
        os.getenv("LANGFUSE_SECRET_KEY")
    ]):
        return None

    try:
        # In Langfuse 3.x, CallbackHandler reads credentials from environment
        # Metadata like session_id, user_id are set via langchain config
        handler = CallbackHandler()

        return handler
    except Exception as e:
        print(f"Warning: Failed to initialize Langfuse: {str(e)}")
        return None


# ============================================================================
# Tools Definition
# ============================================================================

@tool
def search_knowledge_base(query: str) -> str:
    """
    PRIMARY TOOL: Search the internal knowledge base using Azure AI Search.

    This is your PRIMARY source of information. Use this tool FIRST for:
    - Industry analysis, market research, business strategies, and competitive insights
    - Company information, policies, products, services, or procedures
    - Historical data, case studies, research reports, or archived content
    - Domain-specific knowledge, specialized topics, or expert analysis
    - Any question that could be answered by existing documents or reports
    - DEFAULT choice when unsure - always check internal knowledge first

    Args:
        query: The search query to find relevant information in the knowledge base

    Returns:
        Relevant information from the knowledge base
    """
    try:
        search_client = SearchClient(
            endpoint=os.getenv("AZURE_SEARCH_ENDPOINT"),
            index_name=os.getenv("AZURE_SEARCH_INDEX_NAME"),
            credential=AzureKeyCredential(os.getenv("AZURE_SEARCH_API_KEY"))
        )
        
        # Perform hybrid search (semantic + keyword)
        results = search_client.search(
            search_text=query,
            query_type="semantic",
            semantic_configuration_name=os.getenv("AZURE_SEARCH_SEMANTIC_CONFIG", "alpha-sense-semantic-config"),
            top=5,
            select=["content", "title"]  # Adjust based on your index schema
        )
        
        formatted_results = []
        for idx, result in enumerate(results, 1):
            title = result.get("title", "Untitled")
            content = result.get("content", "")[:1000]  # Truncate long content
            formatted_results.append(
                f"[{idx}] **{title}**\n{content}\n"
            )
        
        if not formatted_results:
            return "No relevant results found in the knowledge base."
        
        return "\n---\n".join(formatted_results)
        
    except Exception as e:
        return f"Error searching knowledge base: {str(e)}"


@tool
def search_web(query: str) -> str:
    """
    SECONDARY TOOL: Search the web for real-time information using Gemini's grounded search.

    Use this tool ONLY when:
    - Query explicitly asks for "current", "latest", "today", "recent", or "breaking" information
    - Real-time data is required (live stock prices, current weather, today's news)
    - User specifically requests web search or external sources
    - Knowledge base returned no results AND external information is needed

    DO NOT use for general queries, industry analysis, or business questions that could be in the knowledge base.

    Args:
        query: The search query to find information on the web

    Returns:
        Relevant information from web search results
    """
    try:
        client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))
        
        # Use Gemini with Google Search grounding
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=query,
            config={
                "tools": [GeminiTool(google_search=GoogleSearch())],
                "temperature": 1
            }
        )
        
        # Extract grounded response
        result_text = response.text
        
        # Include grounding metadata if available
        grounding_info = ""
        if hasattr(response, 'candidates') and response.candidates:
            candidate = response.candidates[0]
            if hasattr(candidate, 'grounding_metadata') and candidate.grounding_metadata:
                metadata = candidate.grounding_metadata
                if hasattr(metadata, 'search_entry_point'):
                    grounding_info = "\n\n[Grounded search performed]"
        
        return result_text + grounding_info
        
    except Exception as e:
        return f"Error performing web search: {str(e)}"


# ============================================================================
# Agent State
# ============================================================================

class AgentState(MessagesState):
    """Extended state for tracking tool usage and routing decisions."""
    tool_calls_count: int = 0
    last_tool_used: str = ""


# ============================================================================
# Agent Configuration
# ============================================================================

SYSTEM_PROMPT = """You are an intelligent research assistant with access to two powerful tools:

1. **search_knowledge_base**: Searches an internal Azure AI Search index containing organization-specific documents, policies, and historical data.

2. **search_web**: Performs real-time web searches using Gemini's grounded search for current events, general knowledge, and external information.

## Tool Selection Strategy - IMPORTANT

**DEFAULT BEHAVIOR: Always start with search_knowledge_base FIRST** unless the query explicitly requires real-time or external information.

### ALWAYS use search_knowledge_base FIRST for:
- Any query that could potentially be answered by internal documents
- Industry analysis, market research, or business insights
- Product information, company policies, or organizational data
- Historical context, case studies, or archived content
- Domain-specific knowledge or specialized topics
- Questions about companies, industries, or business strategies
- **DEFAULT for most queries** - when in doubt, check the knowledge base first

### ONLY use search_web when:
- The query explicitly asks for "current", "latest", "today", "recent news", or real-time information
- User specifically requests web search or external sources
- The query is clearly about breaking news or live events (e.g., "today's stock price", "current weather")
- The knowledge base search returned no useful results AND the query requires external information

### Use BOTH tools when:
- The knowledge base provides partial information but needs current/external context
- User explicitly asks to compare internal data with external benchmarks
- Query requires both historical (knowledge base) and current (web) information

## Execution Flow

1. **First, determine the query intent**: Does this ask for real-time/current information, or could it be answered by existing documents?
2. **If unsure, default to search_knowledge_base** - it's better to check internal sources first
3. **Only call search_web** if the query explicitly needs current/external data OR if knowledge base returns insufficient results
4. **Always cite your sources** - indicate whether information came from the knowledge base or web search

## Response Guidelines

1. **Prioritize internal knowledge base** - it contains curated, organization-specific insights
2. **Be transparent** about which source you used and why
3. **Cite sources clearly** - indicate knowledge base vs web search results
4. **If knowledge base has no results**, mention this before using web search

Think step-by-step about which tool(s) to use before responding."""


def create_agent(model_name: str = "gpt-4o", langfuse_handler: Optional[CallbackHandler] = None):
    """
    Create the ReAct agent with dual tool routing.

    Args:
        model_name: The Azure OpenAI model deployment name
        langfuse_handler: Optional Langfuse callback handler for observability

    Returns:
        Compiled LangGraph agent
    """
    # Initialize the LLM (Azure OpenAI)
    llm = AzureChatOpenAI(
        azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT", model_name),
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-08-01-preview"),
        temperature=1
    )

    # Bind tools to the LLM
    tools = [search_knowledge_base, search_web]
    llm_with_tools = llm.bind_tools(tools)
    
    # Create tool node
    tool_node = ToolNode(tools)
    
    # ========================================================================
    # Graph Nodes
    # ========================================================================
    
    def agent_node(state: AgentState) -> AgentState:
        """Main agent reasoning node."""
        messages = state["messages"]

        # Add system prompt if not present
        if not any(isinstance(m, SystemMessage) for m in messages):
            messages = [SystemMessage(content=SYSTEM_PROMPT)] + messages

        # Invoke with Langfuse callback if available
        if langfuse_handler:
            response = llm_with_tools.invoke(messages, config={"callbacks": [langfuse_handler]})
        else:
            response = llm_with_tools.invoke(messages)

        return {
            "messages": [response],
            "tool_calls_count": state.get("tool_calls_count", 0)
        }
    
    def should_continue(state: AgentState) -> Literal["tools", "end"]:
        """Determine if we should continue to tools or end."""
        messages = state["messages"]
        last_message = messages[-1]
        
        # Check if the agent wants to use tools
        if isinstance(last_message, AIMessage) and last_message.tool_calls:
            return "tools"
        
        return "end"
    
    def tool_execution_node(state: AgentState) -> AgentState:
        """Execute tools and track usage."""
        messages = state["messages"]
        last_message = messages[-1]
        
        # Track which tools are being called
        tool_names = [tc["name"] for tc in last_message.tool_calls] if last_message.tool_calls else []
        
        # Execute tools via ToolNode
        tool_results = tool_node.invoke(state)
        
        return {
            "messages": tool_results["messages"],
            "tool_calls_count": state.get("tool_calls_count", 0) + len(tool_names),
            "last_tool_used": ", ".join(tool_names)
        }
    
    # ========================================================================
    # Build the Graph
    # ========================================================================
    
    workflow = StateGraph(AgentState)
    
    # Add nodes
    workflow.add_node("agent", agent_node)
    workflow.add_node("tools", tool_execution_node)
    
    # Add edges
    workflow.add_edge(START, "agent")
    workflow.add_conditional_edges(
        "agent",
        should_continue,
        {
            "tools": "tools",
            "end": END
        }
    )
    workflow.add_edge("tools", "agent")
    
    # Compile with memory for conversation persistence
    memory = MemorySaver()
    agent = workflow.compile(checkpointer=memory)
    
    return agent


# ============================================================================
# Alternative: Using Anthropic Claude as the main LLM
# ============================================================================

def create_agent_with_claude():
    """
    Create the ReAct agent using Anthropic Claude as the reasoning LLM.
    """
    from langchain_anthropic import ChatAnthropic
    
    llm = ChatAnthropic(
        model="claude-sonnet-4-20250514",
        api_key=os.getenv("ANTHROPIC_API_KEY"),
        temperature=0.1
    )
    
    tools = [search_knowledge_base, search_web]
    llm_with_tools = llm.bind_tools(tools)
    tool_node = ToolNode(tools)
    
    def agent_node(state: AgentState) -> AgentState:
        messages = state["messages"]
        if not any(isinstance(m, SystemMessage) for m in messages):
            messages = [SystemMessage(content=SYSTEM_PROMPT)] + messages
        response = llm_with_tools.invoke(messages)
        return {"messages": [response], "tool_calls_count": state.get("tool_calls_count", 0)}
    
    def should_continue(state: AgentState) -> Literal["tools", "end"]:
        last_message = state["messages"][-1]
        if isinstance(last_message, AIMessage) and last_message.tool_calls:
            return "tools"
        return "end"
    
    def tool_execution_node(state: AgentState) -> AgentState:
        tool_results = tool_node.invoke(state)
        return {
            "messages": tool_results["messages"],
            "tool_calls_count": state.get("tool_calls_count", 0) + 1
        }
    
    workflow = StateGraph(AgentState)
    workflow.add_node("agent", agent_node)
    workflow.add_node("tools", tool_execution_node)
    workflow.add_edge(START, "agent")
    workflow.add_conditional_edges("agent", should_continue, {"tools": "tools", "end": END})
    workflow.add_edge("tools", "agent")
    
    return workflow.compile(checkpointer=MemorySaver())


# ============================================================================
# Main execution
# ============================================================================

def run_agent(
    query: str,
    thread_id: str = "default",
    user_id: Optional[str] = None,
    enable_langfuse: bool = True
):
    """
    Run a query through the agent.

    Args:
        query: User's question
        thread_id: Conversation thread ID for memory persistence
        user_id: Optional user ID for Langfuse tracking
        enable_langfuse: Whether to enable Langfuse tracing (default: True)

    Returns:
        Agent's response
    """
    # Initialize Langfuse handler if enabled
    langfuse_handler = None
    if enable_langfuse:
        langfuse_handler = get_langfuse_handler()

    agent = create_agent(langfuse_handler=langfuse_handler)

    config = {"configurable": {"thread_id": thread_id}}

    # Add callbacks and metadata to config if Langfuse is enabled
    if langfuse_handler:
        config["callbacks"] = [langfuse_handler]
        config["metadata"] = {
            "session_id": thread_id,
            "user_id": user_id,
            "trace_name": "react_agent_query"
        }

    result = agent.invoke(
        {"messages": [HumanMessage(content=query)]},
        config=config
    )

    # Flush Langfuse to ensure data is sent
    if langfuse_handler:
        langfuse_handler.client.flush()

    # Extract final response
    final_message = result["messages"][-1]
    return {
        "response": final_message.content,
        "tool_calls_count": result.get("tool_calls_count", 0),
        "messages": result["messages"]
    }


async def run_agent_stream(
    query: str,
    thread_id: str = "default",
    user_id: Optional[str] = None,
    enable_langfuse: bool = True
):
    """
    Run a query through the agent with streaming output.

    Args:
        query: User's question
        thread_id: Conversation thread ID for memory persistence
        user_id: Optional user ID for Langfuse tracking
        enable_langfuse: Whether to enable Langfuse tracing (default: True)

    Yields:
        Streaming chunks of the response
    """
    # Initialize Langfuse handler if enabled
    langfuse_handler = None
    if enable_langfuse:
        langfuse_handler = get_langfuse_handler()

    agent = create_agent(langfuse_handler=langfuse_handler)
    config = {"configurable": {"thread_id": thread_id}}

    # Add callbacks and metadata to config if Langfuse is enabled
    if langfuse_handler:
        config["callbacks"] = [langfuse_handler]
        config["metadata"] = {
            "session_id": thread_id,
            "user_id": user_id,
            "trace_name": "react_agent_stream"
        }

    async for event in agent.astream_events(
        {"messages": [HumanMessage(content=query)]},
        config=config,
        version="v2"
    ):
        kind = event["event"]

        if kind == "on_chat_model_stream":
            content = event["data"]["chunk"].content
            if content:
                yield {"type": "token", "content": content}

        elif kind == "on_tool_start":
            yield {
                "type": "tool_start",
                "tool": event["name"],
                "input": event["data"].get("input", {})
            }

        elif kind == "on_tool_end":
            yield {
                "type": "tool_end",
                "tool": event["name"],
                "output": event["data"].get("output", "")[:500]  # Truncate for display
            }

    # Flush Langfuse to ensure data is sent
    if langfuse_handler:
        langfuse_handler.client.flush()


if __name__ == "__main__":
    # Example usage
    test_queries = [
        # Knowledge base only
        # "Explain about the strategic shifts in footwear and apparel companies?",
        
        # # Web search only
        # "What are the latest developments in AI regulation in 2024?",
        #
        # # Combination query
        "Explain about the strategic shifts in footwear and apparel companies and what are the latest developments in AI regulation in 2024?"
    ]
    
    for query in test_queries:
        print(f"\n{'='*60}")
        print(f"Query: {query}")
        print('='*60)
        
        result = run_agent(query)
        print(f"\nResponse:\n{result['response']}")
        print(f"\nTool calls made: {result['tool_calls_count']}")
