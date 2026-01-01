"""
Enhanced ReAct Agent with Query Analysis for Intelligent Tool Routing.

This version adds a query analyzer that pre-classifies queries to help
the agent make better tool selection decisions.
"""

import os
from typing import Annotated, Literal, TypedDict, Optional
from enum import Enum
from dotenv import load_dotenv

from langchain_core.messages import (
    AIMessage, 
    HumanMessage, 
    SystemMessage, 
    ToolMessage,
    BaseMessage
)
from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import AzureChatOpenAI
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver
from pydantic import BaseModel, Field

from azure.search.documents import SearchClient
from azure.core.credentials import AzureKeyCredential
from google import genai
from google.genai.types import Tool as GeminiTool, GoogleSearch

load_dotenv()


# ============================================================================
# Query Classification
# ============================================================================

class QueryType(str, Enum):
    """Types of queries for routing."""
    KNOWLEDGE_BASE = "knowledge_base"
    WEB_SEARCH = "web_search"
    COMBINED = "combined"
    CONVERSATIONAL = "conversational"


class QueryAnalysis(BaseModel):
    """Structured output for query analysis."""
    query_type: QueryType = Field(
        description="The type of query determining which tools to use"
    )
    reasoning: str = Field(
        description="Brief explanation of why this classification was chosen"
    )
    suggested_tools: list[str] = Field(
        default_factory=list,
        description="List of recommended tools to use"
    )
    reformulated_query: Optional[str] = Field(
        None,
        description="Optimized query for tool execution if needed"
    )


QUERY_ANALYZER_PROMPT = """Analyze the user's query to determine the best tool routing strategy.

Available tools:
1. search_knowledge_base - Internal Azure AI Search index with company documents, policies, procedures
2. search_web - Real-time web search via Gemini for current events, general knowledge, external info

Classification criteria:

**KNOWLEDGE_BASE** - Use when query:
- References internal documents, policies, or procedures
- Asks about company-specific information
- Uses internal terminology or project names
- Asks about historical organizational data

**WEB_SEARCH** - Use when query:
- Asks about current events or news
- Requires real-time or up-to-date external information
- Is general knowledge not organization-specific
- Asks about external products, services, or public figures

**COMBINED** - Use when query:
- Needs comparison between internal and external data
- Requires both historical context and current information
- Explicitly asks for comprehensive research
- Involves benchmarking or competitive analysis

**CONVERSATIONAL** - Use when query:
- Is a greeting or casual conversation
- Asks about the agent's capabilities
- Is a follow-up clarification
- Doesn't require any tool usage

Analyze the following query and provide structured output."""


def create_query_analyzer(llm):
    """Create a query analyzer with structured output."""
    analyzer_llm = llm.with_structured_output(QueryAnalysis)
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", QUERY_ANALYZER_PROMPT),
        ("human", "{query}")
    ])
    
    return prompt | analyzer_llm


# ============================================================================
# Tools (Same as before but with enhanced error handling)
# ============================================================================

@tool
def search_knowledge_base(query: str) -> str:
    """
    Search the internal knowledge base using Azure AI Search.
    Use for company-specific information, policies, documentation, and internal data.
    
    Args:
        query: The search query for the knowledge base
    
    Returns:
        Relevant information from the knowledge base
    """
    try:
        endpoint = os.getenv("AZURE_SEARCH_ENDPOINT")
        index_name = os.getenv("AZURE_SEARCH_INDEX_NAME")
        api_key = os.getenv("AZURE_SEARCH_API_KEY")
        
        if not all([endpoint, index_name, api_key]):
            return "Error: Azure Search configuration incomplete. Please check environment variables."
        
        search_client = SearchClient(
            endpoint=endpoint,
            index_name=index_name,
            credential=AzureKeyCredential(api_key)
        )
        
        # Try semantic search first, fall back to simple search
        try:
            results = search_client.search(
                search_text=query,
                query_type="semantic",
                semantic_configuration_name=os.getenv("AZURE_SEARCH_SEMANTIC_CONFIG", "default"),
                top=5,
                select=["content", "title", "source", "chunk_id"]
            )
        except Exception:
            # Fallback to simple search if semantic isn't configured
            results = search_client.search(
                search_text=query,
                top=5
            )
        
        formatted_results = []
        for idx, result in enumerate(results, 1):
            title = result.get("title", "Untitled")
            content = result.get("content", str(result))[:1500]
            source = result.get("source", "Internal Knowledge Base")
            score = result.get("@search.score", "N/A")
            
            formatted_results.append(
                f"**Result {idx}** (Score: {score})\n"
                f"Title: {title}\n"
                f"Source: {source}\n"
                f"Content: {content}\n"
            )
        
        if not formatted_results:
            return "No relevant results found in the knowledge base for this query."
        
        return "\n---\n".join(formatted_results)
        
    except Exception as e:
        return f"Error searching knowledge base: {type(e).__name__}: {str(e)}"


@tool
def search_web(query: str) -> str:
    """
    Search the web for current information using Gemini's grounded search.
    Use for current events, general knowledge, and external information.
    
    Args:
        query: The search query for web search
    
    Returns:
        Relevant information from web search
    """
    try:
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            return "Error: Google API key not configured."
        
        client = genai.Client(api_key=api_key)
        
        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=f"Search the web and provide accurate, up-to-date information about: {query}",
            config={
                "tools": [GeminiTool(google_search=GoogleSearch())],
                "temperature": 0.1
            }
        )
        
        result_text = response.text
        
        # Extract citation information if available
        citations = []
        if hasattr(response, 'candidates') and response.candidates:
            candidate = response.candidates[0]
            if hasattr(candidate, 'grounding_metadata'):
                metadata = candidate.grounding_metadata
                if hasattr(metadata, 'grounding_chunks'):
                    for chunk in metadata.grounding_chunks[:3]:
                        if hasattr(chunk, 'web') and chunk.web:
                            citations.append(f"- {chunk.web.title}: {chunk.web.uri}")
        
        if citations:
            result_text += "\n\n**Sources:**\n" + "\n".join(citations)
        
        return result_text
        
    except Exception as e:
        return f"Error performing web search: {type(e).__name__}: {str(e)}"


# ============================================================================
# Enhanced Agent State
# ============================================================================

class EnhancedAgentState(MessagesState):
    """Extended state with query analysis."""
    query_analysis: Optional[dict] = None
    tool_calls_count: int = 0
    tools_used: list[str] = []


# ============================================================================
# Agent with Query Pre-Analysis
# ============================================================================

AGENT_SYSTEM_PROMPT = """You are an intelligent research assistant with access to two specialized tools:

1. **search_knowledge_base**: Azure AI Search index with internal company documents
2. **search_web**: Real-time web search via Gemini for current/external information

{analysis_context}

## Response Guidelines

1. **Use the appropriate tool(s)** based on the query analysis provided
2. **Cite your sources** clearly - indicate KB vs Web results
3. **Synthesize information** when using multiple tools
4. **Be concise but thorough** - provide actionable answers
5. **Acknowledge limitations** if information is incomplete

Respond helpfully to the user's query."""


def create_enhanced_agent(model_name: str = "gpt-4o"):
    """
    Create an enhanced ReAct agent with query pre-analysis.
    """
    # Initialize LLM
    llm = AzureChatOpenAI(
        azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT", model_name),
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-08-01-preview"),
        temperature=0.1
    )
    
    # Create query analyzer
    query_analyzer = create_query_analyzer(llm)
    
    # Tools setup
    tools = [search_knowledge_base, search_web]
    llm_with_tools = llm.bind_tools(tools)
    tool_node = ToolNode(tools)
    
    # ========================================================================
    # Graph Nodes
    # ========================================================================
    
    def analyze_query_node(state: EnhancedAgentState) -> EnhancedAgentState:
        """Analyze the incoming query to determine routing strategy."""
        messages = state["messages"]
        
        # Get the last human message
        last_human_msg = None
        for msg in reversed(messages):
            if isinstance(msg, HumanMessage):
                last_human_msg = msg.content
                break
        
        if not last_human_msg:
            return state
        
        try:
            analysis: QueryAnalysis = query_analyzer.invoke({"query": last_human_msg})
            return {
                "query_analysis": {
                    "query_type": analysis.query_type.value,
                    "reasoning": analysis.reasoning,
                    "suggested_tools": analysis.suggested_tools,
                    "reformulated_query": analysis.reformulated_query
                }
            }
        except Exception as e:
            # If analysis fails, proceed without it
            return {
                "query_analysis": {
                    "query_type": "combined",
                    "reasoning": f"Analysis failed: {str(e)}, defaulting to combined approach",
                    "suggested_tools": ["search_knowledge_base", "search_web"]
                }
            }
    
    def agent_node(state: EnhancedAgentState) -> EnhancedAgentState:
        """Main agent reasoning node."""
        messages = state["messages"]
        analysis = state.get("query_analysis", {})
        
        # Build context from analysis
        analysis_context = ""
        if analysis:
            query_type = analysis.get("query_type", "unknown")
            reasoning = analysis.get("reasoning", "")
            suggested = analysis.get("suggested_tools", [])
            
            analysis_context = f"""
## Query Analysis Results
- **Query Type**: {query_type}
- **Reasoning**: {reasoning}
- **Suggested Tools**: {', '.join(suggested) if suggested else 'None specified'}

Based on this analysis, use the appropriate tool(s) to answer the query."""
        
        system_prompt = AGENT_SYSTEM_PROMPT.format(analysis_context=analysis_context)
        
        # Ensure system message is present
        if not any(isinstance(m, SystemMessage) for m in messages):
            messages = [SystemMessage(content=system_prompt)] + list(messages)
        
        response = llm_with_tools.invoke(messages)
        
        return {"messages": [response]}
    
    def should_continue(state: EnhancedAgentState) -> Literal["tools", "end"]:
        """Determine if we should continue to tools or end."""
        messages = state["messages"]
        last_message = messages[-1]
        
        if isinstance(last_message, AIMessage) and last_message.tool_calls:
            return "tools"
        
        return "end"
    
    def tool_execution_node(state: EnhancedAgentState) -> EnhancedAgentState:
        """Execute tools and track usage."""
        messages = state["messages"]
        last_message = messages[-1]
        
        tool_names = []
        if isinstance(last_message, AIMessage) and last_message.tool_calls:
            tool_names = [tc["name"] for tc in last_message.tool_calls]
        
        # Execute tools
        tool_results = tool_node.invoke(state)
        
        # Update tracking
        current_tools = state.get("tools_used", [])
        current_tools.extend(tool_names)
        
        return {
            "messages": tool_results["messages"],
            "tool_calls_count": state.get("tool_calls_count", 0) + len(tool_names),
            "tools_used": current_tools
        }
    
    # ========================================================================
    # Build Graph
    # ========================================================================
    
    workflow = StateGraph(EnhancedAgentState)
    
    # Add nodes
    workflow.add_node("analyze", analyze_query_node)
    workflow.add_node("agent", agent_node)
    workflow.add_node("tools", tool_execution_node)
    
    # Add edges
    workflow.add_edge(START, "analyze")
    workflow.add_edge("analyze", "agent")
    workflow.add_conditional_edges(
        "agent",
        should_continue,
        {"tools": "tools", "end": END}
    )
    workflow.add_edge("tools", "agent")
    
    # Compile
    memory = MemorySaver()
    return workflow.compile(checkpointer=memory)


# ============================================================================
# Convenience function for running queries
# ============================================================================

def query_agent(
    query: str, 
    thread_id: str = "default",
    use_enhanced: bool = True
) -> dict:
    """
    Run a query through the agent.
    
    Args:
        query: User's question
        thread_id: Conversation thread ID
        use_enhanced: Whether to use the enhanced agent with query analysis
    
    Returns:
        Dict with response, analysis, and metadata
    """
    if use_enhanced:
        agent = create_enhanced_agent()
    else:
        from agent import create_agent
        agent = create_agent()
    
    config = {"configurable": {"thread_id": thread_id}}
    
    result = agent.invoke(
        {"messages": [HumanMessage(content=query)]},
        config=config
    )
    
    final_message = result["messages"][-1]
    
    return {
        "response": final_message.content,
        "query_analysis": result.get("query_analysis"),
        "tool_calls_count": result.get("tool_calls_count", 0),
        "tools_used": result.get("tools_used", []),
        "thread_id": thread_id
    }


# ============================================================================
# Example usage
# ============================================================================

if __name__ == "__main__":
    test_queries = [
        # Should route to knowledge base
        "What are our company's guidelines for code reviews?",
        
        # Should route to web search
        "What are the latest LangGraph updates in 2024?",
        
        # Should use both
        "How does our ML pipeline compare to industry best practices for MLOps?",
        
        # Conversational - no tools needed
        "Hello, what can you help me with?"
    ]
    
    for query in test_queries:
        print(f"\n{'='*70}")
        print(f"Query: {query}")
        print('='*70)
        
        result = query_agent(query)
        
        if result.get("query_analysis"):
            print(f"\n📊 Analysis: {result['query_analysis']}")
        
        print(f"\n🔧 Tools used: {result['tools_used']}")
        print(f"\n💬 Response:\n{result['response'][:500]}...")
