"""
FastAPI server for the ReAct Agent with streaming support.
"""

import asyncio
import json
import uuid
from typing import Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from agent import create_agent, run_agent, AgentState, get_langfuse_handler
from langchain_core.messages import HumanMessage, AIMessage


# ============================================================================
# Pydantic Models
# ============================================================================

class ChatRequest(BaseModel):
    """Request model for chat endpoint."""
    message: str = Field(..., description="User's message")
    thread_id: Optional[str] = Field(None, description="Conversation thread ID")
    user_id: Optional[str] = Field(None, description="User ID for tracking")
    stream: bool = Field(False, description="Enable streaming response")


class ChatResponse(BaseModel):
    """Response model for chat endpoint."""
    response: str
    thread_id: str
    tool_calls_count: int
    tools_used: list[str] = []


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    version: str


# ============================================================================
# Application Setup
# ============================================================================

# Global agent instance
_agent = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler for startup/shutdown."""
    global _agent
    print("🚀 Starting ReAct Agent server...")
    _agent = create_agent()
    print("✅ Agent initialized successfully")
    yield
    print("👋 Shutting down...")


app = FastAPI(
    title="ReAct Agent API",
    description="Intelligent agent with Azure AI Search and Gemini Web Search tools",
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================================
# Endpoints
# ============================================================================

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(status="healthy", version="1.0.0")


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Non-streaming chat endpoint.

    Sends a message to the agent and returns the complete response.
    """
    global _agent

    if _agent is None:
        raise HTTPException(status_code=503, detail="Agent not initialized")

    thread_id = request.thread_id or str(uuid.uuid4())

    # Initialize Langfuse handler
    langfuse_handler = get_langfuse_handler(
        session_id=thread_id,
        user_id=request.user_id,
        trace_name="chat_endpoint"
    )

    config = {"configurable": {"thread_id": thread_id}}

    # Add Langfuse callback if available
    if langfuse_handler:
        config["callbacks"] = [langfuse_handler]

    try:
        result = _agent.invoke(
            {"messages": [HumanMessage(content=request.message)]},
            config=config
        )

        # Extract response and tool usage info
        final_message = result["messages"][-1]
        tools_used = []

        for msg in result["messages"]:
            if isinstance(msg, AIMessage) and msg.tool_calls:
                tools_used.extend([tc["name"] for tc in msg.tool_calls])

        # Flush Langfuse
        if langfuse_handler:
            langfuse_handler.flush()

        return ChatResponse(
            response=final_message.content,
            thread_id=thread_id,
            tool_calls_count=len(tools_used),
            tools_used=list(set(tools_used))
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/chat/stream")
async def chat_stream(request: ChatRequest):
    """
    Streaming chat endpoint.

    Streams the agent's response including tool invocations.
    """
    global _agent

    if _agent is None:
        raise HTTPException(status_code=503, detail="Agent not initialized")

    thread_id = request.thread_id or str(uuid.uuid4())

    # Initialize Langfuse handler
    langfuse_handler = get_langfuse_handler(
        session_id=thread_id,
        user_id=request.user_id,
        trace_name="chat_stream_endpoint"
    )

    config = {"configurable": {"thread_id": thread_id}}

    # Add Langfuse callback if available
    if langfuse_handler:
        config["callbacks"] = [langfuse_handler]

    async def event_generator():
        """Generate SSE events from agent execution."""
        try:
            # Send thread_id first
            yield f"data: {json.dumps({'type': 'thread_id', 'thread_id': thread_id})}\n\n"

            async for event in _agent.astream_events(
                {"messages": [HumanMessage(content=request.message)]},
                config=config,
                version="v2"
            ):
                kind = event["event"]

                if kind == "on_chat_model_stream":
                    content = event["data"]["chunk"].content
                    if content:
                        yield f"data: {json.dumps({'type': 'token', 'content': content})}\n\n"

                elif kind == "on_tool_start":
                    yield f"data: {json.dumps({'type': 'tool_start', 'tool': event['name'], 'input': str(event['data'].get('input', {}))[:200]})}\n\n"

                elif kind == "on_tool_end":
                    output = str(event["data"].get("output", ""))
                    yield f"data: {json.dumps({'type': 'tool_end', 'tool': event['name'], 'output_preview': output[:300]})}\n\n"

            yield f"data: {json.dumps({'type': 'done'})}\n\n"

            # Flush Langfuse after streaming completes
            if langfuse_handler:
                langfuse_handler.flush()

        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        }
    )


@app.get("/threads/{thread_id}/history")
async def get_thread_history(thread_id: str):
    """
    Get conversation history for a thread.
    """
    global _agent
    
    if _agent is None:
        raise HTTPException(status_code=503, detail="Agent not initialized")
    
    config = {"configurable": {"thread_id": thread_id}}
    
    try:
        state = _agent.get_state(config)
        
        if not state.values:
            return {"thread_id": thread_id, "messages": []}
        
        messages = []
        for msg in state.values.get("messages", []):
            messages.append({
                "role": "user" if isinstance(msg, HumanMessage) else "assistant",
                "content": msg.content,
                "tool_calls": msg.tool_calls if hasattr(msg, "tool_calls") and msg.tool_calls else None
            })
        
        return {"thread_id": thread_id, "messages": messages}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/threads/{thread_id}")
async def delete_thread(thread_id: str):
    """
    Delete a conversation thread.
    """
    # With MemorySaver, threads are in-memory and will be cleared on restart
    # For production, you'd use a persistent checkpointer
    return {"message": f"Thread {thread_id} marked for deletion"}


# ============================================================================
# Run server
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
