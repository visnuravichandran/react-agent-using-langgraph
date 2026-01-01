"""
Trace Extraction for Agent Evaluation

This module extracts structured execution traces from LangGraph agent runs for evaluation.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from datetime import datetime

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage


# ============================================================================
# Data Structures
# ============================================================================

@dataclass
class ToolCall:
    """Represents a single tool call made by the agent."""

    name: str
    input: Dict[str, Any]
    output: str
    timestamp: Optional[datetime] = None
    id: Optional[str] = None

    def __str__(self) -> str:
        return f"ToolCall(name={self.name}, input={self.input})"


@dataclass
class ReasoningStep:
    """Represents a reasoning step by the agent."""

    content: str
    timestamp: Optional[datetime] = None
    has_tool_calls: bool = False
    tool_calls: List[ToolCall] = field(default_factory=list)

    def __str__(self) -> str:
        if self.has_tool_calls:
            return f"Reasoning with {len(self.tool_calls)} tool call(s): {self.content[:50]}..."
        return f"Reasoning: {self.content[:50]}..."


@dataclass
class ExecutionTrace:
    """Complete execution trace for an agent run."""

    query: str
    final_answer: str
    tool_calls: List[ToolCall] = field(default_factory=list)
    reasoning_steps: List[ReasoningStep] = field(default_factory=list)
    total_steps: int = 0
    execution_time: float = 0.0
    tool_calls_count: int = 0
    all_messages: List[Any] = field(default_factory=list)

    def get_tool_names(self) -> List[str]:
        """Get list of tool names called in order."""
        return [tc.name for tc in self.tool_calls]

    def get_unique_tools(self) -> List[str]:
        """Get list of unique tools called."""
        return list(dict.fromkeys(self.get_tool_names()))

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "query": self.query,
            "final_answer": self.final_answer,
            "tool_calls": [
                {
                    "name": tc.name,
                    "input": tc.input,
                    "output": tc.output[:200] + "..." if len(tc.output) > 200 else tc.output,
                }
                for tc in self.tool_calls
            ],
            "reasoning_steps": [
                {
                    "content": rs.content[:200] + "..." if len(rs.content) > 200 else rs.content,
                    "has_tool_calls": rs.has_tool_calls,
                }
                for rs in self.reasoning_steps
            ],
            "total_steps": self.total_steps,
            "execution_time": self.execution_time,
            "tool_calls_count": self.tool_calls_count,
            "tool_names": self.get_tool_names(),
            "unique_tools": self.get_unique_tools(),
        }

    def __str__(self) -> str:
        tools_str = ", ".join(self.get_tool_names()) if self.tool_calls else "none"
        return (
            f"ExecutionTrace(\n"
            f"  query={self.query[:50]}...\n"
            f"  tools={tools_str}\n"
            f"  steps={self.total_steps}\n"
            f"  execution_time={self.execution_time:.2f}s\n"
            f")"
        )


# ============================================================================
# Trace Extraction Functions
# ============================================================================

def extract_tool_calls_from_message(message: AIMessage) -> List[Dict[str, Any]]:
    """
    Extract tool calls from an AIMessage.

    Args:
        message: AIMessage that may contain tool_calls

    Returns:
        List of tool call dictionaries
    """
    if not isinstance(message, AIMessage):
        return []

    if not hasattr(message, "tool_calls") or not message.tool_calls:
        return []

    return message.tool_calls


def extract_tool_result_from_message(message: ToolMessage) -> str:
    """
    Extract tool result from a ToolMessage.

    Args:
        message: ToolMessage containing tool output

    Returns:
        Tool output as string
    """
    if not isinstance(message, ToolMessage):
        return ""

    return message.content


def extract_reasoning_from_message(message: AIMessage) -> str:
    """
    Extract reasoning content from an AIMessage.

    Args:
        message: AIMessage containing reasoning

    Returns:
        Reasoning content as string
    """
    if not isinstance(message, AIMessage):
        return ""

    return message.content if message.content else ""


def extract_execution_trace(
    result: Dict[str, Any],
    start_time: Optional[datetime] = None,
    end_time: Optional[datetime] = None
) -> ExecutionTrace:
    """
    Extract structured execution trace from agent result.

    Args:
        result: Agent invocation result containing messages and state
        start_time: Optional start time of execution
        end_time: Optional end time of execution

    Returns:
        ExecutionTrace: Structured trace of the execution
    """
    messages = result.get("messages", [])
    tool_calls_count = result.get("tool_calls_count", 0)

    # Initialize trace
    trace = ExecutionTrace(
        query="",
        final_answer="",
        all_messages=messages,
        tool_calls_count=tool_calls_count,
    )

    # Calculate execution time
    if start_time and end_time:
        trace.execution_time = (end_time - start_time).total_seconds()

    # Extract query from first HumanMessage
    for msg in messages:
        if isinstance(msg, HumanMessage):
            trace.query = msg.content
            break

    # Process messages chronologically to build trace
    tool_call_map = {}  # Map tool_call_id to ToolCall object

    for i, msg in enumerate(messages):
        # Skip system messages
        if isinstance(msg, SystemMessage):
            continue

        # Process AIMessage
        if isinstance(msg, AIMessage):
            reasoning_content = extract_reasoning_from_message(msg)
            tool_calls_in_msg = extract_tool_calls_from_message(msg)

            # Create reasoning step
            step = ReasoningStep(
                content=reasoning_content,
                has_tool_calls=len(tool_calls_in_msg) > 0,
            )

            # Extract tool calls from this message
            for tc_dict in tool_calls_in_msg:
                tool_call = ToolCall(
                    name=tc_dict.get("name", ""),
                    input=tc_dict.get("args", {}),
                    output="",  # Will be filled when ToolMessage is processed
                    id=tc_dict.get("id", ""),
                )
                tool_call_map[tool_call.id] = tool_call
                step.tool_calls.append(tool_call)
                trace.tool_calls.append(tool_call)

            if reasoning_content or tool_calls_in_msg:
                trace.reasoning_steps.append(step)

            # If this is the last AIMessage and has no tool calls, it's the final answer
            if i == len(messages) - 1 and not tool_calls_in_msg:
                trace.final_answer = reasoning_content

        # Process ToolMessage
        elif isinstance(msg, ToolMessage):
            tool_output = extract_tool_result_from_message(msg)
            tool_call_id = getattr(msg, "tool_call_id", None)

            # Match ToolMessage to its ToolCall by ID
            if tool_call_id and tool_call_id in tool_call_map:
                tool_call_map[tool_call_id].output = tool_output

    # Count total reasoning steps
    trace.total_steps = len(trace.reasoning_steps)

    return trace


def extract_trace_from_state_snapshot(
    state_snapshot: Dict[str, Any],
    start_time: Optional[datetime] = None,
    end_time: Optional[datetime] = None
) -> ExecutionTrace:
    """
    Extract trace from a LangGraph state snapshot.

    This is useful when you have access to the full state history.

    Args:
        state_snapshot: State snapshot from agent
        start_time: Optional start time
        end_time: Optional end time

    Returns:
        ExecutionTrace: Structured trace
    """
    # State snapshots have 'values' key containing the state
    if "values" in state_snapshot:
        return extract_execution_trace(state_snapshot["values"], start_time, end_time)
    else:
        return extract_execution_trace(state_snapshot, start_time, end_time)


# ============================================================================
# Helper Functions for Evaluation
# ============================================================================

def trace_to_deepeval_context(trace: ExecutionTrace) -> List[str]:
    """
    Convert execution trace to DeepEval retrieval_context format.

    DeepEval metrics use retrieval_context to provide additional context
    about the agent's execution for evaluation.

    Args:
        trace: Execution trace

    Returns:
        List of context strings for DeepEval
    """
    context = []

    # Add tool call information
    if trace.tool_calls:
        tools_summary = f"Tools called: {', '.join(trace.get_tool_names())}"
        context.append(tools_summary)

        for tc in trace.tool_calls:
            tool_context = f"Tool: {tc.name}\nInput: {tc.input}\nOutput: {tc.output[:300]}..."
            context.append(tool_context)

    # Add reasoning steps
    for i, step in enumerate(trace.reasoning_steps, 1):
        if step.content:
            reasoning_context = f"Step {i}: {step.content[:300]}..."
            context.append(reasoning_context)

    # Add execution metadata
    metadata = f"Total steps: {trace.total_steps}, Execution time: {trace.execution_time:.2f}s"
    context.append(metadata)

    return context


def trace_to_string(trace: ExecutionTrace) -> str:
    """
    Convert execution trace to human-readable string.

    Args:
        trace: Execution trace

    Returns:
        Formatted string representation
    """
    lines = []
    lines.append("=" * 80)
    lines.append("EXECUTION TRACE")
    lines.append("=" * 80)
    lines.append(f"\nQuery: {trace.query}")
    lines.append(f"\nTools Called: {', '.join(trace.get_tool_names()) if trace.tool_calls else 'none'}")
    lines.append(f"Total Steps: {trace.total_steps}")
    lines.append(f"Execution Time: {trace.execution_time:.2f}s")

    lines.append("\n" + "-" * 80)
    lines.append("REASONING STEPS")
    lines.append("-" * 80)

    for i, step in enumerate(trace.reasoning_steps, 1):
        lines.append(f"\nStep {i}:")
        if step.content:
            lines.append(f"  Reasoning: {step.content[:200]}...")
        if step.has_tool_calls:
            lines.append(f"  Tool Calls: {len(step.tool_calls)}")
            for tc in step.tool_calls:
                lines.append(f"    - {tc.name}({tc.input})")

    lines.append("\n" + "-" * 80)
    lines.append("FINAL ANSWER")
    lines.append("-" * 80)
    lines.append(f"\n{trace.final_answer}")

    lines.append("\n" + "=" * 80)

    return "\n".join(lines)


# ============================================================================
# Validation Functions
# ============================================================================

def validate_trace(trace: ExecutionTrace) -> bool:
    """
    Validate that a trace has minimum required information.

    Args:
        trace: Execution trace to validate

    Returns:
        bool: True if valid, False otherwise
    """
    if not trace.query:
        return False

    if not trace.final_answer:
        return False

    if trace.total_steps == 0:
        return False

    return True


def compare_tool_calls(
    actual_tools: List[str],
    expected_tools: List[str],
    check_order: bool = False
) -> float:
    """
    Compare actual tool calls with expected tool calls.

    Args:
        actual_tools: List of actual tool names called
        expected_tools: List of expected tool names
        check_order: Whether to check the order of tool calls

    Returns:
        float: Similarity score between 0.0 and 1.0
    """
    if not expected_tools and not actual_tools:
        return 1.0

    if not expected_tools or not actual_tools:
        return 0.0

    if check_order:
        # Check exact match including order
        return 1.0 if actual_tools == expected_tools else 0.0
    else:
        # Check set match (ignoring order and duplicates)
        actual_set = set(actual_tools)
        expected_set = set(expected_tools)

        if actual_set == expected_set:
            return 1.0

        # Calculate Jaccard similarity
        intersection = len(actual_set & expected_set)
        union = len(actual_set | expected_set)

        return intersection / union if union > 0 else 0.0