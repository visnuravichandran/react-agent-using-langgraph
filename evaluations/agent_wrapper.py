"""
Agent Wrapper for Evaluation

This module wraps the LangGraph ReAct agent to capture detailed execution traces
for evaluation purposes.
"""

import time
import uuid
from datetime import datetime
from typing import Optional, Dict, Any, List

from langchain_core.messages import HumanMessage
from langfuse.langchain import CallbackHandler

# Import from main agent module
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from agent import create_agent, get_langfuse_handler

# Import trace extraction utilities
from evaluations.trace_extractor import (
    ExecutionTrace,
    extract_execution_trace,
)


# ============================================================================
# Agent Wrapper
# ============================================================================

class EvaluationAgentWrapper:
    """
    Wrapper for the LangGraph ReAct agent that captures execution traces.

    This wrapper:
    1. Creates an agent instance using the existing create_agent() function
    2. Captures full execution traces including messages, tool calls, timing
    3. Provides a consistent interface for evaluation
    """

    def __init__(
        self,
        model_name: str = "gpt-4o",
        enable_langfuse: bool = False,
        user_id: Optional[str] = None
    ):
        """
        Initialize the agent wrapper.

        Args:
            model_name: Azure OpenAI model deployment name
            enable_langfuse: Whether to enable Langfuse tracing
            user_id: Optional user ID for Langfuse tracking
        """
        self.model_name = model_name
        self.enable_langfuse = enable_langfuse
        self.user_id = user_id
        self.langfuse_handler = None

        # Initialize Langfuse handler if enabled
        if self.enable_langfuse:
            self.langfuse_handler = get_langfuse_handler(
                user_id=user_id
            )

        # Create the agent
        self.agent = create_agent(
            model_name=model_name,
            langfuse_handler=self.langfuse_handler
        )

    def invoke_with_trace(
        self,
        query: str,
        thread_id: Optional[str] = None,
        timeout: Optional[float] = None
    ) -> tuple[Dict[str, Any], ExecutionTrace]:
        """
        Invoke the agent and capture execution trace.

        Args:
            query: User query to process
            thread_id: Optional thread ID for conversation persistence
            timeout: Optional timeout in seconds (not currently enforced)

        Returns:
            tuple: (agent_result, execution_trace)
                - agent_result: Raw result from agent invocation
                - execution_trace: Structured ExecutionTrace object
        """
        # Generate thread_id if not provided
        if thread_id is None:
            thread_id = str(uuid.uuid4())

        # Prepare input
        input_message = {"messages": [HumanMessage(content=query)]}

        # Prepare config for agent invocation
        config = {
            "configurable": {"thread_id": thread_id},
        }

        # Add Langfuse callback if enabled
        if self.langfuse_handler:
            config["callbacks"] = [self.langfuse_handler]

        # Execute agent and measure time
        start_time = datetime.now()
        start_timestamp = time.time()

        try:
            result = self.agent.invoke(input_message, config=config)
        except Exception as e:
            # If agent fails, return empty result and trace
            end_time = datetime.now()
            execution_time = time.time() - start_timestamp

            error_trace = ExecutionTrace(
                query=query,
                final_answer=f"Error during execution: {str(e)}",
                execution_time=execution_time
            )

            return {}, error_trace

        end_time = datetime.now()

        # Extract execution trace
        trace = extract_execution_trace(result, start_time, end_time)

        # Flush Langfuse if enabled
        if self.langfuse_handler and hasattr(self.langfuse_handler, "client"):
            self.langfuse_handler.client.flush()

        return result, trace

    def invoke_batch(
        self,
        queries: List[str],
        thread_ids: Optional[List[str]] = None
    ) -> List[tuple[Dict[str, Any], ExecutionTrace]]:
        """
        Invoke the agent on multiple queries sequentially.

        Args:
            queries: List of user queries
            thread_ids: Optional list of thread IDs (one per query)

        Returns:
            List of (result, trace) tuples
        """
        if thread_ids is None:
            thread_ids = [str(uuid.uuid4()) for _ in queries]

        if len(thread_ids) != len(queries):
            raise ValueError("Number of thread_ids must match number of queries")

        results = []
        for query, thread_id in zip(queries, thread_ids):
            result, trace = self.invoke_with_trace(query, thread_id)
            results.append((result, trace))

        return results

    def get_conversation_history(self, thread_id: str) -> List[Any]:
        """
        Get conversation history for a specific thread.

        Args:
            thread_id: Thread ID to retrieve history for

        Returns:
            List of messages in the conversation
        """
        config = {"configurable": {"thread_id": thread_id}}

        try:
            # Get state for this thread
            state = self.agent.get_state(config)
            if state and "values" in state:
                return state["values"].get("messages", [])
        except Exception:
            pass

        return []


# ============================================================================
# Helper Functions
# ============================================================================

def create_evaluation_agent(
    model_name: str = "gpt-4o",
    enable_langfuse: bool = False,
    user_id: Optional[str] = None
) -> EvaluationAgentWrapper:
    """
    Create an agent wrapper configured for evaluation.

    Args:
        model_name: Azure OpenAI model deployment name
        enable_langfuse: Whether to enable Langfuse tracing
        user_id: Optional user ID for Langfuse tracking

    Returns:
        EvaluationAgentWrapper: Configured agent wrapper
    """
    return EvaluationAgentWrapper(
        model_name=model_name,
        enable_langfuse=enable_langfuse,
        user_id=user_id
    )


def run_test_query(
    query: str,
    model_name: str = "gpt-4o",
    enable_langfuse: bool = False,
    verbose: bool = True
) -> tuple[Dict[str, Any], ExecutionTrace]:
    """
    Run a single test query and return the result and trace.

    Args:
        query: User query to test
        model_name: Model name to use
        enable_langfuse: Whether to enable Langfuse
        verbose: Whether to print verbose output

    Returns:
        tuple: (result, trace)
    """
    agent = create_evaluation_agent(
        model_name=model_name,
        enable_langfuse=enable_langfuse
    )

    if verbose:
        print(f"\n{'='*80}")
        print(f"Running query: {query}")
        print(f"{'='*80}\n")

    result, trace = agent.invoke_with_trace(query)

    if verbose:
        print(f"\nTools called: {', '.join(trace.get_tool_names()) if trace.tool_calls else 'none'}")
        print(f"Steps: {trace.total_steps}")
        print(f"Execution time: {trace.execution_time:.2f}s")
        print(f"\nFinal answer: {trace.final_answer}\n")

    return result, trace


# ============================================================================
# Testing/Demo
# ============================================================================

def main():
    """Demo usage of the agent wrapper."""
    import argparse

    parser = argparse.ArgumentParser(description="Test the evaluation agent wrapper")
    parser.add_argument("--query", default="What are the strategic shifts in footwear companies?", help="Query to test")
    parser.add_argument("--model", default="gpt-4o", help="Model to use")
    parser.add_argument("--langfuse", action="store_true", help="Enable Langfuse tracing")

    args = parser.parse_args()

    # Run test query
    result, trace = run_test_query(
        query=args.query,
        model_name=args.model,
        enable_langfuse=args.langfuse,
        verbose=True
    )

    # Print detailed trace
    print("\n" + "="*80)
    print("DETAILED EXECUTION TRACE")
    print("="*80)
    print(f"\nQuery: {trace.query}")
    print(f"\nTools: {trace.get_tool_names()}")
    print(f"Total steps: {trace.total_steps}")
    print(f"Execution time: {trace.execution_time:.2f}s")

    print("\nReasoning Steps:")
    for i, step in enumerate(trace.reasoning_steps, 1):
        print(f"\n  Step {i}:")
        if step.content:
            print(f"    Content: {step.content[:150]}...")
        if step.has_tool_calls:
            print(f"    Tool calls: {[tc.name for tc in step.tool_calls]}")

    print(f"\nFinal Answer:\n{trace.final_answer}")
    print("\n" + "="*80)


if __name__ == "__main__":
    main()
