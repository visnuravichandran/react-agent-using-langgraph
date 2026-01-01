"""
MCP (Model Context Protocol) Integration Module

This module demonstrates how to:
1. Connect to an MCP server
2. Retrieve tools from the MCP server
3. Convert MCP tools to LangChain tools
4. Use them in a LangGraph agent
"""

import asyncio
import json
from typing import Any, Dict, List, Optional
from contextlib import AsyncExitStack

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from langchain_core.tools import tool
from pydantic import BaseModel, Field


class MCPClientManager:
    """
    Manager for MCP client connections and tool conversion.

    This class handles:
    - Connecting to MCP servers via stdio
    - Listing available tools from the MCP server
    - Converting MCP tools to LangChain-compatible tools
    """

    def __init__(self):
        self.session: Optional[ClientSession] = None
        self.exit_stack = AsyncExitStack()
        self.mcp_tools: List[Dict[str, Any]] = []

    async def connect_to_server(
        self,
        command: str,
        args: Optional[List[str]] = None,
        env: Optional[Dict[str, str]] = None
    ):
        """
        Connect to an MCP server using stdio transport.

        Args:
            command: The command to run the MCP server (e.g., "python", "node")
            args: Arguments to pass to the command (e.g., ["server.py"])
            env: Environment variables for the server process
        """
        server_params = StdioServerParameters(
            command=command,
            args=args or [],
            env=env
        )

        # Create stdio client connection
        stdio_transport = await self.exit_stack.enter_async_context(
            stdio_client(server_params)
        )
        self.read, self.write = stdio_transport

        # Create and initialize session
        self.session = await self.exit_stack.enter_async_context(
            ClientSession(self.read, self.write)
        )

        await self.session.initialize()

        # List available tools
        response = await self.session.list_tools()
        self.mcp_tools = [
            {
                "name": tool.name,
                "description": tool.description,
                "input_schema": tool.inputSchema
            }
            for tool in response.tools
        ]

        print(f"✓ Connected to MCP server")
        print(f"✓ Found {len(self.mcp_tools)} tools: {[t['name'] for t in self.mcp_tools]}")

        return self.mcp_tools

    async def call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Any:
        """
        Call an MCP tool by name with the given arguments.

        Args:
            tool_name: Name of the MCP tool to call
            arguments: Dictionary of arguments to pass to the tool

        Returns:
            The result from the MCP tool
        """
        if not self.session:
            raise RuntimeError("Not connected to MCP server. Call connect_to_server first.")

        result = await self.session.call_tool(tool_name, arguments=arguments)

        # Extract content from result
        if result.content:
            if len(result.content) == 1:
                return result.content[0].text
            else:
                return "\n".join(item.text for item in result.content)
        return "No result"

    async def disconnect(self):
        """Disconnect from the MCP server."""
        await self.exit_stack.aclose()
        print("✓ Disconnected from MCP server")


def create_langchain_tool_from_mcp(
    mcp_client: MCPClientManager,
    tool_name: str,
    tool_description: str,
    input_schema: Dict[str, Any],
    event_loop: asyncio.AbstractEventLoop
):
    """
    Convert an MCP tool to a LangChain tool.

    Args:
        mcp_client: The MCP client manager instance
        tool_name: Name of the MCP tool
        tool_description: Description of what the tool does
        input_schema: JSON schema for the tool's input parameters
        event_loop: The event loop where MCP client was created

    Returns:
        A LangChain tool that wraps the MCP tool
    """

    # Create a dynamic Pydantic model from the JSON schema
    # This is a simplified version - you can enhance it for complex schemas
    properties = input_schema.get("properties", {})
    required = input_schema.get("required", [])

    # Build field definitions
    fields = {}
    for prop_name, prop_schema in properties.items():
        prop_type = prop_schema.get("type", "string")
        prop_description = prop_schema.get("description", "")

        # Map JSON schema types to Python types
        type_mapping = {
            "string": str,
            "number": float,
            "integer": int,
            "boolean": bool,
        }
        python_type = type_mapping.get(prop_type, str)

        # Determine if optional
        is_required = prop_name in required
        if is_required:
            fields[prop_name] = (python_type, Field(description=prop_description))
        else:
            fields[prop_name] = (Optional[python_type], Field(default=None, description=prop_description))

    # Create dynamic Pydantic model
    InputModel = type(f"{tool_name.title()}Input", (BaseModel,), {
        "__annotations__": {k: v[0] for k, v in fields.items()},
        **{k: v[1] for k, v in fields.items()}
    })

    # Create the tool function
    @tool(tool_name, args_schema=InputModel)
    def mcp_tool(**kwargs) -> str:
        """
        This docstring will be replaced by the actual tool description.
        """
        # Call the MCP tool asynchronously using the original event loop
        # This ensures we use the same loop where the MCP client was created
        future = asyncio.run_coroutine_threadsafe(
            mcp_client.call_tool(tool_name, kwargs),
            event_loop
        )
        # Wait for the result with a timeout
        result = future.result(timeout=30)
        return str(result)

    # Update the docstring
    mcp_tool.__doc__ = tool_description
    mcp_tool.description = tool_description

    return mcp_tool


async def get_mcp_tools_as_langchain(
    command: str,
    args: Optional[List[str]] = None,
    env: Optional[Dict[str, str]] = None
) -> tuple[MCPClientManager, List, asyncio.AbstractEventLoop]:
    """
    Connect to an MCP server and get all its tools as LangChain tools.

    Args:
        command: The command to run the MCP server
        args: Arguments to pass to the command
        env: Environment variables for the server

    Returns:
        Tuple of (MCP client manager, list of LangChain tools, event loop)
    """
    client = MCPClientManager()
    await client.connect_to_server(command, args, env)

    # Get the current event loop (where the MCP client is running)
    event_loop = asyncio.get_running_loop()

    # Convert all MCP tools to LangChain tools
    langchain_tools = []
    for mcp_tool_def in client.mcp_tools:
        lc_tool = create_langchain_tool_from_mcp(
            client,
            mcp_tool_def["name"],
            mcp_tool_def["description"],
            mcp_tool_def["input_schema"],
            event_loop  # Pass the event loop
        )
        langchain_tools.append(lc_tool)

    return client, langchain_tools, event_loop


# ============================================================================
# Example: Simple Calculator MCP Server
# ============================================================================

# You can use this as a reference to create your own MCP server
# or connect to existing MCP servers like:
# - @modelcontextprotocol/server-brave-search
# - @modelcontextprotocol/server-filesystem
# - @modelcontextprotocol/server-github
# - etc.

if __name__ == "__main__":
    # Example usage
    print("MCP Integration Module")
    print("=" * 60)
    print("\nThis module provides utilities to:")
    print("1. Connect to MCP servers")
    print("2. Convert MCP tools to LangChain tools")
    print("3. Use MCP tools in LangGraph agents")
    print("\nSee agent_with_mcp.py for a complete example.")
