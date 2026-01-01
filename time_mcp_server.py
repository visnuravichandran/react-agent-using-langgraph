"""
Simple Time & Date MCP Server (Python)

A demonstration MCP server that provides time and date information.
No external API keys required - uses Python's built-in datetime module.
"""

import asyncio
from datetime import datetime, timezone
import pytz
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent


# Create MCP server instance
app = Server("time-server")


@app.list_tools()
async def list_tools() -> list[Tool]:
    """List available time/date tools."""
    return [
        Tool(
            name="get_current_time",
            description="Get the current time in a specific timezone. Defaults to UTC if no timezone is specified.",
            inputSchema={
                "type": "object",
                "properties": {
                    "timezone": {
                        "type": "string",
                        "description": "The timezone name (e.g., 'America/New_York', 'Europe/London', 'Asia/Tokyo'). Defaults to 'UTC'.",
                        "default": "UTC"
                    }
                }
            }
        ),
        Tool(
            name="get_current_date",
            description="Get the current date in YYYY-MM-DD format for a specific timezone.",
            inputSchema={
                "type": "object",
                "properties": {
                    "timezone": {
                        "type": "string",
                        "description": "The timezone name (e.g., 'America/New_York'). Defaults to 'UTC'.",
                        "default": "UTC"
                    }
                }
            }
        ),
        Tool(
            name="convert_timezone",
            description="Convert a time from one timezone to another.",
            inputSchema={
                "type": "object",
                "properties": {
                    "time_str": {
                        "type": "string",
                        "description": "The time string in HH:MM format (24-hour)"
                    },
                    "from_timezone": {
                        "type": "string",
                        "description": "Source timezone (e.g., 'America/New_York')"
                    },
                    "to_timezone": {
                        "type": "string",
                        "description": "Target timezone (e.g., 'Asia/Tokyo')"
                    }
                },
                "required": ["time_str", "from_timezone", "to_timezone"]
            }
        ),
    ]


@app.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    """Execute time/date tool calls."""

    if name == "get_current_time":
        timezone_name = arguments.get("timezone", "UTC")
        try:
            tz = pytz.timezone(timezone_name)
            current_time = datetime.now(tz)
            formatted_time = current_time.strftime("%H:%M:%S %Z")
            return [TextContent(
                type="text",
                text=f"Current time in {timezone_name}: {formatted_time}"
            )]
        except Exception as e:
            return [TextContent(
                type="text",
                text=f"Error: Invalid timezone '{timezone_name}'. {str(e)}"
            )]

    elif name == "get_current_date":
        timezone_name = arguments.get("timezone", "UTC")
        try:
            tz = pytz.timezone(timezone_name)
            current_date = datetime.now(tz)
            formatted_date = current_date.strftime("%Y-%m-%d")
            day_name = current_date.strftime("%A")
            return [TextContent(
                type="text",
                text=f"Current date in {timezone_name}: {formatted_date} ({day_name})"
            )]
        except Exception as e:
            return [TextContent(
                type="text",
                text=f"Error: Invalid timezone '{timezone_name}'. {str(e)}"
            )]

    elif name == "convert_timezone":
        time_str = arguments["time_str"]
        from_tz_name = arguments["from_timezone"]
        to_tz_name = arguments["to_timezone"]

        try:
            # Parse time
            today = datetime.now().date()
            time_obj = datetime.strptime(time_str, "%H:%M").time()
            dt = datetime.combine(today, time_obj)

            # Convert timezones
            from_tz = pytz.timezone(from_tz_name)
            to_tz = pytz.timezone(to_tz_name)

            # Localize and convert
            dt_from = from_tz.localize(dt)
            dt_to = dt_from.astimezone(to_tz)

            result_time = dt_to.strftime("%H:%M:%S")

            return [TextContent(
                type="text",
                text=f"{time_str} {from_tz_name} is {result_time} in {to_tz_name}"
            )]
        except Exception as e:
            return [TextContent(
                type="text",
                text=f"Error converting timezone: {str(e)}"
            )]

    else:
        raise ValueError(f"Unknown tool: {name}")


async def main():
    """Run the MCP server using stdio transport."""
    async with stdio_server() as (read_stream, write_stream):
        await app.run(
            read_stream,
            write_stream,
            app.create_initialization_options()
        )


if __name__ == "__main__":
    asyncio.run(main())
