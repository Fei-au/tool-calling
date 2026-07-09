import argparse
import json
import operator
import os
from typing import Annotated, Any, TypedDict

from dotenv import load_dotenv
from openai import OpenAI

from weather_tool import hourly_forecast_tool, weather_tool

load_dotenv()

MODEL = "gpt-4o-mini"

SYSTEM_PROMPT = (
    "You are a helpful weather assistant. Today's date is 2026-07-08. Answer in Chinese.\n"
    "You have two tools:\n"
    "- lookup_weather(location, timeframe): a DAILY summary (high/low temp, rain chance) "
    "for today / 7days / 14days / 30days.\n"
    "- get_hourly_forecast(location, date): the HOUR-BY-HOUR detail (temp, feels-like, "
    "rain chance, UV) for ONE specific date.\n"
    "When the user wants to pick a good day and a good time for an outdoor activity, first "
    "call lookup_weather to see the whole period, reason about which day is best (e.g. lowest "
    "rain chance, comfortable temperature), then call get_hourly_forecast for THAT day to "
    "recommend the best hours. Call tools as many times as needed before answering."
)

# ---------------------------------------------------------------------------
# Shared tool definitions (OpenAI tool-calling schema + a name->fn registry)
# ---------------------------------------------------------------------------
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "lookup_weather",
            "description": (
                "Daily weather summary (high/low temperature, rain probability) for a "
                "location over a time frame. Use this first to survey a whole period."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {"type": "string", "description": "City or location to query"},
                    "timeframe": {
                        "type": "string",
                        "description": "One of: today, 7days, 14days, 30days",
                    },
                },
                "required": ["location", "timeframe"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_hourly_forecast",
            "description": (
                "Hour-by-hour forecast (temperature, feels-like, rain probability, UV index) "
                "for ONE specific date. Call this after lookup_weather to drill into the day "
                "you picked and recommend the best hours."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {"type": "string", "description": "City or location to query"},
                    "date": {"type": "string", "description": "The target date, format YYYY-MM-DD"},
                },
                "required": ["location", "date"],
            },
        },
    },
]

TOOL_REGISTRY = {
    "lookup_weather": weather_tool,
    "get_hourly_forecast": hourly_forecast_tool,
}


def get_client() -> OpenAI:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("Please set OPENAI_API_KEY in your environment or .env file")
    return OpenAI(api_key=api_key)


def execute_tool_call(tool_call: Any) -> dict[str, Any]:
    """Run one tool call and return the corresponding `tool` role message."""
    name = tool_call.function.name
    fn = TOOL_REGISTRY.get(name)
    try:
        arguments = json.loads(tool_call.function.arguments or "{}")
        if fn is None:
            result: Any = {"error": f"Unknown tool: {name}"}
        else:
            result = fn(arguments)
    except Exception as exc:  # surface errors back to the model instead of crashing
        result = {"error": str(exc)}

    return {
        "role": "tool",
        "tool_call_id": tool_call.id,
        "content": json.dumps(result, ensure_ascii=False),
    }


def assistant_message_to_dict(message: Any) -> dict[str, Any]:
    """Serialize an OpenAI assistant message (with tool_calls) back into a dict."""
    payload: dict[str, Any] = {"role": "assistant", "content": message.content}
    if message.tool_calls:
        payload["tool_calls"] = [
            {
                "id": tc.id,
                "type": tc.type,
                "function": {"name": tc.function.name, "arguments": tc.function.arguments},
            }
            for tc in message.tool_calls
        ]
    return payload


# ---------------------------------------------------------------------------
# Mode 1: plain OpenAI agent loop (no orchestration framework)
# ---------------------------------------------------------------------------
def run_direct_openai_tool_call(user_message: str, max_steps: int = 5) -> str:
    client = get_client()
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_message},
    ]

    # for + max_steps guarantees the loop terminates (no infinite calls).
    for _ in range(max_steps):
        response = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            tools=TOOLS,
            tool_choice="auto",
        )
        assistant_message = response.choices[0].message
        messages.append(assistant_message_to_dict(assistant_message))

        # No tool calls -> the model produced its final answer, stop the loop.
        if not assistant_message.tool_calls:
            return assistant_message.content or "No answer generated."

        # Execute every requested tool call and feed the results back in.
        for tool_call in assistant_message.tool_calls:
            messages.append(execute_tool_call(tool_call))

    return "Reached max steps without a final answer."


# ---------------------------------------------------------------------------
# Mode 2: the same agent loop, but orchestrated by LangGraph
# ---------------------------------------------------------------------------
class AgentState(TypedDict):
    messages: Annotated[list[dict[str, Any]], operator.add]


def run_langgraph(user_message: str, max_steps: int = 5) -> str:
    from langgraph.graph import END, StateGraph

    client = get_client()

    def agent(state: AgentState) -> dict[str, Any]:
        """LLM decides whether to call a tool or answer."""
        response = client.chat.completions.create(
            model=MODEL,
            messages=state["messages"],
            tools=TOOLS,
            tool_choice="auto",
        )
        assistant_message = response.choices[0].message
        return {"messages": [assistant_message_to_dict(assistant_message)]}

    def tools(state: AgentState) -> dict[str, Any]:
        """Execute the tool calls requested in the last assistant message."""
        last = state["messages"][-1]
        tool_messages: list[dict[str, Any]] = []
        for tc in last.get("tool_calls", []):
            name = tc["function"]["name"]
            fn = TOOL_REGISTRY.get(name)
            try:
                arguments = json.loads(tc["function"].get("arguments") or "{}")
                result: Any = fn(arguments) if fn else {"error": f"Unknown tool: {name}"}
            except Exception as exc:
                result = {"error": str(exc)}
            tool_messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tc["id"],
                    "content": json.dumps(result, ensure_ascii=False),
                }
            )
        return {"messages": tool_messages}

    def should_continue(state: AgentState) -> str:
        """Loop back to tools if the model asked for one, otherwise finish."""
        last = state["messages"][-1]
        # Guard against infinite loops: cap the number of tool rounds.
        tool_rounds = sum(1 for m in state["messages"] if m.get("role") == "tool")
        if last.get("tool_calls") and tool_rounds < max_steps:
            return "tools"
        return END

    builder = StateGraph(AgentState)
    builder.add_node("agent", agent)
    builder.add_node("tools", tools)
    builder.set_entry_point("agent")
    builder.add_conditional_edges("agent", should_continue, {"tools": "tools", END: END})
    builder.add_edge("tools", "agent")  # <- the loop: tool result goes back to the LLM

    graph = builder.compile()
    result = graph.invoke(
        {
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ]
        }
    )

    for message in reversed(result["messages"]):
        if message.get("role") == "assistant" and message.get("content"):
            return message["content"]
    return "No answer generated."


def main() -> None:
    parser = argparse.ArgumentParser(description="Weather tool calling demo")
    parser.add_argument("--mode", choices=["direct", "langgraph"], default="direct")
    parser.add_argument("--message", default="今天上海的天气怎么样")
    args = parser.parse_args()

    if args.mode == "direct":
        print(run_direct_openai_tool_call(args.message))
    else:
        print(run_langgraph(args.message))


if __name__ == "__main__":
    main()
