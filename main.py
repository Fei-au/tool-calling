import argparse
import json
import os
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI

from weather_tool import extract_location_and_timeframe, weather_tool

load_dotenv()


def run_direct_openai_tool_call(user_message: str) -> str:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("Please set OPENAI_API_KEY in your environment or .env file")

    client = OpenAI(api_key=api_key)
    tools = [
        {
            "type": "function",
            "function": {
                "name": "lookup_weather",
                "description": "Look up weather for a location and a requested time frame.",
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
        }
    ]

    messages = [
        {"role": "system", "content": "You are a helpful assistant. Use the weather tool when the user asks about weather."},
        {"role": "user", "content": user_message},
    ]
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        tools=tools,
        tool_choice="auto",
    )

    assistant_message = response.choices[0].message
    if not assistant_message.tool_calls:
        return assistant_message.content or "No tool call was needed."

    tool_call = assistant_message.tool_calls[0]
    arguments = json.loads(tool_call.function.arguments)
    tool_result = weather_tool(arguments)

    follow_up_messages = [
        *messages,
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": tool_call.id,
                    "type": tool_call.type,
                    "function": {
                        "name": tool_call.function.name,
                        "arguments": tool_call.function.arguments,
                    },
                }
            ],
        },
        {"role": "tool", "tool_call_id": tool_call.id, "content": json.dumps(tool_result, ensure_ascii=False)},
    ]

    final_response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=follow_up_messages,
    )
    return final_response.choices[0].message.content or "No answer generated."


def run_langgraph(user_message: str) -> str:
    from langgraph.graph import END, StateGraph

    class WeatherState(dict):
        pass

    def parse_request(state: WeatherState) -> dict[str, Any]:
        last_message = state["messages"][-1]["content"]
        location, timeframe = extract_location_and_timeframe(last_message)
        return {"location": location, "timeframe": timeframe}

    def call_weather_tool(state: WeatherState) -> dict[str, Any]:
        tool_result = weather_tool({"location": state["location"], "timeframe": state["timeframe"]})
        return {"tool_result": json.dumps(tool_result, ensure_ascii=False)}

    def respond(state: WeatherState) -> dict[str, Any]:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("Please set OPENAI_API_KEY in your environment or .env file")

        client = OpenAI(api_key=api_key)
        prompt = (
            "You are a helpful assistant. Use the weather tool result to answer the user's question in Chinese. "
            f"User question: {state['messages'][-1]['content']}\n"
            f"Tool result: {state['tool_result']}"
        )
        response = client.chat.completions.create(model="gpt-4o-mini", messages=[{"role": "user", "content": prompt}])
        return {"final_answer": response.choices[0].message.content or "No answer generated."}

    builder = StateGraph(WeatherState)
    builder.add_node("parse_request", parse_request)
    builder.add_node("call_weather_tool", call_weather_tool)
    builder.add_node("respond", respond)
    builder.set_entry_point("parse_request")
    builder.add_edge("parse_request", "call_weather_tool")
    builder.add_edge("call_weather_tool", "respond")
    builder.add_edge("respond", END)

    graph = builder.compile()
    result = graph.invoke({"messages": [{"role": "user", "content": user_message}]})
    return result.get("final_answer", "No answer generated.")


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
