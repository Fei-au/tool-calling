import json
import os
import re
from typing import Any

import requests
from dotenv import load_dotenv

load_dotenv()

CITY_COORDINATES = {
    "北京": {"name": "Beijing", "country": "China", "latitude": 39.9075, "longitude": 116.39723},
    "上海": {"name": "Shanghai", "country": "China", "latitude": 31.22222, "longitude": 121.45806},
    "广州": {"name": "Guangzhou", "country": "China", "latitude": 23.12911, "longitude": 113.26438},
    "深圳": {"name": "Shenzhen", "country": "China", "latitude": 22.5431, "longitude": 114.05787},
    "杭州": {"name": "Hangzhou", "country": "China", "latitude": 30.29365, "longitude": 120.16142},
    "南京": {"name": "Nanjing", "country": "China", "latitude": 32.06167, "longitude": 118.77778},
    "武汉": {"name": "Wuhan", "country": "China", "latitude": 30.58333, "longitude": 114.26667},
    "成都": {"name": "Chengdu", "country": "China", "latitude": 30.65984, "longitude": 104.06573},
    "西安": {"name": "Xi'an", "country": "China", "latitude": 34.25833, "longitude": 108.92861},
    "重庆": {"name": "Chongqing", "country": "China", "latitude": 29.56026, "longitude": 106.55771},
}


def extract_location_and_timeframe(text: str) -> tuple[str, str]:
    lowered = text.lower()
    if any(keyword in lowered for keyword in ["7天", "7 day", "7 days", "future week", "未来一周", "next week"]):
        timeframe = "7days"
    elif any(keyword in lowered for keyword in ["14天", "14 day", "14 days", "future two weeks", "未来两周", "next 2 weeks"]):
        timeframe = "14days"
    elif any(keyword in lowered for keyword in ["一个月", "30天", "30 day", "30 days", "month", "未来一个月", "next month"]):
        timeframe = "30days"
    else:
        timeframe = "today"

    cleaned = re.sub(
        r"(今天|明天|后天|天气|怎么样|如何|请问|帮我|查|下|的|请|for|weather|forecast|what's|what is|the|is)",
        "",
        text,
        flags=re.IGNORECASE,
    )
    cleaned = re.sub(r"[，。！？、；:：\s]+", " ", cleaned).strip()
    location = cleaned or "Beijing"
    return location, timeframe


def geocode_location(location: str) -> dict[str, Any]:
    normalized = (location or "").strip()
    if not normalized:
        raise ValueError("Location is empty")

    if normalized in CITY_COORDINATES:
        return CITY_COORDINATES[normalized]

    response = requests.get(
        "https://geocoding-api.open-meteo.com/v1/search",
        params={"name": normalized, "count": 1, "language": "en", "format": "json"},
        timeout=10,
    )
    response.raise_for_status()
    payload = response.json()
    if payload.get("results"):
        result = payload["results"][0]
        return {
            "name": result.get("name"),
            "country": result.get("country"),
            "latitude": result.get("latitude"),
            "longitude": result.get("longitude"),
        }

    if normalized in {"上海市", "北京市", "广州市", "深圳市", "杭州市", "南京市", "武汉市", "成都市", "西安市", "重庆市"}:
        short_name = normalized[:-1]
        if short_name in CITY_COORDINATES:
            return CITY_COORDINATES[short_name]

    raise ValueError(f"Could not find location: {location}")


def lookup_weather(location: str, timeframe: str = "today") -> dict[str, Any]:
    coordinates = geocode_location(location)
    mapped_days = {"today": 1, "7days": 7, "14days": 14, "30days": 16}
    forecast_days = mapped_days.get(timeframe, 1)

    response = requests.get(
        "https://api.open-meteo.com/v1/forecast",
        params={
            "latitude": coordinates["latitude"],
            "longitude": coordinates["longitude"],
            "daily": "temperature_2m_max,temperature_2m_min,precipitation_probability_max,weathercode",
            "timezone": "auto",
            "forecast_days": forecast_days,
        },
        timeout=15,
    )
    response.raise_for_status()
    payload = response.json()
    daily = payload.get("daily", {})

    summary = []
    for index, day in enumerate(daily.get("time", [])[:forecast_days]):
        summary.append(
            {
                "date": day,
                "high_c": daily.get("temperature_2m_max", [None])[index],
                "low_c": daily.get("temperature_2m_min", [None])[index],
                "rain_prob": daily.get("precipitation_probability_max", [None])[index],
            }
        )

    return {
        "location": coordinates["name"],
        "country": coordinates.get("country"),
        "timeframe": timeframe,
        "forecast_days": forecast_days,
        "forecast": summary,
        "provider": "Open-Meteo",
    }


def weather_tool(tool_input: dict[str, Any]) -> dict[str, Any]:
    location = tool_input.get("location", "Beijing")
    timeframe = tool_input.get("timeframe", "today")
    return lookup_weather(location, timeframe)


if __name__ == "__main__":
    print(json.dumps(lookup_weather("Shanghai", "today"), ensure_ascii=False, indent=2))
