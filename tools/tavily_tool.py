import os
import random
import re

from dotenv import load_dotenv
from langchain_core.tools import tool

load_dotenv()

TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", "").strip()


def _mock_options(source: str, destination: str, date: str) -> dict:
    random.seed(f"{source}-{destination}-{date}")

    base_km = random.randint(300, 1800)
    return {
        "flight": [
            {
                "provider": "IndiGo",
                "price": round(base_km * random.uniform(4.5, 6.5)),
                "duration_hours": round(base_km / 700 + 1, 1),
                "booking_link": "https://www.goindigo.in",
            }
        ],
        "train": [
            {
                "provider": "Indian Railways (via IRCTC)",
                "price": round(base_km * random.uniform(0.8, 1.4)),
                "duration_hours": round(base_km / 60, 1),
                "booking_link": "https://www.irctc.co.in",
            }
        ],
        "bus": [
            {
                "provider": "RedBus / VRL Travels",
                "price": round(base_km * random.uniform(1.2, 2.0)),
                "duration_hours": round(base_km / 50, 1),
                "booking_link": "https://www.redbus.in",
            }
        ],
        "source_note": "mock-data (set TAVILY_API_KEY in .env for live search results)",
    }


_MODE_QUERIES = {
    "flight": "flight from {source} to {destination} airline name price",
    "train": "train name and number from {source} to {destination} IRCTC",
    "bus": "bus operator from {source} to {destination} fare booking",
}


def _is_grounded(name: str, source_text: str) -> bool:
    name_norm = re.sub(r"[^a-z0-9 ]", " ", name.lower())
    source_norm = re.sub(r"[^a-z0-9 ]", " ", source_text.lower())

    digits = re.findall(r"\d{3,}", name)
    if digits:
        return any(d in source_text for d in digits)

    words = name_norm.split()
    for n in range(min(len(words), 5), 2, -1):
        for i in range(len(words) - n + 1):
            phrase = " ".join(words[i : i + n])
            if len(phrase) > 8 and phrase in source_norm:
                return True
    return False


def _extract_provider_name(mode: str, source: str, destination: str, results: list, fallback: str) -> str:
    if not results:
        return fallback

    snippets = "\n\n".join(
        f"Title: {r.get('title', '')}\nContent: {(r.get('content') or '')[:400]}" for r in results
    )
    prompt = (
        f"Below are web search results about {mode} travel from {source} to {destination}. "
        f"Quote the single most specific, real {mode} provider or service name that is "
        f"EXPLICITLY WRITTEN in the text below — copy it verbatim, do not paraphrase or guess. "
        f"(for trains: an actual train name and/or number; for flights: an airline name; "
        f"for buses: an operator name). "
        f"Respond with ONLY that exact text and nothing else. If none of the results "
        f"explicitly name one, respond with exactly: NONE\n\n{snippets}"
    )
    try:
        from llm.model import get_shared_llm

        response = get_shared_llm().invoke(prompt)
        name = response.content.strip().strip('"')
        if not name or name.upper() == "NONE" or len(name) > 100:
            return fallback
        if not _is_grounded(name, snippets):
            return fallback
        return name
    except Exception:
        return fallback


def _tavily_enrich(source: str, destination: str, date: str, base: dict) -> dict:
    from tavily import TavilyClient

    client = TavilyClient(api_key=TAVILY_API_KEY)
    enriched = {mode: [dict(option) for option in options] for mode, options in base.items() if mode != "source_note"}

    for mode in ["flight", "train", "bus"]:
        query = _MODE_QUERIES[mode].format(source=source, destination=destination)
        try:
            resp = client.search(query=query, max_results=3, search_depth="basic")
            results = resp.get("results", [])
        except Exception:
            results = []

        if results:
            enriched[mode][0]["provider"] = _extract_provider_name(
                mode, source, destination, results, enriched[mode][0]["provider"]
            )
            enriched[mode][0]["booking_link"] = results[0].get("url") or enriched[mode][0]["booking_link"]
            enriched[mode][0]["search_snippet"] = (results[0].get("content") or "")[:250]

    enriched["source_note"] = (
        "tavily-enriched (price/duration are estimates; provider name and "
        "booking link come from live Tavily search)"
    )
    return enriched


@tool
def search_travel_options_tool(source: str, destination: str, date: str) -> dict:
    """Search for flight, train, and bus options between two cities on a given date."""
    base = _mock_options(source, destination, date)
    if TAVILY_API_KEY:
        try:
            return _tavily_enrich(source, destination, date, base)
        except Exception:
            return base
    return base
