import os
from functools import cache

from tavily import TavilyClient

MAX_CHARS_PER_SOURCE = 2000

SEARCH_WEB_TOOL = {
    "name": "search_web",
    "description": "Searches the internet for information about cars. Use it when you need data you don't have or want to verify something with real sources.",
    "input_schema": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "The search terms",
            }
        },
        "required": ["query"],
    },
}


@cache
def tavily() -> TavilyClient:
    return TavilyClient(api_key=os.environ["TAVILY_API_KEY"])


def search_web(query: str) -> str:
    """Runs the search and returns the results as text."""
    response = tavily().search(query=query, max_results=3)
    return "\n\n".join(
        [
            f"Source: {m["url"]}\n{m["content"][:MAX_CHARS_PER_SOURCE]}"
            for m in response["results"]
        ]
    )
