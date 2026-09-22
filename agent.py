import os
from dotenv import load_dotenv
from anthropic import Anthropic
from tavily import TavilyClient
from rich import print as rprint
import json
from langfuse import get_client
from typing import Iterator
from pydantic import BaseModel
from tools import CONVERT_UNITS_TOOL, convert_units

load_dotenv()

langfuse = get_client()

MAX_CHARS_PER_SOURCE = 2000

tavily = TavilyClient(api_key=os.environ["TAVILY_API_KEY"])
claude = Anthropic()


class AgentResult(BaseModel):
    answer: str | None = None
    error: str | None = None
    tokens: int = 0
    searches: int = 0


tools = [
    {
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
    },
    CONVERT_UNITS_TOOL,
]


def search_web(query: str) -> str:
    """Runs the search and returns the results as text."""
    response = tavily.search(query=query, max_results=3)
    return "\n\n".join(
        [
            f"Source: {m["url"]}\n{m["content"][:MAX_CHARS_PER_SOURCE]}"
            for m in response["results"]
        ]
    )


SYSTEM = """You are a research assistant about cars.

Rules:
- Answer ONLY with information from the searches. If something isn't there, say so.
- Cite the source URL after each claim with concrete data.
- If the sources contradict each other, mention it instead of picking one.
- Don't make up figures. If you didn't find a fact, say you didn't find it.
- Normalize all units to the metric system. When a source gives a non-metric value
  (mph, mi, mpg, gal, hp, lb, in, psi, °F), call convert_units to get the metric
  value — never convert it yourself."""

MAX_ITERATIONS = 10
TOKEN_BUDGET = 50_000


def run_agent_stream(question: str) -> Iterator[dict]:
    root = langfuse.start_observation(name="run-agent", as_type="span", input=question)
    try:
        messages = [{"role": "user", "content": question}]
        used_tokens = 0
        searches = 0
        for i in range(MAX_ITERATIONS):
            gen = root.start_observation(
                name=f"llm-call-{i + 1}", as_type="generation", model="claude-haiku-4-5"
            )
            try:
                with claude.messages.stream(
                    model="claude-haiku-4-5",
                    max_tokens=2000,
                    tools=tools,
                    messages=messages,
                    system=SYSTEM,
                ) as stream:
                    for text in stream.text_stream:
                        yield {
                            "type": "token",
                            "text": text,
                        }
                    response = stream.get_final_message()
                gen.update(
                    usage_details={
                        "input": response.usage.input_tokens,
                        "output": response.usage.output_tokens,
                    },
                    output=response.stop_reason,
                )
            finally:
                gen.end()
            used_tokens += response.usage.input_tokens + response.usage.output_tokens
            messages.append({"role": "assistant", "content": response.content})

            if used_tokens > TOKEN_BUDGET:
                root.update(output="budget_exceeded")
                yield {
                    "type": "result",
                    "answer": None,
                    "error": "budget_exceeded",
                    "tokens": used_tokens,
                    "searches": searches,
                }
                return

            if response.stop_reason == "end_turn":
                answer = "\n".join(
                    content.text
                    for content in response.content
                    if content.type == "text"
                )
                root.update(output=answer)
                yield {
                    "type": "result",
                    "answer": answer,
                    "error": None,
                    "tokens": used_tokens,
                    "searches": searches,
                }
                return

            if response.stop_reason != "tool_use":
                error = f"unexpected_stop: {response.stop_reason}"
                root.update(output=error)
                yield {
                    "type": "result",
                    "answer": None,
                    "error": error,
                    "tokens": used_tokens,
                    "searches": searches,
                }
                return

            tool_results = []
            for content in response.content:
                if content.type == "tool_use":
                    if content.name == "search_web":
                        query = content.input["query"]
                        searches += 1
                        yield {"type": "searching", "query": query}
                        span = root.start_observation(
                            name="search", as_type="span", input=query
                        )
                        result = search_web(query)
                        span.update(output=f"{len(result)} chars")
                        span.end()
                        tool_results.append(
                            {
                                "type": "tool_result",
                                "tool_use_id": content.id,
                                "content": result,
                            }
                        )
                    elif content.name == "convert_units":
                        unit_from = content.input["unit_from"]
                        unit_to = content.input["unit_to"]
                        value = content.input["value"]
                        try:
                            result = convert_units(
                                unit_from=unit_from, unit_to=unit_to, value=value
                            )
                            tool_results.append(
                                {
                                    "type": "tool_result",
                                    "tool_use_id": content.id,
                                    "content": result,
                                }
                            )
                        except ValueError as e:
                            tool_results.append(
                                {
                                    "type": "tool_result",
                                    "tool_use_id": content.id,
                                    "content": str(e),
                                    "is_error": True,
                                }
                            )

            messages.append({"role": "user", "content": tool_results})

        root.update(output="max_iterations")
        yield {
            "type": "result",
            "answer": None,
            "error": "max_iterations",
            "tokens": used_tokens,
            "searches": searches,
        }
    finally:
        root.end()


def run_agent(question: str, verbose: bool = False) -> AgentResult:
    for event in run_agent_stream(question=question):
        if verbose and event["type"] == "searching":
            rprint(f"  🔍 {event['query']}")

        elif event["type"] == "result":
            return AgentResult(
                answer=event["answer"],
                error=event["error"],
                tokens=event["tokens"],
                searches=event["searches"],
            )

    return AgentResult(error="no_result")


if __name__ == "__main__":
    result = run_agent(
        question="Is a 2015 Golf or a 2016 Focus better for city driving?", verbose=True
    )
    rprint(f"Tokens: {result.tokens} | Searches: {result.searches}")
    rprint(f"Answer: {result.answer}")
    langfuse.flush()
