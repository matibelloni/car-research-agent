import os
from dotenv import load_dotenv
from anthropic import Anthropic
from tavily import TavilyClient
from rich import print as rprint
import json
from langfuse import get_client

load_dotenv()

langfuse = get_client()

MAX_CHARS_PER_SOURCE = 2000

tavily = TavilyClient(api_key=os.environ["TAVILY_API_KEY"])
claude = Anthropic()

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
    }
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
- Normalize all units to the metric system"""

MAX_ITERATIONS = 10
TOKEN_BUDGET = 50_000


def run_agent(question: str, verbose: bool = False):
    with langfuse.start_as_current_observation(
        as_type="span", name="run-agent", input=question
    ) as root:
        messages = [{"role": "user", "content": question}]
        used_tokens = 0
        searches = 0
        for i in range(MAX_ITERATIONS):
            with langfuse.start_as_current_observation(
                as_type="generation", name=f"llm-call-{i + 1}", model="claude-haiku-4-5"
            ) as gen:
                response = claude.messages.create(
                    model="claude-haiku-4-5",
                    max_tokens=2000,
                    tools=tools,
                    messages=messages,
                    system=SYSTEM,
                )
                gen.update(
                    usage_details={
                        "input": response.usage.input_tokens,
                        "output": response.usage.output_tokens,
                    },
                    output=response.stop_reason,
                )
            # rprint(response)
            used_tokens += response.usage.input_tokens + response.usage.output_tokens

            # print(f"i:{i + 1}: used tokens: {used_tokens}")
            messages.append({"role": "assistant", "content": response.content})

            if used_tokens > TOKEN_BUDGET:
                root.update(output="budget_exceeded")
                return {
                    "answer": None,
                    "error": "budget_exceeded",
                    "tokens": used_tokens,
                    "searches": searches,
                }

            if response.stop_reason == "end_turn":
                answer = "\n".join(
                    content.text
                    for content in response.content
                    if content.type == "text"
                )
                root.update(output=answer)
                return {
                    "answer": answer,
                    "error": None,
                    "tokens": used_tokens,
                    "searches": searches,
                }

            if response.stop_reason != "tool_use":
                root.update(output=f"unexpected_stop: {response.stop_reason}")
                return {
                    "answer": None,
                    "error": f"unexpected_stop: {response.stop_reason}",
                    "tokens": used_tokens,
                    "searches": searches,
                }

            tool_results = []
            # rprint(response.content)
            for content in response.content:
                if content.type == "tool_use" and content.name == "search_web":
                    query = content.input["query"]
                    searches += 1
                    if verbose:
                        print(f"  🔍 {query}")
                    with langfuse.start_as_current_observation(
                        as_type="span", name="search", input=query
                    ) as span:
                        result = search_web(query)
                        span.update(output=f"{len(result)} chars")
                        tool_results.append(
                            {
                                "type": "tool_result",
                                "tool_use_id": content.id,
                                "content": result,
                            }
                        )
            messages.append({"role": "user", "content": tool_results})

        root.update(output="max_iterations")
        return {
            "answer": None,
            "error": "max_iterations",
            "tokens": used_tokens,
            "searches": searches,
        }


if __name__ == "__main__":
    result = run_agent(
        question="Is a 2015 Golf or a 2016 Focus better for city driving?", verbose=True
    )
    rprint(f"Tokens: {result["tokens"]} | Searches: {result["searches"]}")
    rprint(f"Answer: {result["answer"]}")
    langfuse.flush()
