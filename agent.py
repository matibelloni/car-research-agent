import json
from typing import Any, Callable, Generator, Iterator

import anthropic
from dotenv import load_dotenv
from langfuse import get_client
from pydantic import BaseModel
from rich import print as rprint

from tools import TOOLS, convert_units, get_recalls_ar, search_web, search_manual_tool

load_dotenv()

langfuse = get_client()


def classify_error(e: anthropic.APIError) -> str:
    if isinstance(e, anthropic.RateLimitError):
        return "rate_limited"
    if isinstance(e, anthropic.APIConnectionError):
        return "provider_unreachable"
    if isinstance(e, anthropic.APIStatusError) and e.status_code >= 500:
        return "provider_unavailable"
    return "provider_error"


claude = anthropic.Anthropic(max_retries=3, timeout=60.0)


class AgentResult(BaseModel):
    answer: str | None = None
    error: str | None = None
    tokens: int = 0
    searches: int = 0


# Each handler is a generator: it yields stream events for the client and
# returns the tool_result fields (content, is_error) for the model.
ToolHandler = Callable[[dict, Any], Generator[dict, None, dict]]


def handle_search_web(input: dict, trace) -> Generator[dict, None, dict]:
    query = input["query"]
    yield {"type": "searching", "query": query}
    span = trace.start_observation(name="search", as_type="span", input=query)
    try:
        result = search_web(query)
        span.update(output=f"{len(result)} chars")
        return {"content": result}
    except Exception as e:
        span.update(level="ERROR", status_message=str(e))
        return {
            "content": f"Search failed: {e}. Try a different query or answer with what you have.",
            "is_error": True,
        }
    finally:
        span.end()


def handle_convert_units(input: dict, trace) -> Generator[dict, None, dict]:
    try:
        result = convert_units(
            unit_from=input["unit_from"], unit_to=input["unit_to"], value=input["value"]
        )
        return {"content": result}
    except ValueError as e:
        return {"content": str(e), "is_error": True}
    yield  # no events, but keeps the handler a generator like the others


def handle_get_recalls_ar(input: dict, trace) -> Generator[dict, None, dict]:
    brand = input["brand"]
    keyword = input.get("keyword")
    span = trace.start_observation(
        name="search", as_type="span", input={"brand": brand, "keyword": keyword}
    )
    try:
        result = get_recalls_ar(brand, keyword)
        yield {
            "type": "recalls_lookup",
            "brand": brand,
            "keyword": keyword,
            "total": result.get("total"),
            "error": result.get("error"),
        }
        span.update(output=result.get("error") or f"{result['total']} recalls")
        return {
            "content": json.dumps(result, ensure_ascii=False),
            "is_error": "error" in result,
        }
    except ValueError as e:
        span.update(level="ERROR", status_message=str(e))
        return {"content": f"Recall lookup failed: {e}.", "is_error": True}
    finally:
        span.end()


def handle_search_manual(input: dict, trace) -> Generator[dict, None, dict]:
    model = input["model"]
    year = input.get("year")
    query = input["query"]
    yield {"type": "manual_search", "model": model, "year": year, "query": query}
    span = trace.start_observation(
        name="search",
        as_type="span",
        input={"model": model, "year": year, "query": query},
    )
    try:
        result = search_manual_tool(model, query, year)
        # Which sections the model got, without their text, to read off the trace.
        span.update(
            output=result.get("error")
            or {
                "count": len(result["sections"]),
                "sections": [
                    {"citation": s["citation"], "heading": s["heading"]}
                    for s in result["sections"]
                ],
            }
        )
        return {
            "content": json.dumps(result, ensure_ascii=False),
            "is_error": "error" in result,
        }
    except Exception as e:
        span.update(level="ERROR", status_message=str(e))
        return {
            "content": f"Manual search failed: {e}. Search the web instead.",
            "is_error": True,
        }
    finally:
        span.end()


HANDLERS: dict[str, ToolHandler] = {
    "search_web": handle_search_web,
    "convert_units": handle_convert_units,
    "get_recalls_ar": handle_get_recalls_ar,
    "search_manual": handle_search_manual,
}


def run_tool(content, trace) -> Generator[dict, None, dict]:
    handler = HANDLERS.get(content.name)
    if handler is None:
        outcome = {"content": f"Unknown tool: {content.name}", "is_error": True}
    else:
        outcome = yield from handler(content.input, trace)
    return {"type": "tool_result", "tool_use_id": content.id, **outcome}


SYSTEM = """You are a research assistant about cars.

Rules:
- Answer ONLY with information from your tool results. If something isn't there,
  say so. Don't add facts from your own knowledge, even if you believe they're true.
- Cite the source after each claim with concrete data:
  - For web results, the source URL right after each figure — even when
    several figures come from the same source. In bulleted lists, every
    bullet with a figure ends with its own URL. Summary figures and ranges
    ("around 28-30 mpg") need a source too. Never group sources at the end.
  - For recalls from get_recalls_ar, "Defensa del Consumidor" and the publication
    date (e.g. "Defensa del Consumidor, 29/01/2026").
  - For owner's manual sections from search_manual, the section's citation
    field, exactly as given (e.g. "Manual del Fiat Cronos 2021, p. G-14").
- If the sources contradict each other, mention it instead of picking one.
- Don't make up figures. If you didn't find a fact, say you didn't find it.
- If search_manual's sections don't answer the question, say the manual doesn't
  cover it and call search_web before answering. Never end with advice from your
  own knowledge instead.
- Always answer in the same language as the user's question.
- Normalize all units to the metric system. When a source gives a non-metric value
  (mph, mi, mpg, gal, hp, lb, in, psi, °F), call convert_units to get the metric
  value — never convert it yourself.
- When reporting recalls, always state how many were found in total. If you
  summarize or list only some of them, say how many you're leaving out."""

MODEL = "claude-haiku-4-5"
MAX_ITERATIONS = 10
TOKEN_BUDGET = 50_000
MAX_OUTPUT_TOKENS = 2000
# Models that accept tool_choice "any". Opus 5.5, Sonnet 5.5, Fable 5.1 and
# Mythos 5.1 reject it with a 400:
# https://platform.claude.com/docs/en/agents-and-tools/tool-use/define-tools#forcing-tool-use
FORCED_TOOL_MODELS = {"claude-haiku-4-5"}


def call_model(
    messages: list, tool_choice: dict, trace, name: str
) -> Generator[dict, None, anthropic.types.Message]:
    """One streamed call: yields its tokens and returns the final message."""
    gen = trace.start_observation(name=name, as_type="generation", model=MODEL)
    try:
        with claude.messages.stream(
            model=MODEL,
            max_tokens=MAX_OUTPUT_TOKENS,
            tools=TOOLS,
            tool_choice=tool_choice,
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
    return response


def run_agent_stream(question: str) -> Iterator[dict]:
    root = langfuse.start_observation(name="run-agent", as_type="span", input=question)
    try:
        messages = [{"role": "user", "content": question}]
        used_tokens = 0
        searches = 0
        for i in range(MAX_ITERATIONS):
            response = yield from call_model(
                messages, {"type": "auto"}, root, f"llm-call-{i + 1}"
            )
            used_tokens += response.usage.input_tokens + response.usage.output_tokens

            # Some runs answered a price question with "I have no tools for that"
            # without trying any. Forcing a tool on every first call fixed that
            # but sent price questions to the manual only, so the call is
            # repeated with a forced tool only when it used none.
            if (
                i == 0
                and response.stop_reason == "end_turn"
                and MODEL in FORCED_TOOL_MODELS
                and used_tokens <= TOKEN_BUDGET
            ):
                root.update(metadata={"forced_tool_retry": True})
                yield {"type": "retry"}  # the streamed text is discarded
                response = yield from call_model(
                    messages, {"type": "any"}, root, "llm-call-1-forced"
                )
                used_tokens += response.usage.input_tokens + response.usage.output_tokens

            messages.append({"role": "assistant", "content": response.content})

            if used_tokens > TOKEN_BUDGET:
                root.update(
                    output="budget_exceeded",
                    level="ERROR",
                    status_message=f"{used_tokens} tokens over a budget of {TOKEN_BUDGET}",
                )
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
                # A truncated answer reads like a complete one, so it's an error,
                # with its own code so the client can say what happened.
                if response.stop_reason == "max_tokens":
                    error = "answer_too_long"
                else:
                    error = f"unexpected_stop: {response.stop_reason}"
                root.update(output=error, level="ERROR", status_message=error)
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
                        searches += 1
                    tool_results.append((yield from run_tool(content, root)))

            messages.append({"role": "user", "content": tool_results})

        root.update(
            output="max_iterations",
            level="ERROR",
            status_message=f"no answer after {MAX_ITERATIONS} iterations",
        )
        yield {
            "type": "result",
            "answer": None,
            "error": "max_iterations",
            "tokens": used_tokens,
            "searches": searches,
        }
    except anthropic.APIError as e:
        code = classify_error(e)
        root.update(output=code, level="ERROR", status_message=str(e))
        yield {
            "type": "result",
            "answer": None,
            "error": code,
            "tokens": used_tokens,
            "searches": searches,
        }
    finally:
        root.end()


def run_agent(question: str, verbose: bool = False) -> AgentResult:
    for event in run_agent_stream(question=question):
        if verbose and event["type"] == "searching":
            rprint(f"  🔍 {event['query']}")

        elif verbose and event["type"] == "manual_search":
            year = f" {event['year']}" if event["year"] else ""
            rprint(f"  📖 {event['model']}{year}: {event['query']}")

        elif event["type"] == "result":
            return AgentResult(
                answer=event["answer"],
                error=event["error"],
                tokens=event["tokens"],
                searches=event["searches"],
            )

    return AgentResult(error="no_result")


if __name__ == "__main__":
    questions = [
        # question="Is a 2015 Golf or a 2016 Focus better for city driving?",
        # question="¿Hay recalls de airbags para el Volkswagen Gol en Argentina?",
        # question="¿Cada cuántos km se cambia la correa de accesorios del Fiat Cronos 2021?",
        # "¿Cuánto cuesta el service de los 10.000 km del Fiat Cronos 2021?",
        "¿Cada cuántos km se cambia la correa de accesorios del Fiat Cronos 2021?",
    ]
    for question in questions:
        for run in range(1, 4):
            rprint(f"\n[bold]{question} — run {run}[/bold]")
            result = run_agent(question=question, verbose=True)
            first_line = (result.answer or result.error or "").split("\n")[0]
            rprint(f"Tokens: {result.tokens} | Searches: {result.searches}")
            rprint(f"Answer: {first_line}")
    langfuse.flush()
