# Car Research Agent

An AI agent that answers car-related questions by searching the web and citing its sources.

Ask *"Should I buy a Golf 2015 or a Focus 2016 for city driving?"* and it breaks the question into multiple searches, reads the results, and returns an answer with a URL next to every factual claim.

## How it works

```
question → [agent loop] → search → read → decide if more is needed → answer
                ↑                                    │
                └────────────────────────────────────┘
```

The model decides **what** to search, **how many times**, and **when it has enough**. That loop is what makes it an agent rather than a fixed pipeline: a single-shot RAG would search once with the raw question, which is a poor search query.

## Stack

| Component | Choice | Why |
|---|---|---|
| Model | Anthropic Claude (Haiku) | Cheap enough for iteration |
| Search | Tavily | 1,000 free credits/month, built for LLM retrieval |
| Unit conversion | pint | Deterministic metric conversion — the model calls a tool instead of doing the math itself |
| API | FastAPI | Pydantic validation, auto-generated docs, native SSE streaming |
| Observability | Langfuse | Open source, OpenTelemetry-based, no framework lock-in |
| Orchestration | Custom loop | See below |

## Design decisions

**No agent framework.** The loop is ~40 lines of plain Python. LangChain would add abstraction without solving anything here, and debugging an opaque loop is worse than writing an obvious one.

**Web search is not delegated to the model provider.** Anthropic ships a built-in `web_search` tool, but its results come back encrypted — you can't inspect, filter or cache them. Owning the search step means owning the trade-offs.

**Unit conversion is a tool, not a prompt instruction.** Asking the model to convert mph → km/h or mpg → l/100km inline invites silent arithmetic errors. `convert_units` (built on `pint`) does the math; the model just calls it.

**Hard limits on cost.** The agent stops at 10 iterations or 50,000 tokens, whichever comes first. One early run consumed 115,000 tokens against a ~26,000 average, so this isn't theoretical.

## Measuring quality

The agent is evaluated on a hand-written dataset using LLM-as-judge, on two criteria: every numeric fact must carry its source URL (**citations**), and comparisons must use equivalent bases — same unit, same measurement basis, same conditions (**comparability**).

Two "obvious" optimizations were tested and **both made the system worse**:

| Config | Score | Tokens |
|---|---|---|
| 2000 chars/source, 10 iterations | **3.50** | 35,766 |
| 600 chars/source, 10 iterations | 3.23 | 26,950 |
| 600 chars/source, 4 iterations | 3.07 | 33,490 |

Limiting iterations to force fewer searches backfired: the agent parallelized more searches per turn, spending *more* tokens and scoring *lower*. Without evals, that change would have shipped as an improvement.

Identical runs varied up to 6× in cost, so every configuration is measured across multiple runs and reported as a range — a single run proves nothing.

Two smaller, standalone evals cover the deterministic pieces:

- `python eval_units.py` — checks `convert_units` against known conversions
- `python eval_retrieval.py` — checks embedding-based retrieval precision on a hand-written question/document set

## Running it

```bash
pip install -r requirements.txt
cp .env.example .env    # add your API keys, including APP_API_KEY (pick any string)
fastapi dev api.py
```

Interactive docs at `http://localhost:8000/docs`. Both endpoints below require an `x-api-key` header matching `APP_API_KEY`.

- `POST /ask` — blocking, returns the full answer once it's ready
- `POST /ask/stream` — Server-Sent Events, streams tokens and search progress as they happen

```bash
python run_evals.py     # run the answer-quality eval suite
```

## Known limitations

- **Source quality is unfiltered.** The agent has cited dealership pages and Instagram reels as authorities. Domain filtering is the next fix.
