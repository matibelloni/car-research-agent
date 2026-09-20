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
| API | FastAPI | Pydantic validation, auto-generated docs |
| Observability | Langfuse | Open source, OpenTelemetry-based, no framework lock-in |
| Orchestration | Custom loop | See below |

## Design decisions

**No agent framework.** The loop is ~40 lines of plain Python. LangChain would add abstraction without solving anything here, and debugging an opaque loop is worse than writing an obvious one.

**Web search is not delegated to the model provider.** Anthropic ships a built-in `web_search` tool, but its results come back encrypted — you can't inspect, filter or cache them. Owning the search step means owning the trade-offs.

**Hard limits on cost.** The agent stops at 10 iterations or 50,000 tokens, whichever comes first. One early run consumed 115,000 tokens against a ~26,000 average, so this isn't theoretical.

## Measuring quality

The agent is evaluated on a hand-written dataset using LLM-as-judge. The criterion is strict: *every numeric fact must carry its source URL*.

Two "obvious" optimizations were tested and **both made the system worse**:

| Config | Score | Tokens |
|---|---|---|
| 2000 chars/source, 10 iterations | **3.50** | 35,766 |
| 600 chars/source, 10 iterations | 3.23 | 26,950 |
| 600 chars/source, 4 iterations | 3.07 | 33,490 |

Limiting iterations to force fewer searches backfired: the agent parallelized more searches per turn, spending *more* tokens and scoring *lower*. Without evals, that change would have shipped as an improvement.

Identical runs varied up to 6× in cost, so every configuration is measured across multiple runs and reported as a range — a single run proves nothing.

## Running it

```bash
pip install -r requirements.txt
cp .env.example .env    # add your API keys
fastapi dev api.py
```

Interactive docs at `http://localhost:8000/docs`.

```bash
python run_evals.py     # run the eval suite
```

## Known limitations

- **Source quality is unfiltered.** The agent has cited dealership pages and Instagram reels as authorities. Domain filtering is the next fix.
- **No unit normalization across sources.** One source reports metres, another millimetres, and the agent presents them side by side.
- **No streaming.** A request takes 20–40 seconds with no intermediate feedback.
