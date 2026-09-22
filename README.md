# Car Research Agent

An AI agent that answers car-related questions by searching the web and citing its sources.

**Live demo:** https://car-research-agent.onrender.com *(API key required — the free tier sleeps after 15 minutes idle, so the first request can take up to a minute)*

Ask *"Should I buy a Golf 2015 or a Focus 2016 for city driving?"* and it breaks the question into multiple searches, streams its progress live, and returns an answer with a URL next to every factual claim.

## How it works

```
question → [agent loop] → search → read → decide if more is needed → answer
                ↑                                    │
                └────────────────────────────────────┘
```

The model decides **what** to search, **how many times**, and **when it has enough**. That loop is what makes it an agent rather than a fixed pipeline: a single-shot RAG would search once with the raw question, which is a poor search query.

The agent has two tools:

- **`search_web`** — Tavily web search
- **`convert_units`** — unit conversion done in code. The model used to convert mpg to L/100km on its own and got it wrong; arithmetic doesn't get fixed by prompting, it gets taken away from the model.

## Architecture

```
run_agent_stream()  ──►  /ask/stream  ──►  SSE  ──►  browser
   (generator)                                        (fetch + buffer)
        │
        └──►  run_agent()  ──►  /ask  (non-streaming)
```

The agent loop exists **once**, as a generator that emits `searching`, `token` and `result` events. The streaming endpoint forwards them as Server-Sent Events; the non-streaming path consumes the same generator and returns only the final result.

## Stack

| Component | Choice | Why |
|---|---|---|
| Model | Anthropic Claude (Haiku) | Cheap enough for iteration |
| Search | Tavily | 1,000 free credits/month, built for LLM retrieval |
| API | FastAPI | Pydantic validation, auto-generated docs |
| Frontend | Vanilla HTML/JS | No build step; served by FastAPI from the same origin |
| Observability | Langfuse | Open source, OpenTelemetry-based, no framework lock-in |
| Hosting | Render | Free tier without a credit card |

## Design decisions

**No agent framework.** The loop is plain Python. LangChain would add abstraction without solving anything here, and debugging an opaque loop is worse than writing an obvious one.

**Web search is not delegated to the model provider.** Anthropic ships a built-in `web_search` tool, but its results come back encrypted — you can't inspect, filter or cache them. Owning the search step means owning the trade-offs.

**Hard limits on cost.** The agent stops at 10 iterations or 50,000 tokens. One early run consumed 115,000 tokens against a ~26,000 average, so this isn't theoretical.

**The LLM's output is treated as untrusted input.** The agent reads arbitrary web pages, and a malicious page can induce the model to repeat executable HTML. Rendered markdown goes through DOMPurify before touching the DOM — this is indirect prompt injection turning into XSS, not a hypothetical.

**Tracing inside a generator uses manual observations.** OpenTelemetry context managers lose their context across `yield` when FastAPI iterates the generator in a threadpool. The fix is explicit `start_observation()` calls with `try/finally`, so spans close even when the client disconnects mid-stream.

**Endpoints that spend money are authenticated.** Keeping keys out of the repo stops people from *reading* them; it doesn't stop them from *using* a public endpoint that uses them.

## Measuring quality

The agent is evaluated on a hand-written dataset using LLM-as-judge, on two independent criteria:

- **Citations** — every numeric fact must carry its source URL
- **Comparability** — compared figures must share the same unit and measurement basis

| Metric | Mean | Range |
|---|---|---|
| Citations | 3.87 / 5 | 3.60 – 4.20 |
| Comparability | 4.53 / 5 | 4.20 – 5.00 |

Identical runs varied up to 6× in cost, so every configuration is measured across multiple runs and reported as a range — a single run proves nothing.

### Experiments that didn't ship

| Change | Result | Decision |
|---|---|---|
| 2000 vs 600 chars per source | +0.27 score, +33% cost | Kept 2000 — quality over cost at this scale |
| Limit to 4 iterations | Lower score **and** higher cost — the agent parallelized more searches per turn | Reverted |
| "Omit figures without a source" | Citations +0.20 (not significant), comparability **−0.86** | Reverted |

The last one is the reason for tracking two metrics. Citations went up, so the change looked like a win — the damage only showed up because comparability was measured separately. With a single metric, it would have shipped.

## Running it

```bash
pip install -r requirements.txt
cp .env.example .env    # add your API keys
fastapi dev api.py
```

App at `http://localhost:8000`, interactive API docs at `http://localhost:8000/docs`.

```bash
python run_evals.py     # run the eval suite
```

## Known limitations

- **Synchronous streaming.** Each open stream holds a worker thread for most of its duration (the default pool is 40). Fine for a demo; real traffic would need an async rewrite, since the agent spends almost all its time waiting on external APIs.
- **No conversation memory.** Every question starts from scratch. The design is clear — store only question and final answer per turn, not the search payloads — but it isn't built.
- **Source quality is unfiltered.** The agent has cited dealership pages and Instagram reels as authorities.
- **Measurement bases can still get mixed.** It once compared one car's curb weight against another's gross vehicle weight. The comparability eval shows this is uncommon, not solved.
- **Cold starts** on the free tier.