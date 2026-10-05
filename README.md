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

The agent has three tools:

- **`search_web`** — Tavily web search
- **`convert_units`** — unit conversion done in code. The model used to convert mpg to L/100km on its own and got it wrong; arithmetic doesn't get fixed by prompting, it gets taken away from the model.
- **`get_recalls_ar`** — official vehicle recalls published in Argentina by Defensa del Consumidor, from a local dataset built by `scripts/ingest_recalls.py`

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

**Endpoints that spend money are authenticated.** Keeping keys out of the repo stops people from *reading* them; it doesn't stop them from *using* a public endpoint that uses them. For the same reason, questions are capped at 1,000 characters and rejected when empty, before they reach the model.

## Measuring quality

There are two eval suites in [`evals/`](evals/), both on hand-written datasets.

### Answer quality — `evals/answers.py`

General car questions scored by an LLM judge on two independent criteria:

- **Citations** — every numeric fact must carry its source URL (judged by Sonnet 5.5)
- **Comparability** — compared figures must share the same unit and measurement basis (judged by Haiku 4.5)

| Metric | Mean | Range |
|---|---|---|
| Citations | 3.87 / 5 | 3.60 – 4.20 |
| Comparability | 4.53 / 5 | 4.20 – 5.00 |

*Measured with the earlier Haiku-only judge and citations prompt. Calibration later showed that judge was too lenient on citations (see below), so the citations figure is likely inflated; it needs a re-run with the current judge.*

Identical runs varied up to 6× in cost, so every configuration is measured across multiple runs and reported as a range — a single run proves nothing.

### Experiments that didn't ship

| Change | Result | Decision |
|---|---|---|
| 2000 vs 600 chars per source | +0.27 score, +33% cost | Kept 2000 — quality over cost at this scale |
| Limit to 4 iterations | Lower score **and** higher cost — the agent parallelized more searches per turn | Reverted |
| "Omit figures without a source" | Citations +0.20 (not significant), comparability **−0.86** | Reverted |

The last one is the reason for tracking two metrics. Citations went up, so the change looked like a win — the damage only showed up because comparability was measured separately. With a single metric, it would have shipped.

### Recalls — `evals/recalls.py`

Checks how the agent uses `get_recalls_ar`, mostly with deterministic checks; the LLM judge only covers what code can't check:

- **Lookup** (deterministic) — the agent called the tool with the expected brand and keyword (e.g. "brake" → `freno`, "Hilux" → Toyota)
- **Total** (deterministic) — the answer states how many recalls the lookup found
- **Behavior** (LLM judge, some cases) — no invented recalls when there are none, and never claiming the user's specific car is affected

### Calibrating the judges

Each suite ships hand-written answers with a known verdict. Before trusting a judge on the agent, run it on those: if it disagrees, the criterion gets fixed first.

The first calibration set only had easy citation cases, and Haiku passed them. Adding borderline cases (one URL after two figures, a URL in the next sentence, a figure inside advice) exposed it:

| Judge | Citations | Comparability |
|---|---|---|
| Haiku 4.5, original prompt | 6/9 | 8/8 |
| Haiku 4.5, clarified prompt (3 runs) | 6–7/9 — the same 2 cases wrong in every run | 7–8/8 |
| Sonnet 5.5, clarified prompt (3 runs) | **9/9** every run | 6/8 — docks points for details the answer never mentions |

Haiku's errors were misreadings, not ambiguity: it reported a URL from the next sentence as "right after" the figure, and invented a rule that one URL covers several figures — the prompt says the opposite. So each criterion gets the judge that calibrates best: Sonnet for citations, Haiku for comparability. The judge also writes its reasoning *before* the score, since structured output generates fields in order.

## Running it

```bash
pip install -r requirements.txt
cp .env.example .env    # add your API keys
fastapi dev api.py
```

App at `http://localhost:8000`, interactive API docs at `http://localhost:8000/docs`.

Evals (from the project root; they call the real APIs and cost money):

```bash
python -m evals.answers                  # answer quality suite
python -m evals.recalls                  # recalls suite (or pass case ids to run only those)
python -m evals.answers --calibrate      # check a judge against hand-written answers
python -m evals.recalls --calibrate
```

Rebuilding the recalls dataset (`data/recalls_clean.json` is committed, so this is only needed to refresh it):

```bash
python -m scripts.ingest_recalls     # download the Defensa del Consumidor sheet and clean it
python -m scripts.resolve_brands     # narrow multi-brand recalls with an LLM (cached)
```

## Known limitations

- **Synchronous streaming.** Each open stream holds a worker thread for most of its duration (the default pool is 40). Fine for a demo; real traffic would need an async rewrite, since the agent spends almost all its time waiting on external APIs.
- **No conversation memory.** Every question starts from scratch. The design is clear — store only question and final answer per turn, not the search payloads — but it isn't built.
- **Source quality is unfiltered.** The agent has cited dealership pages and Instagram reels as authorities.
- **Measurement bases can still get mixed.** It once compared one car's curb weight against another's gross vehicle weight. The comparability eval shows this is uncommon, not solved.
- **Credits live in memory.** Each API key's credits reset when the server restarts, and a credit is spent before the agent runs, so a request that fails on the provider's side still costs one.
- **Cold starts** on the free tier.

### Recalls data

- **It can't tell whether a specific car is affected.** Only 14 of the 486 recalls mention a model year, and recalls cover production ranges, not model years. The agent is instructed to send the user to check their VIN instead of guessing, and an eval case checks that it does.
- **The source itself is inconsistent.** Row 176 lists Audi models (A4, A6, A8, TT) under Peugeot Citroën Argentina as the publisher. Brand mapping is publisher-based, so that recall is tagged Peugeot/Citroën and an Audi lookup misses it.
- **Keyword search over-includes.** Every word of the keyword must appear as a substring, so short words match far too much: "Clase E" returns 43 of the 85 Mercedes-Benz recalls, and only 16 of them mention the E-Class (the lone "e" matches almost any text). The model filters the rows when it answers, but the result is noisier and uses more tokens than it should.
- **Some multi-brand recalls stay ambiguous.** When a publisher covers several brands (Peugeot Citroën, Toyota/Lexus, FCA, Volkswagen Argentina), an LLM narrows each recall down using the product text. In 25 rows the text isn't enough, so they keep every candidate brand on purpose — mostly Peugeot Citroën (14) and Toyota/Lexus (6). A Citroën lookup can return a Peugeot-only recall; over-including is safer than silently dropping a recall.