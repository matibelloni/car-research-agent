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

The eval suites in [`evals/`](evals/) all run on hand-written datasets.

### Answer quality — `evals/answers.py`

General car questions scored by an LLM judge on two independent criteria:

- **Citations** — every concrete data point (figures, including those inside advice, and recall facts) must have its source right after it: a URL for web facts, "Defensa del Consumidor" plus publication date for recalls; advice without figures, numbers that restate the question, and unit conversions of a sourced figure need none (judged by Sonnet 5.5)
- **Comparability** — compared figures must share the same unit and measurement basis (judged by Haiku 4.5)

| Metric | Baseline | Per-figure citation rule |
|---|---|---|
| Citations | 3.53 / 5 (3.40 – 3.80) | 4.20 / 5 (3.80 – 4.60) |
| Comparability | 3.87 / 5 (3.80 – 4.00) | 4.47 / 5 (4.40 – 4.60) |

*3 runs × 5 questions each, measured on 2026-10-05, both columns with the same judges (citations prompt including the restated-question and variant-name rules, which calibration showed the earlier prompt scored wrong). The only change between columns is the agent's system prompt: it now spells out that every bullet with a figure ends with its own URL and that summary figures and ranges need a source too. The citation ranges only touch at 3.80, and the Corolla question went from 5/3/3 to 5/5/5. Comparability's judge and prompt didn't change, so its rise comes from the agent or from web results drifting between runs. A citation rule shouldn't move it, so it isn't claimed as an effect. The recalls suite still passes every case after the change.*

*Earlier figures, each with a judge that has since been replaced, so not comparable: citations 3.87 (3.60 – 4.20) with a Haiku-only judge that was too lenient, then 3.40 (2.60 – 4.20) with Sonnet before the restated-question rule.*

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
- **Behavior** (LLM judge, some cases) — no invented recalls when there are none; for a brand the data doesn't cover, saying so and marking any recall found on the web as from another market; and never claiming the user's specific car is affected

Latest run (12 cases × 3 runs, 2026-10-05): lookup **36/36**, total **30/30**, behavior **9/9**. Cases where a check doesn't apply (e.g. no total for an unknown brand) aren't counted.

### Owner's manual, end to end — `evals/manual.py`

Questions about a manual, run through the whole agent:

- **Route** (deterministic) — manual questions cite the expected page label and make no web search; price and owner-opinion questions, which the manual can't answer, make at least one web search
- **Facts** (LLM judge, Sonnet 5.5) — every fact of the case is stated, with the same figures. Facts are copied from the chunks' text, so web cases have none

| Case | Route | All facts | Baseline route | Baseline facts |
|---|---|---|---|---|
| Cronos — accessory belts (G-14) | 5/5 | 0/5 | 3/3 | 1/3 |
| Cronos — taxi use (G-18 / G-14) | 5/5 | 0/5 | 3/3 | 1/3 |
| Cronos — brake fluid (G-14) | 5/5 | 1/5 | 3/3 | 1/3 |
| Cronos — 10,000 km service price (web) | 5/5 | n/a | 3/3 | n/a |
| Cronos — owners on highway consumption (web) | 2/3, 3/3 | n/a | 3/3 | n/a |

*Latest: Haiku 4.5 as the agent, 5 runs per case, 2026-10-21, 153,446 tokens for the four cases above, facts re-judged by Sonnet 5.5; highway consumption is the two earlier 3-run passes. Baseline: 3 runs per case, 2026-10-18, 103,562 tokens, judged by Haiku 4.5. The taxi-use case is stricter than in the baseline: its second fact now names the checks (brake pads, engine oil and filter, air filter) instead of "checks such as", and the judge asks for every named item. Re-judged that way, the run before had 0/3 too.*

The main interval is stated in every run. The second fact is what goes missing, and each case misses it for a different reason:

- **Accessory belts — a Haiku limit, not the data.** The maintenance plan now attaches each footnote to the tasks that carry its marker (`attach_footnotes`), so "Sustitución de las correas (\*): cada 48 meses…" is followed by the (\*) note that halves it under severe use. Haiku still quotes the (\*\*) note, the dusty-road check of the *visual inspection* row, in 8 of 8 runs. Sonnet 5.5 as the agent states the halving in 3 of 3. A Haiku judge passed only 1 of those, so the facts judge is now Sonnet, calibrated on the two answers it failed.
- **Taxi use — retrieval.** The agent never sends "remis" or "taxi" to `search_manual`: it rewrites the question as "intervalos de mantenimiento service". That query returns G-11 and G-14, not G-18, where the list of checks is. The only run that got G-18 (a query with "uso intensivo remis") stated the list and missed G-14's taxi footnote instead.
- **Brake fluid — completeness.** The clutch fluid is in the same line as the interval; the agent drops it in 4 of 5 runs.

Tried and dropped: an instruction in the tool result instead of the system prompt, since it only applies when the agent has the manual's sections in front of it. "Answer only from these sections…" made the agent skip the web search on the price question. A softer version ("if these sections answer the question, include every condition…; if they don't, search the web") got brake fluid to 3/3 once, but still lost the web search on the price question in 1 of 3 runs and made the agent apply the severe-use halving to the brake fluid, which has no marker. Attaching the footnotes costs some retrieval: G-14 grew from 5,397 to 8,983 chars, and rerank MRR went from 0.97 to 0.95 (fuel filter from rank 1 to 2, still in the top 3).

Known limitations:

- **Answers that need two different sections.** `search_manual` returns the top 3, and the taxi-use answer needs both G-14 (taxi counts as severe use) and G-18 (the checks done more often). With the agent's rewritten queries they never come back together. Next experiment: raise `TOP_K`, and measure what it costs in tokens and in irrelevant sections.
- **A run without tools.** On the owner-opinions question, 1 of 6 runs answered "I have no tools to access owner opinions" without calling any tool, `search_web` included. It wasn't the tool result's wording: the run never got one.

### Maintenance tables — `scripts/extract_maintenance_llm.py`

The maintenance plan is a grid of marks, and its meaning is in the columns, so plain PDF text loses it. The first extractor reads Fiat's grid with pdfplumber and only works for that layout. The generic one sends the plan's pages to Claude as a PDF, asks for each row's structure (every revision / some revisions / its own interval / footnote), and builds the same sentences in code. Checked against the grid extractor on the Cronos (27 rows, footnotes compared as one text):

| Model | Rows identical | Runs | Tokens per run |
|---|---|---|---|
| Haiku 4.5 | 19/27 — the same 8 rows' revision columns wrong in every run | 3 | ~11,400 |
| Sonnet 5.5 | **27/27** | 3 | ~14,400 |

Haiku's errors are all shifted or skipped columns (e.g. 1ª/3ª/7ª instead of 2ª/4ª/8ª), never text, so it reads the table but not its grid. Sonnet is the default.

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
python -m evals.manual                   # owner's manual suite, end to end
python -m evals.manual --calibrate
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
- **Keyword search is literal.** Each keyword word must appear as a whole word, in singular or plural. This used to be a substring match, where "Clase E" returned 43 of the 85 Mercedes-Benz recalls — the lone "e" matched almost any text — and "Gol" returned Golf recalls; whole-word matching brought "Clase E" down to exactly the 16 E-Class recalls. It still misses other word forms and spellings: "freno" skips 5 recalls that only say "frenado", and "C300" doesn't match "C 300".
- **Some multi-brand recalls stay ambiguous.** When a publisher covers several brands (Peugeot Citroën, Toyota/Lexus, FCA, Volkswagen Argentina), an LLM narrows each recall down using the product text. In 25 rows the text isn't enough, so they keep every candidate brand on purpose — mostly Peugeot Citroën (14) and Toyota/Lexus (6). A Citroën lookup can return a Peugeot-only recall; over-including is safer than silently dropping a recall.