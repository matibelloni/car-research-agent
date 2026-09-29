from fastapi import FastAPI, HTTPException, Header, responses, Depends
from pydantic import BaseModel
from agent import run_agent, run_agent_stream, AgentResult
import os
import json

app = FastAPI(title="Auto Research Agent")

ADMIN_API_KEY = os.environ.get("ADMIN_API_KEY")
API_KEY_CREDITS = {os.environ["APP_API_KEY"]: 5}

API_KEY = os.environ["APP_API_KEY"]

ERROR_STATUS = {
    "rate_limited": 429,
    "provider_unavailable": 503,
    "provider_unreachable": 503,
    "provider_error": 502,
}


class AskRequest(BaseModel):
    question: str


def verify_api_key(x_api_key: str = Header(None)) -> str:
    if x_api_key == ADMIN_API_KEY:
        return x_api_key
    credits = API_KEY_CREDITS.get(x_api_key, 0)
    if credits <= 0:
        raise HTTPException(status_code=401, detail="Invalid API KEY, or no credits")
    return x_api_key


def consume_credit(x_api_key: str) -> None:
    if x_api_key != ADMIN_API_KEY:
        API_KEY_CREDITS[x_api_key] -= 1


@app.post("/ask", response_model=AgentResult)
def ask(request: AskRequest, x_api_key: str = Depends(verify_api_key)) -> AgentResult:
    consume_credit(x_api_key)
    result = run_agent(request.question)
    if result.error in ERROR_STATUS:
        raise HTTPException(status_code=ERROR_STATUS[result.error], detail=result.error)
    return result


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/ask/stream")
def ask_stream(request: AskRequest, x_api_key: str = Depends(verify_api_key)):
    consume_credit(x_api_key)

    def event_stream():
        for event in run_agent_stream(request.question):
            yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"

    return responses.StreamingResponse(event_stream(), media_type="text/event-stream")


@app.get("/")
def index():
    return responses.FileResponse("static/index.html")
