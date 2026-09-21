from fastapi import FastAPI, HTTPException, Header, responses
from pydantic import BaseModel
from agent import run_agent, run_agent_stream, AgentResult
import os
import json

app = FastAPI(title="Auto Research Agent")

API_KEY = os.environ["APP_API_KEY"]


class AskRequest(BaseModel):
    question: str


@app.post("/ask", response_model=AgentResult)
def ask(request: AskRequest, x_api_key: str = Header(None)) -> AgentResult:
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API KEY")
    return run_agent(request.question)


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/ask/stream")
def ask_stream(request: AskRequest, x_api_key: str = Header(None)):
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API KEY")

    def event_stream():
        for event in run_agent_stream(request.question):
            yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"

    return responses.StreamingResponse(event_stream(), media_type="text/event-stream")


@app.get("/")
def index():
    return responses.FileResponse("static/index.html")
