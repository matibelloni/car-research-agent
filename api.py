from fastapi import FastAPI
from pydantic import BaseModel
from agent import run_agent

app = FastAPI(title="Auto Research Agent")


class AskRequest(BaseModel):
    question: str


class AskResponse(BaseModel):
    answer: str | None
    error: str | None
    tokens: int
    searches: int


@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest) -> AskResponse:
    result = run_agent(request.question)
    return AskResponse(**result)


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}
