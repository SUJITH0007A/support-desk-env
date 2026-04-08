from __future__ import annotations

from typing import Dict

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, ConfigDict

from server.env import SupportDeskEnv
from server.models import StepResult, TicketAction, TicketObservation


class ResetRequest(BaseModel):
    task: str

    model_config = ConfigDict(json_schema_extra={"example": {"task": "task1_classify"}})


class HealthResponse(BaseModel):
    status: str


class RootResponse(BaseModel):
    name: str
    version: str
    description: str
    endpoints: Dict[str, str]


app = FastAPI(title="SupportDeskEnv", version="1.0.0")
app.state.env = SupportDeskEnv()


@app.get("/", response_model=RootResponse)
def root() -> RootResponse:
    return RootResponse(
        name="support-desk-env",
        version="1.0.0",
        description="Customer support ticket triage and resolution environment",
        endpoints={"reset": "/reset", "step": "/step", "state": "/state", "health": "/health"},
    )


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok")


@app.post("/reset", response_model=TicketObservation)
def reset(payload: ResetRequest, request: Request) -> TicketObservation:
    try:
        return request.app.state.env.reset(task_name=payload.task)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/step", response_model=StepResult)
def step(action: TicketAction, request: Request) -> StepResult:
    try:
        return request.app.state.env.step(action=action)
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/state", response_model=dict)
def state(request: Request) -> dict:
    return request.app.state.env.state()

