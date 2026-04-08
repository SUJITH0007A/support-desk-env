from __future__ import annotations

from typing import Dict, List, Literal, Optional, Union

from pydantic import BaseModel, ConfigDict, Field, field_validator


CATEGORY_VALUES = {"billing", "technical", "general", "abuse"}
URGENCY_VALUES = {"low", "medium", "high", "critical"}


class TicketObservation(BaseModel):
    ticket_id: str
    subject: str
    body: str
    customer_tier: Literal["free", "pro", "enterprise"]
    created_at: str
    previous_actions: List[str]
    current_step: int
    max_steps: int
    task_name: str
    task_instructions: str
    last_reward: float
    cumulative_reward: float

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "ticket_id": "TCK-1001",
                "subject": "Charged twice for March",
                "body": "I see two charges on my card for the same month.",
                "customer_tier": "pro",
                "created_at": "2026-04-01T09:15:00Z",
                "previous_actions": [],
                "current_step": 0,
                "max_steps": 3,
                "task_name": "task2_respond",
                "task_instructions": "Draft a professional response.",
                "last_reward": 0.0,
                "cumulative_reward": 0.0,
            }
        }
    )

    @field_validator("created_at")
    @classmethod
    def validate_iso_date(cls, value: str) -> str:
        if "T" not in value or not (value.endswith("Z") or "+" in value):
            raise ValueError("created_at must be ISO-like datetime string")
        return value

    @field_validator("current_step")
    @classmethod
    def validate_current_step(cls, value: int) -> int:
        if value < 0:
            raise ValueError("current_step must be >= 0")
        return value

    @field_validator("max_steps")
    @classmethod
    def validate_max_steps(cls, value: int) -> int:
        if value <= 0:
            raise ValueError("max_steps must be > 0")
        return value

    @field_validator("last_reward", "cumulative_reward")
    @classmethod
    def validate_rewards(cls, value: float) -> float:
        if value < 0.0:
            raise ValueError("reward values must be >= 0.0")
        return value


class TicketAction(BaseModel):
    category: Optional[str] = None
    urgency: Optional[str] = None
    response_draft: Optional[str] = None
    escalate: Optional[bool] = None
    resolve: Optional[bool] = None
    reasoning: Optional[str] = None
    internal_summary: Optional[str] = None
    sla_breach: Optional[bool] = None

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "category": "billing",
                "urgency": "high",
                "response_draft": "Hello, thanks for reporting this issue...",
                "escalate": False,
                "resolve": False,
                "reasoning": "The issue appears billing-related due to duplicate charge.",
                "internal_summary": "Investigated duplicate billing complaint.",
                "sla_breach": False,
            }
        }
    )

    @field_validator("category")
    @classmethod
    def validate_category(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return value
        lowered = value.strip().lower()
        if lowered not in CATEGORY_VALUES:
            raise ValueError(f"category must be one of {sorted(CATEGORY_VALUES)}")
        return lowered

    @field_validator("urgency")
    @classmethod
    def validate_urgency(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return value
        lowered = value.strip().lower()
        if lowered not in URGENCY_VALUES:
            raise ValueError(f"urgency must be one of {sorted(URGENCY_VALUES)}")
        return lowered

    @field_validator("response_draft", "reasoning", "internal_summary")
    @classmethod
    def normalize_text(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return value
        text = value.strip()
        if not text:
            return None
        return text


class TicketReward(BaseModel):
    value: float = Field(ge=0.0, le=1.0)
    breakdown: Dict[str, float]
    feedback: str

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "value": 0.72,
                "breakdown": {
                    "classification": 0.5,
                    "urgency": 0.2,
                    "reasoning_bonus": 0.02,
                },
                "feedback": "Category correct; urgency slightly under-estimated.",
            }
        }
    )

    @field_validator("breakdown")
    @classmethod
    def validate_breakdown(cls, value: Dict[str, float]) -> Dict[str, float]:
        for key, score in value.items():
            if not isinstance(key, str) or not key:
                raise ValueError("breakdown keys must be non-empty strings")
            if score < -1.0 or score > 1.0:
                raise ValueError("breakdown values must be in [-1.0, 1.0]")
        return value


class StepResult(BaseModel):
    observation: TicketObservation
    reward: TicketReward
    done: bool
    info: Dict[str, Union[str, int, float, bool]]

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "observation": TicketObservation.model_config["json_schema_extra"]["example"],
                "reward": TicketReward.model_config["json_schema_extra"]["example"],
                "done": False,
                "info": {"task": "task2_respond", "attempt": 1},
            }
        }
    )

