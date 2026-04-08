from __future__ import annotations

import json
import random
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

from server.graders.grader_classify import grade_classification
from server.graders.grader_resolve import grade_resolve_step
from server.graders.grader_respond import grade_response
from server.models import StepResult, TicketAction, TicketObservation, TicketReward
from server.tasks import task1_classify, task2_respond, task3_resolve


TASK_SPECS: Dict[str, Dict[str, Any]] = {
    "task1_classify": {"max_steps": task1_classify.MAX_STEPS, "instructions": task1_classify.INSTRUCTIONS},
    "task2_respond": {"max_steps": task2_respond.MAX_STEPS, "instructions": task2_respond.INSTRUCTIONS},
    "task3_resolve": {"max_steps": task3_resolve.MAX_STEPS, "instructions": task3_resolve.INSTRUCTIONS},
}


class SupportDeskEnv:
    def __init__(self, data_path: str | None = None) -> None:
        self._lock = threading.Lock()
        resolved = data_path or str(Path(__file__).parent / "data" / "tickets.json")
        with Path(resolved).open("r", encoding="utf-8") as handle:
            self._tickets: List[Dict[str, Any]] = json.load(handle)
        self._rng = random.Random(42)
        self._episode: Dict[str, Any] = {}

    def _select_ticket(self, task_name: str) -> Dict[str, Any]:
        if task_name == "task1_classify":
            pool = self._tickets[:10]
        else:
            pool = self._tickets
        return dict(self._rng.choice(pool))

    def _observation(self) -> TicketObservation:
        ticket = self._episode["ticket"]
        return TicketObservation(
            ticket_id=ticket["id"],
            subject=ticket["subject"],
            body=ticket["body"],
            customer_tier=ticket["customer_tier"],
            created_at=self._episode["created_at"],
            previous_actions=self._episode["action_history"],
            current_step=self._episode["current_step"],
            max_steps=self._episode["max_steps"],
            task_name=self._episode["task_name"],
            task_instructions=self._episode["task_instructions"],
            last_reward=self._episode["last_reward"],
            cumulative_reward=self._episode["cumulative_reward"],
        )

    def reset(self, task_name: str) -> TicketObservation:
        if task_name not in TASK_SPECS:
            raise ValueError(f"Unknown task '{task_name}'")

        with self._lock:
            ticket = self._select_ticket(task_name)
            self._episode = {
                "task_name": task_name,
                "task_instructions": TASK_SPECS[task_name]["instructions"],
                "max_steps": TASK_SPECS[task_name]["max_steps"],
                "current_step": 0,
                "created_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
                "ticket": ticket,
                "action_history": [],
                "last_reward": 0.0,
                "cumulative_reward": 0.0,
                "done": False,
                "has_response": False,
            }
            if task_name == "task2_respond":
                self._episode["action_history"].append(
                    f"prefill: category={ticket['correct_category']}, urgency={ticket['correct_urgency']}"
                )
            return self._observation()

    def _compute_common_adjustments(self, action: TicketAction, done: bool) -> Dict[str, float]:
        adjustments: Dict[str, float] = {}
        if (
            action.category is None
            and action.urgency is None
            and action.response_draft is None
            and action.escalate is None
            and action.resolve is None
            and action.reasoning is None
            and action.internal_summary is None
            and action.sla_breach is None
        ):
            adjustments["no_op_penalty"] = -0.15
        if action.reasoning is not None and len(action.reasoning.split()) > 20:
            adjustments["consistency_bonus"] = 0.05
        if done and self._episode["current_step"] < self._episode["max_steps"]:
            adjustments["speed_bonus"] = 0.05
        return adjustments

    def step(self, action: TicketAction) -> StepResult:
        with self._lock:
            if not self._episode:
                raise RuntimeError("Environment not initialized. Call reset first.")
            if self._episode["done"]:
                return StepResult(
                    observation=self._observation(),
                    reward=TicketReward(value=0.0, breakdown={"episode_done": 0.0}, feedback="Episode already done."),
                    done=True,
                    info={"task": self._episode["task_name"], "message": "Episode already finished"},
                )

            self._episode["current_step"] += 1
            step_index = self._episode["current_step"]
            task_name = self._episode["task_name"]
            ticket = self._episode["ticket"]

            if action.response_draft:
                self._episode["has_response"] = True

            if task_name == "task1_classify":
                reward = grade_classification(ticket, action)
            elif task_name == "task2_respond":
                reward = grade_response(ticket, action)
            else:
                reward = grade_resolve_step(ticket, action, step_index, self._episode["has_response"])

            done = False
            if self._episode["current_step"] >= self._episode["max_steps"]:
                done = True
            if action.resolve is True or action.escalate is True:
                done = True

            adjustments = self._compute_common_adjustments(action, done=done)
            adjusted_value = reward.value + sum(adjustments.values())
            final_value = max(0.0, min(1.0, adjusted_value))

            breakdown = dict(reward.breakdown)
            breakdown.update(adjustments)
            final_reward = TicketReward(
                value=final_value,
                breakdown=breakdown,
                feedback=reward.feedback,
            )

            self._episode["last_reward"] = final_reward.value
            self._episode["cumulative_reward"] = max(0.0, min(1.0, self._episode["cumulative_reward"] + final_reward.value))
            self._episode["done"] = done

            self._episode["action_history"].append(action.model_dump_json(exclude_none=True))

            obs = self._observation()
            return StepResult(
                observation=obs,
                reward=final_reward,
                done=done,
                info={"task": task_name, "step": step_index, "max_steps": self._episode["max_steps"]},
            )

    def state(self) -> Dict[str, Any]:
        with self._lock:
            if not self._episode:
                return {"initialized": False}
            ticket = self._episode["ticket"]
            return {
                "initialized": True,
                "task_name": self._episode["task_name"],
                "current_step": self._episode["current_step"],
                "max_steps": self._episode["max_steps"],
                "cumulative_reward": self._episode["cumulative_reward"],
                "done": self._episode["done"],
                "ticket_id": ticket["id"],
                "action_count": len(self._episode["action_history"]),
            }

