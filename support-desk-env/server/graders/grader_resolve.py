from __future__ import annotations

from typing import Any, Dict

from server.graders.grader_classify import grade_classification
from server.graders.grader_respond import grade_response
from server.models import TicketAction, TicketReward


def grade_resolve_step(
    ticket: Dict[str, Any], action: TicketAction, step_index: int, has_response: bool
) -> TicketReward:
    """Deterministic per-step grading for task3 pipeline."""
    breakdown: Dict[str, float] = {}
    feedback: list[str] = []

    if step_index == 1:
        cls_reward = grade_classification(ticket, action)
        value = 0.25 * cls_reward.value
        breakdown = {f"classify_{k}": v * 0.25 for k, v in cls_reward.breakdown.items()}
        feedback.append("Step 1 classification evaluated.")
        return TicketReward(value=value, breakdown=breakdown, feedback=" ".join(feedback))

    if step_index == 2:
        resp_reward = grade_response(ticket, action)
        value = 0.30 * resp_reward.value
        breakdown = {f"respond_{k}": v * 0.30 for k, v in resp_reward.breakdown.items()}
        feedback.append("Step 2 response evaluated.")
        return TicketReward(value=value, breakdown=breakdown, feedback=" ".join(feedback))

    if step_index == 3:
        expected_breach = ticket["sla_hours"] <= 8
        given = action.sla_breach
        score = 0.0
        if given is None:
            feedback.append("SLA breach check missing.")
        elif given == expected_breach:
            score = 0.2
            feedback.append("SLA breach assessment correct.")
        else:
            feedback.append("SLA breach assessment incorrect.")
        return TicketReward(
            value=score,
            breakdown={"sla_check": score},
            feedback=" ".join(feedback),
        )

    if step_index == 4:
        expected_escalate = bool(ticket["is_escalation_required"])
        value = 0.0
        if action.escalate is not None:
            if action.escalate == expected_escalate:
                value += 0.15
                feedback.append("Escalation decision correct.")
            elif action.escalate and not expected_escalate:
                value -= 0.2
                feedback.append("Unnecessary escalation penalty applied.")
            else:
                feedback.append("Escalation decision incorrect.")
        if action.resolve is True and has_response:
            value += 0.10
            feedback.append("Resolution marked with prior response.")
        elif action.resolve is True and not has_response:
            value -= 0.3
            feedback.append("Resolved without responding penalty applied.")
        return TicketReward(
            value=max(0.0, min(1.0, value)),
            breakdown={"decision": value},
            feedback=" ".join(feedback),
        )

    if step_index == 5:
        summary = (action.internal_summary or "").strip()
        words = len(summary.split())
        score = 0.0
        if words >= 20:
            score = 0.15
            feedback.append("Internal summary is sufficiently detailed.")
        elif words >= 8:
            score = 0.07
            feedback.append("Internal summary partially detailed.")
        else:
            feedback.append("Internal summary too short.")
        return TicketReward(
            value=score,
            breakdown={"internal_summary": score},
            feedback=" ".join(feedback),
        )

    return TicketReward(value=0.0, breakdown={"invalid_step": 0.0}, feedback="Invalid step.")
