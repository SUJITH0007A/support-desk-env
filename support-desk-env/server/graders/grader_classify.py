from __future__ import annotations

from typing import Any, Dict

from server.models import TicketAction, TicketReward

URGENCY_ORDER = {"low": 0, "medium": 1, "high": 2, "critical": 3}


def grade_classification(ticket: Dict[str, Any], action: TicketAction) -> TicketReward:
    breakdown: Dict[str, float] = {"category": 0.0, "urgency": 0.0}
    feedback_parts: list[str] = []

    if action.category == ticket["correct_category"]:
        breakdown["category"] = 0.5
        feedback_parts.append("Category is correct.")
    else:
        feedback_parts.append(
            f"Category mismatch: expected {ticket['correct_category']}."
        )

    if action.urgency is not None:
        expected = URGENCY_ORDER[ticket["correct_urgency"]]
        got = URGENCY_ORDER[action.urgency]
        diff = abs(expected - got)
        if diff == 0:
            breakdown["urgency"] = 0.5
            feedback_parts.append("Urgency is correct.")
        elif diff == 1:
            breakdown["urgency"] = 0.25
            feedback_parts.append("Urgency close (within one level).")
        else:
            feedback_parts.append(
                f"Urgency too far: expected {ticket['correct_urgency']}."
            )
    else:
        feedback_parts.append("Urgency missing.")

    total = max(0.0, min(1.0, sum(breakdown.values())))
    return TicketReward(value=total, breakdown=breakdown, feedback=" ".join(feedback_parts))
