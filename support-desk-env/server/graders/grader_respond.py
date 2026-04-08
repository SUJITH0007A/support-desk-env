from __future__ import annotations

from typing import Any, Dict

from server.models import TicketAction, TicketReward

RUDE_WORDS = {"idiot", "stupid", "dumb", "useless", "shut up"}
GREETING_WORDS = {"hello", "hi", "dear", "thanks", "thank you"}
SIGNOFF_WORDS = {"regards", "best", "sincerely", "thank you", "thanks"}


def _contains_any(text: str, words: set[str]) -> bool:
    lowered = text.lower()
    return any(word in lowered for word in words)


def grade_response(ticket: Dict[str, Any], action: TicketAction) -> TicketReward:
    text = (action.response_draft or "").strip()
    words = [w for w in text.split() if w]
    word_count = len(words)

    breakdown: Dict[str, float] = {
        "issue_addressed": 0.0,
        "tone": 0.0,
        "escalation": 0.0,
        "length": 0.0,
    }
    feedback: list[str] = []

    overlap_tokens = {
        token.lower().strip(".,!?:;()[]{}")
        for token in (ticket["subject"] + " " + ticket["body"]).split()
        if len(token) > 4
    }
    response_tokens = {
        token.lower().strip(".,!?:;()[]{}") for token in words if len(token) > 4
    }
    overlap_ratio = 0.0
    if overlap_tokens:
        overlap_ratio = len(overlap_tokens & response_tokens) / len(overlap_tokens)

    if overlap_ratio >= 0.12:
        breakdown["issue_addressed"] = 0.3
        feedback.append("Response addresses the issue details.")
    elif overlap_ratio >= 0.06:
        breakdown["issue_addressed"] = 0.15
        feedback.append("Response partially addresses issue details.")
    else:
        feedback.append("Response should reference ticket specifics more directly.")

    has_greeting = _contains_any(text, GREETING_WORDS)
    has_signoff = _contains_any(text, SIGNOFF_WORDS)
    has_rude = _contains_any(text, RUDE_WORDS)
    if has_greeting and has_signoff and not has_rude:
        breakdown["tone"] = 0.2
        feedback.append("Tone is professional and complete.")
    elif not has_rude and (has_greeting or has_signoff):
        breakdown["tone"] = 0.1
        feedback.append("Tone is acceptable but missing greeting or sign-off.")
    else:
        feedback.append("Tone needs improvement.")

    expected_escalate = bool(ticket["is_escalation_required"])
    if action.escalate is None:
        feedback.append("Escalation decision missing.")
    elif action.escalate == expected_escalate:
        breakdown["escalation"] = 0.3
        feedback.append("Escalation decision is correct.")
    else:
        feedback.append(
            f"Escalation decision incorrect; expected escalate={expected_escalate}."
        )

    if 50 <= word_count <= 400:
        breakdown["length"] = 0.2
        feedback.append("Length is appropriate.")
    elif 30 <= word_count < 50 or 401 <= word_count <= 500:
        breakdown["length"] = 0.1
        feedback.append("Length is slightly off target.")
    else:
        feedback.append("Length should be between 50 and 400 words.")

    total = max(0.0, min(1.0, sum(breakdown.values())))
    return TicketReward(value=total, breakdown=breakdown, feedback=" ".join(feedback))
