from __future__ import annotations

from server.env import SupportDeskEnv
from server.models import TicketAction


def test_reset_returns_valid_observation() -> None:
    env = SupportDeskEnv()
    obs = env.reset("task1_classify")
    assert obs.ticket_id.startswith("TCK-")
    assert obs.task_name == "task1_classify"
    assert obs.max_steps == 1


def test_step_changes_state_and_reward_is_bounded() -> None:
    env = SupportDeskEnv()
    env.reset("task1_classify")
    action = TicketAction(category="billing", urgency="medium", reasoning="This is billing related due to invoice issue details.")
    result = env.step(action)
    assert result.observation.current_step == 1
    assert 0.0 <= result.reward.value <= 1.0
    assert 0.0 <= result.observation.cumulative_reward <= 1.0


def test_done_set_when_budget_exceeded() -> None:
    env = SupportDeskEnv()
    env.reset("task2_respond")
    action = TicketAction(response_draft="Hello team, thank you for reaching out. We are reviewing your issue in detail and will share next steps soon. Regards, Support team.", escalate=False)
    r1 = env.step(action)
    assert not r1.done
    r2 = env.step(action)
    assert not r2.done
    r3 = env.step(action)
    assert r3.done


def test_done_set_on_escalate_or_resolve() -> None:
    env = SupportDeskEnv()
    env.reset("task3_resolve")
    action = TicketAction(escalate=True, reasoning="Escalating due to potential major impact and strict SLA constraints requiring specialist intervention.")
    result = env.step(action)
    assert result.done


def test_grader_scores_in_range() -> None:
    env = SupportDeskEnv()
    env.reset("task3_resolve")
    a1 = env.step(TicketAction(category="technical", urgency="high"))
    assert 0.0 <= a1.reward.value <= 1.0
    if not a1.done:
        a2 = env.step(
            TicketAction(
                response_draft=(
                    "Hello, thank you for reporting this issue. We reviewed the details from your message "
                    "and started investigation with our engineering team. We will keep you updated and "
                    "share a clear timeline for the fix. Best regards, Support Team"
                ),
                escalate=True,
            )
        )
        assert 0.0 <= a2.reward.value <= 1.0
