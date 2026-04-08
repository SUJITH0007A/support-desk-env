from __future__ import annotations

import json
import os
import statistics
from typing import Dict, List, Tuple

import httpx
from openai import OpenAI

BASE_URL = os.getenv("SUPPORT_DESK_BASE_URL", "http://127.0.0.1:7860")
MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

TASKS: List[Tuple[str, str]] = [
    ("task1_classify", "Task 1 (Easy)"),
    ("task2_respond", "Task 2 (Medium)"),
    ("task3_resolve", "Task 3 (Hard)"),
]


def build_prompt(obs: Dict[str, object]) -> str:
    return (
        "You are a support agent in a deterministic environment. Return ONLY a JSON object "
        "matching the action schema.\\n"
        f"Task: {obs['task_name']}\\n"
        f"Instructions: {obs['task_instructions']}\\n"
        f"Ticket Subject: {obs['subject']}\\n"
        f"Ticket Body: {obs['body']}\\n"
        f"Tier: {obs['customer_tier']}\\n"
        f"Current Step: {obs['current_step']}/{obs['max_steps']}\\n"
        f"Previous Actions: {obs['previous_actions']}\\n"
        "Action schema keys: category, urgency, response_draft, escalate, resolve, reasoning, internal_summary, sla_breach."
    )


def get_llm_action(client: OpenAI, obs: Dict[str, object]) -> Dict[str, object]:
    completion = client.chat.completions.create(
        model=MODEL,
        temperature=0.2,
        messages=[
            {
                "role": "system",
                "content": "Output valid JSON only. No markdown. No extra keys.",
            },
            {
                "role": "user",
                "content": build_prompt(obs),
            },
        ],
    )
    raw = completion.choices[0].message.content or "{}"
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {"reasoning": raw[:400]}


def run_episode(http: httpx.Client, llm: OpenAI, task_name: str) -> float:
    obs = http.post(f"{BASE_URL}/reset", json={"task": task_name}, timeout=30.0).json()
    done = False
    final_score = 0.0

    while not done:
        action = get_llm_action(llm, obs)
        step_result = http.post(f"{BASE_URL}/step", json=action, timeout=30.0).json()
        obs = step_result["observation"]
        done = bool(step_result["done"])
        final_score = float(obs["cumulative_reward"])

    return max(0.0, min(1.0, final_score))


def main() -> None:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is required")

    base = os.getenv("OPENAI_BASE_URL")
    llm = OpenAI(api_key=api_key, base_url=base)

    task_scores: Dict[str, List[float]] = {}
    with httpx.Client() as http:
        for task_name, label in TASKS:
            scores = [run_episode(http, llm, task_name) for _ in range(5)]
            task_scores[label] = scores

    all_scores = [score for values in task_scores.values() for score in values]

    for label in ["Task 1 (Easy)", "Task 2 (Medium)", "Task 3 (Hard)"]:
        vals = task_scores[label]
        mean = statistics.fmean(vals)
        std = statistics.pstdev(vals)
        print(f"{label}:   {mean:.2f} ± {std:.2f}")

    overall = statistics.fmean(all_scores)
    print(f"Overall Score:   {overall:.2f}")


if __name__ == "__main__":
    main()
