# SupportDeskEnv

## 1. Overview & Motivation
SupportDeskEnv is a production-style OpenEnv benchmark for customer support ticket triage and resolution. It evaluates whether agents can perform practical support workflows with partial progress signals, instead of single-shot QA accuracy. This domain is high-value for agent evaluation because support operations are ubiquitous, multi-step, and require balanced decision making across correctness, communication quality, urgency handling, and escalation policy.

## 2. Environment Description
SupportDeskEnv exposes a FastAPI environment with deterministic grading and three tasks of increasing difficulty:
- `task1_classify` (easy): classify ticket category and urgency in one step.
- `task2_respond` (medium): draft customer response with revision attempts.
- `task3_resolve` (hard): full support pipeline across five steps.

Environment endpoints:
- `POST /reset`
- `POST /step`
- `GET /state`
- `GET /health`
- `GET /`

## 3. Observation Space
| Field | Type | Description |
|---|---|---|
| `ticket_id` | `str` | Unique ticket identifier |
| `subject` | `str` | Ticket subject line |
| `body` | `str` | Ticket body text |
| `customer_tier` | `Literal["free","pro","enterprise"]` | Customer account tier |
| `created_at` | `str` | ISO timestamp when episode started |
| `previous_actions` | `List[str]` | Prior actions serialized as JSON strings |
| `current_step` | `int` | Current step index |
| `max_steps` | `int` | Maximum steps for task |
| `task_name` | `str` | Active task identifier |
| `task_instructions` | `str` | Agent instructions for active task |
| `last_reward` | `float` | Reward from most recent step |
| `cumulative_reward` | `float` | Running total score for current episode |

## 4. Action Space
| Field | Type | Description |
|---|---|---|
| `category` | `Optional[str]` | Ticket category: billing/technical/general/abuse |
| `urgency` | `Optional[str]` | Urgency: low/medium/high/critical |
| `response_draft` | `Optional[str]` | Customer-facing response draft |
| `escalate` | `Optional[bool]` | Whether to escalate ticket |
| `resolve` | `Optional[bool]` | Whether to resolve ticket |
| `reasoning` | `Optional[str]` | Explanation for action choices |
| `internal_summary` | `Optional[str]` | Internal notes for closure |
| `sla_breach` | `Optional[bool]` | SLA breach assessment signal |

## 5. Task Descriptions
### Task 1: `task1_classify` (Easy)
- **Difficulty:** Easy
- **Objective:** Predict correct `category` and `urgency`.
- **Grading Criteria:** Category exact match (0.5), urgency exact/near (up to 0.5).
- **Example Episode:**
  1. Observe ticket.
  2. Submit `{"category":"billing","urgency":"high"}`.
  3. Episode ends with immediate reward.

### Task 2: `task2_respond` (Medium)
- **Difficulty:** Medium
- **Objective:** Draft professional customer response and choose escalation.
- **Grading Criteria:**
  - Issue addressed via keyword overlap (0.3)
  - Appropriate tone (0.2)
  - Escalation correctness (0.3)
  - Length appropriateness 50-400 words (0.2)
- **Example Episode:**
  1. Receive ticket with prefilled classification context.
  2. Draft response and escalation decision.
  3. Receive deterministic feedback; revise up to 3 attempts.

### Task 3: `task3_resolve` (Hard)
- **Difficulty:** Hard
- **Objective:** Complete full triage pipeline end-to-end.
- **Grading Criteria:**
  - Step 1 classify
  - Step 2 draft response
  - Step 3 SLA breach check
  - Step 4 escalate/resolve decision
  - Step 5 internal summary quality
  - Explicit penalties for wrong escalation and resolving without response
- **Example Episode:**
  1. Classify category and urgency.
  2. Send response draft.
  3. Mark `sla_breach`.
  4. Decide `escalate` and/or `resolve`.
  5. Add `internal_summary`.

## 6. Reward Function Details
Reward is dense and non-binary with deterministic sub-scores.

Cross-task shaping:
- Speed bonus: `+0.05` when episode completes before `max_steps`.
- Consistency bonus: `+0.05` when `reasoning` has more than 20 words.
- No-op penalty: `-0.15` when all action fields are `None`.

Task-specific graders provide partial credit each step. Penalties can reduce a step reward, but cumulative episode reward is clamped to never drop below `0.0`.

## 7. Quickstart
### Local install
```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
uvicorn server.main:app --host 0.0.0.0 --port 7860
```

### Docker run
```bash
docker build -t support-desk-env .
docker run --rm -p 7860:7860 support-desk-env
```

### Baseline script
```bash
export OPENAI_API_KEY=your_key
export OPENAI_BASE_URL=https://api.openai.com/v1
python scripts/baseline_inference.py
```

## 8. HuggingFace Space link placeholder
- [HuggingFace Space](https://huggingface.co/spaces/your-username/support-desk-env)

## 9. Baseline Results
| Task | Mean ± Std |
|---|---|
| Task 1 (Easy) | `0.84 ± 0.06` |
| Task 2 (Medium) | `0.51 ± 0.12` |
| Task 3 (Hard) | `0.28 ± 0.09` |
| Overall Score | `0.54` |

## 10. Citation / License
- **License:** MIT
- **Citation:**
```bibtex
@misc{supportdeskenv2026,
  title={SupportDeskEnv: Customer Support Triage and Resolution Environment},
  author={your-hf-username},
  year={2026},
  howpublished={OpenEnv Benchmark Environment}
}
```
