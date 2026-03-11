from __future__ import annotations

import json
import logging
from textwrap import dedent
from urllib.parse import quote_plus

from openai import AsyncOpenAI

from agent.config import get_settings
from agent.models import Plan, PlanStep

logger = logging.getLogger(__name__)
cfg = get_settings()

_SYSTEM_PROMPT = dedent("""
You are a strategic planning agent. Decompose the user task into concrete browser steps.

Rules:
- Maximum {max_steps} steps. Each step is atomic with one action and one expected outcome.

- FOR WEB SEARCHES: Use a direct search URL — never navigate to a homepage and
  interact with a search box. Use this format:
    https://www.google.com/search?q=your+search+terms
  Replace spaces with +. Put the full URL in the navigate step's description.
  Then use a SEPARATE step to read the results.

- FOR READING: After navigating to a search result page or article, use action
  type "done" and write ALL extracted information completely in the rationale field.
  Never say "visible on screen" — write out the actual content.

- ALWAYS add a final "compile" step:
    description: "Compile all gathered information into a complete structured final answer."
    expected_outcome: "Full answer written completely in the rationale field."

- Output ONLY valid JSON, no markdown.

Output schema:
{{
  "goal": "<restatement of task>",
  "steps": [
    {{
      "index": 1,
      "description": "<imperative action>",
      "expected_outcome": "<what the screen shows after success>"
    }}
  ]
}}
""").strip()


class Planner:
    def __init__(self) -> None:
        self._client = AsyncOpenAI(api_key=cfg.openai_api_key)

    async def create_plan(
        self,
        task: str,
        context: str = "",
        memory_snippets: list[str] | None = None,
    ) -> Plan:
        memory_block = ""
        if memory_snippets:
            memory_block = "\n\nRelevant past experience:\n" + "\n".join(f"- {s}" for s in memory_snippets)

        user_prompt = f"Task: {task}\nCurrent context: {context or 'none'}{memory_block}"
        logger.info("Planner creating plan for: %s", task[:80])

        response = await self._client.chat.completions.create(
            model=cfg.planner_model,
            temperature=cfg.planner_temperature,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT.format(max_steps=cfg.max_plan_steps)},
                {"role": "user", "content": user_prompt},
            ],
        )

        data = json.loads(response.choices[0].message.content)
        steps = [
            PlanStep(index=s["index"], description=s["description"], expected_outcome=s["expected_outcome"])
            for s in data["steps"]
        ]
        plan = Plan(goal=data["goal"], steps=steps)
        logger.info("Plan created with %d steps", len(steps))
        return plan

    async def replan(self, original_plan: Plan, completed_indices: list[int], failure_description: str, current_observation: str) -> Plan:
        completed_steps = [s for s in original_plan.steps if s.index in completed_indices]
        prompt = dedent(f"""
            Original goal: {original_plan.goal}
            Completed: {json.dumps([s.model_dump(include={{'index','description'}}) for s in completed_steps], indent=2)}
            Failed: {failure_description}
            Page: {current_observation}
            Generate remaining steps from index {(completed_steps[-1].index if completed_steps else 0) + 1}.
            Use direct Google search URLs. End with a compile step.
        """).strip()

        response = await self._client.chat.completions.create(
            model=cfg.planner_model,
            temperature=cfg.planner_temperature,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT.format(max_steps=cfg.max_plan_steps)},
                {"role": "user", "content": prompt},
            ],
        )

        data = json.loads(response.choices[0].message.content)
        new_steps = [PlanStep(index=s["index"], description=s["description"], expected_outcome=s["expected_outcome"]) for s in data["steps"]]
        return Plan(goal=original_plan.goal, steps=completed_steps + new_steps)