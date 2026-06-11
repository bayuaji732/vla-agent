from __future__ import annotations

import json
import logging
import re
from textwrap import dedent

from openai import AsyncOpenAI

from agent.config import get_settings
from agent.models import Plan, PlanStep

logger = logging.getLogger(__name__)
cfg = get_settings()

# Regex to detect if the task already contains a concrete URL or well-known domain.
# If it does, the planner must navigate there directly — never route through Google.
_DIRECT_URL_RE = re.compile(
    r"(https?://\S+)|"                          # explicit http(s) URL
    r"\b("
    r"wikipedia\.org|github\.com|youtube\.com|"
    r"reddit\.com|twitter\.com|x\.com|"
    r"linkedin\.com|stackoverflow\.com|"
    r"news\.ycombinator\.com|hckrnews\.com|"
    r"google\.com|bing\.com|duckduckgo\.com|"
    r"amazon\.com|ebay\.com|imdb\.com|"
    r"example\.com|w3schools\.com|mdn\."
    r")\b",
    re.IGNORECASE,
)

_SYSTEM_PROMPT = dedent("""
You are a strategic planning agent. Decompose the user task into concrete, atomic browser steps.

════════════════════════════════════════════════════════
RULE 1 — DIRECT NAVIGATION (highest priority)
════════════════════════════════════════════════════════
If the task names a specific website or URL (e.g. "go to wikipedia.org",
"open github.com/trending", "visit https://example.com"), generate a step:
  description: "Navigate directly to <full URL>"
  expected_outcome: "The page loads successfully"

NEVER route a direct-URL task through Google Search.
NEVER add a Google search step before a direct navigation.
Bad ✗:  navigate to https://www.google.com/search?q=wikipedia
Good ✓: navigate to https://en.wikipedia.org/wiki/Main_Page

════════════════════════════════════════════════════════
RULE 2 — WEB SEARCH (only when no URL is known)
════════════════════════════════════════════════════════
When the task requires finding something and NO specific URL is given,
use a direct search URL — never navigate to a search engine homepage:
  https://www.google.com/search?q=your+search+terms   (replace spaces with +)
Follow with a separate step to read the results.

════════════════════════════════════════════════════════
RULE 3 — READING / EXTRACTION
════════════════════════════════════════════════════════
After navigating to a page, use action type "done" with ALL extracted
information written in full in the rationale. Never say "visible on screen".

════════════════════════════════════════════════════════
RULE 4 — FINAL COMPILE STEP (always required)
════════════════════════════════════════════════════════
Always end with:
  description: "Compile all gathered information into a complete structured final answer."
  expected_outcome: "Full answer written completely in the rationale field."

════════════════════════════════════════════════════════
GENERAL
════════════════════════════════════════════════════════
- Maximum {max_steps} steps. Each step is atomic.
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
            memory_block = "\n\nRelevant past experience:\n" + "\n".join(
                f"- {s}" for s in memory_snippets
            )

        # Give the LLM an explicit hint when a direct URL/domain is detected,
        # so it never second-guesses and routes through Google.
        direct_hint = ""
        if _DIRECT_URL_RE.search(task):
            direct_hint = (
                "\n\n⚠️  IMPORTANT: This task contains a specific URL or website name. "
                "You MUST navigate there directly. Do NOT use Google Search."
            )

        user_prompt = (
            f"Task: {task}\n"
            f"Current context: {context or 'none'}"
            f"{direct_hint}"
            f"{memory_block}"
        )
        logger.info("Planner creating plan for: %s", task[:80])

        response = await self._client.chat.completions.create(
            model=cfg.planner_model,
            temperature=cfg.planner_temperature,
            response_format={"type": "json_object"},
            messages=[
                {
                    "role": "system",
                    "content": _SYSTEM_PROMPT.format(max_steps=cfg.max_plan_steps),
                },
                {"role": "user", "content": user_prompt},
            ],
        )

        data = json.loads(response.choices[0].message.content)
        steps = [
            PlanStep(
                index=s["index"],
                description=s["description"],
                expected_outcome=s["expected_outcome"],
            )
            for s in data["steps"]
        ]
        plan = Plan(goal=data["goal"], steps=steps)
        logger.info("Plan created with %d steps", len(steps))
        return plan

    async def replan(
        self,
        original_plan: Plan,
        completed_indices: list[int],
        failure_description: str,
        current_observation: str,
    ) -> Plan:
        completed_steps = [
            s for s in original_plan.steps if s.index in completed_indices
        ]
        next_index = (completed_steps[-1].index if completed_steps else 0) + 1

        prompt = dedent(f"""
            Original goal: {original_plan.goal}
            Completed: {json.dumps(
                [s.model_dump(include={{"index", "description"}}) for s in completed_steps],
                indent=2,
            )}
            Failed: {failure_description}
            Page: {current_observation}
            Generate remaining steps starting from index {next_index}.
            Follow RULE 1 (direct navigation) and RULE 2 (search) above.
            End with a compile step.
        """).strip()

        response = await self._client.chat.completions.create(
            model=cfg.planner_model,
            temperature=cfg.planner_temperature,
            response_format={"type": "json_object"},
            messages=[
                {
                    "role": "system",
                    "content": _SYSTEM_PROMPT.format(max_steps=cfg.max_plan_steps),
                },
                {"role": "user", "content": prompt},
            ],
        )

        data = json.loads(response.choices[0].message.content)
        new_steps = [
            PlanStep(
                index=s["index"],
                description=s["description"],
                expected_outcome=s["expected_outcome"],
            )
            for s in data["steps"]
        ]
        return Plan(goal=original_plan.goal, steps=completed_steps + new_steps)