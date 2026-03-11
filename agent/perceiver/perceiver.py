from __future__ import annotations

import json
import logging
from textwrap import dedent

from openai import AsyncOpenAI

from agent.config import get_settings
from agent.models import Action, ActionType, MarkedElement, Observation
from agent.perceiver.som import mark_screenshot

logger = logging.getLogger(__name__)
cfg = get_settings()

_SYSTEM_PROMPT = dedent("""
You are a Vision-Language Agent Perceiver. You receive a screenshot with numbered
Set-of-Marks labels on interactable elements, plus the current step and history.

Your job: decide the SINGLE best next action to complete the step.

CRITICAL RULES:

1. NAVIGATION: If the step says to "navigate to a URL", always use action type
   "navigate" with the full URL in the "url" field. Never click address bars.

2. SEARCH BOXES: Never interact with search engine homepage search boxes.
   The planner will always use direct search URLs like
   https://www.bing.com/search?q=your+query — use "navigate" for those.

3. READING RESULTS: If the step says to "read", "extract", or "note" information
   from the current page, use action type "done" and write the COMPLETE extracted
   information in the "rationale" field. Never say "visible on screen" — always
   write out the actual content you can read.

4. COMPILE STEP: If the step says "compile all information into a final answer",
   write the FULL structured answer in "rationale" using all prior gathered data
   from the action_history. Be complete and detailed.

5. DONE vs FAIL: Use "done" when the step's expected outcome is achieved.
   Use "fail" only if the step is truly impossible.

6. ONE ACTION: Output exactly one action per response.

Output ONLY valid JSON:
{
  "type": "<navigate|click|type|scroll|hover|key_press|wait|done|fail>",
  "element_id": <int or null>,
  "text": "<string or null>",
  "url": "<full URL or null>",
  "direction": "<up|down or null>",
  "key": "<key name or null>",
  "rationale": "<complete extracted answer for done; one-sentence reason for actions>"
}
""").strip()


class Perceiver:
    def __init__(self) -> None:
        self._client = AsyncOpenAI(api_key=cfg.openai_api_key)

    async def perceive(
        self,
        observation: Observation,
        step_description: str,
        expected_outcome: str,
        action_history: list[str],
        reflection: str = "",
    ) -> Action:
        marked_b64 = mark_screenshot(
            observation.screenshot_b64,
            observation.marked_elements,
        )

        elements_index = self._format_elements(observation.marked_elements)
        history_block = "\n".join(action_history[-8:]) or "None"
        reflection_block = f"\nReflection on last failure:\n{reflection}" if reflection else ""

        user_text = dedent(f"""
            Current URL   : {observation.url}
            Page title    : {observation.page_title}

            Current step     : {step_description}
            Expected outcome : {expected_outcome}

            Recent action history:
            {history_block}
            {reflection_block}

            Interactable elements (SoM id → tag/role: text):
            {elements_index}

            Choose the single best next action.
            For "done" steps: write ALL extracted content in rationale, never say "visible on screen".
            For "navigate" steps: put the full URL in the url field.
        """).strip()

        response = await self._client.chat.completions.create(
            model=cfg.vlm_model,
            temperature=cfg.vlm_temperature,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{marked_b64}",
                                "detail": cfg.vlm_image_detail,
                            },
                        },
                        {"type": "text", "text": user_text},
                    ],
                },
            ],
        )

        raw = response.choices[0].message.content
        data = json.loads(raw)

        action = Action(
            type=ActionType(data["type"]),
            element_id=data.get("element_id"),
            text=data.get("text"),
            url=data.get("url"),
            direction=data.get("direction"),
            key=data.get("key"),
            rationale=data.get("rationale", ""),
        )
        logger.debug("Perceiver → %s (elem=%s): %s",
                     action.type, action.element_id, action.rationale[:80])
        return action

    @staticmethod
    def _format_elements(elements: list[MarkedElement]) -> str:
        if not elements:
            return "(no interactable elements found)"
        lines = []
        for e in elements:
            desc = e.text or e.role or e.tag
            href = f" → {e.href}" if e.href else ""
            lines.append(f"  [{e.id}] {e.tag}/{e.role}: {desc[:60]}{href}")
        return "\n".join(lines)