from __future__ import annotations

import json
import logging
import re
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

════════════════════════════════════════════════════════════════════
NAVIGATION RULES (read carefully — these override everything else)
════════════════════════════════════════════════════════════════════

1. DIRECT NAVIGATION
   If the step description says "navigate to <URL>" or contains a full URL
   (starting with https:// or http://), output:
     { "type": "navigate", "url": "<exact URL from the step description>" }
   Copy the URL verbatim. Do NOT substitute a Google search URL.
   Do NOT click anything — just navigate.

2. WEB SEARCH (only when the step explicitly says to search)
   If the step says "search for X", use a direct search URL:
     https://www.google.com/search?q=X+encoded
   Never interact with a search box on a homepage.

3. ADDRESS BAR
   Never click the browser address bar. Use "navigate" action type with "url" field.

════════════════════════════════════════════════════════════════════
READING / EXTRACTION RULES
════════════════════════════════════════════════════════════════════

4. READING RESULTS
   If the step says "read", "extract", "note", or "summarize" information,
   use action type "done" and write ALL extracted content verbatim in the
   "rationale" field. Never say "visible on screen" — write the actual text.

5. COMPILE STEP
   If the step says "compile all information into a final answer", write the
   FULL structured answer in "rationale" using all data from action_history.

════════════════════════════════════════════════════════════════════
GENERAL
════════════════════════════════════════════════════════════════════

6. ONE ACTION per response.
7. Use "done" when the step's expected outcome is achieved.
8. Use "fail" only if the step is truly impossible.

Output ONLY valid JSON (no markdown, no extra keys):
{
  "type": "<navigate|click|type|scroll|hover|key_press|wait|done|fail>",
  "element_id": <int or null>,
  "text": "<string or null>",
  "url": "<full URL or null>",
  "direction": "<up|down or null>",
  "key": "<key name or null>",
  "rationale": "<for done: complete extracted answer; for actions: one-sentence reason>"
}
""").strip()

# Regex to pull the first URL out of a step description so we can
# pass it to the LLM as an explicit reminder.
_URL_IN_TEXT_RE = re.compile(r"https?://\S+", re.IGNORECASE)


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
        # ── Fast-path: if the step contains an explicit URL, skip the VLM call
        # entirely and return a navigate action directly.  This is 100% reliable
        # and costs zero tokens.
        url_match = _URL_IN_TEXT_RE.search(step_description)
        if url_match and self._step_is_navigate(step_description):
            url = url_match.group(0).rstrip(".,;)")  # strip trailing punctuation
            logger.info("Perceiver fast-path: direct navigate → %s", url)
            return Action(
                type=ActionType.NAVIGATE,
                url=url,
                rationale=f"Direct navigation to {url} as specified in step.",
            )

        marked_b64 = mark_screenshot(
            observation.screenshot_b64,
            observation.marked_elements,
        )

        elements_index = self._format_elements(observation.marked_elements)
        history_block = "\n".join(action_history[-8:]) or "None"
        reflection_block = (
            f"\nReflection on last failure:\n{reflection}" if reflection else ""
        )

        # Explicit URL reminder to reduce LLM hallucination
        url_reminder = ""
        if url_match:
            url_reminder = (
                f"\n⚠️  URL IN STEP: {url_match.group(0).rstrip('.,;)')} — "
                "use action type 'navigate' with this exact URL."
            )

        user_text = dedent(f"""
            Current URL   : {observation.url}
            Page title    : {observation.page_title}

            Current step     : {step_description}
            Expected outcome : {expected_outcome}
            {url_reminder}

            Recent action history:
            {history_block}
            {reflection_block}

            Interactable elements (SoM id → tag/role: text):
            {elements_index}

            Choose the single best next action.
            • For navigate steps: put the EXACT URL from the step in the url field.
            • For done steps: write ALL extracted content in rationale — never say "visible on screen".
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

        # ── Safety net: if the LLM returned a Google search URL but the step
        # clearly intended a direct navigation, override it.
        if (
            action.type == ActionType.NAVIGATE
            and action.url
            and "google.com/search" in action.url
            and url_match
            and "google.com/search" not in url_match.group(0)
        ):
            corrected_url = url_match.group(0).rstrip(".,;)")
            logger.warning(
                "Perceiver overrode Google search with direct URL: %s → %s",
                action.url,
                corrected_url,
            )
            action = action.model_copy(
                update={
                    "url": corrected_url,
                    "rationale": f"Corrected to direct navigation: {corrected_url}",
                }
            )

        logger.debug(
            "Perceiver → %s (elem=%s url=%s): %s",
            action.type,
            action.element_id,
            action.url,
            action.rationale[:80],
        )
        return action

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _step_is_navigate(description: str) -> bool:
        """Return True if the step description is primarily a navigate instruction."""
        nav_keywords = (
            "navigate", "go to", "open", "visit", "load", "access",
            "browse to", "head to", "direct",
        )
        lower = description.lower()
        return any(kw in lower for kw in nav_keywords)

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