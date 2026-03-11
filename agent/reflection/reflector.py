from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from textwrap import dedent

from openai import AsyncOpenAI

from agent.config import get_settings
from agent.models import Action, Observation, PlanStep

logger = logging.getLogger(__name__)
cfg = get_settings()

_REFLECT_SYSTEM = dedent("""
You are an expert agent debugger. An AI browser agent attempted an action and it failed.
Your job:
1. Diagnose WHY it failed based on the screenshot and action attempted.
2. Suggest a CONCRETE corrective action or reinterpretation.
3. Flag if the step should be SKIPPED or the whole task ABORTED.

Output ONLY valid JSON:
{
  "diagnosis": "<one sentence root cause>",
  "corrective_hint": "<what the agent should try instead>",
  "skip_step": <true|false>,
  "abort_task": <true|false>,
  "confidence": <0.0-1.0>
}
""").strip()


class Reflector:
    def __init__(self) -> None:
        self._client = AsyncOpenAI(api_key=cfg.openai_api_key)

    async def reflect(
        self,
        step: PlanStep,
        failed_action: Action,
        error_message: str,
        observation: Observation,
    ) -> ReflectionResult:
        user_prompt = dedent(f"""
            Step goal:       {step.description}
            Expected result: {step.expected_outcome}
            Action tried:    type={failed_action.type} element_id={failed_action.element_id} text={failed_action.text!r}
            Action rationale:{failed_action.rationale}
            Error:           {error_message}
            Retry #:         {step.retries}
            Current URL:     {observation.url}
        """).strip()

        response = await self._client.chat.completions.create(
            model=cfg.reflection_model,
            temperature=0.1,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": _REFLECT_SYSTEM},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{observation.screenshot_b64}",
                                "detail": "low",
                            },
                        },
                        {"type": "text", "text": user_prompt},
                    ],
                },
            ],
        )

        content = response.choices[0].message.content
        if not content:
            logger.warning("Reflector got empty response, using safe defaults")
            return ReflectionResult(
                diagnosis="Unknown error — empty model response",
                corrective_hint="Retry the step with a different approach",
                skip_step=False,
                abort_task=False,
                confidence=0.3,
            )

        data = json.loads(content)
        result = ReflectionResult(
            diagnosis=data.get("diagnosis", "Unknown"),
            corrective_hint=data.get("corrective_hint", "Try a different approach"),
            skip_step=data.get("skip_step", False),
            abort_task=data.get("abort_task", False),
            confidence=data.get("confidence", 0.5),
        )
        logger.info("Reflection: diagnosis='%s' skip=%s abort=%s",
                    result.diagnosis, result.skip_step, result.abort_task)
        return result

    async def summarize_trajectory(
        self,
        task: str,
        step_summaries: list[str],
        outcome: str,
    ) -> TrajectorySummary:
        # Guard: if no steps were executed, return a minimal summary
        if not step_summaries:
            return TrajectorySummary(
                summary=f"Task '{task}' ended with outcome '{outcome}' but no steps were executed.",
                key_actions=[],
                lessons=["Task failed before any actions could be taken"],
            )

        steps_text = "\n".join(f"  {i+1}. {s}" for i, s in enumerate(step_summaries))
        prompt = dedent(f"""
            Task: {task}
            Outcome: {outcome}
            Steps taken:
            {steps_text}

            Produce a memory summary. Output ONLY valid JSON:
            {{
              "summary": "<2-3 sentence summary of what happened>",
              "key_actions": ["<action 1>", "<action 2>"],
              "lessons": ["<lesson 1>", "<lesson 2>"]
            }}
        """).strip()

        response = await self._client.chat.completions.create(
            model=cfg.reflection_model,
            temperature=0.2,
            response_format={"type": "json_object"},
            messages=[{"role": "user", "content": prompt}],
        )

        content = response.choices[0].message.content
        if not content:
            logger.warning("summarize_trajectory got empty response, using fallback")
            return TrajectorySummary(
                summary=f"Task '{task}' completed with outcome '{outcome}'.",
                key_actions=[s[:80] for s in step_summaries[:3]],
                lessons=["Review trajectory logs for details"],
            )

        data = json.loads(content)
        return TrajectorySummary(
            summary=data.get("summary", ""),
            key_actions=data.get("key_actions", []),
            lessons=data.get("lessons", []),
        )


@dataclass
class ReflectionResult:
    diagnosis: str
    corrective_hint: str
    skip_step: bool
    abort_task: bool
    confidence: float


@dataclass
class TrajectorySummary:
    summary: str
    key_actions: list[str]
    lessons: list[str]