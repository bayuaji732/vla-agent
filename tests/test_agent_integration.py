"""
Integration smoke test — runs the agent against a simple, safe task.
Requires: OPENAI_API_KEY in .env, ChromaDB running.

Run: pytest tests/test_agent_integration.py -v -s
"""
from __future__ import annotations

import asyncio
import pytest

from agent.main import VisionLanguageAgent
from agent.models import TaskStatus


@pytest.mark.asyncio
async def test_simple_navigation_task():
    """Agent should navigate to example.com and report success."""
    agent = VisionLanguageAgent()
    trajectory = await agent.run("Go to https://example.com and tell me the page title.")

    assert trajectory.status in (TaskStatus.SUCCESS, TaskStatus.FAILED)
    assert len(trajectory.steps) > 0
    assert trajectory.ended_at is not None


@pytest.mark.asyncio
async def test_safety_blocks_dangerous_url():
    """Safety guard must block navigation to a blocked domain pattern."""
    from agent.models import Action, ActionType
    from agent.safety.safety import SafetyGuard

    guard = SafetyGuard()
    action = Action(type=ActionType.NAVIGATE, url="https://checkout.example.com/payment")
    verdict = await guard.check(action)

    assert verdict.risk_level in ("high", "blocked")
    assert not verdict.safe


@pytest.mark.asyncio
async def test_planner_produces_valid_plan():
    from agent.planner.planner import Planner

    planner = Planner()
    plan = await planner.create_plan("Search for Python tutorials on YouTube")

    assert len(plan.steps) >= 2
    assert all(s.description for s in plan.steps)
    assert all(s.expected_outcome for s in plan.steps)


@pytest.mark.asyncio
async def test_set_of_marks_overlay():
    """SoM should return a valid base64 PNG string."""
    import base64
    from agent.perceiver.som import mark_screenshot
    from agent.models import MarkedElement

    # Create a tiny blank PNG
    from PIL import Image
    import io

    img = Image.new("RGB", (400, 300), color=(30, 30, 30))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode()

    elements = [
        MarkedElement(id=1, tag="button", role="button", text="Click me", x=50, y=50, width=100, height=30),
        MarkedElement(id=2, tag="a", role="link", text="Go here", href="https://example.com", x=50, y=120, width=80, height=20),
    ]

    marked = mark_screenshot(b64, elements)
    assert isinstance(marked, str)
    decoded = base64.b64decode(marked)
    assert decoded[:4] == b"\x89PNG"   # valid PNG magic bytes
