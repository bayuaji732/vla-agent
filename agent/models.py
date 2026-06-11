from __future__ import annotations

import uuid
from datetime import datetime
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field


# ── Enums ─────────────────────────────────────────────────────────────────────

class ActionType(str, Enum):
    CLICK        = "click"
    TYPE         = "type"
    SCROLL       = "scroll"
    NAVIGATE     = "navigate"
    HOVER        = "hover"
    KEY_PRESS    = "key_press"
    SCREENSHOT   = "screenshot"
    WAIT         = "wait"
    DONE         = "done"
    FAIL         = "fail"


class StepStatus(str, Enum):
    PENDING   = "pending"
    RUNNING   = "running"
    SUCCESS   = "success"
    FAILED    = "failed"
    SKIPPED   = "skipped"


class TaskStatus(str, Enum):
    RUNNING   = "running"
    SUCCESS   = "success"
    FAILED    = "failed"
    ABORTED   = "aborted"


# ── Core data models ──────────────────────────────────────────────────────────

class Action(BaseModel):
    type: ActionType
    element_id: Optional[int]   = None   # Set-of-Marks label
    selector: Optional[str]     = None   # CSS / XPath fallback
    text: Optional[str]         = None   # for TYPE
    url: Optional[str]          = None   # for NAVIGATE
    x: Optional[int]            = None   # absolute coords fallback
    y: Optional[int]            = None
    direction: Optional[str]    = None   # "up" / "down" for SCROLL
    key: Optional[str]          = None   # for KEY_PRESS
    delay_ms: int               = 0
    rationale: str              = ""     # LLM's reasoning


class Observation(BaseModel):
    screenshot_b64: str                   # base64-encoded PNG
    url: str         = ""
    page_title: str  = ""
    marked_elements: list[MarkedElement] = Field(default_factory=list)
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class MarkedElement(BaseModel):
    id: int
    tag: str
    role: str        = ""
    text: str        = ""
    href: Optional[str] = None
    x: int; y: int; width: int; height: int


class PlanStep(BaseModel):
    index: int
    description: str
    expected_outcome: str
    status: StepStatus = StepStatus.PENDING
    actions_taken: list[Action]  = Field(default_factory=list)
    retries: int = 0


class Plan(BaseModel):
    goal: str
    steps: list[PlanStep]
    created_at: datetime = Field(default_factory=datetime.utcnow)


class StepResult(BaseModel):
    step: PlanStep
    action: Action
    observation: Observation
    success: bool
    error: Optional[str] = None
    reflection: Optional[str] = None


class Trajectory(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    task: str
    status: TaskStatus = TaskStatus.RUNNING
    plan: Optional[Plan] = None
    steps: list[StepResult] = Field(default_factory=list)
    final_answer: Optional[str] = None
    started_at: datetime = Field(default_factory=datetime.utcnow)
    ended_at: Optional[datetime] = None
    total_tokens_used: int = 0


class MemoryEntry(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    task: str
    summary: str
    outcome: str      # "success" | "failure"
    key_actions: list[str]
    lessons: list[str]
    embedding_text: str   # text used to embed
    created_at: datetime = Field(default_factory=datetime.utcnow)


class SafetyVerdict(BaseModel):
    safe: bool
    risk_level: str   # "low" | "medium" | "high" | "blocked"
    reason: str
    requires_human_approval: bool = False