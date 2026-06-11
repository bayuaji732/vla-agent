from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import BackgroundTasks, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from agent.main import VisionLanguageAgent
from agent.models import TaskStatus, Trajectory
from agent.utils.trajectory_logger import TrajectoryLogger

logger = logging.getLogger(__name__)

# In-memory job store (use Redis in production)
_jobs: dict[str, Trajectory] = {}
_agents: dict[str, VisionLanguageAgent] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("VLA Agent API starting up")
    yield
    logger.info("VLA Agent API shutting down")
    for agent in _agents.values():
        await agent.actor.stop()


app = FastAPI(
    title="Vision-Language Autonomous Agent API",
    version="1.0.0",
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Request / Response models ─────────────────────────────────────────────────

class TaskRequest(BaseModel):
    task: str
    headless: bool = True


class TaskResponse(BaseModel):
    job_id: str
    status: str
    message: str


class JobStatusResponse(BaseModel):
    job_id: str
    task: str
    status: str
    steps_completed: int
    total_steps: Optional[int]
    started_at: Optional[str]
    ended_at: Optional[str]
    final_answer: Optional[str]


# ── Background task runner ────────────────────────────────────────────────────

async def _run_agent(job_id: str, task: str) -> None:
    agent = VisionLanguageAgent()
    _agents[job_id] = agent
    try:
        trajectory = await agent.run(task)
        _jobs[job_id] = trajectory
    except Exception as exc:
        logger.error("Agent job %s failed: %s", job_id, exc)
        if job_id in _jobs:
            _jobs[job_id].status = TaskStatus.FAILED
    finally:
        await agent.actor.stop()
        _agents.pop(job_id, None)


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.post("/tasks", response_model=TaskResponse)
async def create_task(req: TaskRequest, background_tasks: BackgroundTasks):
    """Submit a new agent task for async execution."""
    import uuid
    job_id = str(uuid.uuid4())

    placeholder = Trajectory(task=req.task)
    _jobs[job_id] = placeholder

    background_tasks.add_task(_run_agent, job_id, req.task)
    logger.info("Task submitted: %s → %s", job_id, req.task)

    return TaskResponse(
        job_id=job_id,
        status="queued",
        message="Task accepted. Poll /tasks/{job_id} for status.",
    )


@app.get("/tasks/{job_id}", response_model=JobStatusResponse)
async def get_task_status(job_id: str):
    """Poll the status of a running or completed task."""
    traj = _jobs.get(job_id)
    if traj is None:
        raise HTTPException(status_code=404, detail="Job not found")

    return JobStatusResponse(
        job_id=job_id,
        task=traj.task,
        status=traj.status.value,
        steps_completed=len(traj.steps),
        total_steps=len(traj.plan.steps) if traj.plan else None,
        started_at=traj.started_at.isoformat() if traj.started_at else None,
        ended_at=traj.ended_at.isoformat() if traj.ended_at else None,
        final_answer=traj.final_answer,
    )


@app.get("/tasks/{job_id}/trajectory")
async def get_trajectory(job_id: str):
    """Return full trajectory for a completed task."""
    traj = _jobs.get(job_id)
    if traj is None:
        raise HTTPException(status_code=404, detail="Job not found")
    return traj.model_dump()


@app.post("/tasks/{job_id}/abort")
async def abort_task(job_id: str):
    """Abort a running agent task."""
    agent = _agents.get(job_id)
    if agent:
        await agent.actor.stop()
        _agents.pop(job_id, None)
    traj = _jobs.get(job_id)
    if traj:
        traj.status = TaskStatus.ABORTED
    return {"job_id": job_id, "status": "aborted"}


@app.get("/health")
async def health():
    return {"status": "ok", "active_jobs": len(_agents)}