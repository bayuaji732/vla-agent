from __future__ import annotations

import logging
from datetime import datetime, timezone
from functools import partial
from typing import Optional, TypedDict

from langgraph.graph import END, StateGraph

from agent.actor.actor import Actor
from agent.config import get_settings
from agent.memory.episodic import EpisodicMemory
from agent.models import (
    Action,
    ActionType,
    MemoryEntry,
    Observation,
    Plan,
    StepResult,
    StepStatus,
    TaskStatus,
    Trajectory,
)
from agent.perceiver.perceiver import Perceiver
from agent.planner.planner import Planner
from agent.reflection.reflector import Reflector
from agent.safety.safety import SafetyGuard
from agent.utils.trajectory_logger import TrajectoryLogger

logger = logging.getLogger(__name__)
cfg = get_settings()


# ── Graph state ───────────────────────────────────────────────────────────────

class AgentState(TypedDict):
    task: str
    trajectory: Trajectory
    plan: Optional[Plan]
    current_step_idx: int
    observation: Optional[Observation]
    last_action: Optional[Action]
    last_reflection: str
    action_history: list[str]
    error_count: int


# ── Node functions ────────────────────────────────────────────────────────────

async def node_plan(state: AgentState, *, planner: Planner, memory: EpisodicMemory) -> AgentState:
    snippets = await memory.build_memory_snippets(state["task"])
    plan = await planner.create_plan(state["task"], memory_snippets=snippets)
    trajectory = state["trajectory"]
    trajectory.plan = plan
    return {**state, "plan": plan, "current_step_idx": 0, "trajectory": trajectory}


async def node_observe(state: AgentState, *, actor: Actor) -> AgentState:
    obs = await actor.observe()
    return {**state, "observation": obs}


async def node_perceive(state: AgentState, *, perceiver: Perceiver) -> AgentState:
    step = state["plan"].steps[state["current_step_idx"]]
    action = await perceiver.perceive(
        observation=state["observation"],
        step_description=step.description,
        expected_outcome=step.expected_outcome,
        action_history=state["action_history"],
        reflection=state["last_reflection"],
    )
    return {**state, "last_action": action}


async def node_safety_check(state: AgentState, *, guard: SafetyGuard) -> AgentState:
    action = state["last_action"]
    obs = state["observation"]
    verdict = await guard.check(action, obs.url if obs else "")

    if not verdict.safe:
        if verdict.requires_human_approval:
            approved = await SafetyGuard.request_human_approval(action)
            if not approved:
                action = Action(type=ActionType.FAIL, rationale=f"Human denied: {verdict.reason}")
        else:
            action = Action(type=ActionType.FAIL, rationale=f"Safety blocked: {verdict.reason}")

    return {**state, "last_action": action}


async def node_act(state: AgentState, *, actor: Actor, reflector: Reflector) -> AgentState:
    action = state["last_action"]
    obs = state["observation"]
    step = state["plan"].steps[state["current_step_idx"]]
    trajectory = state["trajectory"]
    history = list(state["action_history"])
    error_count = state["error_count"]
    reflection_text = ""
    success = False
    error_msg = None

    if action.type in (ActionType.DONE, ActionType.FAIL):
        success = action.type == ActionType.DONE
        # Capture the rationale as the final answer when task is done
        if success and action.rationale:
            trajectory.final_answer = action.rationale
        if not success:
            trajectory.status = TaskStatus.FAILED
    else:
        try:
            await actor.execute(action, obs.marked_elements if obs else [])
            success = True
            step.status = StepStatus.SUCCESS
            history.append(f"[step {step.index}] {action.type} → SUCCESS: {action.rationale}")
        except Exception as exc:
            error_msg = str(exc)
            error_count += 1
            step.retries += 1
            logger.warning("Action failed (retry %d/%d): %s", step.retries, cfg.max_retries_per_step, error_msg)

            new_obs = await actor.observe()
            reflection = await reflector.reflect(step, action, error_msg, new_obs)
            reflection_text = f"Diagnosis: {reflection.diagnosis} | Try instead: {reflection.corrective_hint}"
            history.append(f"[step {step.index}] {action.type} → FAIL ({error_msg[:60]}) | reflect: {reflection.diagnosis}")

            if reflection.abort_task:
                step.status = StepStatus.FAILED
                trajectory.status = TaskStatus.FAILED
            elif reflection.skip_step:
                step.status = StepStatus.SKIPPED
                success = True
            else:
                step.status = StepStatus.FAILED if step.retries >= cfg.max_retries_per_step else StepStatus.RUNNING

    if obs:
        trajectory.steps.append(StepResult(
            step=step, action=action, observation=obs,
            success=success, error=error_msg, reflection=reflection_text or None,
        ))

    return {
        **state,
        "trajectory": trajectory,
        "action_history": history,
        "last_reflection": reflection_text,
        "error_count": error_count,
    }


async def node_advance_step(state: AgentState) -> AgentState:
    return {**state, "current_step_idx": state["current_step_idx"] + 1, "last_reflection": ""}


async def node_finalize(
    state: AgentState, *, memory: EpisodicMemory, reflector: Reflector, traj_logger: TrajectoryLogger
) -> AgentState:
    trajectory = state["trajectory"]
    if trajectory.status == TaskStatus.RUNNING:
        trajectory.status = TaskStatus.SUCCESS
    outcome = trajectory.status.value
    trajectory.ended_at = datetime.now(timezone.utc)

    step_summaries = [f"[{r.step.status.value}] {r.step.description}" for r in trajectory.steps]
    ts = await reflector.summarize_trajectory(state["task"], step_summaries, outcome)
    mem_entry = MemoryEntry(
        task=state["task"],
        summary=ts.summary,
        outcome=outcome,
        key_actions=ts.key_actions,
        lessons=ts.lessons,
        embedding_text=f"{state['task']} | {ts.summary}",
    )
    await memory.store(mem_entry)
    traj_logger.save(trajectory)
    logger.info("Task complete. Status=%s", trajectory.status)
    return {**state, "trajectory": trajectory}


# ── Routing ───────────────────────────────────────────────────────────────────

def route_after_act(state: AgentState) -> str:
    if state["trajectory"].status == TaskStatus.FAILED:
        return "finalize"
    action = state["last_action"]
    step = state["plan"].steps[state["current_step_idx"]]
    if action.type == ActionType.DONE or step.status == StepStatus.SUCCESS:
        return "advance"
    if action.type == ActionType.FAIL:
        return "finalize"
    if step.retries >= cfg.max_retries_per_step:
        return "finalize"
    return "observe"


def route_after_advance(state: AgentState) -> str:
    if state["current_step_idx"] >= len(state["plan"].steps):
        return "finalize"
    return "observe"


# ── Graph builder ─────────────────────────────────────────────────────────────

def build_agent_graph(
    planner: Planner,
    perceiver: Perceiver,
    actor: Actor,
    reflector: Reflector,
    memory: EpisodicMemory,
    guard: SafetyGuard,
    traj_logger: TrajectoryLogger,
) -> StateGraph:
    g = StateGraph(AgentState)

    g.add_node("plan",         partial(node_plan,         planner=planner, memory=memory))
    g.add_node("observe",      partial(node_observe,      actor=actor))
    g.add_node("perceive",     partial(node_perceive,     perceiver=perceiver))
    g.add_node("safety_check", partial(node_safety_check, guard=guard))
    g.add_node("act",          partial(node_act,          actor=actor, reflector=reflector))
    g.add_node("advance",      node_advance_step)
    g.add_node("finalize",     partial(node_finalize,     memory=memory, reflector=reflector, traj_logger=traj_logger))

    g.set_entry_point("plan")
    g.add_edge("plan",         "observe")
    g.add_edge("observe",      "perceive")
    g.add_edge("perceive",     "safety_check")
    g.add_edge("safety_check", "act")
    g.add_conditional_edges("act",     route_after_act,     {"observe": "observe", "advance": "advance", "finalize": "finalize"})
    g.add_conditional_edges("advance", route_after_advance, {"observe": "observe", "finalize": "finalize"})
    g.add_edge("finalize", END)

    return g