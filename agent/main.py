from __future__ import annotations

import asyncio
import logging
import sys
from datetime import datetime, timezone

from agent.actor.actor import Actor
from agent.config import get_settings
from agent.graph.agent_graph import AgentState, build_agent_graph
from agent.memory.episodic import EpisodicMemory
from agent.models import TaskStatus, Trajectory
from agent.perceiver.perceiver import Perceiver
from agent.planner.planner import Planner
from agent.reflection.reflector import Reflector
from agent.safety.safety import SafetyGuard
from agent.utils.trajectory_logger import TrajectoryLogger

cfg = get_settings()

logging.basicConfig(
    level=getattr(logging, cfg.log_level),
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


class VisionLanguageAgent:
    """
    Top-level agent orchestrator.
    Wires all components and runs the LangGraph execution loop.
    """

    def __init__(self) -> None:
        self.planner     = Planner()
        self.perceiver   = Perceiver()
        self.actor       = Actor()
        self.reflector   = Reflector()
        self.memory      = EpisodicMemory()
        self.guard       = SafetyGuard()
        self.traj_logger = TrajectoryLogger()

        # Compile graph — nodes are async, use ainvoke at runtime
        self._graph = build_agent_graph(
            planner     = self.planner,
            perceiver   = self.perceiver,
            actor       = self.actor,
            reflector   = self.reflector,
            memory      = self.memory,
            guard       = self.guard,
            traj_logger = self.traj_logger,
        ).compile()

    async def run(self, task: str) -> Trajectory:
        """
        Execute a natural-language task end-to-end.

        Args:
            task: e.g. "Find the cheapest flight from Jakarta to Singapore"

        Returns:
            Completed Trajectory with all steps and final status.
        """
        logger.info("━" * 60)
        logger.info("TASK: %s", task)
        logger.info("━" * 60)

        trajectory = Trajectory(
            task=task,
            started_at=datetime.now(timezone.utc),
        )

        initial_state: AgentState = {
            "task":             task,
            "trajectory":       trajectory,
            "plan":             None,
            "current_step_idx": 0,
            "observation":      None,
            "last_action":      None,
            "last_reflection":  "",
            "action_history":   [],
            "error_count":      0,
        }

        await self.actor.start()
        try:
            # ainvoke runs the full graph asynchronously — no thread gymnastics needed
            final_state = await self._graph.ainvoke(initial_state)
            return final_state["trajectory"]
        finally:
            await self.actor.stop()


# ── CLI entrypoint ────────────────────────────────────────────────────────────

async def _main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python -m agent.main '<task>'")
        print("Example: python -m agent.main 'Go to example.com and tell me the page title'")
        sys.exit(1)

    task = " ".join(sys.argv[1:])
    agent = VisionLanguageAgent()

    try:
        trajectory = await agent.run(task)
    except KeyboardInterrupt:
        logger.warning("Interrupted by user")
        return

    ended = trajectory.ended_at or datetime.now(timezone.utc)
    started = trajectory.started_at or ended
    elapsed = int((ended - started).total_seconds())

    print("\n" + "━" * 60)
    print(f"RESULT: {trajectory.status.value.upper()}")
    if trajectory.final_answer:
        print(f"ANSWER: {trajectory.final_answer}")
    print(f"STEPS : {len(trajectory.steps)} executed")
    print(f"TIME  : {elapsed}s")
    print("━" * 60)


if __name__ == "__main__":
    asyncio.run(_main())