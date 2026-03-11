from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # ── LLM ──────────────────────────────────────────────────────────────────
    openai_api_key: str = ""
    anthropic_api_key: str = ""

    planner_model: str = "gpt-4o"
    vlm_model: str = "gpt-4o"
    reflection_model: str = "gpt-4o"

    planner_temperature: float = 0.2
    vlm_temperature: float = 0.1
    max_plan_steps: int = 30
    max_retries_per_step: int = 3

    # ── Speed ─────────────────────────────────────────────────────────────────
    # action_delay_ms  : pause after EVERY action. 100 is safe, 0 may miss renders.
    # navigate_delay_ms: pause after page navigation. 800 minimum for JS-heavy pages.
    # vlm_image_detail : "low" = 85 tokens/fast, "high" = 1275 tokens/slow/accurate
    # mouse_steps      : steps in human-mouse curve. 1 = instant, 8 = realistic.
    action_delay_ms: int = 100          # was 500 — saves ~400ms per action
    navigate_delay_ms: int = 800        # was 1500-3000 — saves 1-2s per navigation
    vlm_image_detail: str = "low"       # was "high" — saves ~1s + 75% token cost
    mouse_steps: int = 3                # was 8-15 — saves ~300ms per click

    # ── Browser / actor ───────────────────────────────────────────────────────
    headless: bool = False
    browser_timeout_ms: int = 30_000
    viewport_width: int = 1280
    viewport_height: int = 800

    # ── Safety ───────────────────────────────────────────────────────────────
    safe_mode: bool = True
    human_in_loop: bool = False
    allowed_domains: list[str] = ["*"]
    blocked_domains: list[str] = ["bank", "payment", "checkout", "delete", "admin"]

    # ── ChromaDB ─────────────────────────────────────────────────────────────
    chroma_mode: Literal["local", "http"] = "local"
    chroma_persist_dir: str = "./chroma_db"
    chroma_host: str = "localhost"
    chroma_port: int = 8001
    episodic_collection: str = "episodic_memory"
    semantic_collection: str = "semantic_memory"
    memory_top_k: int = 5
    memory_similarity_threshold: float = 0.75

    # ── Logging / Tracing ────────────────────────────────────────────────────
    trajectory_dir: str = "./trajectories"
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    enable_otel: bool = False
    otel_endpoint: str = "http://localhost:4317"

    # ── Set-of-Marks ─────────────────────────────────────────────────────────
    som_label_font_size: int = 14
    som_max_elements: int = 40          # was 60 — fewer elements = faster VLM parse

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()