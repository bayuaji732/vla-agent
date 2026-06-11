# Vision-Language Autonomous Agent

A production-grade browser agent that perceives its environment through screenshots, reasons with a Vision-Language Model (VLM), and executes multi-step tasks autonomously — built from scratch with a hierarchical planner-actor architecture, episodic memory, self-reflection, and a safety layer.

---

## Architecture

```
Task (natural language)
        │
        ▼
┌───────────────┐
│    Planner    │  GPT-4o decomposes goal into PlanSteps
│               │  + consults EpisodicMemory for past experience
└──────┬────────┘
       │ Plan
       ▼
┌──────────────────────────────────────────────────┐
│               Perception-Action Loop              │
│                                                  │
│  ┌──────────┐    ┌───────────┐    ┌───────────┐ │
│  │  Actor   │───▶│ Perceiver │───▶│  Safety   │ │
│  │Playwright│◀───│  VLM+SoM  │    │   Guard   │ │
│  └──────────┘    └───────────┘    └─────┬─────┘ │
│       │                                 │       │
│       │  failure                        │ safe  │
│       ▼                                 ▼       │
│  ┌──────────┐                    ┌───────────┐  │
│  │Reflector │                    │   Actor   │  │
│  │self-crit │                    │  execute  │  │
│  └──────────┘                    └───────────┘  │
└──────────────────────────────────────────────────┘
       │ trajectory
       ▼
┌───────────────┐
│EpisodicMemory │  ChromaDB — stores run summary + lessons
│TrajectoryLog  │  JSON logs for offline eval / few-shot learning
└───────────────┘
```

The entire loop is orchestrated as a **LangGraph state machine** with conditional routing between nodes.

---

## Key Features

**Hierarchical Planning** — GPT-4o breaks any natural-language task into atomic, ordered browser steps. On unrecoverable failure, the planner can `replan()` from the point of failure without restarting.

**Set-of-Marks (SoM) Perception** — Before every VLM call, interactable DOM elements are extracted and numbered labels are overlaid on the screenshot using Pillow. The VLM references elements by ID rather than coordinates, giving precise grounding without fine-tuning.

**Self-Reflection Loop** — On each failed action, a second VLM call diagnoses the root cause, proposes a corrective action, and decides whether to retry, skip the step, or abort the task entirely.

**Episodic Memory** — Past trajectories are embedded via `text-embedding-3-small` and stored in ChromaDB. Before planning, the agent retrieves similar past tasks and injects their lessons as planner context — enabling it to avoid repeated mistakes across sessions.

**Safety Guard** — Every action passes through a rule-based safety layer before execution. Risky actions (payments, deletions, blocked domains) are either blocked outright or escalated to a human-in-the-loop approval flow.

**Full Trajectory Logging** — Every run is serialized to JSON with the full action log, screenshots at each step, reflections, and final outcome — ready for offline evaluation or few-shot learning.

---

## Project Structure

```
vla-agent/
├── agent/
│   ├── config.py                  # Pydantic settings (env-driven)
│   ├── models.py                  # Shared data models (Action, Plan, Trajectory …)
│   ├── main.py                    # VisionLanguageAgent orchestrator + CLI
│   ├── api.py                     # FastAPI async REST server
│   ├── planner/
│   │   └── planner.py             # LLM task decomposition + replanning
│   ├── perceiver/
│   │   ├── perceiver.py           # VLM action selection
│   │   └── som.py                 # Set-of-Marks screenshot annotation
│   ├── actor/
│   │   └── actor.py               # Playwright browser executor
│   ├── memory/
│   │   └── episodic.py            # ChromaDB episodic memory store
│   ├── reflection/
│   │   └── reflector.py           # Self-reflection + trajectory summarizer
│   ├── safety/
│   │   └── safety.py              # Pre-execution action validator
│   ├── graph/
│   │   └── agent_graph.py         # LangGraph state machine
│   └── utils/
│       └── trajectory_logger.py   # JSON trajectory persistence
├── tests/
│   └── test_agent_integration.py  # Integration smoke tests
├── trajectories/                  # Auto-created: run logs
├── .env.example
├── requirements.txt
├── Dockerfile
└── docker-compose.yml
```

---

## Quickstart

### 1. Prerequisites

- Python 3.11+
- Docker & Docker Compose
- OpenAI API key (GPT-4o access required for vision)

### 2. Clone & configure

```bash
git clone https://github.com/bayuaji732/vla-agent.git
cd vla-agent

cp .env.example .env
# Edit .env and set OPENAI_API_KEY=sk-...
```

### 3. Start infrastructure

```bash
docker compose up -d chromadb
```

### 4. Install Python dependencies

```bash
uv add -r requirements.txt
uv run playwright install chromium
```

### 5. Run a task (CLI)

```bash
uv run python -m agent.main "Go to news.ycombinator.com and tell me the top 3 stories"
```

### 6. Run the API server

```bash
uv run uvicorn agent.api:app --reload --port 8000
```

Submit a task:

```bash
curl -X POST http://localhost:8000/tasks \
  -H "Content-Type: application/json" \
  -d '{"task": "Search for Python tutorials on YouTube", "headless": true}'
```

Poll for status:

```bash
curl http://localhost:8000/tasks/{job_id}
```

---

## Configuration

All settings are driven by environment variables (or `.env`). The full reference is in `agent/config.py`.

| Variable               | Default          | Description                                   |
| ---------------------- | ---------------- | --------------------------------------------- |
| `OPENAI_API_KEY`       | —                | Required. GPT-4o API key                      |
| `ANTHROPIC_API_KEY`    | —                | Optional. For Claude-based models             |
| `PLANNER_MODEL`        | `gpt-4o`         | LLM used for task decomposition               |
| `VLM_MODEL`            | `gpt-4o`         | VLM used for perception                       |
| `HEADLESS`             | `true`           | Run browser in headless mode                  |
| `SAFE_MODE`            | `true`           | Enable safety validation layer                |
| `HUMAN_IN_LOOP`        | `false`          | Pause on high-risk actions for human approval |
| `CHROMA_HOST`          | `localhost`      | ChromaDB host                                 |
| `CHROMA_PORT`          | `8001`           | ChromaDB port                                 |
| `MAX_PLAN_STEPS`       | `15`             | Max steps the planner can generate            |
| `MAX_RETRIES_PER_STEP` | `3`              | Max reflection-retry cycles per step          |
| `LOG_LEVEL`            | `INFO`           | Logging verbosity                             |
| `TRAJECTORY_DIR`       | `./trajectories` | Directory for trajectory JSON files           |

---

## API Reference

| Method | Endpoint                     | Description                        |
| ------ | ---------------------------- | ---------------------------------- |
| `POST` | `/tasks`                     | Submit a new agent task (async)    |
| `GET`  | `/tasks/{job_id}`            | Poll task status and step count    |
| `GET`  | `/tasks/{job_id}/trajectory` | Full trajectory JSON when complete |
| `POST` | `/tasks/{job_id}/abort`      | Abort a running task               |
| `GET`  | `/health`                    | Health check + active job count    |

---

## Running Tests

```bash
pytest tests/ -v -s
```

The integration tests cover:

- End-to-end agent run against a live URL
- Safety guard blocking dangerous URLs
- Planner producing valid structured plans
- Set-of-Marks overlay returning a valid PNG

> Note: integration tests require `OPENAI_API_KEY` and ChromaDB running.

---

## How Each Component Works

### Planner (`planner/planner.py`)

Sends a structured JSON prompt to GPT-4o with the task, optional episodic memory snippets, and the current page context. Returns a `Plan` containing ordered `PlanStep` objects, each with a description and expected outcome. If a step fails unrecoverably, `replan()` regenerates only the remaining steps from the failure point — preserving completed work.

### Set-of-Marks (`perceiver/som.py`)

Extracts all interactable DOM elements via a Playwright `page.evaluate()` call, assigns sequential integer IDs, and draws coloured bounding boxes with numeric labels directly onto the screenshot using Pillow. The VLM then references elements by `element_id` rather than pixel coordinates, dramatically improving grounding accuracy without any fine-tuning.

### Perceiver (`perceiver/perceiver.py`)

Sends the annotated screenshot + step context to GPT-4o Vision. The model returns a single action as JSON: type, target element ID, optional text/URL, and a rationale sentence. The perceiver also receives the last reflection critique so it can learn from its previous failure in the same context window.

### Actor (`actor/actor.py`)

Wraps Playwright's async API. Resolves action targets in priority order: SoM element ID → CSS selector → absolute (x, y) coordinates. Handles all action types: `click`, `type`, `scroll`, `navigate`, `hover`, `key_press`, `wait`. After each action, a configurable stabilisation pause lets the page settle before the next observation.

### Reflector (`reflection/reflector.py`)

On action failure, sends the failed action, error message, and a fresh screenshot to the VLM. Returns a structured diagnosis (`skip_step`, `abort_task`, `corrective_hint`, `confidence`). Also contains `summarize_trajectory()` which condenses a completed run into a `MemoryEntry` — extracting key actions and lessons learned.

### Episodic Memory (`memory/episodic.py`)

Embeds task text using `text-embedding-3-small` and stores it in a ChromaDB collection with cosine similarity. On each new task, retrieves the top-K most similar past runs above a similarity threshold and formats them as planner context snippets (outcome + lessons). This gives the agent cross-session learning without any weight updates.

### Safety Guard (`safety/safety.py`)

Pure rule-based, zero-latency validation before every action. Checks: URL scheme and domain against block-list patterns, typed text against high-risk keyword regex, and action rationale for dangerous intent signals. Returns a `SafetyVerdict` with a four-tier risk level. In `HUMAN_IN_LOOP` mode, high-risk actions pause execution and prompt the operator for approval via stdin (swap for Slack/webhook in production).

### LangGraph Graph (`graph/agent_graph.py`)

Defines the agent as a typed state machine with seven nodes: `plan → observe → perceive → safety_check → act → advance → finalize`. Conditional edges handle the three outcomes of `act`: retry (re-observe), advance to next step, or finalize (failure/completion). This structure makes the control flow explicit, inspectable, and easy to extend with new nodes.

---

## Demo Scenarios

**Basic web research**

```bash
uv run python -m agent.main "Go to Wikipedia.org and summarize the article on Large Language Models in 3 bullet points"
```

**Multi-step navigation**

```bash
uv run python -m agent.main "Go to github.com/trending and find the top Python repository this week"
```

**Self-reflection demo** — deliberately give an ambiguous task to observe the retry + reflection loop in the logs:

```bash
LOG_LEVEL=DEBUG python -m agent.main "Click the blue button on example.com"
```

**Memory demo** — run the same task twice and observe the planner receiving past-experience snippets on the second run:

```bash
uv run python -m agent.main "Search Hacker News for AI agent articles"
uv run python -m agent.main "Search Hacker News for RAG articles"   # planner now has memory context
```

---

## Deployment (Full Docker)

```bash
docker compose up --build
```

This starts ChromaDB and the FastAPI agent server. The agent API is available at `http://localhost:8000`. Interactive docs at `http://localhost:8000/docs`.

---

## Roadmap

- [ ] Semantic memory layer (persistent tool/site knowledge)
- [ ] WebArena benchmark evaluation harness
- [ ] Trajectory viewer UI (React)
- [ ] Multi-tab support in Actor
- [ ] Claude Vision as alternative VLM backend
- [ ] OpenTelemetry tracing integration
- [ ] Kubernetes deployment manifests

---

## License

MIT
