from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path

import chromadb

from agent.config import get_settings
from agent.models import MemoryEntry

logger = logging.getLogger(__name__)
cfg = get_settings()


def _get_chroma_client():
    """
    Return a ChromaDB client.

    - CHROMA_MODE=http  → HttpClient (points at Docker / remote server)
    - CHROMA_MODE=local → PersistentClient (local disk, no server needed)

    Default is 'local' so the agent works out-of-the-box without Docker.
    """
    if cfg.chroma_mode == "http":
        logger.info("ChromaDB: connecting to HTTP server %s:%s", cfg.chroma_host, cfg.chroma_port)
        return chromadb.HttpClient(host=cfg.chroma_host, port=cfg.chroma_port)

    persist_dir = str(Path(cfg.chroma_persist_dir).resolve())
    logger.info("ChromaDB: using local PersistentClient at %s", persist_dir)
    return chromadb.PersistentClient(path=persist_dir)


class EpisodicMemory:
    """
    Stores and retrieves summaries of past agent runs (trajectories).
    Uses ChromaDB for vector similarity search.
    """

    def __init__(self) -> None:
        from openai import AsyncOpenAI
        self._chroma = _get_chroma_client()
        self._col = self._chroma.get_or_create_collection(
            name=cfg.episodic_collection,
            metadata={"hnsw:space": "cosine"},
        )
        self._openai = AsyncOpenAI(api_key=cfg.openai_api_key)

    # ──────────────────────────────────────────────────────────────────────────

    async def store(self, entry: MemoryEntry) -> None:
        embedding = await self._embed(entry.embedding_text)
        self._col.add(
            ids=[entry.id],
            embeddings=[embedding],
            documents=[entry.embedding_text],
            metadatas=[{
                "task":        entry.task,
                "summary":     entry.summary,
                "outcome":     entry.outcome,
                "key_actions": "||".join(entry.key_actions),
                "lessons":     "||".join(entry.lessons),
                "created_at":  entry.created_at.isoformat(),
            }],
        )
        logger.info("EpisodicMemory: stored entry id=%s outcome=%s", entry.id, entry.outcome)

    async def retrieve(self, query: str, top_k: int | None = None) -> list[MemoryEntry]:
        k = top_k or cfg.memory_top_k
        count = self._col.count()
        if count == 0:
            return []

        embedding = await self._embed(query)
        results = self._col.query(
            query_embeddings=[embedding],
            n_results=min(k, count),
            include=["documents", "metadatas", "distances"],
        )

        entries: list[MemoryEntry] = []
        for meta, dist in zip(results["metadatas"][0], results["distances"][0]):
            similarity = 1.0 - dist
            if similarity < cfg.memory_similarity_threshold:
                continue
            entries.append(MemoryEntry(
                task=meta["task"],
                summary=meta["summary"],
                outcome=meta["outcome"],
                key_actions=meta["key_actions"].split("||"),
                lessons=meta["lessons"].split("||"),
                embedding_text=meta.get("embedding_text", meta["summary"]),
                created_at=datetime.fromisoformat(meta["created_at"]),
            ))

        logger.info("EpisodicMemory: retrieved %d memories for query", len(entries))
        return entries

    async def build_memory_snippets(self, task: str) -> list[str]:
        memories = await self.retrieve(task)
        snippets = []
        for m in memories:
            outcome_tag = "✓" if m.outcome == "success" else "✗"
            lessons = "; ".join(m.lessons[:3])
            snippets.append(f"[{outcome_tag} similar task] {m.summary} | Lessons: {lessons}")
        return snippets

    async def _embed(self, text: str) -> list[float]:
        resp = await self._openai.embeddings.create(
            model="text-embedding-3-small",
            input=text[:8192],
        )
        return resp.data[0].embedding