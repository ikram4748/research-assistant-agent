"""Settings, read from `.env` (see `.env.example`)."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")


def _list(name: str, default: str) -> list[str]:
    return [x.strip() for x in os.getenv(name, default).split(",") if x.strip()]


@dataclass(frozen=True)
class Settings:
    api_keys: list[str] = field(default_factory=lambda: [
        k for k in (os.getenv("GEMINI_API_KEY", ""), os.getenv("GEMINI_API_KEY_2", "")) if k.strip()])
    # the agents' model
    models: list[str] = field(default_factory=lambda: _list("MODELS", "gemini-3.1-flash-lite"))
    # the evaluation judge: a different model from the agents, so the pipeline is not graded by the critic it optimised against
    judge_models: list[str] = field(default_factory=lambda: _list("JUDGE_MODELS", "gemini-3.5-flash-lite"))
    requests_per_minute: int = int(os.getenv("REQUESTS_PER_MINUTE", "12"))
    cache_dir: Path = field(default_factory=lambda: Path(os.getenv("CACHE_DIR", str(ROOT / ".cache"))))
    embed_model: str = os.getenv("EMBED_MODEL", "BAAI/bge-small-en-v1.5")


@lru_cache
def get_settings() -> Settings:
    return Settings()
