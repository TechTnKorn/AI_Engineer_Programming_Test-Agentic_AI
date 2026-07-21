"""Define the configurable parameters for the Retriever/Generator agent pair."""

from __future__ import annotations

import os
from dataclasses import dataclass, field, fields

from . import prompts


@dataclass(kw_only=True)
class Context:
    """Runtime-configurable settings for the RAG multi-agent graph."""

    retriever_system_prompt: str = field(
        default=prompts.RETRIEVER_SYSTEM_PROMPT,
        metadata={"description": "The system prompt for the Data Retriever agent."},
    )

    generator_system_prompt: str = field(
        default=prompts.GENERATOR_SYSTEM_PROMPT,
        metadata={
            "description": "The system prompt template for the Report Generator agent."
        },
    )

    top_k: int = field(
        default=3,
        metadata={
            "description": "The maximum number of knowledge-base snippets the "
            "Data Retriever tool should return per query."
        },
    )

    def __post_init__(self) -> None:
        """Fetch env vars for attributes that were not passed as args."""
        for f in fields(self):
            if not f.init:
                continue

            if getattr(self, f.name) == f.default:
                setattr(self, f.name, os.environ.get(f.name.upper(), f.default))
