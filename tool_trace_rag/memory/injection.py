from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class MemoryPromptContext:
    prompt_section: str
    metadata: dict[str, Any]

    @property
    def has_prompt(self) -> bool:
        return bool(self.prompt_section.strip())
