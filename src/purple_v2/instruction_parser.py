"""Parse messages from the green agent (evaluator)."""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional

from .grid import Grid, GridConfig


@dataclass
class ParsedInstruction:
    """Structured representation of a green agent message."""
    instruction_text: str
    speaker: str = ""
    start_structure: str = ""
    start_grid: Grid = field(default_factory=Grid)
    task_description: str = ""
    is_feedback: bool = False
    feedback_text: str = ""
    _answered_count: int | None = None


_TASK_RE = re.compile(r"\[TASK_DESCRIPTION\]\s*(.*?)(?=\[SPEAKER\]|\[START_STRUCTURE\]|$)", re.DOTALL)
_SPEAKER_RE = re.compile(r"\[SPEAKER\]\s*(\S+)")
_START_RE = re.compile(r"\[START_STRUCTURE\]\s*(.*?)(?:\n|$)")


def parse_green_message(text: str, config: GridConfig | None = None) -> ParsedInstruction:
    """Parse a green agent message into structured data.

    Expected format:
        [TASK_DESCRIPTION] ...
        [SPEAKER] Name
        [START_STRUCTURE] Color,x,y,z;Color,x,y,z;...
        Instruction text here

    Also handles feedback/answer messages and transition messages.
    """
    config = config or GridConfig()
    text = text.strip()

    # Detect feedback messages
    if text.startswith("Feedback:") or text.startswith("A new task is starting"):
        return ParsedInstruction(
            instruction_text=text,
            is_feedback=True,
            feedback_text=text,
        )

    # Detect answer messages
    if text.startswith("Answer:"):
        return ParsedInstruction(
            instruction_text=text,
            is_feedback=False,
            feedback_text=text,
        )

    result = ParsedInstruction(instruction_text=text)

    # Extract task description
    m = _TASK_RE.search(text)
    if m:
        result.task_description = m.group(1).strip()

    # Extract speaker
    m = _SPEAKER_RE.search(text)
    if m:
        result.speaker = m.group(1).strip()

    # Extract start structure
    m = _START_RE.search(text)
    if m:
        result.start_structure = m.group(1).strip()
        result.start_grid = Grid.from_str(result.start_structure, config=config)

    # Extract the actual instruction (everything after the last tag)
    # Find the position after the last tag
    last_tag_end = 0
    for pattern in [_TASK_RE, _SPEAKER_RE, _START_RE]:
        for match in pattern.finditer(text):
            end = match.end()
            if end > last_tag_end:
                last_tag_end = end

    if last_tag_end > 0:
        instruction = text[last_tag_end:].strip()
        if instruction:
            result.instruction_text = instruction

    return result
