"""
Trace Models

A Trace represents a complete agent interaction consisting of multiple steps.
"""

from typing import Sequence

from pydantic import BaseModel, Field

from .steps import Step


class Trace(BaseModel):
    """
    A complete agent trace consisting of multiple steps.

    A trace represents the full record of an agent's interaction,
    from initial prompt through reasoning, function calls, and final answer.
    """

    steps: list[Step] = Field(default_factory=list, description="Ordered list of steps in the trace")

    def append(self, step: Step) -> None:
        """Add a step to the trace."""
        self.steps.append(step)

    def to_string(self) -> str:
        """Convert trace to a string by concatenating all step diffs."""
        return "".join(step.diff for step in self.steps)

    def __len__(self) -> int:
        return len(self.steps)

    def __getitem__(self, index: int) -> Step:
        return self.steps[index]

    def __iter__(self):
        return iter(self.steps)

    @property
    def last_step(self) -> Step | None:
        """Get the last step in the trace."""
        return self.steps[-1] if self.steps else None
