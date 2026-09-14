"""Undo/redo history for pipeline edits.

The history stores snapshots of the pipeline (its steps and parameters), never
image arrays. Undo and redo only swap the stored snapshots; the caller is
responsible for re-running the pipeline to refresh the preview.
"""

from dataclasses import replace

from openvision_lab.pipeline import PipelineStep


def clone_steps(steps: list[PipelineStep]) -> list[PipelineStep]:
    """Return an independent copy of a pipeline step list.

    ``PipelineStep`` is mutable, so snapshots must be copied to keep the stored
    states from changing when the live pipeline is edited.
    """
    return [replace(step) for step in steps]


class PipelineHistory:
    """Undo/redo history built from pipeline snapshots.

    Snapshot 0 is the oldest state and the last snapshot is the most recent.
    Recording after an undo drops the states that were ahead (the redo branch).
    """

    def __init__(self, steps: list[PipelineStep]) -> None:
        self._states: list[list[PipelineStep]] = [clone_steps(steps)]
        self._index = 0

    @property
    def can_undo(self) -> bool:
        """Whether there is an older snapshot to return to."""
        return self._index > 0

    @property
    def can_redo(self) -> bool:
        """Whether there is a newer snapshot to return to."""
        return self._index < len(self._states) - 1

    def current(self) -> list[PipelineStep]:
        """Return a copy of the current snapshot."""
        return clone_steps(self._states[self._index])

    def record(self, steps: list[PipelineStep]) -> None:
        """Append a new snapshot, discarding any redo branch."""
        del self._states[self._index + 1 :]
        self._states.append(clone_steps(steps))
        self._index = len(self._states) - 1

    def undo(self) -> list[PipelineStep] | None:
        """Move one snapshot back and return it, or ``None`` at the start."""
        if not self.can_undo:
            return None
        self._index -= 1
        return self.current()

    def redo(self) -> list[PipelineStep] | None:
        """Move one snapshot forward and return it, or ``None`` at the end."""
        if not self.can_redo:
            return None
        self._index += 1
        return self.current()

    def reset(self, steps: list[PipelineStep]) -> None:
        """Replace the whole history with a single snapshot."""
        self._states = [clone_steps(steps)]
        self._index = 0
