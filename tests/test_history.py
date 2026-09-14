from openvision_lab.history import PipelineHistory
from openvision_lab.pipeline import PipelineStep, Processor, default_pipeline


def _processors(steps: list[PipelineStep]) -> list[Processor]:
    return [step.processor for step in steps]


def test_new_history_cannot_undo_or_redo():
    history = PipelineHistory(default_pipeline())

    assert not history.can_undo
    assert not history.can_redo


def test_record_then_undo_and_redo_restore_states():
    history = PipelineHistory([PipelineStep(Processor.GRAYSCALE)])
    history.record(
        [
            PipelineStep(Processor.GRAYSCALE),
            PipelineStep(Processor.GAUSSIAN_BLUR),
        ]
    )

    assert history.can_undo
    undone = history.undo()
    assert _processors(undone) == [Processor.GRAYSCALE]
    assert history.can_redo
    redone = history.redo()
    assert _processors(redone) == [Processor.GRAYSCALE, Processor.GAUSSIAN_BLUR]


def test_undo_at_start_returns_none():
    history = PipelineHistory(default_pipeline())

    assert history.undo() is None


def test_redo_at_end_returns_none():
    history = PipelineHistory(default_pipeline())

    assert history.redo() is None


def test_new_record_discards_redo_branch():
    history = PipelineHistory([PipelineStep(Processor.GRAYSCALE)])
    history.record(
        [
            PipelineStep(Processor.GRAYSCALE),
            PipelineStep(Processor.GAUSSIAN_BLUR),
        ]
    )
    history.undo()
    assert history.can_redo

    history.record([PipelineStep(Processor.BINARY_THRESHOLD)])

    assert not history.can_redo
    assert _processors(history.current()) == [Processor.BINARY_THRESHOLD]


def test_snapshots_are_independent_from_the_original_list():
    steps = [PipelineStep(Processor.GAUSSIAN_BLUR, kernel_size=5)]
    history = PipelineHistory(steps)

    steps[0].kernel_size = 9

    assert history.current()[0].kernel_size == 5


def test_snapshots_are_independent_from_returned_copies():
    history = PipelineHistory([PipelineStep(Processor.GAUSSIAN_BLUR, kernel_size=5)])

    returned = history.current()
    returned[0].kernel_size = 3

    assert history.current()[0].kernel_size == 5


def test_reset_replaces_the_history():
    history = PipelineHistory([PipelineStep(Processor.GRAYSCALE)])
    history.record([PipelineStep(Processor.BINARY_THRESHOLD)])

    history.reset([PipelineStep(Processor.GAUSSIAN_BLUR)])

    assert not history.can_undo
    assert not history.can_redo
    assert _processors(history.current()) == [Processor.GAUSSIAN_BLUR]
