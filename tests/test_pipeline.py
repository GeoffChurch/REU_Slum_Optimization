from reblock.pipeline import RunOutput, sample


def test_sample_takes_first_n_of_a_list() -> None:
    assert sample(["a", "b", "c", "d"], 2) == ["a", "b"]


def test_sample_passes_through_all() -> None:
    assert sample(None, 5) is None            # None = ALL


def test_sample_n_larger_than_selection() -> None:
    assert sample(["a", "b"], 10) == ["a", "b"]


def test_runoutput_holds_selection_and_results() -> None:
    out = RunOutput(selection=["a", "b", "c"], results=[])
    assert out.selection == ["a", "b", "c"] and out.results == []
