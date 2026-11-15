import pytest

from run_6p_next import KS, VARIANTS, jobs, window_groups
from src.evaluation.stability import nogueira_stability, selection_matrix


def test_job_sets_have_the_planned_sizes_and_no_duplicates():
    sizes = {"splits": 2 * (6 + 3 * 6), "variants": 5 * 2 * 6, "boot": 6 * 20, "speed": 1}
    for name, size in sizes.items():
        lines = jobs(name)
        assert len(lines) == size
        assert len(set(lines)) == size
    assert all(line.split()[2] in VARIANTS for line in jobs("variants"))
    assert KS == sorted(KS)


def test_window_groups_stay_inside_one_excitation_row():
    names = ["E280_em375", "E280_em379", "E280_em380", "E285_em375"]
    assert window_groups(names, 5) == ["E280_w0", "E280_w0", "E280_w1", "E285_w0"]


def test_nogueira_stability_bounds():
    features = list("abcdefgh")
    assert nogueira_stability([["a", "b"], ["a", "b"], ["a", "b"]], features) == pytest.approx(1.0)
    # Наборы без общих признаков — хуже случайного выбора.
    assert nogueira_stability([["a", "b"], ["c", "d"], ["e", "f"], ["g", "h"]], features) < 0
    partial = nogueira_stability([["a", "b", "c"], ["a", "b", "d"], ["a", "c", "d"]], features)
    assert 0 < partial < 1
    assert selection_matrix([["a"], ["b", "c"]], features).sum() == 3
    with pytest.raises(ValueError):
        nogueira_stability([["a"]], features)
    with pytest.raises(ValueError):
        selection_matrix([["z"]], features)
