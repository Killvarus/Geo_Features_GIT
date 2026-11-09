import pandas as pd

from run_6p_selnorm_grid import KS, describe, jobs, ordered_for


def test_jobs_cover_the_grid_once():
    lines = jobs()
    assert len(lines) == 6 + 5 * 4 * 2 * 6
    assert len(set(lines)) == len(lines)
    assert not any("raman" in line for line in lines)
    assert lines[0].startswith("full ")


def test_describe_reads_folder_names():
    assert describe(("max__robust", "fbsbs", "Ni")) == {
        "spectrum_norm": "max",
        "channel_norm": "robust",
        "method": "fbsbs",
        "ion": "Ni",
    }
    assert describe(("all_channels", "Cu"))["method"] == "all_channels"


def test_floating_sets_go_first_and_nothing_is_lost():
    ranking = pd.DataFrame({"Feature": ["a", "b", "c", "d", "e"], "Individual_Rank": [1, 2, 3, 4, 5]})
    sets = {2: ["d", "a"]}
    assert ordered_for("fbsbs", ranking, sets, 2) == ["d", "a", "b", "c", "e"]
    assert ordered_for("bsbs", ranking, {}, 2) == ["a", "b", "c", "d", "e"]
    assert KS == sorted(KS)
