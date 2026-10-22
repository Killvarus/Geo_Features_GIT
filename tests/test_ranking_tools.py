import pandas as pd

from src.models.ranking_tools import spaced_order, spaced_ranking


def test_neighbours_are_postponed_but_kept():
    features = ["E350_em450", "E350_em451", "E350_em452", "E300_em500", "E350_em470"]
    order = spaced_order(features, 10.0)
    assert order[:3] == ["E350_em450", "E300_em500", "E350_em470"]
    assert sorted(order) == sorted(features)


def test_zero_distance_keeps_order():
    features = ["E350_em450", "E350_em451", "E300_em500"]
    assert spaced_order(features, 0.0) == features


def test_spaced_ranking_has_consecutive_ranks():
    ranking = pd.DataFrame(
        {"Feature": ["E350_em451", "E350_em450", "E300_em500"], "Individual_Rank": [2, 1, 3], "Group_Rank": [2, 1, 3]}
    )
    frame = spaced_ranking(ranking, 5.0)
    assert frame["Feature"].tolist() == ["E350_em450", "E300_em500", "E350_em451"]
    assert frame["Individual_Rank"].tolist() == [1, 2, 3]
    assert frame["Score"].tolist() == [-1, -3, -2]
