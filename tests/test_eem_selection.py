import numpy as np
import pandas as pd

from plot_6p_selection_eem import METHODS
from src.visualization.eem_selection import (
    N_EM,
    channel_coordinates,
    color_limits,
    first_packet,
    mean_train_matrix,
    save_selection_map,
)


def test_named_channel_matches_flattened_column_number():
    excitation, emission = channel_coordinates("E395_em397")
    assert (excitation, emission) == (395, 397)
    row = (395 - 280) // 5
    col = 397 - 375
    number = row * N_EM + col + 1
    assert channel_coordinates(str(number)) == (395, 397)
    assert channel_coordinates("1") == (280, 375)
    assert channel_coordinates("202") == (285, 375)


def test_first_packet_keeps_only_the_best_group_in_rank_order():
    ranking = pd.DataFrame({
        "Feature": ["E280_em376", "E280_em375", "E285_em375"],
        "Group_Rank": [1, 1, 2],
        "Individual_Rank": [2, 1, 3],
        "Score": [0.2, 0.9, 0.1],
    })
    packet = first_packet(ranking)
    assert packet["Feature"].tolist() == ["E280_em375", "E280_em376"]


def test_mean_matrix_uses_train_rows_and_emission_fastest():
    X = np.zeros((10, 27 * 201))
    index = 23 * 201 + 22
    X[:, index] = 50
    matrix = mean_train_matrix(X, n_train=4)
    assert matrix.shape == (27, 201)
    assert matrix[23, 22] == 50
    assert matrix[0, 0] == 0


def test_save_map_marks_requested_channels(tmp_path):
    matrix = np.linspace(0, 10, 27 * 201).reshape(27, 201)
    features = ["E280_em375", "E410_em575"]
    coords = save_selection_map(matrix, features, "проверка", tmp_path / "map.png")
    assert coords == [(280, 375), (410, 575)]
    assert (tmp_path / "map.png").stat().st_size > 1000
    assert color_limits(matrix)[0] == 0.0


def test_method_list_covers_recent_spectroscopy_rankings():
    folders = [item[0] for item in METHODS]
    assert folders == [
        "6p_tbs_per_ion",
        "6p_tbs_per_ion_valid",
        "6p_forward_ridge_valid",
        "6p_forward_ols_valid",
        "6p_ridge_rfe_backward",
        "6p_ridge_rfe_forward",
    ]
