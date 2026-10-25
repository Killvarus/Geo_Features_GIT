from run_6p_norm_grid import CF_SKIP, cf_combos, describe, jobs


def test_job_sets_have_expected_sizes():
    assert len(jobs("base")) == 6 * 6 + 6 * 4 * 2 * 6
    assert len(jobs("chfirst")) == 13 * 6 + 13 * 2 * 6
    assert len(jobs("all")) == len(jobs("base")) + len(jobs("chfirst"))
    assert len(set(jobs("all"))) == len(jobs("all"))
    assert {("zscore", "area"), ("robust", "area")} == CF_SKIP
    assert not CF_SKIP & set(cf_combos())


def test_describe_tells_the_two_orders_apart():
    assert describe(("max__zscore", "rfe", "Cu"))["order"] == "spectrum_first"
    assert describe(("max__all_channels", "Cu"))["method"] == "all_channels"
    variant = describe(("max__pareto__chfirst", "all_channels", "Ni"))
    assert variant == {
        "order": "channels_first",
        "spectrum_norm": "max",
        "channel_norm": "pareto",
        "method": "all_channels",
        "ion": "Ni",
    }
    assert describe(("row_max__robust__chfirst", "ppop", "Al"))["spectrum_norm"] == "row_max"
