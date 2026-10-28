"""test_add_permutation_scores.py - both p value paths on three positions

input:   ACGT counts small enough to enumerate and one deep enough to draw
output:  pytest assertions
usage:   pytest tests/test_add_permutation_scores.py
env:     numpy and scipy, reached through the module under test
notes:   the pmf takes integer counts while the scorer takes floats, and the
         enumerated path is where the two meet
"""
import numpy as np
import pandas as pd
import pytest
from add_permutation_scores import counts_for, permutation_test

COUNTS1 = np.array([[8, 1, 0, 1], [30, 2, 1, 1], [400, 60, 20, 20]],
                   dtype=float)
COUNTS2 = np.array([[7, 2, 1, 0], [28, 9, 1, 2], [380, 20, 30, 70]],
                   dtype=float)
DRAWS = 50


@pytest.fixture(scope="module")
def result():
    return permutation_test((COUNTS1, COUNTS2), qual=20, seed=42, draws=DRAWS)


def test_both_paths_run(result):
    assert result["p_exact"].tolist() == [True, True, False]


def test_every_p_is_a_probability(result):
    assert np.all(np.isfinite(result["pvalue"]))
    assert np.all((result["pvalue"] > 0) & (result["pvalue"] <= 1))


def test_a_drawn_p_stops_at_the_sampling_limit(result):
    assert result["pvalue"][2] >= 1 / (DRAWS + 1)


def test_the_margin_is_the_score_above_the_null(result):
    assert result["score_minus_max_null"][2] > 0
    assert np.all(result["score_minus_max_null"] >= 0)


@pytest.mark.parametrize("error_type", ["mismatch", "deletion", "insertion"])
def test_replicates_count_as_one_pooled_sample(error_type):
    replicated = pd.DataFrame({
        "bases11": ["8,1,0,1"], "bases12": ["2,0,1,0"], "bases21": ["7,2,1,0"],
        "deletions11": [3], "deletions12": [1], "deletions21": [2],
        "insertions11": [1], "insertions12": [2], "insertions21": [0]})
    pooled = pd.DataFrame({
        "bases11": ["10,1,1,1"], "bases21": ["7,2,1,0"],
        "deletions11": [4], "deletions21": [2],
        "insertions11": [3], "insertions21": [0]})

    for got, want in zip(counts_for(replicated, error_type),
                         counts_for(pooled, error_type)):
        np.testing.assert_array_equal(got, want)
