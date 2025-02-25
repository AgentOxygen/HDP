from hdp.metric import index_heatwaves
import numpy as np
import pytest


class TestIndexHeatwave:
    def test_index_heatwaves_null_case(self):
        hot_day_null = np.zeros(100, dtype=bool)
        assert np.array_equal(index_heatwaves(hot_day_null, 1, 1, 1), np.zeros(hot_day_null.size))
        assert np.array_equal(index_heatwaves(hot_day_null, 1, 0, 1), np.zeros(hot_day_null.size))
        assert np.array_equal(index_heatwaves(hot_day_null, 0, 0, 1), np.zeros(hot_day_null.size))

    def test_index_heatwaves_full_case(self):
        hot_day_full = np.ones(100, dtype=bool)
        assert np.array_equal(index_heatwaves(hot_day_full, 1, 1, 1), np.ones(hot_day_full.size))
        assert np.array_equal(index_heatwaves(hot_day_full, 1, 0, 1), np.ones(hot_day_full.size))
        assert np.array_equal(index_heatwaves(hot_day_full, 0, 0, 1), np.ones(hot_day_full.size))

    def test_index_heatwaves_case1(self):
        hot_day_case1 = np.array(
            [0, 1, 1, 1, 1, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0],
            dtype=bool)
        assert np.array_equal(index_heatwaves(hot_day_case1, 1, 1, 1), [0, 1, 1, 1, 1, 0, 1, 1, 1, 0, 0, 0, 2, 2, 2, 2, 0, 0, 0, 0])
        assert np.array_equal(index_heatwaves(hot_day_case1, 1, 0, 1), [0, 1, 1, 1, 1, 0, 2, 2, 2, 0, 0, 0, 3, 3, 3, 3, 0, 0, 0, 0])
        assert np.array_equal(index_heatwaves(hot_day_case1, 0, 0, 1), [0, 1, 1, 1, 1, 0, 2, 2, 2, 0, 0, 0, 3, 3, 3, 3, 0, 0, 0, 0])

    def test_index_heatwaves_case2(self):
        hot_day_case2 = np.array(
            [0, 1, 0, 1, 1, 0, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 1],
            dtype=bool)
        assert np.array_equal(index_heatwaves(hot_day_case2, 1, 1, 1), [0, 1, 0, 1, 1, 0, 2, 2, 0, 0, 0, 0, 3, 3, 3, 3, 0, 0, 0, 4])
        assert np.array_equal(index_heatwaves(hot_day_case2, 1, 0, 1), [0, 1, 0, 2, 2, 0, 3, 3, 0, 0, 0, 0, 4, 4, 4, 4, 0, 0, 0, 5])
        assert np.array_equal(index_heatwaves(hot_day_case2, 0, 0, 1), [0, 1, 0, 2, 2, 0, 3, 3, 0, 0, 0, 0, 4, 4, 4, 4, 0, 0, 0, 5])

    def test_index_heatwaves_case3(self):
        hot_day_case3 = np.array(
            [0, 0, 0, 1, 1, 0, 1, 0, 1, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0],
            dtype=bool)
        assert np.array_equal(index_heatwaves(hot_day_case3, 1, 1, 1), [0, 0, 0, 1, 1, 0, 1, 0, 2, 0, 0, 0, 0, 3, 3, 3, 0, 0, 0, 0])
        assert np.array_equal(index_heatwaves(hot_day_case3, 1, 0, 1), [0, 0, 0, 1, 1, 0, 2, 0, 3, 0, 0, 0, 0, 4, 4, 4, 0, 0, 0, 0])
        assert np.array_equal(index_heatwaves(hot_day_case3, 0, 0, 1), [0, 0, 0, 1, 1, 0, 2, 0, 3, 0, 0, 0, 0, 4, 4, 4, 0, 0, 0, 0])