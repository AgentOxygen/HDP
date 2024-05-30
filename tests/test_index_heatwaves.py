import sys  
sys.path.insert(1, '../src')

import heat_stats
import numpy as np
import numba as nb
import pytest


class TestIndexHeatwave:
    def test_index_heatwaves_gufunc(self):
        assert type(heat_stats.index_heatwaves) is nb.np.ufunc.gufunc.GUFunc

    def test_index_heatwaves_null_case(self):
        hot_day_null = np.zeros(100, dtype=bool)
        return_values = np.zeros(hot_day_null.size, dtype=int)
        assert np.array_equal(heat_stats.index_heatwaves(hot_day_null, 1, 1, return_values), np.zeros(hot_day_null.size))
        assert np.array_equal(heat_stats.index_heatwaves(hot_day_null, 0, 1, return_values), np.zeros(hot_day_null.size))
        assert np.array_equal(heat_stats.index_heatwaves(hot_day_null, 0, 0, return_values), np.zeros(hot_day_null.size))

    def test_index_heatwaves_full_case(self):
        hot_day_full = np.ones(100, dtype=bool)
        return_values = np.zeros(hot_day_full.size, dtype=int)
        assert np.array_equal(heat_stats.index_heatwaves(hot_day_full, 1, 1, return_values), np.ones(hot_day_full.size))
        assert np.array_equal(heat_stats.index_heatwaves(hot_day_full, 0, 1, return_values), np.ones(hot_day_full.size))
        assert np.array_equal(heat_stats.index_heatwaves(hot_day_full, 0, 0, return_values), np.ones(hot_day_full.size))

    def test_index_heatwaves_nan_pure_case(self):
        hot_day_nans = np.zeros(100, dtype=bool)*np.nan
        return_values = np.zeros(hot_day_nans.size, dtype=int)
        with pytest.raises(TypeError):
            assert np.array_equal(heat_stats.index_heatwaves(hot_day_nans, 1, 1, return_values), np.zeros(hot_day_nans.size))
        with pytest.raises(TypeError):
            assert np.array_equal(heat_stats.index_heatwaves(hot_day_nans, 0, 1, return_values), np.zeros(hot_day_nans.size))
        with pytest.raises(TypeError):
            assert np.array_equal(heat_stats.index_heatwaves(hot_day_nans, 0, 0, return_values), np.zeros(hot_day_nans.size))

    def test_index_heatwaves_nan_mixed_case(self):
        hot_day_nans = np.zeros(100, dtype=bool)*np.nan
        hot_day_mixednans = np.zeros(100, dtype=bool)*np.nan
        hot_day_mixednans[5:10] = np.ones(5, dtype=bool)
        hot_day_mixednans[50:65] = np.ones(15, dtype=bool)
        return_values = np.zeros(hot_day_mixednans.size, dtype=int)
        with pytest.raises(TypeError):
            assert np.array_equal(heat_stats.index_heatwaves(hot_day_mixednans, 1, 1, return_values), heat_stats.index_heatwaves(hot_day_nans, 1, 1, return_values))
        with pytest.raises(TypeError):
            assert np.array_equal(heat_stats.index_heatwaves(hot_day_mixednans, 0, 1, return_values), heat_stats.index_heatwaves(hot_day_nans, 0, 1, return_values))
        with pytest.raises(TypeError):
            assert np.array_equal(heat_stats.index_heatwaves(hot_day_mixednans, 0, 0, return_values), heat_stats.index_heatwaves(hot_day_nans, 0, 0, return_values))

    def test_index_heatwaves_case1(self):
        hot_day_case1 = np.array(
            [0, 1, 1, 1, 1, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0],
            dtype=bool)
        return_values = np.zeros(hot_day_case1.size, dtype=int)
        assert np.array_equal(heat_stats.index_heatwaves(hot_day_case1, 1, 1, return_values), [0, 1, 1, 1, 1, 0, 1, 1, 1, 0, 0, 0, 2, 2, 2, 2, 0, 0, 0, 0])
        assert np.array_equal(heat_stats.index_heatwaves(hot_day_case1, 0, 1, return_values), [0, 1, 1, 1, 1, 0, 2, 2, 2, 0, 0, 0, 3, 3, 3, 3, 0, 0, 0, 0])
        assert np.array_equal(heat_stats.index_heatwaves(hot_day_case1, 0, 0, return_values), [0, 1, 1, 1, 1, 0, 2, 2, 2, 0, 0, 0, 3, 3, 3, 3, 0, 0, 0, 0])

    def test_index_heatwaves_case2(self):
        hot_day_case2 = np.array(
            [0, 1, 0, 1, 1, 0, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 1],
            dtype=bool)
        return_values = np.zeros(hot_day_case2.size, dtype=int)
        assert np.array_equal(heat_stats.index_heatwaves(hot_day_case2, 1, 1, return_values), [0, 1, 0, 1, 1, 0, 2, 2, 0, 0, 0, 0, 3, 3, 3, 3, 0, 0, 0, 4])
        assert np.array_equal(heat_stats.index_heatwaves(hot_day_case2, 0, 1, return_values), [0, 1, 0, 2, 2, 0, 3, 3, 0, 0, 0, 0, 4, 4, 4, 4, 0, 0, 0, 5])
        assert np.array_equal(heat_stats.index_heatwaves(hot_day_case2, 0, 0, return_values), [0, 1, 0, 2, 2, 0, 3, 3, 0, 0, 0, 0, 4, 4, 4, 4, 0, 0, 0, 5])

    def test_index_heatwaves_case3(self):
        hot_day_case3 = np.array(
            [0, 0, 0, 1, 1, 0, 1, 0, 1, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0],
            dtype=bool)
        return_values = np.zeros(hot_day_case3.size, dtype=int)
        assert np.array_equal(heat_stats.index_heatwaves(hot_day_case3, 1, 1, return_values), [0, 0, 0, 1, 1, 0, 1, 0, 2, 0, 0, 0, 0, 3, 3, 3, 0, 0, 0, 0])
        assert np.array_equal(heat_stats.index_heatwaves(hot_day_case3, 0, 1, return_values), [0, 0, 0, 1, 1, 0, 2, 0, 3, 0, 0, 0, 0, 4, 4, 4, 0, 0, 0, 0])
        assert np.array_equal(heat_stats.index_heatwaves(hot_day_case3, 0, 0, return_values), [0, 0, 0, 1, 1, 0, 2, 0, 3, 0, 0, 0, 0, 4, 4, 4, 0, 0, 0, 0])