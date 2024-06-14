from src import heat_stats 
import numpy as np
import numba as nb
import pytest


class TestHeatwaveFrequency:
    def test_heatwave_frequency_gufunc(self):
        assert type(heat_stats.heatwave_frequency) is nb.np.ufunc.gufunc.GUFunc

    def test_heatwave_frequency_null_case(self):
        hot_day_null = np.zeros(100, dtype=int)
        season_range = np.array([[0, 100]], dtype=int)
        return_values = np.zeros(season_range.shape[0], dtype=int)
        assert np.array_equal(heat_stats.heatwave_frequency(hot_day_null, season_range, return_values), np.array([0]))


    def test_heatwave_frequency_full_case(self):
        hot_day_full = np.ones(100, dtype=bool)
        season_range = np.array([[0, 100]], dtype=int)
        return_values = np.zeros(season_range.shape[0], dtype=int)
        assert np.array_equal(heat_stats.heatwave_frequency(hot_day_full, season_range, return_values), np.array([100]))


    def test_heatwave_frequency_nan_mixed_case(self):
        hot_day_nans = np.zeros(100, dtype=int)*np.nan
        hot_day_mixednans = np.zeros(100, dtype=int)*np.nan
        hot_day_mixednans[5:10] = np.ones(5, dtype=int)
        hot_day_mixednans[50:65] = np.ones(15, dtype=int)
        season_range = np.array([[0, 100]], dtype=int)
        return_values = np.zeros(season_range.shape[0], dtype=int)
        with pytest.raises(TypeError):
            assert np.array_equal(heat_stats.heatwave_frequency(hot_day_mixednans, season_range, return_values), np.array([20]))

    
    def test_heatwave_frequency_case1(self):
        hot_day_case1 = np.array(
            [0, 1, 1, 1, 1, 0, 1, 1, 1, 0, 0, 0, 2, 2, 2, 2, 0, 0, 0, 0],
            dtype=int)
        season_range = np.array([[0, hot_day_case1.size]], dtype=int)
        return_values = np.zeros(season_range.shape[0], dtype=int)
        assert np.array_equal(heat_stats.heatwave_frequency(hot_day_case1, season_range, return_values), np.array([11]))

    
    def test_heatwave_frequency_case2(self):
        hot_day_case2 = np.array(
            [0, 1, 0, 1, 1, 0, 1, 1, 0, 0, 0, 0, 2, 2, 2, 2, 0, 0, 0, 3],
            dtype=int)
        season_range = np.array([[0, hot_day_case2.size]], dtype=int)
        return_values = np.zeros(season_range.shape[0], dtype=int)
        assert np.array_equal(heat_stats.heatwave_frequency(hot_day_case2, season_range, return_values), np.array([10]))


    def test_heatwave_frequency_case3(self):
        hot_day_case3 = np.array(
            [0, 0, 0, 1, 1, 0, 2, 0, 3, 0, 0, 0, 0, 4, 4, 4, 0, 0, 0, 0],
            dtype=int)
        season_range = np.array([[0, hot_day_case3.size]], dtype=int)
        return_values = np.zeros(season_range.shape[0], dtype=int)
        assert np.array_equal(heat_stats.heatwave_frequency(hot_day_case3, season_range, return_values), np.array([7]))


    def test_heatwave_frequency_season_full_case(self):
        hot_day_full = np.ones(100, dtype=bool)
        season_range = np.array([[0, 5], [0, 10], [20, 30], [42, 50]], dtype=int)
        return_values = np.zeros(season_range.shape[0], dtype=int)
        assert np.array_equal(heat_stats.heatwave_frequency(hot_day_full, season_range, return_values), np.array([5, 10, 10, 8]))
    

    def test_heatwave_frequency_season_case1(self):
        hot_day_case3 = np.array(
            [0, 1, 1, 1, 1, 0, 1, 1, 1, 0, 0, 0, 2, 2, 2, 2, 0, 0, 0, 0],
            dtype=int)
        season_range = np.array([[0, 8], [8, hot_day_case3.size]], dtype=int)
        return_values = np.zeros(season_range.shape[0], dtype=int)
        assert np.array_equal(heat_stats.heatwave_frequency(hot_day_case3, season_range, return_values), np.array([6, 5]))

    def test_heatwave_frequency_season_case2(self):
        hot_day_case3 = np.array(
            [0, 0, 0, 1, 1, 0, 2, 0, 3, 0, 0, 0, 0, 4, 4, 4, 0, 0, 0, 0],
            dtype=int)
        season_range = np.array([[0, 8], [8, hot_day_case3.size]], dtype=int)
        return_values = np.zeros(season_range.shape[0], dtype=int)
        assert np.array_equal(heat_stats.heatwave_frequency(hot_day_case3, season_range, return_values), np.array([3, 4]))
