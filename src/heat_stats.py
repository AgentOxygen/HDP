#!/usr/bin/env python
"""
heat_core.py

Contains function definitions for various heatwave statistics.
All methods are static and should be called when computing heatwave metrics.

Developer: Cameron Cummins
Contact: cameron.cummins@utexas.edu
2/8/24
"""
from numba import njit
import numba as nb
import numpy as np


class HeatStats:
    @staticmethod
    @njit
    def index_heatwaves(hot_days_ts: np.ndarray, max_break: int=1, min_duration: int=3) -> np.ndarray:
        """
        Identifies the heatwaves in the timeseries using the specified heatwave definition

        Keyword arguments:
        timeseries -- integer array of ones and zeros where ones indicates a hot day (numpy.ndarray)
        max_break -- the maximum number of days between hot days within one heatwave event (default 1)
        min_duration -- the minimum number of hot days to constitute a heatwave event, including after breaks (default 3)
        """
        timeseries = np.zeros(hot_days_ts.shape[0] + 2, dtype=nb.int64)
        timeseries[1:timeseries.shape[0]-1] = hot_days_ts

        diff_indices = np.where(np.diff(timeseries) != 0)[0] + 1

        in_heatwave = False
        current_hw_index = 1

        hw_indices = np.zeros(timeseries.shape, dtype=nb.int64)

        broken = False
        for i in range(diff_indices.shape[0]-1):
            index = diff_indices[i]
            next_index = diff_indices[i+1]

            if timeseries[index] == 1 and in_heatwave:
                hw_indices[index:next_index] = current_hw_index
            elif timeseries[index] == 0 and in_heatwave and next_index-index <= max_break and not broken:
                hw_indices[index:next_index] = current_hw_index
                broken = True
            elif timeseries[index] == 1 and not in_heatwave and next_index-index >= min_duration:
                in_heatwave = True
                hw_indices[index:next_index] = current_hw_index
            elif in_heatwave:
                current_hw_index += 1
                in_heatwave = False
                broken = False
        return timeseries[1:-1]*hw_indices[1:-1]

    @staticmethod
    @njit
    def heatwave_frequency(hw_ts: np.array) -> int:
        return np.sum(hw_ts > 0)

    @staticmethod
    @njit
    def heatwave_duration(hw_ts: np.array) -> int:
        hwd = 0
        for value in np.unique(hw_ts):
            index_count = 0
            if value != 0:
                for day in hw_ts:
                    if day == value:
                        index_count += 1
            if index_count > hwd:
                hwd = index_count
        return hwd