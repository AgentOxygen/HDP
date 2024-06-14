#!/usr/bin/env python
"""
heat_core.py

Contains function definitions for various heatwave statistics.
All methods are static and should be called when computing heatwave metrics.

Developer: Cameron Cummins
Contact: cameron.cummins@utexas.edu
"""
from numba import njit
import numba as nb
import numpy as np


@njit
def index_heatwaves(hot_days_ts: np.ndarray, min_duration: int, max_break: int, max_subs: int) -> np.ndarray:
    """
    Identifies the heatwaves in the timeseries using the specified heatwave definition

    Keyword arguments:
    timeseries -- integer array of ones and zeros where ones indicates a hot day (numpy.ndarray)
    max_break -- the maximum number of days between hot days within one heatwave event (default 1)
    min_duration -- the minimum number of hot days to constitute a heatwave event, including after breaks (default 3)
    max_subs -- the maximum number of subsequent events allowed to be apart of the initial consecutive hot days
    """
    ts = np.zeros(hot_days_ts.size + 2, dtype=nb.int64)
    for i in range(0, hot_days_ts.size):
        if hot_days_ts[i]:
            ts[i + 1] = 1
    diff_ts = np.diff(ts)
    diff_indices = np.where(diff_ts != 0)[0]

    in_heatwave = False
    current_hw_index = 0
    sub_events = 0
    hw_indices = np.zeros(diff_ts.size, dtype=nb.int64)

    for i in range(diff_indices.size-1):
        index = diff_indices[i]
        next_index = diff_indices[i+1]

        if diff_ts[index] == 1 and next_index - index >= min_duration and not in_heatwave:
            current_hw_index += 1
            in_heatwave = True
            hw_indices[index:next_index] = current_hw_index
        elif diff_ts[index] == -1 and next_index - index > max_break:
            in_heatwave = False
        elif diff_ts[index] == 1 and in_heatwave and sub_events < max_subs:
            sub_events += 1
            hw_indices[index:next_index] = current_hw_index
        elif diff_ts[index] == 1 and in_heatwave and sub_events >= max_subs:
            if next_index - index >= min_duration:
                current_hw_index += 1
                hw_indices[index:next_index] = current_hw_index
            else:
                in_heatwave = False
            sub_events = 0

    return hw_indices[0:hw_indices.size-1]

@njit
def heatwave_frequency(hw_ts: np.ndarray, season_ranges: np.ndarray) -> np.ndarray:
    output = np.zeros(season_ranges.shape[0], dtype=nb.int64)
    for y in range(season_ranges.shape[0]):
        end_points = season_ranges[y]
        output[y] = np.sum(hw_ts[end_points[0]:end_points[1]] > 0, dtype=nb.int64)
    return output

@njit
def heatwave_duration(hw_ts: np.ndarray, season_ranges: np.ndarray) -> np.ndarray:
    output = np.zeros(season_ranges.shape[0], dtype=nb.int64)
    for y in range(season_ranges.shape[0]):
        end_points = season_ranges[y]
        hw_ts_slice = hw_ts[end_points[0]:end_points[1]]
        for value in np.unique(hw_ts_slice):
            index_count = 0
            if value != 0:
                for day in hw_ts_slice:
                    if day == value:
                        index_count += 1
            if index_count > output[y]:
                output[y] = index_count
    return output