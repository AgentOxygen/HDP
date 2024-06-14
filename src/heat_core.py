#!/usr/bin/env python
"""
heat_core.py

Contains core functions for analyzing timeseries data. 
All methods are static. The 'HeatCore' class serves as a library for hdp.py

Parallelism is enabled by default.

Developer: Cameron Cummins
Contact: cameron.cummins@utexas.edu
"""
from numba import njit, prange,  guvectorize, int64, float32, boolean
import numba as nb
import numpy as np


@njit(parallel=True)
def compute_heatwave_metric(heat_stat_func, season_ranges: np.array, heatwave_indices: np.array):
    metric = np.zeros((season_ranges.shape[0], season_ranges.shape[2], season_ranges.shape[3]), dtype=nb.int64)
    for y in prange(metric.shape[0]):
        for i in prange(metric.shape[1]):
            for j in prange(metric.shape[2]):
                start = season_ranges[y, 0, i, j]
                end = season_ranges[y, 1, i, j]
                metric[y, i, j] = heat_stat_func(heatwave_indices[start:end, i, j])

    return metric


@njit(parallel=True)
def compute_int64_spatial_func(ts_spatial_array, func):
    results = np.zeros(ts_spatial_array.shape, nb.int64)

    for i in prange(ts_spatial_array.shape[1]):
        for j in prange(ts_spatial_array.shape[2]):
            results[:, i, j] = func(ts_spatial_array[:, i, j])
    return results


@njit
def indicate_hot_days(temperatures: np.ndarray, threshold: np.ndarray, doy_map: np.ndarray):
    output = np.zeros(temperatures.shape, dtype=nb.boolean)
    for t in range(temperatures.size):
        doy = doy_map[t]
        if temperatures[t] > threshold[doy]:
            output[t] = True
        else:
            output[t] = False
    return output


def datetimes_to_windows(datetimes: np.ndarray, window_radius: int=7) -> np.ndarray:
    """
    Calculates sample windows for array indices from the datetime dimension 

    datetimes - array of datetime objects corresponding to the dataset's time dimension
    window_radius - radius of windows to generate
    """
    day_of_yr_to_index = {}
    for index, date in enumerate(datetimes):
        if date.dayofyr in day_of_yr_to_index.keys():
            day_of_yr_to_index[date.dayofyr].append(index)
        else:
            day_of_yr_to_index[date.dayofyr] = [index]

    time_index = np.zeros((len(day_of_yr_to_index), np.max([len(x) for x in day_of_yr_to_index.values()])), int) - 1

    for index, day_of_yr in enumerate(day_of_yr_to_index):
        for i in range(len(day_of_yr_to_index[day_of_yr])):
            time_index[index, i] = day_of_yr_to_index[day_of_yr][i]

    window_samples = np.zeros((len(day_of_yr_to_index), 2*window_radius+1, time_index.shape[1]), int)

    for day_of_yr in range(window_samples.shape[0]):
        for window_index in range(window_samples.shape[1]):
            sample_index = day_of_yr + window_radius - window_index
            if sample_index >= time_index.shape[0]:
                sample_index = time_index.shape[0] - sample_index
            window_samples[day_of_yr, window_index] = time_index[sample_index]
    return window_samples.reshape((window_samples.shape[0], window_samples.shape[1]*window_samples.shape[2]))


@njit(parallel=True)
def compute_percentiles_nb(temp_data, window_samples, percentiles):
    """
    Computes the temperatures for multiple percentiles using sample index windows.
    Numba parallel version. Works well on multi-core systems that don't use Dask.

    temp_data - dataset containing temperatures to compute percentiles from
    window_samples - array containing "windows" of indices cenetered at each day of the year
    percentiles - array of perecentiles to compute [0, 1]
    """
    percentile_temp = np.zeros((percentiles.shape[0], window_samples.shape[0], temp_data.shape[1], temp_data.shape[2]), np.float32)

    for doy_index in prange(window_samples.shape[0]):
        sample_time_indices = window_samples[doy_index]

        time_index_size = 0
        for sample_time_index in prange(sample_time_indices.shape[0]):
            if sample_time_indices[sample_time_index] != -1:
                time_index_size += 1

        temp_sample = np.zeros((time_index_size, temp_data.shape[1], temp_data.shape[2]), np.float32)

        time_index = 0
        for sample_time_index in prange(sample_time_indices.shape[0]):
            if sample_time_indices[sample_time_index] != -1:
                temp_sample[time_index] = temp_data[sample_time_indices[sample_time_index]]
                time_index += 1

        for i in prange(temp_sample.shape[1]):
            for j in prange(temp_sample.shape[2]):
                percentile_temp[:, doy_index, i, j] = np.quantile(temp_sample[:, i, j], percentiles)
    return percentile_temp


@nb.guvectorize(
    [(nb.float32[:],
      nb.int64[:, :],
      nb.float64[:],
      nb.float64[:, :])],
    '(t), (d, b), (p) -> (d, p)'
)
def compute_percentiles(temperatures, window_samples, percentiles, output):
    """
    Generalized universal function that computes the temperatures for
    multiple percentiles using sample index windows.

    temperatures - dataset containing temperatures to compute percentiles from
    window_samples - array containing "windows" of indices cenetered at each day of the year
    percentiles - array of perecentiles to compute [0, 1]
    """
    for doy_index in range(window_samples.shape[0]):
        doy_temps = np.zeros(window_samples[doy_index].size)
        for index, temperature_index in enumerate(window_samples[doy_index]):
            doy_temps[index] = temperatures[temperature_index]
        output[doy_index] = np.quantile(doy_temps, percentiles)