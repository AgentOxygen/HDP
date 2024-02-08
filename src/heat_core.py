#!/usr/bin/env python
"""
heat_core.py

Contains core functions for analyzing timeseries data. 
All methods are static. The 'HeatCore' class serves as a library for hdp.py

Parallelism is enabled by default.

Developer: Cameron Cummins
Contact: cameron.cummins@utexas.edu
2/8/24
"""
from numba import njit, prange
import numba as nb
import numpy as np


class HeatCore:
    @staticmethod
    @njit(parallel=True)
    def spatial_looper(hw_array, func):
        results = np.zeros(hw_array.shape[:-1], dtype=type(func(np.array([0]))))
        for i in prange(hw_array.shape[0]):
            for j in prange(hw_array.shape[1]):
                results[i, j] = func(hw_array[i, j])
        return results
    
    
    @staticmethod
    @njit(parallel=True)
    def index_heatwaves(hot_days, max_break: int=1, min_duration: int=3) -> np.ndarray:
        """
        Identifies the heatwaves in the timeseries using the specified heatwave definition

        Keyword arguments:
        timeseries -- integer array of ones and zeros where ones indicates a hot day (numpy.ndarray)
        max_break -- the maximum number of days between hot days within one heatwave event (default 1)
        min_duration -- the minimum number of hot days to constitute a heatwave event, including after breaks (default 3)
        """
        indexed_heatwaves = np.zeros(hot_days.shape, dtype=nb.int64)
        for p_index in prange(hot_days.shape[0]):
            for i in prange(hot_days.shape[2]):
                for j in prange(hot_days.shape[3]):                    
                    timeseries = np.zeros(hot_days[p_index, :, i, j].shape[0] + 2, dtype=nb.int64)
                    timeseries[1:timeseries.shape[0] - 1] = hot_days[p_index, :, i, j]
                    
                    timeseries_diff = np.zeros(timeseries.shape[0]-1, dtype=nb.int64)
                    for index in range(timeseries_diff.shape[0]):
                        timeseries_diff[index] = timeseries[index+1] - x[index]
                    
                    diff_indices = np.where(timeseries_diff != 0)[0] + 1

                    in_heatwave = False
                    current_hw_index = 1

                    hw_indices = np.zeros(timeseries.shape, dtype=nb.int64)

                    broken = False
                    for i in range(diff_indices.shape[0]-1):
                        index = diff_indices[i]
                        next_index = diff_indices[i+1]

                        if timeseries[index] == True and in_heatwave:
                            hw_indices[index:next_index] = current_hw_index
                        elif timeseries[index] == False and in_heatwave and next_index-index <= max_break and not broken:
                            hw_indices[index:next_index] = current_hw_index
                            broken = True
                        elif timeseries[index] == True and not in_heatwave and next_index-index >= min_duration:
                            in_heatwave = True
                            hw_indices[index:next_index] = current_hw_index
                        elif in_heatwave:
                            current_hw_index += 1
                            in_heatwave = False
                            broken = False
                    indexed_heatwaves[p_index, :, i, j] = timeseries[1:-1]*hw_indices[1:-1]
        return indexed_heatwaves
    
    
    @staticmethod
    def indicate_hot_days(temp_ds: xarray.DataArray, threshold: xarray.DataArray) -> np.ndarray:
        """
        Marks days in the temperature input that exceed the daily thresholds.

        Keyword arguments:
        temp_ds -- temperature dataset to use as input (xarray.DataArray)
        threshold -- threshold dataset to compare the input against (xarray.DataArray)
        """
        hot_days = np.zeros((threshold.shape[0],) + temp_ds.values.shape, dtype=int)

        for index in range(temp_ds.time.values.size):
            day_number = temp_ds.time.values[index].dayofyr
            hot_days[:, index] = (temp_ds.values[index] > threshold.values[:, day_number-1])
        return hot_days
    
    
    @staticmethod
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


    @staticmethod
    @njit(parallel=True)
    def compute_percentiles(temp_data, window_samples, percentiles):
        """
        Computes the temperatures for multiple percentiles using sample index windows.

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
