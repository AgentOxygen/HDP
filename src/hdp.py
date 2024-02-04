#!/usr/bin/env python
"""
hdp.py

Heatwave Diagnostics Package

Contains all functions necessary for computing heatwave thresholds and metrics using numpy with xarray wrapper functions.
"""
import xarray
import numpy as np
from numba import jit
from datetime import datetime
from scipy import stats


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
    

@jit(nopython=True)
def compute_percentiles(temp_data, window_samples, percentiles):
    """
    Computes the temperatures for multiple percentiles using sample index windows.
    
    temp_data - dataset containing temperatures to compute percentiles from
    window_samples - array containing "windows" of indices cenetered at each day of the year
    percentiles - array of perecentiles to compute [0, 1]
    """
    percentile_temp = np.zeros((percentiles.shape[0], window_samples.shape[0], temp_data.shape[1], temp_data.shape[2]), np.float32)

    for doy_index in range(window_samples.shape[0]):
        sample_time_indices = window_samples[doy_index]
        
        time_index_size = 0
        for sample_time_index in range(sample_time_indices.shape[0]):
            if sample_time_indices[sample_time_index] != -1:
                time_index_size += 1

        temp_sample = np.zeros((time_index_size, temp_data.shape[1], temp_data.shape[2]), np.float32)

        time_index = 0
        for sample_time_index in range(sample_time_indices.shape[0]):
            if sample_time_indices[sample_time_index] != -1:
                temp_sample[time_index] = temp_data[sample_time_indices[sample_time_index]]
                time_index += 1

        for i in range(temp_sample.shape[1]):
            for j in range(temp_sample.shape[2]):
                percentile_temp[:, doy_index, i, j] = np.quantile(temp_sample[:, i, j], percentiles)
        
    return percentile_temp


def compute_threshold(temperature_dataset: xarray.DataArray, percentiles: np.ndarray, temp_path: str="No path provided.") -> xarray.DataArray:
    """
    Computes day-of-year quantile temperatures for given temperature dataset and percentile. The output is used as the threshold input for 'heatwave_metrics.py'.
    
    Keyword arguments:
    temperature_data -- Temperature dataset to compute quantiles from
    percentile -- Percentile to compute the quantile temperatures at
    temp_path -- Path to 'temperature_data' temperature dataset to add to meta-data
    """
    
    window_samples = gen_windowed_samples(temperature_dataset, 7)
    annual_threshold = compute_percentile_thresholds(temperature_dataset.values, window_samples, percentiles)
    
    return xarray.Dataset(
        data_vars=dict(
            threshold=(["percentile", "day", "lat", "lon"], annual_threshold),
        ),
        coords=dict(
            lon=(["lon"], temperature_dataset.lon.values),
            lat=(["lat"], temperature_dataset.lat.values),
            day=np.arange(0, num_days),
            percentile=percentiles
        ),
        attrs={
            "description": f"Percentile temperatures.",
            "percentiles": str(percentile),
            "temperature dataset path": temp_path
        },
    )


def indicate_hot_days(temp_ds: xarray.DataArray, threshold: xarray.DataArray) -> np.ndarray:
    """
    Marks days in the temperature input that exceed the daily thresholds.
    
    Keyword arguments:
    temp_ds -- temperature dataset to use as input (xarray.DataArray)
    threshold -- threshold dataset to compare the input against (xarray.DataArray)
    """
    hot_days = np.zeros(temp_ds.values.shape, dtype=int)
    
    for index in range(temp_ds.time.values.size):
        day_number = temp_ds.time.values[index].dayofyr
        hot_days[index] = (temp_ds.values[index] > threshold.values[day_number-1])

    return hot_days


def index_heatwaves(timeseries: np.ndarray, max_break: int=1, min_duration: int=3) -> np.ndarray:
    """
    Identifies the heatwaves in the timeseries using the specified heatwave definition
    
    Keyword arguments:
    timeseries -- integer array of ones and zeros where ones indicates a hot day (numpy.ndarray)
    max_break -- the maximum number of days between hot days within one heatwave event (default 1)
    min_duration -- the minimum number of hot days to constitute a heatwave event, including after breaks (default 3)
    """
    timeseries = np.pad(timeseries, 1)
    
    diff_indices = np.where(np.diff(timeseries) != 0)[0] + 1

    in_heatwave = False
    current_hw_index = 1

    hw_indices = np.zeros(timeseries.shape, dtype=np.short)

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


def compute_metrics(temp_ds: xarray.DataArray, control_threshold: xarray.DataArray, temp_path: str="No path provided.", control_path: str="No path provided.") -> xarray.Dataset:
    """
    Computes the relevant heatwave metrics for a given temperature dataset and threshold.
    
    Keyword arguments:
    temp_ds -- Temperature dataset to compare against threshold and compute heatwave metrics for
    control_threshold -- Day-of-year temperature dataset to use as threshold for heatwave days
    temp_path -- Path to 'temp_ds' temperature dataset to add to meta-data
    control_path -- Path to 'control_threshold' threshold temperature dataset to add to meta-data
    """
    hot_days = indicate_hot_days(temp_ds, control_threshold)
    indexed_heatwaves = np.zeros(hot_days.shape, dtype=np.short)

    for i in range(hot_days.shape[1]):
        for j in range(hot_days.shape[2]):
            indexed_heatwaves[:, i, j] = index_heatwaves(hot_days[:, i, j])

    num_index_heatwaves = indexed_heatwaves > 0
    years = temp_ds.time.values[-1].year - temp_ds.time.values[0].year + 1

    south_hemisphere = np.ones((int(temp_ds.shape[1]/2), temp_ds.shape[2]), dtype=int)
    south_hemisphere.resize((temp_ds.shape[1], temp_ds.shape[2]))
    north_hemisphere = 1 - south_hemisphere
    
    hwf = np.zeros((years, indexed_heatwaves.shape[1], indexed_heatwaves.shape[2]), dtype=int)
    hwd = np.zeros((years, indexed_heatwaves.shape[1], indexed_heatwaves.shape[2]), dtype=int)
    for index in range(0, years):
        north_lower, north_upper = (365*index + 121, 365*index + 274)
        south_lower, south_upper = (365*index + 304, 365*index + 455)
        
        hwf[index] = north_hemisphere*np.sum(num_index_heatwaves[north_lower:north_upper], axis=0) + south_hemisphere*np.sum(num_index_heatwaves[south_lower:south_upper], axis=0)
        
        north_hw_indices = indexed_heatwaves[north_lower:north_upper]
        south_hw_indices = indexed_heatwaves[south_lower:south_upper]
        
    masked_north = north_hw_indices.astype(np.float16)
    masked_north[north_hw_indices == 0] = np.nan
    
    masked_south = south_hw_indices.astype(np.float16)
    masked_south[south_hw_indices == 0] = np.nan
    
    hwd[index] = north_hemisphere*np.sum((north_hw_indices == stats.mode(masked_north, axis=0, nan_policy="omit")[0].astype(np.short)), axis=0) + south_hemisphere*np.sum((masked_south == stats.mode(south_hw_indices, axis=0, nan_policy="omit")[0].astype(np.short)), axis=0)
    
    meta = {
            "temperature dataset path": temp_path,
            "control dataset path": control_path,
            "time_created": str(datetime.now()),
            "author": "Cameron Cummins",
            "credit": "Original algorithm written by Tammas Loughran and modified by Jane Baldwin",
            "Tammas Loughran's repository": "https://github.com/tammasloughran/ehfheatwaves",
            "script repository": "https://github.austin.utexas.edu/csc3323/heatwave_analysis_package",
            "contact": "cameron.cummins@utexas.edu"
    }
    for key in control_threshold.attrs:
        meta[f"threshold-{key}"] = control_threshold.attrs[key]

    return xarray.Dataset(
        data_vars=dict(
            HWF=(["year", "lat", "lon"], hwf),
            HWD=(["year", "lat", "lon"], hwd)
        ),
        coords=dict(
            lon=(["lon"], temp_ds.lon.values),
            lat=(["lat"], temp_ds.lat.values),
            year=np.arange(temp_ds.time.values[0].year, temp_ds.time.values[-1].year+1),
        ),
        attrs=meta,
        )

