#!/usr/bin/env python
"""
heatwave_metrics.py

Cameron Cummins (cameron.cummins@utexas.edu)
8/4/2023

Python functions for computing the heatwave metrics from a temperature dataset.

The algorithm is encapsulated in a Python function with additional documentation on its use in other scripts and how the metrics are computed. There is an additional wrapper function "metrics_from_path()" to handle string inputs. This allows the user to develop an xarray solution using the primary function "compute_metrics()"
"""
import xarray
import numpy as np
from datetime import datetime


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
        hot_days[index] = (temp_ds.values[index] > threshold.values[day_number-1])*(temp_ds.values[index] >= 273.15)

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
    for index in range(0, years):
        north_lower, north_upper = (365*index + 121, 365*index + 274)
        south_lower, south_upper = (365*index + 304, 365*index + 455)

        hwf[index] = north_hemisphere*np.sum(num_index_heatwaves[north_lower:north_upper], axis=0) + south_hemisphere*np.sum(num_index_heatwaves[south_lower:south_upper], axis=0)

    meta = {
            "temperature dataset path": "none",
            "control dataset path": "none",
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
            HD=(["time", "lat", "lon"], hot_days),
            HWI=(["time", "lat", "lon"], indexed_heatwaves),
            HWF=(["year", "lat", "lon"], hwf)
        ),
        coords=dict(
            lon=(["lon"], temp_ds.lon.values),
            lat=(["lat"], temp_ds.lat.values),
            year=np.arange(temp_ds.time.values[0].year, temp_ds.time.values[-1].year+1),
            time=temp_ds.time.values
        ),
        attrs=meta,
        )
    
    
def metrics_from_path(temp_ds_path: str, temp_var_name: str, control_path: str, control_var_name: str) -> xarray.Dataset:
    return compute_metrics(xarray.open_dataset(temp_ds_path)[temp_var_name], xarray.open_dataset(control_path[control_var_name]), temp_path=temp_path, control_path=control_path)