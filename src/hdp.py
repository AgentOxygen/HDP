#!/usr/bin/env python
"""
hdp.py

Heatwave Diagnostics Package (HDP)

Contains primary functions for computing heatwave thresholds and metrics using numpy with xarray wrapper functions.

Developer: Cameron Cummins
Contact: cameron.cummins@utexas.edu
2/8/24
"""
import xarray
import numpy as np
from datetime import datetime
from scipy import stats
from heat_core import HeatCore
from heat_stats import HeatStats


def get_range_indices(times: np.array, start: tuple, end: tuple):
    num_years = times[-1].year - times[0].year + 1
    ranges = np.zeros((num_years, 2), dtype=int) - 1

    n = 0
    looking_for_start = True
    for t in range(times.shape[0]):
        if looking_for_start:
            if times[t].month == start[0] and times[t].day == start[1]:
                looking_for_start = False
                ranges[n, 0] = t
        else:
            if times[t].month == end[0] and times[t].day == end[1]:
                looking_for_start = True
                ranges[n, 1] = t
                n += 1

    if not looking_for_start:
        ranges[-1, -1] = times.shape[0]

    return ranges


def compute_hemisphere_ranges(temperatures: xarray.DataArray):
    north_ranges = get_range_indices(temperatures.time.values, (5, 1), (10, 1))
    south_ranges = get_range_indices(temperatures.time.values, (10, 1), (3, 1))

    ranges = np.zeros((north_ranges.shape[0], 2, temperatures.shape[1], temperatures.shape[2]), dtype=int) - 1

    for i in range(temperatures.shape[1]):
        for j in range(temperatures.shape[2]):
            if i < ranges.shape[2] / 2:
                ranges[:, :, i, j] = south_ranges
            else:
                ranges[:, :, i, j] = north_ranges

    return ranges


def build_doy_map(temperatures: xarray.DataArray, threshold: xarray.DataArray):
    doy_map = np.zeros(temperatures.shape[0], dtype=int) - 1
    for time_index, time in enumerate(temperatures.time.values):
        doy_map[time_index] = time.dayofyr - 1
    return doy_map


def compute_threshold(temperature_dataset: xarray.DataArray, percentiles: np.ndarray, temp_path: str="No path provided.") -> xarray.DataArray:
    """
    Computes day-of-year quantile temperatures for given temperature dataset and percentile.
    
    Keyword arguments:
    temperature_data -- Temperature dataset to compute quantiles from
    percentile -- Percentile to compute the quantile temperatures at
    temp_path -- Path to 'temperature_data' temperature dataset to add to meta-data
    """
    
    window_samples = HeatCore.datetimes_to_windows(temperature_dataset.time.values, 7)
    annual_threshold = HeatCore.compute_percentiles(temperature_dataset.values, window_samples, percentiles)
    
    return xarray.Dataset(
        data_vars=dict(
            threshold=(["percentile", "day", "lat", "lon"], annual_threshold),
        ),
        coords=dict(
            lon=(["lon"], temperature_dataset.lon.values),
            lat=(["lat"], temperature_dataset.lat.values),
            day=np.arange(0, window_samples.shape[0]),
            percentile=percentiles
        ),
        attrs={
            "description": f"Percentile temperatures.",
            "percentiles": str(percentiles),
            "temperature dataset path": temp_path
        },
    )


def compute_heatwave_metrics(future_dataset: xarray.DataArray, threshold: xarray.DataArray):
    datasets = []
    for perc in threshold.percentile.values:
        print(perc, end=", ")
        doy_map = build_doy_map(future_dataset, threshold["threshold"])
        hot_days = HeatCore.indicate_hot_days(future_dataset.values, threshold["threshold"].sel(percentile=perc).values, doy_map)
        heatwave_indices = HeatCore.compute_int64_spatial_func(hot_days, HeatStats.index_heatwaves)
        season_ranges = compute_hemisphere_ranges(future_dataset)

        metrics_ds = xarray.Dataset(data_vars={
                "HWF": (["year", "lat", "lon"], HeatCore.compute_heatwave_metric(HeatStats.heatwave_frequency, season_ranges, heatwave_indices)),
                "HWD": (["year", "lat", "lon"], HeatCore.compute_heatwave_metric(HeatStats.heatwave_duration, season_ranges, heatwave_indices))
            },
            coords=dict(
                year=np.arange(future_dataset.time.values[0].year, future_dataset.time.values[-1].year + 1),
                lat=future_dataset.lat.values,
                lon=future_dataset.lon.values,
                percentile=perc
            ))
        datasets.append(metrics_ds)

    dataset = xarray.concat(datasets, dim="percentile")
    dataset.attrs = {
        "dev_name" : "Cameron Cummins",
        "dev_affiliation" : "Persad Aero-Climate Lab, Department of Earth and Planetary Sciences, The University of Texas at Austin",
        "dev_email" : "cameron.cummins@utexas.edu",
        "description": "Heatwave metrics.",
        "date_prepared" : str(datetime.now())
    }

    return dataset