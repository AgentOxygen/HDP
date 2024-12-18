#!/usr/bin/env python
"""
hdp.py

Heatwave Diagnostics Package (HDP)

Contains primary functions for computing heatwave thresholds and metrics using numpy with xarray wrapper functions.

Developer: Cameron Cummins
Contact: cameron.cummins@utexas.edu
"""
import xarray
import numpy as np
from datetime import datetime
import hdp.heat_core as heat_core
import hdp.heat_stats as heat_stats
from hdp.hw_dataset import HeatwaveDataset
from numba import njit, int64
from importlib.metadata import version as getVersion
from importlib.metadata import PackageNotFoundError
import cftime


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

    ranges = np.zeros((north_ranges.shape[0], 2, temperatures.lat.size, temperatures.lon.size), dtype=int) - 1

    for i in range(temperatures.lat.size):
        for j in range(temperatures.lon.size):
            if i < ranges.shape[2] / 2:
                ranges[:, :, i, j] = south_ranges
            else:
                ranges[:, :, i, j] = north_ranges

    return ranges


def build_doy_map(temperatures: xarray.DataArray, threshold: xarray.DataArray):
    doy_map = np.zeros(temperatures.time.size, dtype=int) - 1
    for time_index, time in enumerate(temperatures.time.values):
        doy_map[time_index] = time.dayofyr - 1
    return doy_map


def compute_threshold(temperature_dataset: xarray.DataArray, percentiles: np.ndarray, temp_path: str="No path provided.", dask: bool=True) -> xarray.DataArray:
    """
    Computes day-of-year quantile temperatures for given temperature dataset and percentile.

    Keyword arguments:
    temperature_data -- Temperature dataset to compute quantiles from
    percentile -- Percentile to compute the quantile temperatures at
    temp_path -- Path to 'temperature_data' temperature dataset to add to meta-data
    dask -- Boolean indicating whether or not to apply the generalized ufunc using Dask and xarray
    """

    window_samples = heat_core.datetimes_to_windows(temperature_dataset.time.values, 7)
    if dask:
        annual_threshold = xarray.apply_ufunc(heat_core.compute_percentiles,
                                              temperature_dataset,
                                              xarray.DataArray(data=window_samples,
                                                               coords={"day": np.arange(window_samples.shape[0]),
                                                                       "t_index": np.arange(window_samples.shape[1])}),
                                              xarray.DataArray(data=percentiles,
                                                               coords={"percentile": percentiles}),
                                              dask="parallelized",
                                              input_core_dims=[["time"], ["day", "t_index"], ["percentile"]],
                                              output_core_dims=[["day", "percentile"]])
    else:
        annual_threshold = heat_core.compute_percentiles_nb(temperature_dataset.values,
                                                            window_samples,
                                                            percentiles)

    try:
        version = getVersion("hdp")
    except PackageNotFoundError:
        version = "source"

    ds = xarray.Dataset(
        data_vars=dict(
            threshold=(["lat", "lon", "day", "percentile"], annual_threshold.data),
        ),
        coords=dict(
            lon=(["lon"], annual_threshold.lon.values),
            lat=(["lat"], annual_threshold.lat.values),
            day=np.arange(0, window_samples.shape[0]),
            percentile=percentiles
        ),
    )

    ds["threshold"].attrs |= {
        "description": f"Percentile temperatures.",
        "percentiles": str(percentiles),
        "temperature dataset path": temp_path,
        "hdp_version": version
    }
    return ds


@njit
def compute_heatwave_metrics(temperatures: np.ndarray, threshold: np.ndarray, doy_map: np.ndarray,
                             min_duration: int, max_break: int, max_subs: int,
                             season_ranges: np.ndarray) -> np.ndarray:
    hot_days_ts = heat_core.indicate_hot_days(temperatures, threshold, doy_map)
    hw_ts = heat_stats.index_heatwaves(hot_days_ts, min_duration, max_break, max_subs)
    hwf = heat_stats.heatwave_frequency(hw_ts, season_ranges)
    hwd = heat_stats.heatwave_duration(hw_ts, season_ranges)
    hwn = heat_stats.heatwave_number(hw_ts, season_ranges)
    output = np.zeros((3,) + hwf.shape, dtype=int64)
    output[0] = hwf
    output[1] = hwd
    output[2] = hwn
    return output


def sample_heatwave_metrics(future_temps: xarray.DataArray, threshold_ds: xarray.DataArray, hw_definitions: np.array, use_cftime: bool=True):
    percentile_datasets = []
    for perc in threshold_ds.percentile.values:
        perc_threshold = threshold_ds.sel(percentile=perc)
        
        times = future_temps.time.values
        season_ranges = xarray.DataArray(data=compute_hemisphere_ranges(future_temps),
                                         dims=["year", "end_points", "lat", "lon"],
                                         coords={
                                             "year": np.arange(times[0].year, times[-1].year + 1, 1),
                                             "end_points": ["start", "finish"],
                                             "lat": future_temps.lat.values,
                                             "lon": future_temps.lon.values
                                         })
        
        doy_map = xarray.DataArray(
            data=build_doy_map(future_temps, perc_threshold),
            coords={"time": times}
        )
        
        definition_datasets = []
        for hw_def in hw_definitions:
            metric_data = xarray.apply_ufunc(compute_heatwave_metrics, future_temps, perc_threshold, doy_map,
                                             hw_def[0], hw_def[1], hw_def[2],
                                             season_ranges,
                                             vectorize=True, dask="parallelized",
                                             input_core_dims=[["time"], ["day"], ["time"], [], [], [], ["year", "end_points"]],
                                             output_core_dims=[["metric", "year"]],
                                             output_dtypes=int, dask_gufunc_kwargs=dict(output_sizes=dict(metric=3)))
            definition_datasets.append(xarray.Dataset(
                dict(
                    HWF=metric_data.sel(metric=0),
                    HWD=metric_data.sel(metric=1),
                    HWN=metric_data.sel(metric=2)
                ),
                coords=dict(
                    definition=f"{hw_def[0]}-{hw_def[1]}-{hw_def[2]}"
                )
            ))
        percentile_datasets.append(xarray.concat(definition_datasets, dim="definition"))
    try:
        version = getVersion("hdp")
    except PackageNotFoundError:
        version = "source"

    ds = xarray.merge([threshold_ds, xarray.concat(percentile_datasets, dim="percentile")])
    
    if use_cftime:
        start_ts = cftime.datetime(ds.year[0], 1, 1, calendar=future_temps.time.values[0].calendar)
        end_ts = cftime.datetime(ds.year[-1], 1, 1, calendar=future_temps.time.values[0].calendar)
        ds = ds.rename(dict(year="time")).assign_coords(dict(time=xarray.cftime_range(start_ts, end_ts, periods=ds.year.size)))
        
    ds.attrs |= {
        "description": "Heatwave metrics.",
        "hdp_version": version
    }

    ds["HWF"].attrs |= {
        "units": "heatwave days",
        "long_name": "Heatwave Frequency", 
        "description": "Number of days that constitute a heatwave within a heatwave season."
    }
    ds["HWD"].attrs |= {
        "units": "heatwave days", 
        "long_name": "Heatwave Duration", 
        "description": "Length of longest heatwave during a heatwave season."
    }
    ds["HWN"].attrs |= {
        "units": "heatwave events", 
        "long_name": "Heatwave Number", 
        "description": "Number of distinct heatwaves during a heatwave season."
    }
    ds["percentile"].attrs |= {
        "range": "(0, 1)"
    }
    ds["definition"].attrs |= {
        "first_number": "Minimum number of consecutively hot days.",
        "second_number": "Maximum number of break days after first wave.",
        "third_number": "Minimum number of consecutively hot days after the break."
    }
    
    return ds


def output_heatwave_metrics_to_zarr(temps, threshold, definitions, path):
    sample_heatwave_metrics(temps, threshold, definitions).to_zarr(path, compute=True, mode='w')