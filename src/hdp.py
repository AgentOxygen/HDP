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
import heat_core
import heat_stats


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
    doy_map = np.zeros(temperatures.shape[0], dtype=int) - 1
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

    return xarray.Dataset(
        data_vars=dict(
            threshold=(["lat", "lon", "day", "percentile"], annual_threshold.data),
        ),
        coords=dict(
            lon=(["lon"], annual_threshold.lon.values),
            lat=(["lat"], annual_threshold.lat.values),
            day=np.arange(0, window_samples.shape[0]),
            percentile=percentiles
        ),
        attrs={
            "description": f"Percentile temperatures.",
            "percentiles": str(percentiles),
            "temperature dataset path": temp_path
        },
    )


def compute_heatwave_metrics_old(future_dataset: xarray.DataArray, threshold: xarray.DataArray):
    datasets = []
    for perc in threshold.percentile.values:
        doy_map = build_doy_map(future_dataset, threshold["threshold"])
        hot_days = heat_core.indicate_hot_days(future_dataset.values, threshold["threshold"].sel(percentile=perc).values, doy_map)
        heatwave_indices = heat_core.compute_int64_spatial_func(hot_days, heat_stats.index_heatwaves)
        season_ranges = compute_hemisphere_ranges(future_dataset)

        metrics_ds = xarray.Dataset(data_vars={
                "HWF": (["year", "lat", "lon"], heat_core.compute_heatwave_metric(heat_stats.heatwave_frequency, season_ranges, heatwave_indices)),
                "HWD": (["year", "lat", "lon"], heat_core.compute_heatwave_metric(heat_stats.heatwave_duration, season_ranges, heatwave_indices))
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


def compute_heatwave_metrics_dask(future_temps: xarray.DataArray, threshold_ds: xarray.DataArray, hw_definitions: np.array):
    labels = []
    metrics = []
    for hw_def in hw_definitions:
        labels.append(f"{hw_def[0]}-{hw_def[1]}")
        season_ranges = compute_hemisphere_ranges(future_temps)
        doy_map = build_doy_map(future_temps, threshold_ds)

        hot_days = xarray.apply_ufunc(heat_core.indicate_hot_days,
                                      future_temps,
                                      threshold_ds,
                                      xarray.DataArray(data=doy_map, coords={"time": future_temps.time.values}), 
                                      dask="parallelized",
                                      input_core_dims=[["time"], ["day", "percentile"], ["time"]],
                                      output_core_dims=[["time", "percentile"]])

        hw_indices = xarray.apply_ufunc(heat_stats.index_heatwaves,
                                        hot_days,
                                        hw_def[1],
                                        hw_def[0],
                                        dask="parallelized",
                                        input_core_dims=[["time"], [], []],
                                        output_core_dims=[["time"]])


        times = future_temps.time.values
        season_ranges_da = xarray.DataArray(data=season_ranges,
                                            dims=["year", "end_points", "lat", "lon"],
                                            coords={
                                                 "year": np.arange(times[0].year, times[-1].year + 1, 1),
                                                 "end_points": ["start", "finish"],
                                                 "lat": future_temps.lat.values,
                                                 "lon": future_temps.lon.values
                                             })
        hwd = xarray.apply_ufunc(heat_stats.heatwave_duration,
                                 hw_indices,
                                 season_ranges_da,
                                 dask="parallelized",
                                 input_core_dims=[["time"], ["year", "end_points"]],
                                 output_core_dims=[["year"]])

        hwf = xarray.apply_ufunc(heat_stats.heatwave_frequency,
                                 hw_indices,
                                 season_ranges_da,
                                 dask="parallelized",
                                 input_core_dims=[["time"], ["year", "end_points"]],
                                 output_core_dims=[["year"]])
        metrics.append(xarray.Dataset(dict(HWF=hwf, HWD=hwd)))
    return xarray.concat(metrics, dim="definition").assign_coords(dict(definition=labels))

