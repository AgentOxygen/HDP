#!/usr/bin/env python
import xarray
import numpy as np
from importlib.metadata import version as getVersion
import numba as nb
from time import time
import datetime
from os.path import isdir
from os import makedirs
from pathlib import Path


def get_time_stamp():
    """
    Summary

    :return: 
    :rtype: str
    """
    return datetime.datetime.fromtimestamp(time()).strftime('%Y-%m-%d %H:%M')


def datetimes_to_windows(datetimes: np.ndarray, window_radius: int) -> np.ndarray:
    """
    Calculates sample windows for array indices from the datetime dimension 


    :param datetimes: Array of datetime objects corresponding to the dataset's time dimension
    :type datetimes: np.ndarray
    :param window_radius: Radius of windows to generate
    :type window_radius: int
    :return: 
    :rtype: np.ndarray
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


@nb.guvectorize(
    [(nb.float32[:],
      nb.int64[:, :],
      nb.float64[:],
      nb.float64[:, :])],
    '(t), (d, b), (p) -> (d, p)'
)
def compute_percentiles(temperatures: np.ndarray, window_samples: np.ndarray, percentiles: np.ndarray, output: np.ndarray):
    """
    Generalized universal function that computes the temperatures for multiple percentiles using sample index windows.


    :param temperatures: Dataset containing temperatures to compute percentiles from
    :type temperatures: np.ndarray
    :param window_samples: Array containing "windows" of indices cenetered at each day of the year
    :type window_samples: np.ndarray
    :param percentiles: Array of perecentiles to compute [0, 1]
    :type percentiles: np.ndarray
    :param output: Array to write percentiles to
    :type output: np.ndarray
    :return: 
    :rtype: None
    """
    for doy_index in range(window_samples.shape[0]):
        doy_temps = np.zeros(window_samples[doy_index].size)
        for index, temperature_index in enumerate(window_samples[doy_index]):
            doy_temps[index] = temperatures[temperature_index]
        output[doy_index] = np.quantile(doy_temps, percentiles)


def compute_thresholds(baseline_dataset: list[xarray.DataArray], percentiles: np.ndarray, no_season: bool=False, rolling_window_size: int=7, fixed_value: float=None):
    """
    Summary


    :param baseline_data:
    :type baseline_data: xarray.DataArray
    :param percentiles:
    :type percentiles: np.ndarray
    :param no_season:
    :type no_season: bool
    :param rolling_window_size:
    :type rolling_window_size: int
    :param fixed_value:
    :type fixed_value: float
    :return: 
    :rtype: None
    """
    threshold_datasets = []
    for baseline_data in baseline_dataset:
        threshold_datasets.append(compute_threshold(baseline_data, percentiles, no_season, rolling_window_size, fixed_value))
    return xarray.merge(threshold_datasets)


def compute_threshold(baseline_data: xarray.DataArray, percentiles: np.ndarray, no_season: bool=False, rolling_window_size: int=7, fixed_value: float=None):
    """
    Summary


    :param baseline_data:
    :type baseline_data: xarray.DataArray
    :param percentiles:
    :type percentiles: np.ndarray
    :param no_season:
    :type no_season: bool
    :param rolling_window_size:
    :type rolling_window_size: int
    :param fixed_value:
    :type fixed_value: float
    :return: 
    :rtype: None
    """
    percentiles = np.array(percentiles)
    
    rolling_windows_indices = datetimes_to_windows(baseline_data.time.values, rolling_window_size)
    rolling_windows_coords ={
        "doy": np.arange(rolling_windows_indices.shape[0]),
        "t_index": np.arange(rolling_windows_indices.shape[1])
    }
    rolling_windows = xarray.DataArray(data=rolling_windows_indices,
                                       coords=rolling_windows_coords)

    percentiles = xarray.DataArray(data=percentiles,
                                   coords={"percentile": percentiles})
    
    threshold_da = xarray.apply_ufunc(compute_percentiles,
                                      baseline_data,
                                      rolling_windows,
                                      percentiles,
                                      dask="parallelized",
                                      input_core_dims=[["time"], ["doy", "t_index"], ["percentile"]],
                                      output_core_dims=[["doy", "percentile"]],
                                      keep_attrs="override")
    history_str = ""
    if "history" in threshold_da.attrs:
        history_str += threshold_da.attrs["history"]
    
    history_str = f"({get_time_stamp()}) Threshold data computed by HDP v{getVersion('hdp_python')}.\n"
    if "long_name" in threshold_da.attrs:
        history_str += f"({get_time_stamp()}) Metadata updated: 'long_name' value '{threshold_da.attrs["long_name"]}' overwritten by HDP.\n"
    
    threshold_da.attrs |= {
        "long_name": f"Percentile threshold values for baseline variable '{baseline_data.name}'",
        "baseline_variable": baseline_data.name,
        "baseline_start_time": f"{str(baseline_data.time.values[0])}",
        "baseline_end_time": f"{str(baseline_data.time.values[-1])}",
        "baseline_calendar": f"{str(baseline_data.time.values[-1].calendar)}",
        "param_percentiles": str(percentiles.values),
        "param_noseason": str(no_season),
        "param_rolling_window_size": str(rolling_window_size),
        "param_fixed_value": str(fixed_value),
        "hdp_type": "threshold",
        "history": history_str
    }
    
    ds = xarray.Dataset(
        data_vars={
            f"{baseline_data.name}_threshold": threshold_da,
        },
        coords=dict(
            lon=(["lon"], threshold_da.lon.values),
            lat=(["lat"], threshold_da.lat.values),
            doy=np.arange(0, rolling_windows.shape[0]),
            percentile=percentiles
        ),
        attrs=dict(
            description=f"Extreme heat threshold dataset generated by Heatwave Diagnostics Package (HDP v{getVersion('hdp_python')})",
            hdp_version=getVersion('hdp_python'),
        )
    )
    ds["doy"].attrs = dict(units="day_of_year", baseline_calendar=str(baseline_data.time.values[0].calendar))
    return ds


def compute_threshold_io(baseline_path: str,
                         baseline_var: str,
                         output_path: str,
                         percentiles: np.ndarray,
                         no_season: bool=False,
                         rolling_window_size: int=7,
                         fixed_value: float=None,
                         overwrite: bool=False):
    """
    Summary


    :param baseline_path:
    :type baseline_path: str
    :param baseline_var:
    :type baseline_var: str
    :param output_path:
    :type output_path: str
    :param percentiles:
    :type percentiles: np.ndarray
    :param no_season:
    :type no_season: bool
    :param rolling_window_size:
    :type rolling_window_size: int
    :param fixed_value:
    :type fixed_value: float
    :param overwrite:
    :type overwrite: bool
    :return: 
    :rtype: None
    """
    output_path = Path(output_path)
    baseline_path = Path(baseline_path)
    
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"Overwrite parameter set to False and file exists at '{output_path}'.")

    if not output_path.parent.exists():
        if overwrite:
            makedirs(output_path)
        else:
            raise FileExistsError(f"Overwrite parameter set to False and directory '{output_path.parent}' does not exist.")

    if output_path.suffix not in [".zarr", ".nc"]:
        raise ValueError(f"File type '{output_path.suffix}' from '{output_path}' not supported.")
    
    if baseline_path.suffix == ".zarr" and baseline_path.isdir():
        baseline_data = xarray.open_zarr(baseline_path)[baseline_var]
    else:
        baseline_data = xarray.open_dataset(baseline_path)[baseline_var]

    baseline_data.attrs["baseline_source"] = str(baseline_path)
    threshold_ds = compute_threshold(baseline_data, percentiles, no_season, rolling_window_size, fixed_value)

    if output_path.suffix == ".zarr":
        threshold_ds.to_zarr(output_path)
    else:
        threshold_ds.to_netcdf(output_path)