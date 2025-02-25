#!/usr/bin/env python
import xarray
import numpy as np
import numba as nb
from os.path import isdir
from os import makedirs
from pathlib import Path
from hdp.utils import add_history, get_version
import dask.array as da


def datetimes_to_windows(datetimes: np.ndarray, window_radius: int) -> np.ndarray:
    """
    Generate rolling window sample of the day of year for each time index in the specified time series. Example below:

    Day of year values for datetimes: 1 2 3 4 5 6
    Window radius: 1
    Result: [6 1 2], [1 2 3], [2 3 4], [3 4 5], [4 5 6], [5 6 1]
    

    :param datetimes: Array of datetime objects corresponding to the dataset's time dimension
    :type datetimes: np.ndarray
    :param window_radius: Radius of each window
    :type window_radius: int
    :return: Two dimensional array where each index of the first dimension corresponds to the time dimension and the second dimension stores the indices of each window.
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
    Generalized universal function that computes the temperatures for multiple percentiles using time index window samples.

    :param temperatures: Dataset containing temperatures to compute percentiles from
    :type temperatures: np.ndarray
    :param window_samples: Array containing "windows" of indices cenetered at each day of the year
    :type window_samples: np.ndarray
    :param percentiles: Array of perecentiles to compute [0, 1]
    :type percentiles: np.ndarray
    :param output: Array to write percentiles to
    :type output: np.ndarray
    :return: None
    :rtype: None
    """
    for doy_index in range(window_samples.shape[0]):
        doy_temps = np.zeros(window_samples[doy_index].size)
        for index, temperature_index in enumerate(window_samples[doy_index]):
            doy_temps[index] = temperatures[temperature_index]
        output[doy_index] = np.quantile(doy_temps, percentiles)


def compute_percentiles_wrapper(baseline_data, rolling_windows, percentiles):
    threshold_da = xarray.apply_ufunc(compute_percentiles,
                                      baseline_data,
                                      rolling_windows,
                                      percentiles,
                                      dask="parallelized",
                                      input_core_dims=[["time"], ["doy", "t_index"], ["percentile"]],
                                      output_core_dims=[["doy", "percentile"]],
                                      keep_attrs="override",
                                      dask_gufunc_kwargs={
                                          'allow_rechunk': False
                                      })
    return threshold_da


def compute_threshold(baseline_data: xarray.DataArray, percentiles: np.ndarray, no_season: bool=False, rolling_window_size: int=7, fixed_value: float=None) -> xarray.Dataset:
    """
    Computes percentile and, optionally, fixed value thresholds for a baseline measurements.


    :param baseline_data: DataArrays with baseline measurements to calculate thresholds from.
    :type baseline_data: xarray.DataArray
    :param percentiles: List of percentiles to calculate for each baseline.
    :type percentiles: np.ndarray
    :param no_season: (Optional) Instead of taking window samples at each day of year to get a seasonally-varying threshold, calculate a single percentile for the entire year. Defaults to False.
    :type no_season: bool
    :param rolling_window_size: (Optional) Size of rolling windows to use when calculating the percentiles. Defaults to 7.
    :type rolling_window_size: int
    :param fixed_value: (Optional) Value to use for fixed threshold (non-seasonally varying). Defaults to None.
    :type fixed_value: float
    :return: Aggregated dataset of all thresholds generated.
    :rtype: xarray.Dataset
    """
    if "member" in baseline_data.dims:
        member_datasets = []
        for member in baseline_data.member.values:
            member_slice = baseline_data.sel(member=member).drop_vars("member")
            member_datasets.append(member_slice)
        baseline_data = xarray.concat(member_datasets, dim="time")

    baseline_data = baseline_data.chunk(dict(time=-1)).astype(np.float32)
    
    percentiles = np.array(percentiles)
    
    rolling_windows_indices = datetimes_to_windows(baseline_data.time.values, rolling_window_size)
    rolling_windows_coords = {
        "doy": np.arange(rolling_windows_indices.shape[0]),
        "t_index": np.arange(rolling_windows_indices.shape[1])
    }
    rolling_windows = xarray.DataArray(data=rolling_windows_indices,
                                       coords=rolling_windows_coords)

    percentiles = xarray.DataArray(data=percentiles,
                                   coords={"percentile": percentiles})

    da_dims = []
    da_shape = []
    da_chunks = []
    
    for index, dim in enumerate(baseline_data.dims):
        if dim != "time":
            da_dims.append(dim)
            da_shape.append(baseline_data.shape[index])
            da_chunks.append(baseline_data.chunks[index])

    da_dims.extend(["doy", "percentile"])
    da_shape.extend([rolling_windows_indices.shape[0], percentiles.values.size])
    da_chunks.extend([(rolling_windows_indices.shape[0]), (percentiles.values.size)])
    
    
    da_coords = {coord: baseline_data.coords[coord] for coord in baseline_data.coords if coord != "time"}
    da_coords["doy"] = np.arange(rolling_windows_indices.shape[0])
    da_coords["percentile"] = percentiles.values
    
    template = xarray.DataArray(
        da.random.random(da_shape, chunks=da_chunks),
        dims=da_dims,
        coords=da_coords
    )

    threshold_da = xarray.map_blocks(
        compute_percentiles_wrapper,
        obj=baseline_data,
        kwargs={
            "rolling_windows": rolling_windows,
            "percentiles": percentiles
        },
        template=template
    )

    add_history(threshold_da, f"Threshold data computed by HDP v{get_version()}.\n")
    if "long_name" in threshold_da.attrs:
        add_history(threshold_da, f"Metadata updated: 'long_name' value '{threshold_da.attrs["long_name"]}' overwritten by HDP.\n")
    
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
        "hdp_type": "threshold"
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
            description=f"Extreme heat threshold dataset generated by Heatwave Diagnostics Package (HDP v{get_version()})",
            hdp_version=get_version(),
        )
    )
    ds["doy"].attrs = dict(units="day_of_year", baseline_calendar=str(baseline_data.time.values[0].calendar))
    return ds


def compute_thresholds(baseline_dataset: list[xarray.DataArray], percentiles: np.ndarray, no_season: bool=False, rolling_window_size: int=7, fixed_value: float=None) -> xarray.Dataset:
    """
    Wrapper function for generating multiple thresholds with hdp.threshold.compute_threshold. 
    Computes percentile and, optionally, fixed value thresholds for a list of baseline measurements.


    :param baseline_data: List of DataArrays with baseline measurements to calculate thresholds from.
    :type baseline_data: list[xarray.DataArray]
    :param percentiles: List of percentiles to calculate for each baseline.
    :type percentiles: np.ndarray
    :param no_season: (Optional) Instead of taking window samples at each day of year to get a seasonally-varying threshold, calculate a single percentile for the entire year. Defaults to False.
    :type no_season: bool
    :param rolling_window_size: Size of rolling windows to use when calculating the percentiles.
    :type rolling_window_size: int
    :param fixed_value: Value to use for fixed threshold (non-seasonally varying).
    :type fixed_value: float
    :return: Aggregated dataset of all thresholds generated.
    :rtype: xarray.Dataset
    """
    threshold_datasets = []
    for var_name in baseline_dataset:
        threshold_datasets.append(compute_threshold(baseline_dataset[var_name], percentiles, no_season, rolling_window_size, fixed_value))
    return xarray.merge(threshold_datasets)


def compute_threshold_io(baseline_path: str,
                         baseline_var: str,
                         output_path: str,
                         percentiles: np.ndarray,
                         no_season: bool=False,
                         rolling_window_size: int=7,
                         fixed_value: float=None,
                         overwrite: bool=False) -> None:
    """
    Computes thresholds from path inputs instead of manually supplied xarray Datasets/DataArrays (automates reading from and writing to disk).
    Resulting threshold datasets are written directly to disk instead of holding in memory.

    :param baseline_path: Path to netCDF or zarr store containing baseline measurements to compute thresholds from.
    :type baseline_path: str
    :param baseline_var: Variable read from the dataset at the specified path.
    :type baseline_var: str
    :param output_path: Path to write dataset(s) to, can be a zarr store (faster) or netCDF file (slower).
    :type output_path: str
        :param percentiles: List of percentiles to calculate for each baseline.
    :type percentiles: np.ndarray
    :param no_season: (Optional) Instead of taking window samples at each day of year to get a seasonally-varying threshold, calculate a single percentile for the entire year. Defaults to False.
    :type no_season: bool
    :param rolling_window_size: (Optional) Size of rolling windows to use when calculating the percentiles. Defaults to 7.
    :type rolling_window_size: int
    :param fixed_value: (Optional) Value to use for fixed threshold (non-seasonally varying). Defaults to None.
    :type fixed_value: float
    :param overwrite: (Optional) Whether or not to overwrite existing datasets at the output_path. Defaults to False
    :type overwrite: bool
    :return: None
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