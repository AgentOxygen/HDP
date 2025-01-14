import xarray
import numpy as np
import cftime
import numba as nb
from importlib.metadata import version as getVersion


@nb.njit
def index_heatwaves(hot_days_ts: np.ndarray, min_duration: int, max_break: int, max_subs: int) -> np.ndarray:
    """
    Identifies the heatwaves in the timeseries using the specified heatwave definition

    :param hot_days_ts: Integer array of ones and zeros where ones indicates a hot day
    :type hot_days_ts: np.ndarray
    :param min_duration: The minimum number of hot days to constitute a heatwave event, including after breaks
    :type min_duration: int
    :param max_break: The maximum number of days between hot days within one heatwave event
    :type max_break: int
    :param max_subs: the maximum number of subsequent events allowed to be apart of the initial consecutive hot days
    :type max_subs: int
    :return: Timeseries where nonzero integers indicate heatwave indices for each timestep
    :rtype: np.ndarray
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


@nb.njit
def heatwave_number(hw_ts: np.ndarray, season_ranges: np.ndarray) -> np.ndarray:
    """
    Summary

    :param hw_ts: Integer timeseries of indexed heatwave days.
    :type hw_ts: np.ndarray
    :param season_ranges: Range of array indices, corresponding to heatwave season, in indexed heatwave day timeseries to count.
    :type season_ranges: np.ndarray
    :return: Number of heatwaves per heatwave season
    :rtype: np.ndarray
    """
    output = np.zeros(season_ranges.shape[0], dtype=nb.int64)
    for y in range(season_ranges.shape[0]):
        end_points = season_ranges[y]
        output[y] = np.unique(hw_ts[end_points[0]:end_points[1]]).size - 1
    return output


@nb.njit
def heatwave_frequency(hw_ts: np.ndarray, season_ranges: np.ndarray) -> np.ndarray:
    """
    Summary

    :param hw_ts: Integer timeseries of indexed heatwave days.
    :type hw_ts: np.ndarray
    :param season_ranges: Range of array indices, corresponding to heatwave season, in indexed heatwave day timeseries to count.
    :type season_ranges: np.ndarray
    :return: Number of heatwave days per heatwave season.
    :rtype: np.ndarray
    """
    output = np.zeros(season_ranges.shape[0], dtype=nb.int64)
    for y in range(season_ranges.shape[0]):
        end_points = season_ranges[y]
        output[y] = np.sum(hw_ts[end_points[0]:end_points[1]] > 0, dtype=nb.int64)
    return output


@nb.njit
def heatwave_duration(hw_ts: np.ndarray, season_ranges: np.ndarray) -> np.ndarray:
    """
    Summary

    :param hw_ts: Integer timeseries of indexed heatwave days.
    :type hw_ts: np.ndarray
    :param season_ranges: Range of array indices, corresponding to heatwave season, in indexed heatwave day timeseries to count.
    :type season_ranges: np.ndarray
    :return: Length of longest heatwave per heatwave season.
    :rtype: np.ndarray
    """
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


def get_range_indices(times: np.ndarray, start: tuple, end: tuple) -> np.ndarray:
    """
    Summary

    :param times: Array of CFTime objects.
    :type times: np.ndarray
    :param start: Tuple of starting month integer and day integer to compare times array against.
    :type start: tuple
    :param end: Tuple of ending month integer and day integer to compare times array against.
    :type end: tuple
    :return: Range of indices in times array for each season.
    :rtype: np.ndarray
    """
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


def compute_hemisphere_ranges(measure: xarray.DataArray) -> xarray.DataArray:
    """
    Summary

    :param measure: DataArray containing 'lat' and 'lon' variables corresponding to grid.
    :type measure: xarray.DataArray
    :return: Generates summer seasonal ranges by hemisphere for an arbitrary 'lat'-'lon' grid.
    :rtype: xarray.DataArray
    """
    north_ranges = get_range_indices(measure.time.values, (5, 1), (10, 1))
    south_ranges = get_range_indices(measure.time.values, (10, 1), (3, 1))

    ranges = np.zeros((north_ranges.shape[0], 2, measure.lat.size, measure.lon.size), dtype=int) - 1

    for i in range(measure.lat.size):
        for j in range(measure.lon.size):
            if i < ranges.shape[2] / 2:
                ranges[:, :, i, j] = south_ranges
            else:
                ranges[:, :, i, j] = north_ranges

    return xarray.DataArray(data=ranges,
                            dims=["year", "end_points", "lat", "lon"],
                            coords={
                                "year": np.arange(measure.time.values[0].year, measure.time.values[-1].year + 1, 1),
                                "end_points": ["start", "finish"],
                                "lat": measure.lat.values,
                                "lon": measure.lon.values
                            })


def build_doy_map(times: np.ndarray) -> np.ndarray:
    """
    Summary

    :param measure: Array of CFTime objects.
    :type measure: np.ndarray
    :return: Day of year for each timestep.
    :rtype: np.ndarray
    """
    doy_map = np.zeros(times.size, dtype=int) - 1
    for time_index, time in enumerate(times):
        doy_map[time_index] = time.dayofyr - 1
    return doy_map


@nb.njit
def indicate_hot_days(measure: np.ndarray, threshold: np.ndarray, doy_map: np.ndarray) -> np.ndarray:
    """
    Summary

    :param measure: Heat measure to compare against the threshold.
    :type measure: np.ndarray
    :param threshold: Threshold for extreme heat.
    :type threshold: np.ndarray
    :param doy_map: Mapping from index in times array to day of year value.
    :type doy_map: np.ndarray
    :return: Boolean array of days in measure that exceed the threshold.
    :rtype: np.ndarray
    """
    output = np.zeros(measure.shape, dtype=nb.boolean)
    for t in range(measure.size):
        doy = doy_map[t]
        if measure[t] > threshold[doy]:
            output[t] = True
        else:
            output[t] = False
    return output


@nb.njit
def compute_heatwave_metrics(measure: np.ndarray, threshold: np.ndarray, doy_map: np.ndarray,
                             min_duration: int, max_break: int, max_subs: int,
                             season_ranges: np.ndarray) -> np.ndarray:
    """
    Summary

    :param measure: Heat measure to compare against the threshold.
    :type measure: np.ndarray
    :param threshold: Threshold for extreme heat.
    :type threshold: np.ndarray
    :param doy_map: Mapping from index in times array to day of year value.
    :type doy_map: np.ndarray
    :param min_duration: Minimum number of days that exceed the threshold to constitute a heatwave.
    :type min_duration: int
    :param max_break: Maximum number of days below the threshold that can exist in a heatwave.
    :type max_break: int
    :param max_subs: Maximum number of subsequent heatwaves.
    :type max_subs: int
    :param season_ranges: 
    :type season_ranges: np.ndarray
    :return: Array of equal number of dimensions to measure and an additional dimension containing each heatwave metric: HWF, HWD, HWN
    :rtype: np.ndarray
    """
    hot_days_ts = indicate_hot_days(measure, threshold, doy_map)
    hw_ts = index_heatwaves(hot_days_ts, min_duration, max_break, max_subs)
    hwf = heatwave_frequency(hw_ts, season_ranges)
    hwd = heatwave_duration(hw_ts, season_ranges)
    hwn = heatwave_number(hw_ts, season_ranges)
    output = np.zeros((3,) + hwf.shape, dtype=nb.int64)
    output[0] = hwf
    output[1] = hwd
    output[2] = hwn
    return output


def compute_group_metrics(measures: xarray.Dataset, thresholds:xarray.Dataset, hw_definitions: list, include_threshold: bool=False, check_variables: bool=True) -> xarray.Dataset:
    metric_sets = []
    for measure_name in list(measures.keys()):
        measure = measures[measure_name]
        for threshold_name in list(thresholds.keys()):
            threshold = thresholds[threshold_name]
            if threshold.attrs["baseline_variable"] == measure.attrs["input_variable"]:
                hw_metrics = compute_individual_metrics(measure, threshold, hw_definitions, include_threshold, check_variables)
                var_renames = {name:f"{measure_name}.{threshold_name}.{name}" for name in list(hw_metrics.keys())}
                metric_sets.append(hw_metrics.rename(var_renames))

    aggr_ds = xarray.merge(metric_sets)
    aggr_ds.attrs["variable_naming_desc"] = "(heat measure).(threshold used).(heatwave metric)"
    aggr_ds.attrs["variable_naming_delimeter"] = "."
    return aggr_ds


def compute_individual_metrics(measure: xarray.DataArray, threshold: xarray.DataArray, hw_definitions: list, include_threshold: bool=True, check_variables: bool=True) -> xarray.Dataset:
    """
    Summary

    :param measure: 
    :type measure: xarray.DataArray
    :param threshold: 
    :type threshold: xarray.DataArray
    :param hw_definitions: 
    :type hw_definitions: list
    :param include_threshold: 
    :type include_threshold: bool
    :param check_variables: 
    :type check_variables: bool
    :return: 
    :rtype: xarray.Dataset
    """
    if check_variables:
        assert "hdp_type" in threshold.attrs
        assert threshold.attrs["hdp_type"] == "threshold"
        assert threshold.attrs["baseline_variable"] == measure.attrs["input_variable"]
        assert threshold.attrs["baseline_calendar"] == measure.time.values[0].calendar
    
    percentile_datasets = []
    times = measure.time.values
    
    for perc in threshold.percentile.values:
        perc_threshold = threshold.sel(percentile=perc)
        
        season_ranges = xarray.DataArray(data=compute_hemisphere_ranges(measure),
                                         dims=["year", "end_points", "lat", "lon"],
                                         coords={
                                             "year": np.arange(times[0].year, times[-1].year + 1, 1),
                                             "end_points": ["start", "finish"],
                                             "lat": measure.lat.values,
                                             "lon": measure.lon.values
                                         })
        
        doy_map = xarray.DataArray(
            data=build_doy_map(times),
            coords={"time": times}
        )
        
        definition_datasets = []
        for hw_def in hw_definitions:
            metric_data = xarray.apply_ufunc(compute_heatwave_metrics, measure, perc_threshold, doy_map,
                                             hw_def[0], hw_def[1], hw_def[2],
                                             season_ranges,
                                             vectorize=True, dask="parallelized",
                                             input_core_dims=[["time"], ["doy"], ["time"], [], [], [], ["year", "end_points"]],
                                             output_core_dims=[["metric", "year"]],
                                             output_dtypes=[int], dask_gufunc_kwargs=dict(output_sizes=dict(metric=3)))
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

    if include_threshold:
        ds = xarray.merge([threshold, xarray.concat(percentile_datasets, dim="percentile")])
    else:
        ds = xarray.concat(percentile_datasets, dim="percentile")
    
    start_ts = cftime.datetime(ds.year[0], 1, 1, calendar=measure.time.values[0].calendar)
    end_ts = cftime.datetime(ds.year[-1], 1, 1, calendar=measure.time.values[0].calendar)
    ds = ds.rename(dict(year="time")).assign_coords(dict(time=xarray.cftime_range(start_ts, end_ts, periods=ds.year.size)))
        
    ds.attrs |= {
        "description": f"Heatwave metric dataset generated by Heatwave Diagnostics Package (HDP v{getVersion('hdp_python')})",
        "hdp_version": getVersion('hdp_python')
    }

    ds["HWF"].attrs |= {
        "units": "days",
        "long_name": "Heatwave Frequency", 
        "description": "Number of days that constitute a heatwave within a heatwave season"
    }
    ds["HWD"].attrs |= {
        "units": "days", 
        "long_name": "Heatwave Duration", 
        "description": "Length of longest heatwave during a heatwave season"
    }
    ds["HWN"].attrs |= {
        "units": "events", 
        "long_name": "Heatwave Number", 
        "description": "Number of distinct heatwaves during a heatwave season"
    }
    ds["percentile"].attrs |= {
        "range": "(0, 1)"
    }
    ds["definition"].attrs |= {
        "first_number": "Minimum number of consecutively hot days",
        "second_number": "Maximum number of break days after first wave",
        "third_number": "Minimum number of consecutively hot days after the break"
    }
    return ds


def compute_metrics_io(output_path: str,
                       measure_path: str,
                       measure_var: str,
                       threshold_path: str,
                       hw_definitions: list,
                       include_threshold: bool=False,
                       override_threshold_var: str=None):
    """
    Summary

    :param output_path: 
    :type output_path: str
    :param measure_path: 
    :type measure_path: str
    :param measure_var: 
    :type measure_var: str
    :param threshold_path: 
    :type threshold_path: str
    :param hw_definitions: 
    :type hw_definitions: list
    :param include_threshold: 
    :type include_threshold: bool
    :param override_threshold_var: 
    :type override_threshold_var: str
    :return: 
    :rtype: None
    """
    output_path = Path(output_path)
    measure_path = Path(measure_path)
    threshold_path = Path(threshold_path)
    check_variables = True
    
    if override_threshold_var is None:
        threshold_var = f"threshold_{measure_var}"
        check_variables = False
    
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"Overwrite parameter set to False and file exists at '{output_path}'.")

    if not output_path.parent.exists():
        if overwrite:
            makedirs(output_path)
        else:
            raise FileExistsError(f"Overwrite parameter set to False and directory '{output_path.parent}' does not exist.")

    if output_path.suffix not in [".zarr", ".nc"]:
        raise ValueError(f"File type '{output_path.suffix}' from '{output_path}' not supported.")

    if measure_path.suffix == ".zarr" and measure_path.isdir():
        measure_data = xarray.open_zarr(measure_path)[measure_var]
    else:
        measure_data = xarray.open_dataset(measure_path)[measure_var]
    
    if threshold_path.suffix == ".zarr" and threshold_path.isdir():
        threshold_data = xarray.open_zarr(threshold_path)[threshold_var]
    else:
        threshold_data = xarray.open_dataset(threshold_path)[threshold_var]

    metric_ds = compute_metrics(measure_data, threshold_data, hw_definitions, include_threshold=include_threshold, check_variables=check_variables)

    if output_path.suffix == ".zarr":
        metric_ds.to_zarr(output_path)
    else:
        metric_ds.to_netcdf(output_path)