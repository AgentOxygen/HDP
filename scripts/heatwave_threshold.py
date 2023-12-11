#!/usr/bin/env python
"""
heatwave_thresholds.py

Python functions for computing the daily temperature threshold that defines an extreme heat day and may constitute a portion of a heatwave.

The algorithm is encapsulated in a Python function with additional documentation on its use in other scripts and how the threshold is computed. There is an additional wrapper function "threshold_from_path()" to handle string inputs. This allows the user to develop an xarray solution using the primary function "compute_threshold()"
"""
import xarray
import numpy as np
from numba import jit


############################## functions that needs to be optimized ##################################

def gen_windowed_samples(temperature_dataset: xarray.DataArray, window_radius: int=7) -> np.ndarray:
    #### Create time index mapping
    doy_indices = {}
    for index, date in enumerate(temperature_dataset.time.values):
        if date.dayofyr in doy_indices.keys(): 
            doy_indices[date.dayofyr].append(index)
        else:
            doy_indices[date.dayofyr] = [index]

    max_doy_length = 0
    for doy in doy_indices.keys():
        if len(doy_indices[doy]) > max_doy_length:
            max_doy_length = len(doy_indices[doy])

    # time_index records each day of the year (DOY) appearance in time dimension
    # Shape is (day of year, index in time dimension)
    # -1 indicates a missing value (useful for leap year calendar or sparse data)
    time_index = np.zeros((len(doy_indices.keys()), max_doy_length), int) - 1

    for index, doy in enumerate(doy_indices.keys()):
        for i in range(len(doy_indices[doy])):
            time_index[index, i] = doy_indices[doy][i]
    ### Done creating time index mapping
    ### Now, we need to make the window samples over which percentile will be computed
    # Samples are centered on each day of the year and then include the window_radius

    window_samples = np.zeros((len(doy_indices.keys()), 2*window_radius + 1, max_doy_length), int)
    for i in range(window_samples.shape[0]):
        for j in range(window_samples.shape[1]):
            sample_index = i + window_radius - j
            if sample_index >= time_index.shape[0]:
                sample_index = time_index.shape[0] - sample_index

            window_samples[i, j] = time_index[sample_index]

    # You can check this step by inspecting each sample. The center row, first column should be the index provided
    # padded above and below by consecutive DOY rows by an amount equal to window_radius
    # The length of each row should be equal to max_doy_length with -1 for missing dates.
    # print(window_samples[10])
    # The -1 is more efficient than storing NAN because NAN type requires floating points rather than integers
    # These are later looped out in the percentile calculation. This may be interesting to test performance difference
    # with using nanquantile instead.

    # We ultimately only care about the indices, not the order of the indices.
    # Flattening may take time now, but may also improve loop-performance later (may be worthwhile to test this)
    window_samples = window_samples.reshape((window_samples.shape[0], window_samples.shape[1]*window_samples.shape[2]))

    # Check the dimensions to ensure we have the correct number of points for each sample
    assert window_samples.shape == (len(doy_indices.keys()), (2*window_radius+1)*max_doy_length)

    return window_samples
    ### Done creating window samples
    

@jit(nopython=True)
def compute_percentile_thresholds(temp_data: np.ndarray, window_samples: np.ndarray, percentile: float):
    # Assuming order of (time, lat, lon)
    percentile_temp = np.zeros((window_samples.shape[0], temp_data.shape[1], temp_data.shape[2]), np.float32)

    for doy_index in range(window_samples.shape[0]):
        # this stores the index of each time slice for this sample
        sample_time_indices = window_samples[doy_index]

        # This loop may appear redundant, but for numba to correctly generate the next array
        # we need to know the exact size without missing values.
        # The time array is relatively small, so the hit to performance here is minor
        time_index_size = 0
        for sample_time_index in range(sample_time_indices.shape[0]):
            if sample_time_indices[sample_time_index] != -1:
                time_index_size += 1

        # We can then generate the array for our sample of the temperatures
        # This is a chunk of gridded temperature data based on sample_time_indices
        # Essentially concatenated slices of temp_data over the time dimension
        # This is a big array
        temp_sample = np.zeros((time_index_size, temp_data.shape[1], temp_data.shape[2]), np.float32)

        time_index = 0
        for sample_time_index in range(sample_time_indices.shape[0]):
            if sample_time_indices[sample_time_index] != -1:
                temp_sample[time_index] = temp_data[sample_time_indices[sample_time_index]]
                time_index += 1

        # Compute the percentiles for the grid associated with this doy_index
        #percentile_temp[doy_index] = np.quantile(temp_sample, percentile, axis=0)
        for i in range(temp_sample.shape[1]):
            for j in range(temp_sample.shape[2]):
                percentile_temp[doy_index, i, j] = np.quantile(temp_sample[:, i, j], percentile)
        
    return percentile_temp


def compute_threshold(temperature_dataset: xarray.DataArray, percentile: float, temp_path: str="No path provided.") -> xarray.DataArray:
    """
    Computes day-of-year quantile temperatures for given temperature dataset and percentile. The output is used as the threshold input for 'heatwave_metrics.py'.
    
    Keyword arguments:
    temperature_data -- Temperature dataset to compute quantiles from
    percentile -- Percentile to compute the quantile temperatures at
    temp_path -- Path to 'temperature_data' temperature dataset to add to meta-data
    """
    
    window_samples = gen_windowed_samples(temperature_dataset, 7)
    annual_threshold = compute_percentile_thresholds(temperature_dataset.values, window_samples, percentile)
    
    return xarray.Dataset(
        data_vars=dict(
            threshold=(["day", "lat", "lon"], annual_threshold),
        ),
        coords=dict(
            lon=(["lon"], temperature_dataset.lon.values),
            lat=(["lat"], temperature_dataset.lat.values),
            day=np.arange(0, num_days),
        ),
        attrs={
            "description":f"{int(percentile*100)}th percentile temperatures.",
            "temperature dataset path": temp_path
        },
    )