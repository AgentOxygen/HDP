import xarray
import numpy as np
from scipy.optimize import curve_fit
import scipy.stats as stats
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from numba import jit
from time import time

def gen_windowed_samples(temperature_dataset, window_radius):
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
    
### Percentile calculation, everything should be kept in numpy

@jit(nopython=True)
def compute_percentile_thresholds(temp_data, window_samples, percentiles):
    # Assuming order of (time, lat, lon)
    percentile_temp = np.zeros((percentiles.shape[0], window_samples.shape[0], temp_data.shape[1], temp_data.shape[2]), np.float32)

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
                percentile_temp[:, doy_index, i, j] = np.quantile(temp_sample[:, i, j], percentiles)
        
        
    return percentile_temp


if __name__ == "__main__":
    path = "/projects/dgs/persad_research/EDF_MMS_Data/LENS2/TREFHTMN/DAILY/CONCAT_2015_2100/b.e21.BHISTcmip6.f09_g17.LE2-1001.001.cam.h1.TREFHTMN.20150101-20241231.nc"
    temperature_dataset = xarray.open_dataset(path)["TREFHTMN"]
    temperature_dataset = temperature_dataset.sel(time=slice(temperature_dataset.time[0], temperature_dataset.time[365*5]))

    percs = np.arange(0.9, 1, 0.01)
    
    for i in range(5):
        start = time()
        window_samples = gen_windowed_samples(temperature_dataset, 7)
        thresholds = compute_percentile_thresholds(temperature_dataset.values, window_samples, percs)
        duration = time() - start
        print(duration, end=", ")
    print("")