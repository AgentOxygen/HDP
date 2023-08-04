#!/usr/bin/env python
"""
heatwave_thresholds.py

Python functions for computing the daily temperature threshold that defines an extreme heat day and may constitute a portion of a heatwave.

The algorithm is encapsulated in a Python function with additional documentation on its use in other scripts and how the threshold is computed. There is an additional wrapper function "threshold_from_path()" to handle string inputs. This allows the user to develop an xarray solution using the primary function "compute_threshold()"
"""
import xarray
import numpy as np


def compute_threshold(temperature_data: xarray.DataArray, percentile:float=0.9, temp_path="No path provided.") -> xarray.DataArray:
    """
    Computes day-of-year quantile temperatures for given temperature dataset and percentile. The output is used as the threshold input for 'heatwave_metrics.py'.
    
    Keyword arguments:
    temperature_data -- Temperature dataset to compute quantiles from
    percentile -- Percentile to compute the quantile temperatures at
    temp_path -- Path to 'temperature_data' temperature dataset to add to meta-data
    """
    print("Initialize")
    
    init_year = temperature_data.time.values[0].year
    year_range = temperature_data.time.values[-1].year - init_year + 1
    
    annual_threshold = np.zeros((365, temperature_data.lat.size, temperature_data.lon.size))
    
    day_of_year = np.zeros((365, year_range), int)

    for index in range(temperature_data.time.values.size):
        date = temperature_data.time.values[index]
        day_of_year[date.dayofyr-1, date.year - init_year] = index
    
    for index in range(day_of_year.shape[0]):
        init_i = (index - 7) % day_of_year.shape[0]
        fin_i = (index + 8) % day_of_year.shape[0]
        
        if fin_i > init_i:
            indices = np.array([day_of_year[i] for i in range(init_i, fin_i)]).flatten()
        else:
            both_parts = [day_of_year[i] for i in range(init_i, day_of_year.shape[0])] + [day_of_year[i] for i in range(0, fin_i)]
            indices = np.array(both_parts).flatten()
        
        quantiles = np.quantile(np.array([temperature_data.values[i] for i in indices]), percentile, axis=0, method="midpoint")
        annual_threshold[index] = quantiles
    
    return xarray.Dataset(
        data_vars=dict(
            threshold=(["day", "lat", "lon"], annual_threshold),
        ),
        coords=dict(
            lon=(["lon"], temperature_data.lon.values),
            lat=(["lat"], temperature_data.lat.values),
            day=np.arange(0, 365),
        ),
        attrs={
            "description":f"{int(percentile*100)}th percentile temperatures.",
            "temperature dataset path": temp_path
        },
    )


def threshold_from_path(temperature_path: str, percentile: float, percentile_range: int) -> xarray.DataArray:
    return compute_threshold(xarray.open_dataset(temperature_path), percentile, percentile_range, temp_path=temperature_path)
