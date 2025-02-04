import numpy as np
import numba as nb
import xarray
from hdp.utils import add_history, get_version

TEMPERATURE_UNITS = ['degC', 'degK', 'degF', 'C', 'K', 'F']
HUMIDITY_UNITS = ["%", "g/g"]


def kelvin_to_celsius(temp: float) -> float:
    """
    Converts from temperature value from Kelvin to Celsius.
    
    :param temp: Temperature value in degrees Kelvin
    :type temp: float
    :return: Temperature value in degrees Celsius
    :rtype: float
    """
    attrs = temp.attrs
    temp -= 273.15
    attrs["units"] = "degC"
    temp.attrs = attrs
    add_history(temp, "HDP converted units from Kelvin to Celsius.")
    return temp


def fahrenheit_to_celsius(temp: float) -> float:
    """
    Converts from temperature value from Fahrenheit to Celsius.
    
    :param temp: Temperature value in degrees Fahrenheit
    :type temp: float
    :return: Temperature value in degrees Celsius
    :rtype: float
    """
    attrs = temp.attrs
    temp = (temp - 32) / 1.8
    attrs["units"] = "degC"
    temp.attrs = attrs
    add_history(temp, "HDP converted units from Fahrenheit to Celsius.")
    return temp


def celsius_to_fahrenheit(temp: float) -> float:
    """
    Converts from temperature value from Celsius to Fahrenheit.
    
    :param temp: Temperature value in degrees Celsius
    :type temp: float
    :return: Temperature value in degrees Fahrenheit
    :rtype: float
    """
    attrs = temp.attrs
    temp = (temp * 1.8) + 32
    attrs["units"] = "degF"
    temp.attrs = attrs
    add_history(temp, "HDP converted units from Celsius to Fahrenheit.")
    return temp
    

@nb.njit
def heat_index(temp: float, rel_humid: float) -> float:
    """
    Calculates heat index from temperature and relative humidity.

    This function relies on a regression of the National Weather Service heat index.
    It has an error of +/- 1.3 degrees Fahrenheit.
    Source: https://www.weather.gov/ama/heatindex

    :param temp: Temperature value in degrees Fahrenheit
    :type temp: float
    :param rel_humid: Relative humidity on scale from 0 (no moisture) to 100 (fully saturated).
    :type rel_humid: float
    :return: Heat index value in degrees Fahrenheit
    :rtype: float
    """
    hi = -42.379
    hi += 2.04901523*temp
    hi += 10.14333127*rel_humid 
    hi += -0.22475541*temp*rel_humid 
    hi += -0.00683783*(temp**2)
    hi += -0.05481717*(rel_humid**2)
    hi += 0.00122874*(temp**2)*rel_humid
    hi += 0.00085282*temp*(rel_humid**2)
    hi += -0.00000199*((rel_humid*temp)**2)
    
    if rel_humid < 13 and 80 <= temp <= 112:
        hi -= ((13 - rel_humid)/4)*np.sqrt(((17 - np.abs(temp - 95))/17))
    elif rel_humid > 85 and 80 <= temp <= 87:
        hi += ((rel_humid - 85)/10) * ((87 - temp)/5)
    
    if hi < 80:
        hi = 0.5 * (temp + 61.0 + ((temp - 68.0)*1.2) + (rel_humid*0.094))

    return hi


def apply_heat_index(temp: xarray.DataArray, rh: xarray.DataArray) -> xarray.DataArray:
    """
    Calculates heat index from temperature and relative humidity DataArrays leveraging Dask.
    
    :param temp: Temperature DataArray in degrees Fahrenheit
    :type temp: xarray.DataArray
    :param rh: Relative humidity DataArray in percentage [0-100]
    :type rh: xarray.DataArray
    :return: Heat index value in degrees Fahrenheit
    :rtype: xarray.DataArray
    """
    hi_da = xarray.apply_ufunc(heat_index, temp, rh,
                               vectorize=True, dask="parallelized",
                               input_core_dims=[[], []],
                               output_core_dims=[[]],
                               output_dtypes=[float])
    hi_da.attrs = temp.attrs
    hi_da = hi_da.rename(f"{temp.name}_hi")
    hi_da.attrs["baseline_variable"] = hi_da.name
    hi_da = add_history(hi_da, f"Converted to heat index using '{rh.name}' relative humidity, renamed from '{temp.name}' to '{hi_da.name}'.\n")
    return hi_da


def convert_temp_units(temp_ds: xarray.DataArray) -> xarray.DataArray:
    """
    Determines and converts temperature units to degrees Celsius.
    
    :param temp_ds: Temperature DataArray with one of the supported units
    :type temp_ds: xarray.DataArray
    :return: Temperature DataArray in degrees Celsius
    :rtype: xarray.DataArray
    """
    if temp_ds.attrs["units"] == "K" or temp_ds.attrs["units"] == "degK":
        temp_ds = kelvin_to_celsius(temp_ds)
    elif temp_ds.attrs["units"] == "F" or temp_ds.attrs["units"] == "degF":
        temp_ds = fahrenheit_to_celsius(temp_ds)
    return temp_ds


def format_standard_measures(temp_datasets: list[xarray.DataArray], rh: xarray.DataArray = None) -> xarray.Dataset:
    """
    Formats heat measure datasets (base inputs for heatwave diagnostics) for use within the HDP.
    
    :param temp_datasets: List of input temperature DataArrays using supported units
    :type temp_datasets: list[xarray.DataArray]
    :param rh: (Optional) Relative humidity DataArray to compute heat index values with for each temperature DataArray.
    :type rh: xarray.DataArray
    :return: Dataset containing all input variables formatted and initialized for HDP analysis.
    :rtype: xarray.DataArray
    """
    measures = []

    for temp_ds in temp_datasets:
        assert "units" in temp_ds.attrs, f"Attribute 'units' not found in '{temp_ds.name}' dataset."
        assert temp_ds.attrs["units"] in TEMPERATURE_UNITS, f"Units for '{temp_ds.name}' must be one of the following: {TEMPERATURE_UNITS}"
        temp_ds.attrs |= {
            "hdp_type": "measure",
            "input_variable": temp_ds.name,
            "baseline_variable": temp_ds.name
        }
        
        temp_ds = convert_temp_units(temp_ds)
        measures.append(temp_ds)

    if rh is not None:
        assert "units" in rh.attrs, "Attribute 'units' not found in rh dataset."
        assert rh.attrs["units"] in HUMIDITY_UNITS, f"Units for rh must be one of the following: {HUMIDITY_UNITS}"

        if rh.attrs["units"] == "g/g":
            rh *= 100
            rh.attrs["units"] = "%"

        num_measures = len(measures)
        for index in range(num_measures):
            measures.append(fahrenheit_to_celsius(apply_heat_index(celsius_to_fahrenheit(measures[index].copy()), rh)))
    
    agg_ds = xarray.merge(measures)    
    agg_ds.attrs = {
        "description": f"Heat measurement dataset generated by Heatwave Diagnostics Package (HDP v{get_version()})",
        "hdp_version": get_version(),
    }
    add_history(agg_ds, f"Dataset aggregated by HDP with measures: {[ds.name for ds in measures]}")

    return agg_ds