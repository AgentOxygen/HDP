from time import time
from importlib.metadata import version
import numpy as np
import dask.array as da
import xarray
import datetime
import cftime


def get_time_stamp():
    return datetime.datetime.fromtimestamp(time()).strftime('%Y-%m-%d %H:%M')


def add_history(ds, msg):
    if "history" in ds.attrs:
        ds.attrs["history"] += f"({get_time_stamp()}) {msg}\n"
    else:
        ds.attrs["history"] = f"({get_time_stamp()}) History metadata initialized by HDP v{get_version()}.\n"
        ds.attrs["history"] += f"({get_time_stamp()}) {msg}\n"
    return ds


def get_version():
    return version('hdp_python')


def get_func_description(func):
    lines = func.__doc__.split("\n")
    desc = ""
    for line in lines:
        if ":param" in line:
            break
        line = line.strip()
        if line != "":
            desc += line.strip() + " "
    return desc


def generate_synthetic_dataset(center=25, amplitude=10, name="temperature", units="degC"):
    time_values = xarray.cftime_range(
        start=cftime.DatetimeNoLeap(2000, 1, 1),
        end=cftime.DatetimeNoLeap(2009, 12, 31),
        freq="D",
        calendar="noleap"
    )
    
    temp_timeseries = center + amplitude*np.sin(np.pi*np.arange(time_values.size, dtype=float) / 365)
    temp_values = np.broadcast_to(temp_timeseries, (3, 3, temp_timeseries.size))
    
    temp_da = xarray.DataArray(
        data=da.from_array(temp_values),
        dims=["lat", "lon", "time"],
        coords={
            "lat": np.array([-90, 0, 90], dtype=float),
            "lon": np.array([-180, 0, 180], dtype=float),
            "time": time_values
        },
        name=name,
        attrs={
            "units": units
        }
    ).chunk(dict(lat=1, lon=1))
    return xarray.Dataset({name: temp_da})


def generate_exceedance_dataarray(measure, exceedance_pattern, multiplier=1.0):
    tiles = np.ceil(measure.time.size / len(exceedance_pattern)).astype(int)
    pattern_data = np.broadcast_to(np.tile(exceedance_pattern, tiles)[:measure.time.size], measure.shape)*multiplier

    ret_da = None
    with xarray.set_options(keep_attrs=True): 
        ret_da = measure + xarray.DataArray(
            data=da.from_array(pattern_data),
            dims=measure.dims,
            coords=measure.coords,
            name=measure.name,
            attrs=measure.attrs
        ).chunk(measure.chunksizes)
    return ret_da
