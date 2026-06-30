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


def generate_test_warming_dataarray(start_date="2000-01-01", end_date="2049-12-31", grid_shape=(2, 3), warming_period=100, add_noise=False):
    """
    Generates a synthetic warming temperature DataArray for testing.

    Wraps :func:`generate_test_control_dataarray` and adds a linear warming trend
    over the specified warming period (in years).

    :param start_date: Start date of the time series in YYYY-MM-DD format. Defaults to "2000-01-01".
    :type start_date: str
    :param end_date: End date of the time series in YYYY-MM-DD format. Defaults to "2049-12-31".
    :type end_date: str
    :param grid_shape: Tuple of (lat, lon) grid dimensions. Defaults to (2, 3).
    :type grid_shape: tuple
    :param warming_period: Number of years over which 1 degree of warming is applied. Defaults to 100.
    :type warming_period: int
    :param add_noise: If True, adds random noise to create variance. Defaults to False.
    :type add_noise: bool
    :return: Synthetic warming temperature DataArray in degrees Celsius.
    :rtype: xarray.DataArray
    """
    base_data = generate_test_control_dataarray(start_date=start_date, end_date=end_date, grid_shape=grid_shape, add_noise=add_noise)
    base_data += xarray.DataArray(np.arange(base_data["time"].size) / (365*warming_period), dims=["time"], coords={"time": base_data["time"]})
    return base_data


def generate_test_rh_dataarray(start_date="2000-01-01", end_date="2049-12-31", grid_shape=(2, 3)):
    """
    Generates a synthetic relative humidity DataArray for testing.

    Derived from :func:`generate_test_control_dataarray` and scaled to a relative
    humidity range with units of g/g.

    :param start_date: Start date of the time series in YYYY-MM-DD format. Defaults to "2000-01-01".
    :type start_date: str
    :param end_date: End date of the time series in YYYY-MM-DD format. Defaults to "2049-12-31".
    :type end_date: str
    :param grid_shape: Tuple of (lat, lon) grid dimensions. Defaults to (2, 3).
    :type grid_shape: tuple
    :return: Synthetic relative humidity DataArray with units "g/g".
    :rtype: xarray.DataArray
    """
    base_data = generate_test_control_dataarray(start_date=start_date, end_date=end_date, grid_shape=grid_shape)
    base_data = abs(base_data / base_data.max() - 0.3)
    base_data = base_data.rename("test_rh_data")
    base_data.attrs["units"] = 'g/g'
    return base_data


def generate_test_control_dataarray(start_date="1700-01-01", end_date="1749-12-31", grid_shape=(2, 3), add_noise=False, seed=0):
    """
    Generates a synthetic control temperature DataArray with a seasonal cycle for testing.

    Produces a no-leap-calendar daily time series with a sinusoidal temperature pattern
    and an optional latitudinal gradient. Used as the baseline (control) input in unit tests.

    :param start_date: Start date of the time series in YYYY-MM-DD format. Defaults to "1700-01-01".
    :type start_date: str
    :param end_date: End date of the time series in YYYY-MM-DD format. Defaults to "1749-12-31".
    :type end_date: str
    :param grid_shape: Tuple of (lat, lon) grid dimensions. Defaults to (2, 3).
    :type grid_shape: tuple
    :param add_noise: If True, adds random noise to create variance across definitions and percentiles. Defaults to False.
    :type add_noise: bool
    :param seed: Random seed for reproducibility when add_noise is True. Defaults to 0.
    :type seed: int
    :return: Synthetic control temperature DataArray in degrees Celsius.
    :rtype: xarray.DataArray
    """
    time_values = xarray.date_range(
        start=start_date,
        end=end_date,
        freq="D",
        calendar="noleap",
        use_cftime=True
    )
    north_seasonal_ts = 20+2*np.sin(2*np.pi*((270 + np.arange(time_values.size, dtype=float)) / 365))
    north_seasonal_vals = np.broadcast_to(north_seasonal_ts, (grid_shape[0], grid_shape[1], north_seasonal_ts.size))

    south_seasonal_ts = 20+2*np.sin(2*np.pi*((90 + np.arange(time_values.size, dtype=float)) / 365))
    south_seasonal_vals = np.broadcast_to(south_seasonal_ts, (grid_shape[0], grid_shape[1], south_seasonal_ts.size))

    temperature_seasonal_vals = np.zeros((grid_shape[0], grid_shape[1], north_seasonal_ts.size))
    temperature_seasonal_vals[:, grid_shape[1]//2:, :] = north_seasonal_vals[:, grid_shape[1]//2:, :]
    temperature_seasonal_vals[:, :grid_shape[1]//2, :] = south_seasonal_vals[:, :grid_shape[1]//2, :]
    
    if add_noise:
        np.random.seed(seed)
        temperature_seasonal_vals += np.random.random(temperature_seasonal_vals.shape)*(np.std(temperature_seasonal_vals) / 2)

    lat_vals = np.linspace(-90, 90, grid_shape[1], dtype=float)

    lat_grad = np.broadcast_to(np.abs(lat_vals) / 90, grid_shape)
    temperature_seasonal_vals = temperature_seasonal_vals - 10*np.broadcast_to(lat_grad[:, :, None], temperature_seasonal_vals.shape)

    return xarray.DataArray(
        data=temperature_seasonal_vals,
        dims=["lon", "lat", "time"],
        coords={
            "lon": np.linspace(-180, 180, grid_shape[0], dtype=float),
            "lat": lat_vals,
            "time": time_values
        },
        name="test_temperature_data",
        attrs={
            "units": "degC"
        }
    ).chunk(dict(time=-1, lat=2, lon=2))