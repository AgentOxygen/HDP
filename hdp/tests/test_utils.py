from hdp.utils import *
import numpy as np
from math import isclose
from xarray import DataArray

def test_get_time_stamp():
    assert type(get_time_stamp()) is str


def test_get_version():
    assert type(get_version()) is str


def test_control_dataarray():
    control_da = generate_test_control_dataarray(grid_shape=(1,1), start_date="2000", end_date="2100")
    assert type(control_da) is DataArray
    var = control_da.name
    assert control_da.attrs["units"] == "degC"
    assert control_da.dims == ("lon", "lat", "time")
    assert control_da.dtype == float
    assert control_da.time.values[0].calendar == "noleap"
    assert control_da.time.values.size >= 365
    assert np.sum(np.isnan(control_da)) == 0

    control_slope = np.polyfit(np.arange(control_da["time"].size), control_da.mean(dim=["lat", "lon"]).values, 1)[0]
    assert np.abs(control_slope) < 0.01


def test_warming_dataarray():
    warming_da = generate_test_warming_dataarray(grid_shape=(1,1), start_date="2000", end_date="2100")
    avg_da = warming_da.mean(dim=["lat", "lon"]).values
    warm_slope = np.polyfit(np.arange(avg_da.size), avg_da, 1)[0]

    assert warm_slope > 0
    assert not isclose(warm_slope, 0)


def test_rh_dataarray():
    rh_da = generate_test_rh_dataarray()

    assert rh_da.max() <= 1
    assert rh_da.min() >= 0


def test_defaults_compatibility():
    control_da = generate_test_control_dataarray()
    warming_da = generate_test_warming_dataarray()
    rh_da = generate_test_rh_dataarray()

    assert control_da.shape == warming_da.shape
    assert warming_da.shape == rh_da.shape

    assert control_da.dims == warming_da.dims
    assert warming_da.dims == rh_da.dims

    assert control_da.name == warming_da.name
    assert control_da.units == warming_da.units