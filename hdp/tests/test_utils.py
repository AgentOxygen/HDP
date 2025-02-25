import hdp.utils
import xarray
import numpy as np


def test_get_time_stamp():
    assert type(hdp.utils.get_time_stamp()) is str


def test_get_version():
    assert type(hdp.utils.get_version()) is str


def test_synthetic_data_functions():
    var = "test"
    center_val = 10
    amplitude_val = 1
    units = "test_units"
    
    ds = hdp.utils.generate_synthetic_dataset(name=var, units=units, center=center_val, amplitude=amplitude_val)
    assert type(ds) is xarray.Dataset
    assert len(ds.data_vars) == 1
    assert var in ds
    
    assert "units" in ds[var].attrs
    assert ds[var].attrs["units"] == units
    assert ds[var].dims == ("lat", "lon", "time")
    assert ds[var].dtype == float
    assert ds[var].time.values[0].calendar == "noleap"
    assert ds[var].lat.size > 1
    assert ds[var].lon.size > 1
    assert ds[var].time.size >= 2*365

    data = ds[var].compute()
    
    assert np.isclose(data.mean(), center_val)
    assert np.isclose(data.max(), center_val + amplitude_val)
    assert np.isclose(data.values, data.values[0]).all()

    exceed_data = hdp.utils.generate_exceedance_dataarray(ds[var], exceedance_pattern=[1, 0, 1, 0]).compute()

    assert np.isclose(exceed_data.mean(), center_val + 0.5)
    assert exceed_data.attrs == ds[var].attrs
    assert exceed_data.dims == ds[var].dims
    assert exceed_data.shape == ds[var].shape
    assert exceed_data.dtype == ds[var].dtype