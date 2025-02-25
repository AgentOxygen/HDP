import pytest
import hdp
import numpy as np


def test_full_data_workflow():
    baseline_temp = hdp.utils.generate_synthetic_dataset(name="temp")["temp"]
    baseline_rh = hdp.utils.generate_synthetic_dataset(name="rh", units="%", center=90, amplitude=15)["rh"]
    baseline_measures = hdp.measure.format_standard_measures([baseline_temp], rh=baseline_rh)
    
    percentiles = np.arange(0.9, 1, 0.01)
    
    thresholds = hdp.threshold.compute_thresholds(baseline_measures, percentiles=percentiles)
    
    exceedance_pattern = [1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 0, 1]
    
    test_temp = hdp.utils.generate_synthetic_dataset(name="temp")["temp"]
    test_temp = hdp.utils.generate_exceedance_dataarray(test_temp, exceedance_pattern)
    test_rh = hdp.utils.generate_synthetic_dataset(name="rh", units="%", center=90, amplitude=15)["rh"]
    
    hw_definitions = [[3,0,0], [3,1,1], [4,2,0], [4,1,3], [5,0,1], [5,1,4]]
    
    test_measures = hdp.measure.format_standard_measures([test_temp], rh=test_rh)
    metrics = hdp.metric.compute_group_metrics(test_measures, thresholds, hw_definitions)
    
    metrics = metrics.compute()
    thresholds = thresholds.compute()
    
    assert (thresholds.percentile.values == percentiles).all()
    assert len(thresholds.data_vars) == 2
    
    assert metrics.definition.values[0] == "3-0-0"
    assert metrics.definition.values[1] == "3-1-1"
    assert metrics.definition.values[2] == "4-2-0"
    assert metrics.definition.values[3] == "4-1-3"
    assert metrics.definition.values[4] == "5-0-1"
    assert metrics.definition.values[5] == "5-1-4"
    assert (metrics.percentile.values == percentiles).all()

    assert (metrics["temp.temp_threshold.HWF"] == metrics["temp_hi.temp_hi_threshold.HWF"]).all()
    assert (metrics["temp.temp_threshold.HWD"] == metrics["temp_hi.temp_hi_threshold.HWD"]).all()
    assert (metrics["temp.temp_threshold.HWA"] == metrics["temp_hi.temp_hi_threshold.HWA"]).all()
    assert (metrics["temp.temp_threshold.HWN"] == metrics["temp_hi.temp_hi_threshold.HWN"]).all()

    metric_means = metrics.mean()

    assert metric_means["temp.temp_threshold.HWF"] >= metric_means["temp.temp_threshold.HWD"]
    assert metric_means["temp.temp_threshold.HWD"] >= metric_means["temp.temp_threshold.HWA"]
    
    for var in metrics:
        assert metrics[var].shape == (metrics.percentile.size, metrics.definition.size, metrics.lat.size, metrics.lon.size, metrics.time.size) 
        assert metrics[var].dtype == int
        if "HWF" in var or "HWD" in var:
            assert metrics[var].attrs["units"] == 'heatwave days', f"Variable '{var}' has incorrect units '{metrics[var].attrs["units"]}'"
        elif "HWN" in var or "HWA" in var:
            assert metrics[var].attrs["units"] == 'heatwave events', f"Variable '{var}' has incorrect units '{metrics[var].attrs["units"]}'"
        else:
            assert False, f"Cannot determine primary heatwave metric from variable '{var}'."