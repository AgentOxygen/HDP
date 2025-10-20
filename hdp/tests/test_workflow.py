from hdp.graphics.notebook import create_notebook
import pytest
import hdp.utils
import hdp.measure
import hdp.threshold
import hdp.metric
import numpy as np


@pytest.fixture(scope="function")
def temp_output_dir(tmp_path_factory):
    return tmp_path_factory.mktemp("output_dir")


def test_full_data_workflow(temp_output_dir):
    grid_shape = (2, 3)

    baseline_temp = hdp.utils.generate_test_control_dataarray(grid_shape=grid_shape).rename("temp")
    baseline_rh = hdp.utils.generate_test_rh_dataarray(grid_shape=grid_shape).rename("rh")
    baseline_measures = hdp.measure.format_standard_measures([baseline_temp], rh=baseline_rh)
    
    percentiles = np.arange(0.9, 1, 0.01)
    
    thresholds = hdp.threshold.compute_thresholds(baseline_measures, percentiles=percentiles)
    
    exceedance_pattern = [1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 0, 1]
    
    test_temp = hdp.utils.generate_test_warming_dataarray(grid_shape=grid_shape).rename("temp")
    test_rh = baseline_rh
    
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

    metric_means = metrics.mean()

    assert metric_means["temp.temp_threshold.HWF"] >= metric_means["temp.temp_threshold.HWD"]
    assert metric_means["temp.temp_threshold.HWD"] >= metric_means["temp.temp_threshold.HWA"]
    
    for var in metrics:
        assert metrics[var].shape == (metrics.percentile.size, metrics.definition.size, metrics.lon.size, metrics.lat.size, metrics.time.size) 
        assert metrics[var].dtype == int
        if "HWF" in var or "HWD" in var:
            assert metrics[var].attrs["units"] == 'heatwave days', f"Variable '{var}' has incorrect units '{metrics[var].attrs["units"]}'"
        elif "HWN" in var or "HWA" in var:
            assert metrics[var].attrs["units"] == 'heatwave events', f"Variable '{var}' has incorrect units '{metrics[var].attrs["units"]}'"
        else:
            assert False, f"Cannot determine primary heatwave metric from variable '{var}'."

    figure_notebook = create_notebook(metrics)
    figure_notebook.save_notebook(f"{temp_output_dir}/sample_hw_summary_figures.ipynb")