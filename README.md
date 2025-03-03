# HDP: Heatwave Diagnostics Package

[![Available on pypi](https://img.shields.io/pypi/v/HDP-python.svg)](https://pypi.org/project/HDP-python/)
[![Docs](https://readthedocs.org/projects/hdp/badge/?version=latest)](https://hdp.readthedocs.io/en/latest/)
![GitHub License](https://img.shields.io/github/license/AgentOxygen/HDP)

The Heatwave Diagnostics Package (HDP) is an open-source Python project that equips researchers with computationally-efficient tools to quantify heatwave metrics across multiple parameters for daily, gridded data produced by climate model large ensembles.

The HDP offers functions that leverage Xarray, Dask, and Numba to take full advantage of the available hardware capabilites. These functions have been optimized to both run quickly in serial execution and scale effectively in parallel and distributed computing systems. In addition to computing heatwave datasets, the HDP also contains several plotting functions that generate summary figures for quickly evaluating changes in heatwave patterns spatially, temporally, and across the heatwave parameter space. The user can choose to use the resulting matplotlib figures as base templates for creating their own custom figures or opt to create a standardized deck of figures that broadly summarize the different metric trends. All graphical plots can then be stored in a Jupyter notebook for easy viewing and consolidated storage.

# Why create the HDP?

Existing tools used to quantify heatwave metrics (such as ehfheatwaves, heatwave3, nctoolkit) were not designed to sample large sections of the heatwave parameter space. Many of these tools struggle to handle the computational burden of analyzing terabyte-scale datasets and do not offer a complete workflow for generating heatwave diagnostics from daily, gridded climate model output. The HDP expands upon this work to empower the user to conduct parameter-sampling analysis and reduce the computational burden of calculating heatwave metrics from increasingly large model output.

# Documentation

To learn more about the HDP and how to use it, check out the full ReadTheDocs documentation at https://hdp.readthedocs.io/en/latest/user.html#.

# Quick-Start

The code block below showcases an example HDP workflow for a 400 GB high performance computer:

```
from dask.distributed import Client, LocalCluster
import numpy as np
import xarray
import hdp


cluster = LocalCluster(n_workers=10, memory_limit="40GB", threads_per_worker=1, processes=True)
client = Client(cluster)

input_dir = "/local1/climate_model_output/"

baseline_tasmax = xarray.open_zarr(f"{input_dir}CESM2_historical_day_tasmax.zarr")["tasmax"]
test_tasmax = xarray.open_zarr(f"{input_dir}CESM2_ssp370_day_tasmax.zarr")["tasmax"]

baseline_measures = hdp.measure.format_standard_measures(temp_datasets=[baseline_tasmax])
test_measures = hdp.measure.format_standard_measures(temp_datasets=[test_tasmax])

percentiles = np.arange(0.9, 1.0, 0.01)


thresholds_dataset = hdp.threshold.compute_thresholds(
    baseline_measures,
    percentiles
)

definitions = [[3,0,0], [3,1,1], [4,0,0], [4,1,1], [5,0,0], [5,1,1]]

metrics_dataset = hdp.metric.compute_group_metrics(test_measures, thresholds_dataset, definitions)
metrics_dataset = metrics_dataset.to_zarr("/local1/test_metrics.zarr", mode='w')

figure_notebook = hdp.hdp.create_notebook(metrics_dataset)
figure_notebook.save_notebook("/local1/heatwave_summary_figures.ipynb")
```

# Contributing

Please report any bugs, ask questions, and make suggestions through the [GitHub Issues form of this repository](https://github.com/AgentOxygen/HDP/issues).

# Acknowledgements

I would like to acknowledge the following people for their contributions to this project:
1. Dr. Geeta Persad for her guidance, mentorship, and encouragement throughout the development of this package.
2. Dr. Jane Baldwin for sharing her initial heatwave analysis code that inspired the HDP and providing her expertise on the science of extreme heat.
3. Dr. Tammas Loughran for developing [ehfheatwaves](https://github.com/tammasloughran/ehfheatwaves) which served as a comparable project and informed the software design of the HDP.
4. Dr. Ifeanyi Nduka for his design input and expertise in quantifying extreme heat.
5. (Soon to be Dr.) Sebastian Utama for helping me debug my code and brainstorm ideas.
