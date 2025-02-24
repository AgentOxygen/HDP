---
title: 'Heatwave Diagnostics Package: Efficiently Compute Heatwave Metrics Across Parameter Spaces'
tags:
  - heat
  - heatwave
  - diagnostics
  - python
  - xarray
  - dask
  - large ensemble
  - netcdf
authors:
 - name: Cameron Cummins
   affiliation: 1
 - name: Geeta Persad
   affiliation: 1
affiliations:
 - name: Department of Earth and Planetary Sciences, Jackson School of Geoscience, The University of Texas at Austin, Austin, TX, USA
   index: 1
date: 18 February 2025
bibliography: paper.bib
---

# Summary
The heatwave diagnostics package (`HDP`) is a Python package that provides the climate research community with tools to compute heatwave metrics for the large volumes of data produced by earth system model large ensembles, across multiple measures of heat, extreme heat thresholds, and heatwave definitions. The `HDP` leverages performance-oriented design using xarray, Dask, and Numba to maximize the use of available hardware resources while maintaining accessibility through an intuitive interface and well-documented user guide. This approach empowers the user to generate metrics for a wide and diverse range of heatwave types across the parameter space.

# Statement of Need

Accurate quantification of the evolution of heatwave trends in climate model output is critical for evaluating future change in patterns of hazard. The framework for indexing heatwaves by comparing a time-evolving measure of heat against some seasonally-varing percentile threshold is well-established in the literature (@baldwin_temporally_2019; @schoetter_changes_2015; @acero_changes_2024; @argueso_seasonal_2016).
Metrics such as heatwave frequency and duration are commonly used in hazard assessments, but there are few centralized tools and no universal heatwave criteria for computing them. This has resulted in parameter heterogenity across the literature and has prompted some studies to adopt multiple definitions in an effort to build robustness (@perkins_review_2015). However, many studies rely on only a handful of metrics and definitions due to the excessive data management and computational burden of sampling a greater number of parameters (@perkins_measurement_2013). The introduction of higher resolution global climate models and large ensembles has further complicated the development of software tools, which have remained mostly specific to individual studies and specific high performance computing systems. Some generalized tools have been developed to address this problem, but do not contain explicit methods for evaluating the potential sensitivities of heatwave hazard to the choices of heat measure, extreme heat threshold, and heatwave definition.

Development of the `HDP` was started in 2023 primarily to address the computational obstacles around handling terabyte-scale large ensembles, but quickly evolved to also investigate new scientific questions around how the selection of characteristic heatwave parameters may impact subsequent hazard analysis. By enabling the user to explicitly sample a large combination of parameters, the `HDP` can provide insight into how the spatio-temporal response of heatwaves to climate perturbations and forcings depends on the choice of heatwave parameters (e.g. [heatwave3](https://robwschlegel.github.io/heatwave3/index.html), [xclim](https://xclim.readthedocs.io/en/stable/indices.html), [ehfheatwaves](https://tammasloughran.github.io/ehfheatwaves/)).

# Key Features

## Extension of XArray with Implementations of Dask and Numba
`xarray` is a popular Python package used for geospatial analysis and for working with the netCDF files produced by climate models. The `HDP` workflow is based around `xarray` and seemlessly integrates with the `xarray.DataArray` data structure. By utilizing this well-adopted framework, we increase the ease of use and portability of this package. Parallelization of `HDP` functions is achieved through the integeration of `dask` with automated chunking and task-graph construction features that are built into the `xarray` library. Calculations are computed per-grid-cell and compatable with any spatial configuration so long as it is defined by some latitude and longitude.

The boost in computational performance the `HDP` offers over other heatwave diagnostic tools comes from the combination of `dask` and `numba`. The `dask` Python package provides an interface through which `xarray.DataArray` chunks are assigned to task-graphs and then be dispatched across a cluster. The `dask` library handles many different job-dispatchers and can conform to many different types of distributed-computing systems. This ensures the `HDP` can be used on a variety of high performance computers and supercomputing clusters.

The `numba` Python package converts pure Python code and `numpy` function calls into compiled machine code which can be executed much more quickly than the standard Python interpreter. By writing the core heatwave-indexing and heatwave metric algorithms in Python and using `numba` to convert them to machine code, we preserve the readability of the Python syntax while dramatically increasing the computational efficiency of these algorithms both in terms of speed and memory overhead. We then pass these `numba`-compiled functions to the `dask` cluster for execution in parallel to leverage these improvements at scale.

## Heatwave Metrics for Multiple Measures, Thresholds, and Definitions

The "heatwave parameter space" refers to the span of measures, thresholds, and definitions that define individual heatwave "types" as described in Table \ref{table:params}.

| Parameter | Description | Example |
| :-------: | :----------:| :------:|
| Measure | The daily variable used to quantify heat. | Average temperature, minimum temperature, maximum temperature, heat index, etc. |
| Threshold | The minimum value of heat measure that indicates a "hot day." This can be a fixed value or a percentile derived from a baseline dataset. The threshold can be constant or change relative to the day of year and/or location. | 90th percentile temperature for each day of the year derived from observed temperatures from 1961 to 1990. |
| Definition | "X-Y-Z" where X indicates the minimum number of consecutive hot days, Y indicates the maximum number of non-hot days that can break up a heatwave, and Z indicates the maximum number of breaks. | "3-0-0" (three day heatwaves), "3-1-1" (three day heatwaves with possible one day breaks) |

: Parameters that define the "heatwave parameter space" and can be sampled using the HDP. \label{table:params}


Heatwave studies are often based on a limited selection of these parameters (often only one threshold and definition are used). The `HDP` allows the user to test a range of parameter values: for example, heatwaves that exceed 90th, 91st, ... 99th percentile thresholds for 3-day, 4-day, ... 7-day heatwaves. The multidimensional output produced by this sampling is elegantly stored in `xarray.DataArray` structures that can be indexed and sliced for further analysis. Four heatwave metrics that evaluate the temporal patterns in each grid cell are calculated for each measure and aggregated into an `xarray.Dataset`. Detailed descriptions of these metrics are shown in Table \ref{table:metrics}.

| Metric | Long Name | Units | Description |
| :----: | :--------:| :----:| :--------:  |
| HWF | heatwave frequency | days | The number of heatwave days per heatwave season. |
| HWN | heatwave number | events | The number of heatwaves per heatwave season. |
| HWA | heatwave average | days | The average length of heatwaves per heatwave season. |
| HWD | heatwave duration | days | The length of the longest heatwave per heatwave season. |

: Description of the heatwave metrics produced by the HDP. \label{table:metrics}

## Diagnostic Notebooks and Figures

In addition to datasets which can be saved to disk, the `HDP` includes plotting functions and figure decks that summarize various metric diagnostics. These diagnostic plots are designed to give quick insight into potential differences in metric patterns between heatwave parameters. All figure-generating functions return instances of the `matplotlib.figure.Figure` class, allowing the user to modify the attributes and features of the existing plot or add additional features. These functions are contained within the `hdp.graphics` module which can be executed automatically through the full `HDP` workflow or imported by the user to create custom workflows.

The automatic workflow compiles a "figure deck" containing diagnostic plots for multiple heatwave parameters and input variables. The resulting deck may contain dozens of figures that can be difficult to parse through individually. To simplify this process, figure decks are serialized and stored in a single Jupyter Notebook that is separated into descriptive sections. This allows the user to keep all diagnostic figures in a single Notebook file and navigate through the plots using the Notebook interface. Markdown cells are added to the top of each figure that includes a basic description of the plotting function called and the variables used. The `HDPNotebook` class in `hdp.graphics.notebook` is utilized to facilitate the generation of these Notebooks internally, but can be called through the API as well to buid custom notebooks. An example of a Notebook of the standard figure deck is shown in Figure \ref{fig:notebook}.

![Example of an HDP standard figure deck \label{fig:notebook}](HDP_Notebook_Example.png)

# Ongoing Work

This package was used to produce the results featured in a research manuscript that is currently undergoing the peer review process in a scientific journal. Updates to the `HDP` are ongoing and include, but are not limited to, adding new diagnostic plotting functions and developing heatwave metrics that measure spatial patterns. Additionally, we are planning to integrate this diagnostic package with the CESM Unified Post-Processing and Diagnostics suite (CUPiD) being developed by the National Center for Atmospheric Research.

# Acknowledgements

We thank Dr. Tammas Loughran, Dr. Jane Baldwin, and Dr. Sarah Perkins-Kirkpatrick for their work on developing the initial Python software and heatwave analysis framework that inspired this project. Dr. Loughran's Python package is available on [GitHub](https://tammasloughran.github.io/ehfheatwaves/). This work is partially supported by the Modeling, Analysis, Predictions and Projections Award Program under the National Oceanic and Atmospheric Administration (Award Number NA23OAE4310601).

# References