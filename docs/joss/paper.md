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

Accurate quantification of the evolution of heatwave trends in climate model output is critical for evaluating future changes in hazard. The framework for indexing heatwaves by comparing a time-evolving measure of heat against some seasonally-varying percentile threshold is well-established in the literature (@baldwin_temporally_2019; @schoetter_changes_2015; @acero_changes_2024; @argueso_seasonal_2016).
Metrics such as heatwave frequency and duration are commonly used in hazard assessments, but there are few centralized tools and no universal heatwave criteria for computing them. This has resulted in parameter heterogeneity across the literature and has prompted some studies to adopt multiple definitions to build robustness (@perkins_review_2015). However, many studies rely on only a handful of metrics and definitions due to the excessive data management and computational burden of sampling a greater number of parameters (@perkins_measurement_2013). The introduction of large ensembles has further complicated the development of software tools, which have remained mostly specific to individual studies. Some generalized tools have been developed to address this problem, but do not contain explicit methods for evaluating the potential sensitivities of heatwave hazard to the choices of heat measure, extreme heat threshold, and heatwave definition.

Development of the `HDP` was started in 2023 primarily to address the computational obstacles around handling terabyte-scale large ensembles, but quickly evolved to investigate new scientific questions around how the selection of characteristic heatwave parameters may impact hazard analysis. The `HDP` can provide insight into how the spatial-temporal response of heatwaves to climate perturbations depends on the choice of heatwave parameters. Although software does exist for calculating heatwave metrics (e.g. [heatwave3](https://robwschlegel.github.io/heatwave3/index.html), [xclim](https://xclim.readthedocs.io/en/stable/indices.html), [ehfheatwaves](https://tammasloughran.github.io/ehfheatwaves/)), these tools are not optimized to analyze more than a few definitions and thresholds at a time nor do they offer diagnostic plots.

# Key Features

## Extension of XArray with Implementations of Dask and Numba

`xarray` is a popular Python package used for geospatial analysis and for working with the netCDF files produced by climate models. The `HDP` workflow is based around `xarray` and seamlessly integrates with the `xarray.DataArray` data structure. Parallelization of `HDP` functions is achieved through the integration of `dask` with automated chunking and task graph construction features built into the `xarray` library. 

## Heatwave Metrics for Multiple Measures, Thresholds, and Definitions

The "heatwave parameter space" refers to the span of measures, thresholds, and definitions that define individual heatwave "types" as described in Table \ref{table:params}.

| Parameter | Description | Example |
| :-------: | :----------:| :------:|
| Measure | The daily variable used to quantify heat. | Average temperature, minimum temperature, maximum temperature, heat index, etc. |
| Threshold | The minimum value of heat measure that indicates a "hot day." The threshold can be constant or change relative to the day of year and/or location. | 90th percentile temperature for each day of the year derived from observed temperatures from 1961 to 1990. |
| Definition | "X-Y-Z" where X indicates the minimum number of consecutive hot days, Y indicates the maximum number of non-hot days that can break up a heatwave, and Z indicates the maximum number of breaks. | "3-0-0" (three-day heatwaves), "3-1-1" (three-day heatwaves with possible one-day breaks) |

: Parameters that define the "heatwave parameter space" and can be sampled using the HDP. \label{table:params}

The `HDP` allows the user to test a range of parameter values: for example, heatwaves that exceed 90th, 91st, ... 99th percentile thresholds for 3-day, 4-day, ... 7-day heatwaves. Four heatwave metrics that evaluate the temporal patterns in each grid cell are calculated for each measure and aggregated into a `xarray.Dataset`. Detailed descriptions of these metrics are shown in Table \ref{table:metrics}.

| Metric | Long Name | Units | Description |
| :----: | :--------:| :----:| :--------:  |
| HWF | heatwave frequency | days | The number of heatwave days per heatwave season. |
| HWN | heatwave number | events | The number of heatwaves per heatwave season. |
| HWA | heatwave average | days | The average length of heatwaves per heatwave season. |
| HWD | heatwave duration | days | The length of the longest heatwave per heatwave season. |

: Description of the heatwave metrics produced by the HDP. \label{table:metrics}

## Diagnostic Notebooks and Figures

The automatic workflow compiles a "figure deck" containing diagnostic plots for multiple heatwave parameters and input variables. To simplify this process, figure decks are serialized and stored in a single Jupyter Notebook separated into descriptive sections. Basic descriptions are included in markdown cells at the top of each figure. The `HDPNotebook` class in `hdp.graphics.notebook` is utilized to facilitate the generation of these Notebooks internally, but can be called through the API as well to build custom notebooks. An example figure of HWF from the sample figure deck is shown in Figure \ref{fig:notebook}.

![Example of an HDP standard figure deck \label{fig:notebook}](ExampleFigure.png)

# Ongoing Work

This package was used to produce the results featured in a research manuscript currently undergoing the peer-review process in a scientific journal. Updates to the `HDP` are ongoing.

# Acknowledgements

We thank Dr. Tammas Loughran, Dr. Jane Baldwin, and Dr. Sarah Perkins-Kirkpatrick for their work on developing the initial Python software and heatwave analysis framework that inspired this project. Dr. Loughran's Python package is available on [GitHub](https://tammasloughran.github.io/ehfheatwaves/). This work is partially supported by the Modeling, Analysis, Predictions, and Projections Award Program under the National Oceanic and Atmospheric Administration (Award Number NA23OAE4310601).

# References