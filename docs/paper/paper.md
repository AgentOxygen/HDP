---
title: 'Heatwave Diagnostics Package: Computing heatwave metrics over multiple parameters using xarray'
tags:
  - heat
  - heatwave
  - diagnostics
  - python
  - xarray
authors:
 - name: Cameron Cummins
   affiliation: 1
 - name: Geeta Persad
   affiliation: 1
affiliations:
 - name: Department of Earth and Planetary Sciences, Jackson School of Geoscience, The University of Texas at Austin, Austin, TX, USA
   index: 1
date: 22 October 2024
bibliography: paper.bib
---

# Summary
The heatwave diagnostics package (`HDP`) is a Python package that enables users to compute heatwave frequency and duration metrics for output from global climate models across multiple definitions, thresholds, and measures of heat. The `HDP` leverages performance-oriented code using xarray, Dask, and Numba to efficiently process gridded daily datasets while preserving the readability of Python programming. The speed and accessibility of this approach empowers users to generate metrics for a large variety of heatwave types across the parameter space, which may reveal additional insights into the impact of climate forcings on heatwave dynamics and statistics.

# Statement of need

Quantification of heatwaves patterns from climate model output has remained a hot topic in the climate hazard research community for many years. Metrics such as heatwave frequency and duration are commonly used in hazard assessments, but there is no universal heatwave definition or threshold. This has resulted in parameter heterogenity across the literature and has prompted some studies to adopt multiple definitions in an effort to build robustness. The introduction of large ensembles and higher resolution datasets has further complicated the development of software, which has remained mostly specific to individual studies. Some tools have been developed to address this problem, but suffer from computational burdens that create barriers to quantifying metrics for larger datasets, let alone evaluating the parameter space.

Development of the `HDP` was started in 2023 to address the computational burden of handling large ensembles, but quickly evolved to investigate new scientific questions about the impact of the choice of heatwave parameters on subsequent hazard analysis.


# Key Features
## Streamlined Computation in Parallel
The creation of heatwave thresholds and definitions is streamlined through the use of the `HeatwaveDataset` class. The core algorithms are compiled using Numba to maximize computational speed and scaled using Dask to handle large datasets.

## Readability of Core Algorithms
Many heatwave algorithms are difficult to understand. The algorithms used to generate thresholds, identify heatwave days, and compute heatwave metrics are written as one-dimensional time series functions to promote transparency and extensibility. Unit tests are also included with example cases. 

# Acknowledgements

# References