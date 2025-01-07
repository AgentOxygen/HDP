---
title: 'Heatwave Diagnostics Package: Computing heatwave metrics over multiple parameters using xarray'
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
date: 23 October 2024
bibliography: paper.bib
---

# Summary
The heatwave diagnostics package (`HDP`) is a Python package that enables users to compute heatwave frequency and duration metrics for output from global climate models across multiple measures of heat, extreme heat thresholds, and heatwave definitions. The `HDP` leverages performance-oriented code using xarray, Dask, and Numba to efficiently process gridded daily datasets while preserving the readability of Python programming. The speed and accessibility of this approach empowers users to generate metrics for a large variety of heatwave types across the parameter space, which may reveal additional insights into the impact of climate forcings on heatwave dynamics and statistics.

# Statement of Need

Accurate quantification of the evolution of heatwave trends in climate model output is critical for evaluating future change in patterns of hazard. Metrics such as heatwave frequency and duration are commonly used in hazard assessments, but there are few centralized tools and no universal heatwave criteria for computing them. This has resulted in parameter heterogenity across the literature and has prompted some studies to adopt multiple definitions in an effort to build robustness. The introduction of higher resolution global climate models and large ensembles has further complicated the development of software tools, which have remained mostly specific to individual studies and specific high performance computing systems. Some generalized tools have been developed to address this problem, but do not contain explicit methods for evaluating the potential sensitivities of heatwave hazard to the choices of heat measure, extreme heat threshold, and heatwave definition.

Development of the `HDP` was started in 2023 primarily to address the computational obstacles around handling terabyte-scale large ensembles, but quickly evolved to also investigate new scientific questions around how the selection of characteristic heatwave parameters may impact subsequent hazard analysis. By enabling the user to explicitly sample a large combination of parameters, the `HDP` can provide insight into how different characterizations of heatwaves evolve over time and respond to perturbations or forcings.

# Key Features

## Extension of xarray
`xarray` is a popular Python package used for geospatial analysis and for working with the netCDF files produced by climate models. The `HDP` workflow is based around `xarray` and seemlessly integrates with `xarray` functionality and operator overloading. By using `xarray.Dataset` as a base class for `HeatwaveDataset` and relying on the data structure of `xarray.DataArray`, the user can easily use the `HDP` to explore the dimensions of the multi-parameter heatwave metrics if they are familiar with the `xarray` library. This also gives the `HDP` the flexibility to work with multidimensional data of various structures. At a minimum, users must ensure their data has a monotonic `time` dimension and geospatial `lat` and `lon` dimensions. Other named dimensions are iterated over and distributed in parallel according to the chunking of the `dask.array` data variables that are natively supported by the `xarray` library.

## Heatwave Metrics for Multiple Measures, Thresholds, and Definitions
Fundamental heatwave statistics (frequency of heatwave days, duration of longest heatwave, and number of heatwave events) can easily be computed for heatwave parameters simultaneously using the `HeatwaveDataset` class or multiple functions within the API. The parameters made available to the user are defined as follows:

<!---
I plan to change the way thresholds are stored in the dataset to allow for more flexibility.
New user configuration options for the thresholds still need to be implmented, but this is just
a matter of disabling parts of the current threshold generation process (for example, fixed just
fills the array with one value).
-->

| Parameter | Definition | Examples |
|----------|------------|---------|
| Measure | The variable used to quantify heat. | Daily minimum or maximum temperature, heat index, excess heat factor, etc. |
| Threshold | The minimum value of heat measure that indicates a "hot day." This can be a fixed value or a percentile derived from a baseline dataset. The threshold can be constant or change relative to the day of year and/or location. | 90th percentile temperature for each day of the year derived from observed temperatures from 1961 to 1990. |
| Definition | The pattern of hot days that constitutes a heatwave, described as a three number code. | "3-0-0" (three day heatwaves), "3-1-1" (three day heatwaves with possible one day breaks) |

The `HeatwaveDataset` natively supports minimum temperature, maximum temperature, average temperature, and heat index (which can be optionally computed if relative humidity is provided). However, the API supports any variable, regardless of units or structure. Metrics are computed yearly, but may only evaluate a particular time window within each year, hereafter refered to as a "heatwave season." This allows the user to isolate their analysis to heatwaves that occur during particular seasons, such as the summer, and can vary spatially at the grid-cell level. The API supports full customization of the heatwave seasonal definition and has a pre-defined window that isolates metrics to assess only the summer months for the North and South hemispheres separately. A new dimension is created for each of the three parameter types with coordinates that can span multiple values. The computed thresholds and three heatwave metrics are stored within the dataset as `xarray.DataArray` variables:

<!---
HWN and season_bnds are not current output by the HDP, but are calculated (just not stored).
-->

| Variable | Definition |
|----------|------------|
| threshold | The computed minimum values needed to constitute a hot day on each day of the year. |
| HWF | "Heatwave Frequency" measures the number of days per season that make up heatwaves. |
| HWD | "Heatwave Duration" measures the length, in days, of the longest heatwave each season. |
| HWN | "Heatwave Number" measures the number of heatwave events per season. |
| season_bnds | The heatwave season window that metrics are computed over, defined by day of year boundaries. |

## Streamlined Computation in Parallel

<!---
I still need to test the serial version of the HDP.
-->

The creation of heatwave thresholds and definitions is streamlined through the use of the `HeatwaveDataset` class which handles all user inputs. The class is an extension of the `xarray.Dataset` class with custom plotting functions to allow the user to quickly generate summary figures. The core algorithms are compiled using Numba to maximize computational speed and scaled using parallelism through Dask to handle large datasets. Note that while the `HDP` lists Dask as a required dependency, the user does not necessarily need to have access to a Dask cluster to run it.

## Readability of Core Algorithms

Heatwave algorithms have complex source code that is difficult to understand. The `HDP` algorithms used to generate thresholds, identify heatwave days, and compute heatwave metrics are written as one-dimensional time series functions to promote scientific transparency and software extensibility. Unit tests are also included with example test cases. 

# Ongoing Work

Several studies that demonstrate the utility of the `HDP` are ongoing and awaiting publication. 

# Acknowledgements

This work was supported and made possible by funding from the Jackon School of Geosciences at the University of Texas at Austin.

# References