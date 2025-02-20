User Guide
=====

What is the HDP?
----------------
The Heatwave Diagnostics Package (HDP) is a collection of Python tools for computing heatwave metrics and generating summary figures. Functions can be imported in Jupyter notebooks and Python scripts or called in the terminal using the command line interface (CLI). All data uses `xarray <https://docs.xarray.dev/en/stable/>`_ data structures and can be saved to disk as either Zarr stores (default) or netCDF datasets. Summary figures can be generated to describe the output and can be saved to disk in Jupyter notebooks.

The HDP workflow follows three steps:

1. Format both a baseline and test measure of heat

2. Generate an extreme heat threshold from the baseline

3. Compute heatwave metrics by comparing the test measure against the baseline threshold

Throughout this guide, "baseline" refers to a measure of heat that is used to generate the threshold values from. Threshold values are defined as a percentile of all baseline values for each day of the year (see :ref:`Threshold Calculation <threshold_calc>` for more details). The "test" measure is compared against the threshold values to calculate hot days (days that exceed the threshold). Finally, the "heatwave definition" refers to an integer-sequence that describes what patterns of hot days over time are considered heatwaves. The span of different measures, thresholds, and definitions defines the "heatwave parameter space."

Statement of Need
-----------------
Existing tools used to quantify heatwave metrics (such as `ehfheatwaves <http://tammasloughran.github.io/ehfheatwaves/>`_, `heatwave3 <https://robwschlegel.github.io/heatwave3/index.html>`_, `nctoolkit <https://nctoolkit.readthedocs.io/en/latest/>`_) were not designed to sample large sections of the heatwave parameter space. Many of these tools struggle to handle the computational burden of analyzing terabyte-scale datasets and do not offer a complete workflow for generating heatwave diagnostics from daily, gridded climate model output. The HDP expands upon this work to empower the user to conduct parameter-sampling analysis and reduce the computational burden of calculating heatwave metrics from increasingly large model output.

Installation
------------
The HDP can be installed using PyPI. You can view the webpage `here <https://pypi.org/project/HDP-python/>`_.

.. code-block:: console

   $ pip install hdp-python


Quick Start
-----------
Below is example code that computes heatwave metrics for multiple measures, thresholds, and definitions. Hheatwave metrics are obtained for the test dataset by comparing against the thresholds generated from the baseline dataset.

.. code-block:: python

    

Example 1: Generating Heatwave Diagnostics
------------------------------------------
The Regional Aerosol Model Intercomparison Project (RAMIP) is a multi-model large ensemble of earth system model experiments conducted to quantify the role of regional aerosol emissions changes in near-term climate change projections (`Wilcox et al., 2023 <https://gmd.copernicus.org/articles/16/4451/2023/>`_). For the sake of simplicity, we will only investigate CESM2 (one of the 8 models available in this MIP) for this example. For CESM2, there are 10 ensemble members for each of the six model experiments. Each experiment is essentially a different emission scenario where regional aerosol emissions are held constant over different parts of the globe. We will use a historical simulation from 1960 to 1970 run produced by CESM2 from the same ensemble as the baseline for calculating the extreme heat threshold. In this first example, we will produce heatwave metrics for one emission scenario, namely SSP3-7.0. In :ref:`Example 2: RAMIP Analysis <example_2>`, we will explore how comparing the heatwave parameter spaces from two different scientific experiments can generate insightful analysis.

For this first analysis, we will explore the following set of heatwave parameters:

.. list-table:: Example 1 Parameter Space
   :widths: 50 50
   :header-rows: 1

   * - Parameter
     - Range/Values
   * - Measures
     - tas, tasmax, tas_hi, tasmax_hi,
   * - Thresholds
     - [0.9, 0.91, ... 0.99]
   * - Definitions
     - 3-1-1, 3-3-1, 5-1-1, 5-3-1, 7-1-1, 7-3-1

Note that "_hi" refers to the heat index values for those variables. The model does not explicitly output heat index measurements, but we can calculate them from relative humidity (rh) using the HDP. For the thresholds, we select the range of percentiles from 0.9 to 0.99 with steps of 0.01. The heatwave definitions are defined as integer sequences that describe the following criteria (in order of integer placement):

#. The minimum number of hot days to start a heatwave event.
#. The maximum number of non-hot days that can follow the start of a heatwave event (creating a small break).
#. The maximum number of subsequent events that can come after the break (and be considered part of the starting heatwave).

The definition codes may feel confusing at first, but they allow the user to capture many different "types" of heatwave and derive additional heatwave metrics without having to repeat the computationally-expensive analysis. We will investigate an example of derived metrics at the end of this section.



To fully utilize the performance enhancments offered by the HDP, we must first start up a `Dask cluster <https://docs.dask.org/en/stable/deploying.html>`_ to leverage parallel computation. This step is not automated because it requires system-specific configuration. If you are working on a single, local machine, a `LocalCluster <https://docs.dask.org/en/stable/deploying.html#local-machine>`_ typically works best. However, if you are working on a distributed system at a supercomputing center, use the Dask configuration reccomended by your trusted HPC specialist. Below is an example configuration for use on a single-node with at least 30 cores and 200 (20x10 GB) of memory:

.. code-block:: python

    from dask.distributed import Client, LocalCluster
    cluster = LocalCluster(n_workers=20, memory_limit="10GB", threads_per_worker=1, processes=True, dashboard_address=":8004")
    client = Client(cluster)


Once a Dask cluster is initialized, we then need to organize our data into `xarray.DataArray <https://docs.xarray.dev/en/stable/generated/xarray.DataArray.html>`_ objects. The entire HDP is built around xarray data structures to ensure ease of use and remain agnostic to input file types. Since we are working with a large ensemble, we need to make sure to concatenate the ensemble members along a "member" dimension. If we weren't using a large ensemble (a single long-running simulation for example), we would just omit this step. To read data from disk, we can use the `xarray.open_mfdataset <https://docs.xarray.dev/en/stable/generated/xarray.open_mfdataset.html>`_ function. Reading and post-processing data will look different from system to system, but the final format should be the same. Below is a list of xarray.DataArrays with the data structure for baseline_tasmax dataset visualized below:

.. code-block:: python

    baseline_tasmax
    baseline_rh
    ssp370_tasmax
    ssp370_rh
    
    baseline_tasmax

.. image:: assets/tasmax_dataarray_example.png
   :width: 600

The spatial coordinates for latitude and longitude should be named "lat" and "lon" respectively. The "time" coordinates should be decoded into CFTime objects and a "member" dimension should be created if an ensemble is being used.

To begin, we first need to format these measures so that they are in the correct units. This process will also compute heat index values using the relative humidity (rh) datasets.

.. code-block:: python

    baseline_measures = hdp.measure.format_standard_measures(temp_datasets=[baseline_tasmax], rh=baseline_rh)
    ssp370_measures = hdp.measure.format_standard_measures(temp_datasets=[ssp370_tasmax], rh=ssp370_rh)

Now we can generate our range of thresholds:

.. code-block:: python

    percentiles = np.arange(0.9, 1.0, 0.01)
    thresholds = hdp.threshold.compute_thresholds(
        [baseline_measures["tasmax"], baseline_measures["tasmax_hi"]],
        percentiles
    )

The DataArray structure is visualized below:

.. image:: assets/threshold_dataarray_example.png
   :width: 600

:ref:`example_2`
Example 2: RAMIP Analysis
-------------------------

:ref:`threshold_calc`
Threshold Calculation
---------------------


