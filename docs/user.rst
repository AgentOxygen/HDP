User Guide
=====

What is the HDP?
------------
The Heatwave Diagnostics Package (HDP) is a collection of Python tools for computing heatwave metrics and generating summary figures. Functions can be imported in Jupyter notebooks and Python scripts or called in the terminal using the command line interface (CLI). All data uses `xarray <https://docs.xarray.dev/en/stable/>`_ data structures and can be saved to disk as either Zarr stores (default) or netCDF datasets. Summary figures can be generated and saved to disk or displayed interactively within a Jupyter notebook.

The HDP workflow follows three steps:

1. Acquire and format both a baseline and test measure of heat

2. Generate an extreme heat threshold from the baseline

3. Compute heatwave metrics by comparing the test measure against the baseline threshold


Installation
------------

To use the HDP, first install it using pip:

.. code-block:: console

   $ pip install hdp-python


