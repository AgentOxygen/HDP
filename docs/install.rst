Installation
============

The HDP can be installed using PyPI. You can view the webpage `here <https://pypi.org/project/HDP-python/>`_.

.. code-block:: console

   $ pip install hdp-python

Dependencies
------------

The `requirements.txt` list describes all packages and libraries needed to run the HDP in developer mode. PyPI automatically handles dependencies when installing, however the bare-minimum dependencies are listed below if needed (can be pasted after `pip install`):

.. code-block:: console

  numpy \
  "xarray>=2025.1.1" \
  cartopy \
  "numba>=0.60.0" \
  nc_time_axis \
  "dask[complete]" \
  zarr \
  netCDF4 \
  tqdm \
  ipywidgets \
  nbformat