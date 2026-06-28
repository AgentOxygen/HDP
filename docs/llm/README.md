# HDP — Agent Maintenance Guide

This directory is the **developer-facing** companion to the root [`llms.txt`](../../llms.txt).
`llms.txt` teaches an agent how to *use* HDP to write workflows; these files teach an agent
how to **maintain the package and develop new features** correctly.

Read in this order:

1. [`architecture.md`](architecture.md) — the data pipeline, module responsibilities, and the
   invariants that every change must preserve.
2. [`data_model.md`](data_model.md) — the `hdp_type` Dataset contract (measure → threshold →
   metric): dims, attrs, and variable-naming rules that tie the modules together.
3. [`api_map.md`](api_map.md) — every public and internal function, grouped by module, with the
   "scale-up" pattern (Numba kernel → xarray wrapper → IO wrapper) that recurs throughout.
4. [`development.md`](development.md) — environment setup, the testing strategy, CI, code
   conventions, and a checklist for common feature work.

## 30-second mental model

HDP samples a **parameter space** of `measures × percentile thresholds × heatwave definitions`
in a single pass over gridded climate data. The pipeline is always:

```
measure.format_standard_measures  →  threshold.compute_thresholds  →  metric.compute_group_metrics  →  graphics.notebook.create_notebook
   (hdp_type="measure")               (hdp_type="threshold")           (hdp_type="metric")              (figure deck .ipynb)
```

Everything is an `xarray.Dataset` tagged with an `hdp_type` attribute; the heavy numerics are
Numba kernels (`@nb.njit` / `@nb.guvectorize`) parallelized across the lat/lon grid by Dask via
`xarray.apply_ufunc` / `xarray.map_blocks`.

## The one rule you will break first

**The `time` axis must live in a single Dask chunk** (`da.chunk({"time": -1})`); chunk over
`lat`/`lon` instead. Thresholds need the whole time series to take percentiles, and the heatwave
indexing runs per-timeseries. This constraint propagates through every module — preserve it.
