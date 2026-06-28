# Architecture

HDP is a thin orchestration layer over **Numba kernels** (the actual heatwave math) and
**Dask/xarray** (the parallelism). Understanding the layering is the key to changing it safely.

## The three-layer pattern (recurs in every numeric module)

Each expensive computation appears at three layers. When you add or change a computation,
mirror this structure:

1. **Numba kernel** — `@nb.njit` or `@nb.guvectorize`, operates on plain NumPy arrays, no xarray.
   Pure, embarrassingly parallel per timeseries, unit-testable on 1-D arrays.
   Examples: `index_heatwaves`, `heatwave_frequency`/`heatwave_number`/`heatwave_duration`/
   `heatwave_average`, `indicate_hot_days`, `compute_percentiles`, `heat_index`.
2. **xarray wrapper** — builds a Dask `template` DataArray, then calls `xarray.map_blocks` or
   `xarray.apply_ufunc` to broadcast the kernel across the lat/lon grid, attaching coords/dims.
   Examples: `compute_individual_metrics`, `compute_threshold`, `apply_heat_index`.
3. **Group / IO wrapper** — loops the xarray wrapper over the parameter space (multiple measures,
   thresholds) and merges, or reads inputs from disk and writes outputs to disk.
   Examples: `compute_group_metrics`, `compute_thresholds`, `compute_metrics_io`,
   `compute_threshold_io`.

The `*_wrapper` functions (`compute_heatwave_metrics_wrapper`, `compute_percentiles_wrapper`,
`heat_index_map_wrapper`) are the closures passed to `map_blocks`/`apply_ufunc`. They are not
public API, but they hold the dims/core-dims bookkeeping — read them before touching shapes.

## Module responsibilities

| Module | Responsibility | Key public entry points |
|---|---|---|
| `hdp/measure.py` | Validate units, convert to °C, tag HDP metadata, optionally derive heat index. | `format_standard_measures` |
| `hdp/threshold.py` | Seasonally-varying (day-of-year windowed) percentile thresholds from a baseline. | `compute_thresholds`, `compute_threshold`, `compute_threshold_io` |
| `hdp/metric.py` | The heatwave indexing + the four metrics (HWF/HWN/HWD/HWA) over the parameter space. | `compute_group_metrics`, `compute_individual_metrics`, `compute_metrics_io` |
| `hdp/graphics/figure.py` | Pure matplotlib figure builders (return `Figure` objects). | `plot_metric_*`, `plot_multi_measure_metric_comparisons` |
| `hdp/graphics/notebook.py` | Assemble figures into a standardized `.ipynb` deck. | `create_notebook` → `HDPNotebook.save_notebook` |
| `hdp/utils.py` | Version/history metadata helpers + mock-data generators for tests/examples. | `generate_test_*`, `add_history`, `get_version` |
| `hdp/definitions.py` | Package-relative paths (e.g. the matplotlib stylesheet). | `PATH_MPL_STYLESHEET` |
| `hdp/graphics/winkel_tripel.py` | Winkel-Tripel Cartopy projection used by map figures. | `WinkelTripel` |

## The pipeline, stage by stage

The pipeline is a fixed four-stage sequence. Each stage consumes the previous stage's `xarray`
output and is tagged with an `hdp_type` attribute. Full dims/attrs are in
[`data_model.md`](data_model.md).

1. **Format measures** — `measure.format_standard_measures(temp_datasets, rh=None)` takes a list of
   raw temperature DataArrays (any supported unit), validates and converts them to °C, tags
   `hdp_type="measure"`, and merges them into one Dataset. If `rh` is supplied it also derives a
   heat-index measure per temperature. Call it once for the **baseline** and once for the **test**.
2. **Compute thresholds** — `threshold.compute_thresholds(baseline_measures, percentiles)` consumes
   the *baseline* measure Dataset and produces `hdp_type="threshold"`. Internally:
   `datetimes_to_windows` builds per-day-of-year index windows, then the `compute_percentiles`
   guvectorized kernel takes the requested percentiles within each window.
3. **Compute metrics** — `metric.compute_group_metrics(test_measures, thresholds, definitions)`
   consumes the *test* measure Dataset plus the threshold Dataset and produces `hdp_type="metric"`.
   A measure is paired with a threshold only when their `baseline_variable` attrs match. For each
   `(measure, threshold, percentile, definition)` cell it runs `indicate_hot_days` →
   `index_heatwaves` → the four metric kernels.
4. **Build figure deck** — `graphics.notebook.create_notebook(metric_ds)` consumes a metric Dataset
   and writes a standardized `.ipynb`. (Measure/threshold inputs are currently no-ops.)

So the data dependencies are: raw temps → measure; baseline measure → threshold;
(test measure + threshold) → metric; metric → notebook.

## How the parameter space is realized

`compute_individual_metrics` builds a Dask `template` DataArray with dims
`[percentile, definition, <spatial dims>, metric, year]` and uses `map_blocks`, so a single Dask
graph evaluates every `(percentile × definition)` cell. `compute_heatwave_metrics_wrapper` loops
percentiles and definitions, calling the Numba `compute_heatwave_metrics` once per combination via
`apply_ufunc`. The four metrics are stacked on a synthetic `metric` axis (0=HWF, 1=HWN, 2=HWD,
3=HWA) and split back into named variables at the end. **Sampling many definitions in one pass is
HDP's core value — never refactor this into per-definition re-reads of the data.**

## Invariants every change must preserve

- **Single time chunk.** `compute_threshold` and `compute_individual_metrics` both rely on
  `chunk({"time": -1})`. Spatial chunking is fine and expected. `compute_metrics_io` re-applies
  this defensively after opening from disk.
- **cftime time axis.** Kernels read `.dayofyr`, `.year`, `.month`, `.day`, and `.calendar` off
  timestamp objects (`build_doy_map`, `get_range_indices`, `compute_hemisphere_ranges`). Always
  open data with `use_cftime=True`. A NumPy `datetime64` axis will break these.
- **`hdp_type` contract.** Downstream code branches on `attrs["hdp_type"]`
  (`"measure"`/`"threshold"`/`"metric"`). `create_notebook` asserts it; metric compute asserts the
  threshold's. Keep it set on every Dataset you produce. See [`data_model.md`](data_model.md).
- **`baseline_variable` matching.** `compute_group_metrics` pairs a measure with a threshold only
  when `measure.attrs["baseline_variable"] == threshold.attrs["baseline_variable"]`. This is how a
  measure finds "its" threshold. Heat-index measures set `baseline_variable` to `"{name}_hi"`.
- **Hemisphere seasons are hard-coded.** NH = May 1–Oct 1, SH = Nov 1–Apr 1 (see
  `compute_hemisphere_ranges`). Partial leading/trailing seasons are dropped; the per-season `year`
  axis is renamed to `time` in metric output. Changing season bounds is a behavior change with
  test impact.
- **Numba purity.** `@nb.njit` kernels must stay NumPy-only (no xarray, no Python objects beyond
  what Numba supports). They are validated independently on 1-D arrays — keep them that way.

## Known rough edges / gotchas (don't "fix" without checking intent)

- `compute_group_metrics` / `compute_individual_metrics` accept `include_threshold` but currently
  do not actually embed the threshold in the output — the parameter is plumbed but unused.
- HWA/HWN carry `units="heatwave events"` in their attrs, but HWA is a length in days, not an event
  count. The tests assert the current (mislabeled) units string — change both together.
- `compute_metrics_io` derives a default threshold variable name `f"threshold_{measure_var}"`,
  but `compute_threshold` writes variables as `f"{name}_threshold"`. Pass `override_threshold_var`
  explicitly when using the IO path until this mismatch is reconciled.
- `create_notebook` treats `hdp_type="measure"` and `"threshold"` as no-ops (only metric decks are
  implemented). Figure generation for those is an open feature.
- `notebook.py` defines `set_section_label` without a `self` parameter, and `figure.py` has a
  duplicated `add_percentile_colorbar` plus an `add_definitions_colorbar` — dead/buggy helpers;
  verify callers before relying on them.
