# API Map

Every function grouped by module, with its layer (kernel / xarray wrapper / group-IO / helper) and
a one-line contract. Signatures are abbreviated; read the source docstrings for full parameter
docs. Public entry points are **bold**.

## `hdp/measure.py`

| Function | Layer | Contract |
|---|---|---|
| **`format_standard_measures(temp_datasets, rh=None)`** | group | List of temp DataArrays → merged `hdp_type="measure"` Dataset in °C; optional heat-index vars. |
| `convert_temp_units(temp_ds)` | helper | Dispatch to K→C / F→C conversion based on `attrs["units"]`. |
| `kelvin_to_celsius` / `fahrenheit_to_celsius` / `celsius_to_fahrenheit` | helper | Scalar/DataArray unit conversions; update `units` attr + history. |
| `heat_index(temp, rel_humid)` | kernel (`@nb.vectorize`) | NWS heat-index regression; inputs in °F and % → °F. |
| `heat_index_map_wrapper(ds)` | xarray wrapper | `apply_ufunc` of `heat_index` over a `{temp, rh}` Dataset. |
| `apply_heat_index(temp, rh)` | xarray wrapper | Dask-parallel heat index; asserts temp in °F and rh in %; names result `"{temp.name}_hi"`. |

Module constants: `TEMPERATURE_UNITS`, `HUMIDITY_UNITS`.

## `hdp/threshold.py`

| Function | Layer | Contract |
|---|---|---|
| **`compute_thresholds(baseline_dataset, percentiles, no_season=False, rolling_window_size=7, fixed_value=None)`** | group | Loops `compute_threshold` over every variable; merges. |
| **`compute_threshold(baseline_data, percentiles, ...)`** | xarray wrapper | Single baseline DataArray → `hdp_type="threshold"` Dataset; pools `member` over time. |
| **`compute_threshold_io(baseline_path, baseline_var, output_path, percentiles, ..., overwrite=False)`** | IO | Read baseline from disk, compute, write zarr/netCDF. |
| `datetimes_to_windows(datetimes, window_radius)` | helper | Build per-day-of-year rolling index windows from a cftime time axis. |
| `compute_percentiles(temperatures, window_samples, percentiles, output)` | kernel (`@nb.guvectorize`) | Per-doy `np.quantile` over each window sample. |
| `compute_percentiles_wrapper(baseline_data, rolling_windows, percentiles)` | xarray wrapper | `apply_ufunc` closure for `map_blocks`. |

## `hdp/metric.py`

| Function | Layer | Contract |
|---|---|---|
| **`compute_group_metrics(measures, thresholds, hw_definitions, include_threshold=False, check_variables=True)`** | group | For each measure×threshold paired by `baseline_variable`, compute metrics; rename vars to `"{m}.{t}.{metric}"`; merge. |
| **`compute_individual_metrics(measure, threshold, hw_definitions, include_threshold=True, check_variables=True)`** | xarray wrapper | One measure+threshold → `hdp_type="metric"` Dataset over all percentiles×definitions. |
| **`compute_metrics_io(output_path, measure_path, measure_var, threshold_path, hw_definitions, ...)`** | IO | Read measure+threshold from disk, compute, write zarr/netCDF. |
| `index_heatwaves(hot_days_ts, min_duration, max_break, max_subs)` | kernel (`@nb.njit`) | Boolean hot-day ts → integer-indexed heatwave-event ts per the definition triple. |
| `heatwave_frequency / heatwave_number / heatwave_duration / heatwave_average(hw_ts, season_ranges)` | kernel (`@nb.njit`) | The four metrics, aggregated per season range. |
| `indicate_hot_days(measure, threshold, doy_map)` | kernel (`@nb.njit`) | Per-day boolean: measure > threshold-for-that-doy. |
| `compute_heatwave_metrics(measure, threshold, doy_map, min_duration, max_break, max_subs, season_ranges)` | kernel (`@nb.njit`) | Full per-timeseries pipeline → stacked `[4, year]` array. |
| `compute_heatwave_metrics_wrapper(measure, threshold, doy_map, hw_definitions)` | xarray wrapper | Loops percentiles×definitions via `apply_ufunc`; closure for `map_blocks`. |
| `get_range_indices(times, start, end)` | helper | Index ranges for one season window over a cftime axis. |
| `compute_hemisphere_ranges(measure)` | helper | Per-grid-cell season ranges, NH vs SH by `lat` sign; drops partial seasons. |
| `build_doy_map(times)` | helper | cftime axis → 0-indexed day-of-year per timestep. |

## `hdp/graphics/figure.py` (all return matplotlib `Figure`s)

Public plotters: **`plot_metric_timeseries(metric_da)`**, **`plot_metric_decadal_maps(metric_da)`**
(returns a list of figures), **`plot_metric_parameter_comparison(metric_da)`**,
**`plot_multi_measure_metric_comparisons(metric_ds)`**.
Helpers: `compute_weighted_spatial_mean`, `get_decadal_ranges`, `generate_base_figure`,
`convert_axis_to_map`, `add_four_panel`, `get_color_for_value`, `get_metric_axis_label`,
`add_percentile_colorbar`, `add_definitions_colorbar`, `get_metric_name`, `get_unique_metric_names`,
`plot_map`. Map figures use the `WinkelTripel` projection and the `hdp.mplstyle` stylesheet
(`PATH_MPL_STYLESHEET` from `hdp/definitions.py`).

## `hdp/graphics/notebook.py`

- **`create_notebook(hw_ds)`** → `HDPNotebook`. Asserts `hdp_type`; only `"metric"` builds figures
  (measure/threshold are no-ops). Iterates each metric variable, embeds figures as base64 PNGs in
  markdown cells.
- **`HDPNotebook.save_notebook(path, title=None)`** — writes the `.ipynb`. Other methods:
  `create_section`, `add_markdown_cell`, `add_figure_cell`.

## `hdp/utils.py`

- Mock data (cftime, `noleap`, dims `lon, lat, time`, units `degC`, pre-chunked):
  **`generate_test_control_dataarray(start_date, end_date, grid_shape=(2,3), add_noise=False, seed=0)`**,
  **`generate_test_warming_dataarray(..., warming_period=100)`** (adds a linear trend),
  **`generate_test_rh_dataarray(...)`** (units `g/g`).
- Metadata: `add_history(ds, msg)`, `get_version()` (reads installed `hdp_python` version),
  `get_time_stamp()`, `get_func_description(func)` (first docstring paragraph, used for figure
  captions in the notebook deck).
