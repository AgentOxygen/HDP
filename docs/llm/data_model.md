# Data Model — the `hdp_type` contract

Every Dataset HDP produces is a normal `xarray.Dataset` tagged with an `hdp_type` attribute. That
tag is the contract that lets the four stages compose. The three values are `"measure"`,
`"threshold"`, and `"metric"`. This file is the authoritative reference for the dims, coords,
variables, and attrs at each stage — keep it in sync with the source if you change a schema.

## Stage 1 — `hdp_type="measure"` (from `measure.format_standard_measures`)

- **Variables:** one data variable per input measure, keyed by the input DataArray's `.name`. If
  `rh` is supplied, an additional `"{name}_hi"` heat-index variable is added per temperature.
- **Dims:** `time, lat, lon` (+ `member` if present). dtype `float32`.
- **Per-variable attrs:**
  - `units` — always `"degC"` after formatting (auto-converted).
  - `hdp_type="measure"`
  - `input_variable` — original `.name`.
  - `baseline_variable` — the name used to pair with a threshold downstream. Equals
    `input_variable` for plain temperature; equals `"{name}_hi"` for a heat-index measure.
  - `history` — appended by `add_history`.
- **Dataset attrs:** `description`, `hdp_version`, `history`.
- **Input requirements (asserted):** each input DataArray must have a `.name` and
  `attrs["units"]` in `{degC, degK, degF, C, K, F}`. RH (if given) must have units in `{%, g/g}`
  (`g/g` is auto-scaled ×100 to `%`).

## Stage 2 — `hdp_type="threshold"` (from `threshold.compute_thresholds`)

- **Variables:** `"{measure}_threshold"` (one per baseline variable).
- **Dims:** `lat, lon, doy, percentile`. Note: there is **no `time` dim** — time has been collapsed
  into `doy` (day-of-year, 0-indexed) and `percentile`.
- **`member` handling:** if the baseline has a `member` dim, members are **concatenated along
  `time`** before percentiles are taken (pooled sample), so the output has no `member` dim.
- **Per-variable attrs:** `long_name`, `baseline_variable`, `baseline_start_time`,
  `baseline_end_time`, `baseline_calendar`, `param_percentiles`, `param_noseason`,
  `param_rolling_window_size`, `param_fixed_value`, `hdp_type="threshold"`.
- **Coord attrs:** `doy` carries `units="day_of_year"` and `baseline_calendar`.
- **Dataset attrs:** `description`, `hdp_version`.

## Stage 3 — `hdp_type="metric"` (from `metric.compute_group_metrics`)

- **Variables:** `"{measure}.{threshold}.{metric}"` where `metric ∈ {HWF, HWN, HWD, HWA}` and the
  delimiter is a literal `.`. Example: `test_temperature_data.test_temperature_data_threshold.HWF`.
  The Dataset attrs `variable_naming_desc` and `variable_naming_delimeter` document this (note the
  misspelling "delimeter" is the actual attr key — match it exactly).
- **Dims:** `time, lat, lon, percentile, definition` (+ `member` if present). dtype `int`.
  - `time` here is the **per-season year axis**: `compute_individual_metrics` computes one value
    per heatwave season-year, names that axis `year`, then renames it to `time` and assigns a
    cftime coordinate spanning the season years. It is NOT the daily input time axis.
- **`definition` coord:** string `"a-b-c"` built from each `[min_duration, max_break, max_subs]`
  triple (e.g. `[3,1,1]` → `"3-1-1"`).
- **`percentile` coord:** the fractions in `(0, 1)` used for the thresholds.
- **Per-variable attrs:** `units`, `long_name`, `description`, plus combined `history` from the
  source measure and threshold. Units strings: HWF/HWD = `"heatwave days"`,
  HWN/HWA = `"heatwave events"` (HWA is actually a length in days — see the gotcha in
  [`architecture.md`](architecture.md)).
- **Dataset attrs:** `description`, `hdp_version`, `hdp_type="metric"`, `variable_naming_desc`,
  `variable_naming_delimeter`.

## The four metrics (computed per heatwave season-year)

| Var | Name | Definition | Kernel |
|---|---|---|---|
| `HWF` | Heatwave Frequency | Number of heatwave **days** in the season. | `heatwave_frequency` |
| `HWN` | Heatwave Number | Number of distinct heatwave **events**. | `heatwave_number` |
| `HWD` | Heatwave Duration | Length (days) of the **longest** event. | `heatwave_duration` |
| `HWA` | Heatwave Average | **Mean** event length (days). | `heatwave_average` |

Mathematical sanity relations the tests rely on: per season, `HWF ≥ HWD ≥ HWA` (total hot days ≥
longest event ≥ mean event length). Preserve these when editing kernels.

## The heatwave definition triple `[min_duration, max_break, max_subs]`

Stored as the `"a-b-c"` coordinate on the `definition` dimension and consumed by `index_heatwaves`:

1. `min_duration` — minimum consecutive hot days required to **start** a heatwave event.
2. `max_break` — maximum number of non-hot ("break") days tolerated within an event before it ends.
3. `max_subs` — maximum number of subsequent sub-events joined back across breaks into one event.

`index_heatwaves` returns an integer timeseries where each distinct event has a unique nonzero
index; the metric kernels then slice that per season range and aggregate.

## Conventions worth memorizing

- **Percentiles are fractions in `(0, 1)`** (e.g. `0.9`), never `0–100`.
- **Seasons are hemisphere-aware and hard-coded** (NH May 1–Oct 1, SH Nov 1–Apr 1); partial
  leading/trailing seasons are dropped.
- **Saving:** `.to_zarr(path)` (default/faster) or `.to_netcdf(path)`. For zarr, `zarr_format=2`
  may be needed for compatibility (see the CMIP cloud example).
- **`baseline_variable` is the join key** between a measure and its threshold. If a transformation
  renames a measure, update `baseline_variable` or downstream pairing silently drops it.
