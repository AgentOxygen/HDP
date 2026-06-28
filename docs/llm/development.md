# Development

## Environment setup

Three supported paths (from `docs/dev_guide.rst`):

- **Docker (preferred):**
  - Build: `docker build --rm -t hdp .`
  - Full test suite: `docker run -v .:/project -it hdp`
  - One test file: `docker run -v .:/project -it hdp pytest hdp/tests/test_workflow.py`
  - Live docs server (`localhost:7000`):
    `docker run -v .:/project -p 7000:7000 -it hdp sphinx-autobuild docs/ docs/_build/ --host 0.0.0.0 --port 7000`
- **Conda:** `conda env create --file=environment.yml && conda activate hdp_dev && pip install -e . && pytest hdp/tests`
- **Existing env:** `pip install -r requirements.txt && pip install -e . && pytest hdp/tests`

Python ≥ 3.12.3. Core deps (from `pyproject.toml`): numpy, `xarray>=2025.1.1`, cartopy,
`numba>=0.60.0`, nc_time_axis, `dask[complete]`, zarr, netCDF4, tqdm, ipywidgets, nbformat.
Package name on PyPI is `HDP_python` / import name `hdp` (version is read at runtime via
`importlib.metadata.version('hdp_python')` in `get_version()`).

## Testing strategy

Tests live in `hdp/tests/` and use the `hdp.utils.generate_test_*` mock generators (no downloads).
The mock grid is deliberately `(2, 3)` — non-square — so coordinate-handling bugs (lat/lon swaps)
surface. Tests run with `add_noise=False` for reproducibility. CI runs `pytest hdp/tests/` on every
push touching any `**.py` file (`.github/workflows/unit_tests.yml`, Python 3.13).

Two tiers of tests, matching the two-tier architecture:

- **Kernel tests on 1-D arrays** — `test_heatwave_frequency.py`, `test_heatwave_number.py`,
  `test_heatwave_duration.py`, `test_heatwave_average.py`, `test_index_heatwaves.py`. Because the
  Numba kernels are embarrassingly parallel per timeseries, correctness is verified on hand-checked
  1-D inputs (a `season_ranges` array like `[[0, 100]]` plus an integer/boolean day series). This
  is where you add cases for any kernel change — it is the cheapest, sharpest signal.
- **End-to-end / scaling test** — `test_workflow.py` runs the full
  format → threshold → metric → notebook pipeline on mock 3-D data, asserts output dims/dtypes/attrs
  and the `HWF ≥ HWD ≥ HWA` sanity relation, and (via a `tmp_path` fixture) that the figure deck
  saves. `test_utils.py` checks the mock generators themselves.

There are no graphical/visual regression tests for the figure deck yet — that is an open area.

### Adding a test for a kernel change
Mirror the existing pattern: construct an integer `hw_ts` (output of `index_heatwaves`, where each
event has a unique nonzero index) and a `season_ranges` array of `[start, end]` index pairs, then
assert the metric with `np.array_equal`. Cover the null (all-zero), full (all-hot), multi-event,
and multi-season cases as the existing files do.

## Code conventions

- **Docstrings are reStructuredText** (`:param:`/`:type:`/`:return:`/`:rtype:`) — Sphinx renders the
  API reference (`docs/api.rst`) from them. Match this style on new public functions; the notebook
  deck also extracts the first docstring paragraph via `get_func_description` for figure captions.
- **Metadata discipline:** every Dataset HDP emits sets `hdp_type` and appends to `history` via
  `add_history`. Preserve `baseline_variable` through any transformation — it is the join key
  between measures and thresholds. See [`data_model.md`](data_model.md).
- **Numba kernels stay NumPy-pure** (`@nb.njit`/`@nb.guvectorize`/`@nb.vectorize`) — no xarray, no
  unsupported Python objects. Keep them unit-testable on 1-D arrays.
- **Dask correctness:** new array-producing wrappers build an explicit `template` DataArray (dims,
  shape, chunks, coords) before `map_blocks`/`apply_ufunc`. Copy the bookkeeping in
  `compute_individual_metrics` / `compute_threshold` rather than inventing a new shape scheme.
- **Never assume a chunked `time`.** Re-apply `chunk({"time": -1})` if you open data inside a new
  IO entry point (as `compute_metrics_io` does).

## Adding a new feature — quick checklists

**A new heatwave metric (e.g. HWX):**
1. Add the `@nb.njit` kernel in `metric.py` with signature `(hw_ts, season_ranges) -> np.ndarray`.
2. Call it in `compute_heatwave_metrics`, grow the stacked output's first axis beyond 4, and map the
   new index in `compute_individual_metrics` (the `metric=N` selection + the `ds[...]` assignment).
3. Set `units`/`long_name`/`description` attrs on the new variable.
4. Add a 1-D kernel test file mirroring `test_heatwave_frequency.py`; extend `test_workflow.py`'s
   units/dtype assertions.
5. Teach the figure layer (`figure.py` / `notebook.py`) about the new variable if it should appear
   in the deck.

**A new input unit or measure type:** extend `TEMPERATURE_UNITS`/`HUMIDITY_UNITS` and
`convert_temp_units` in `measure.py`; add conversion helpers; cover in a test.

**A new figure:** add a `plot_*` function in `figure.py` returning a `Figure`, give it a descriptive
first docstring paragraph (it becomes the caption), then wire it into `create_notebook`.

## Docs

User docs are Sphinx/reST under `docs/` (`overview.rst`, `examples.rst`, `api.rst`, etc.) and build
on Read the Docs (`.readthedocs.yaml`). The root `llms.txt` is the user-facing LLM context file;
this `docs/llm/` set is the maintenance-facing companion. When you change public API, behavior, or
the data schema, update **all three**: the reST docs, `llms.txt`, and the relevant file here.
