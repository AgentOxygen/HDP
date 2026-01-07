import matplotlib
matplotlib.use('Agg') # Non-interactive backend

from dask.distributed import Client, LocalCluster
import hdp.measure
import hdp.threshold
import hdp.metric
from hdp.graphics.notebook import create_notebook
import xarray as xr
import numpy as np
import pandas as pd
import s3fs


if __name__ == "__main__":
    aws_cmip_index = pd.read_csv("https://cmip6-pds.s3.amazonaws.com/pangeo-cmip6.csv")
    cesm_index = aws_cmip_index.query("table_id=='day' & source_id=='CESM2'")
    cesm_index = cesm_index[cesm_index["member_id"] == 'r4i1p1f1']

    ssp370_tas_info = cesm_index.query("experiment_id=='ssp370' & variable_id=='tas'")
    baseline_tas_info = cesm_index.query("experiment_id=='historical' & variable_id=='tas'")

    cluster = LocalCluster(n_workers=6, memory_limit="16GB", threads_per_worker=2, processes=True)
    client = Client(cluster)

    fs = s3fs.S3FileSystem(anon=True)

    ssp370_tas = xr.open_zarr(fs.get_mapper(ssp370_tas_info["zstore"].iloc[0]), consolidated=True)["tas"]
    baseline_tas = xr.open_zarr(fs.get_mapper(baseline_tas_info["zstore"].iloc[0]), consolidated=True)["tas"]

    ssp370_tas = ssp370_tas.chunk(dict(time=-1, lat=24, lon=18))
    baseline_tas = baseline_tas.chunk(dict(time=-1, lat=24, lon=18)).sel(time=slice("1961-01-01", "1990-12-31"))

    baseline_measures = hdp.measure.format_standard_measures(temp_datasets=[baseline_tas])
    ssp370_measures = hdp.measure.format_standard_measures(temp_datasets=[ssp370_tas])

    percentiles = np.arange(0.9, 1.0, 0.01)
    thresholds = hdp.threshold.compute_thresholds(
        baseline_measures,
        percentiles
    )

    definitions = [[3,1,0], [3,1,1], [4,0,0], [4,1,1], [5,0,0], [5,1,1]]
    metrics_dataset = hdp.metric.compute_group_metrics(ssp370_measures, thresholds, definitions)

    metrics_dataset.to_zarr("cesm2_ssp370_hw_metrics.zarr", mode='w', compute=True, zarr_format=2)

    metrics_dataset_disk = xr.open_zarr("cesm2_ssp370_hw_metrics.zarr")

    figure_notebook = create_notebook(metrics_dataset_disk)
    figure_notebook.save_notebook("cesm2_ssp370_hw_metrics.ipynb")
    
    client.shutdown()