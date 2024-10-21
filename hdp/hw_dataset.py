import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
import nc_time_axis
import xarray
import numpy as np
import dask
import hdp
import hdp.plotting_functions as hplts


class HeatwaveDataset(xarray.Dataset):
    __slots__ = ['dataset',]
    
    def __init__(self,
                 dataset=None,
                 base_tmin=None,
                 base_tmax=None,
                 base_tavg=None,
                 base_rh=None,
                 test_tmin=None,
                 test_tmax=None,
                 test_tavg=None,
                 test_rh=None,
                 percentiles=None,
                 definitions=None,
                 client=None):
        # Implement error checking
        if dataset is not None:
            super(HeatwaveDataset, self).__init__(dataset, attrs=dataset.attrs)
            return

        client = client or dask.distributed.client._get_global_client()
        if client is None:
            print("Warning: Dask Client object not detected. Proceeding in serial.")
        
        measures = {}
        
        if base_tmin is not None:
            measures["tmin"] = {
                "base": base_tmin,
                "test": test_tmin
            }
        if base_tmax is not None:
            measures["tmax"] = {
                "base": base_tmax,
                "test": test_tmax
            }
        if base_tavg is not None:
            measures["tavg"] = {
                "base": base_tavg,
                "test": test_tavg
            }
        if base_rh is not None:
            # Calculate heat index here, will need to implement units handling
            measures["hi"] = {
                "base": base_rh,
                "test": test_rh
            }

        for measure in measures:
            # Crude implementation, just take first variable
            if type(measures[measure]["base"]) is not xarray.DataArray:
                # Implement a dask chunking parameter, chunks = auto?
                ds = xarray.open_dataset(measures[measure]["base"])
                measures[measure]["base"] = ds[list(ds)[0]]
            if type(measures[measure]["test"]) is not xarray.DataArray:
                ds = xarray.open_dataset(measures[measure]["test"])
                measures[measure]["test"] = ds[list(ds)[0]]

        hw_datasets = []
        for measure in measures:
            # Omitting passing path for now, may utilize a check of the encoding attribute later
            threshold = hdp.hdp.compute_threshold(measures[measure]["base"], percentiles, None)
            hw_datasets.append(
                hdp.hdp.sample_heatwave_metrics(measures[measure]["test"], threshold["threshold"], definitions)
            )
        dataset = xarray.concat(
            hw_datasets,
            dim="measure"
        ).assign_coords(dict(measure=list(measures)))
        super(HeatwaveDataset, self).__init__(dataset, attrs=dataset.attrs)
            

    def get_threshold_units(self):
        if "units" in self:
            return self["units"]
        else:
            return "Unknown"
        
    def plot(self):
        total_figures = []
        for measure in self.measure.values:
            figures = []
            ds = self.sel(measure=measure)
            single_perc = "percentile" not in ds or ds.percentile.size == 1
            single_def = "definition" not in ds or ds.definition.size == 1
            
            if single_perc and single_def:
                figures += [hplts.create_single_hw_plot(ds)]
            elif single_perc:
                figures += [hplts.create_single_perc_plot(ds)]
            elif single_def:
                figures += [hplts.create_single_def_plot(ds)]
            else:
                figures += [
                    hplts.create_global_mean_lineplot(ds),
                    hplts.create_global_mean_implot(ds)
                ]
            figures.append(hplts.create_threshold_plot(ds))
            for f in figures:
                total_figures.append(f)
        for f in total_figures:
            f.show()
        return total_figures

    def __sub__(self, other):
        return HeatwaveDataset(super().__sub__(other))

    def __add__(self, other):
        return HeatwaveDataset(super().__sub__(other))