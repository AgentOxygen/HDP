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
    
    def __init__(self, dataset):
        super(HeatwaveDataset, self).__init__(dataset, attrs=dataset.attrs)

    def __init__(self,
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
            threshold = hdp.compute_threshold(measures[measure]["base"], percentiles, None)
            hw_datasets.append(
                hdp.sample_heatwave_metrics(measures[measure]["test"], threshold["threshold"], definitions)
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
        if self.percentile.size == 1 and self.definition.size == 1:
            figures = [hplts.create_single_hw_plot(self)]
        elif self.percentile.size == 1:
            figures = [hplts.create_single_perc_plot(self)]
        elif self.definition.size == 1:
            figures = [hplts.create_single_def_plot(self)]
        else:
            figures = [
                hplts.create_global_mean_lineplot(self),
                hplts.create_global_mean_implot(self)
            ]
        for f in figures:
            f.show()
        return figures + [hplts.create_threshold_plot(self)]