import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
import nc_time_axis
import xarray
import numpy as np
import cartopy.crs as ccrs


plt.style.use("hdp/pacl.mplstyle")


class WinkelTripel(ccrs._WarpedRectangularProjection):
    """
    Winkel-Tripel projection implementation for Cartopy
    """

    def __init__(self, central_longitude=0.0, central_latitude=0.0, globe=None):
        globe = globe or ccrs.Globe(semimajor_axis=ccrs.WGS84_SEMIMAJOR_AXIS)
        proj4_params = [('proj', 'wintri'),
                        ('lon_0', central_longitude),
                        ('lat_0', central_latitude)]

        super(WinkelTripel, self).__init__(proj4_params, central_longitude, globe=globe)

    @property
    def threshold(self):
        return 1e4


class HeatwaveDataset(xarray.Dataset):
    __slots__ = ['dataset',]

    def __init__(self, dataset):
        super(HeatwaveDataset, self).__init__(dataset, attrs=dataset.attrs)

    def global_mean_lineplots(self):
        hwf_spatial_mean = self.HWF.weighted(np.cos(np.deg2rad(self.lat))).mean(dim=["lat", "lon"])
        hwd_spatial_mean = self.HWD.weighted(np.cos(np.deg2rad(self.lat))).mean(dim=["lat", "lon"])
        
        f = plt.figure(figsize=(8.6, 7), dpi=150)
        gridspec = f.add_gridspec(2, 2, height_ratios=[2, 2])
        
        ax1 = f.add_subplot(gridspec[0, 0])
        ax2 = f.add_subplot(gridspec[0, 1])
        ax3 = f.add_subplot(gridspec[1, 0])
        ax4 = f.add_subplot(gridspec[1, 1])
        
        for hw_def in hwf_spatial_mean.definition:
            ax1.plot(
                hwf_spatial_mean.time,
                hwf_spatial_mean.sel(definition=hw_def).mean(dim="percentile"),
                label=hw_def.values
            )
            ax2.plot(
                hwd_spatial_mean.time,
                hwd_spatial_mean.sel(definition=hw_def).mean(dim="percentile"),
                label=hw_def.values
            )
        
        ax1.set_xlim(hwf_spatial_mean.time.values[0], hwf_spatial_mean.time.values[-1])
        ax1.set_title("Global, Percentile Mean HWF")
        
        ax2.set_xlim(hwd_spatial_mean.time.values[0], hwd_spatial_mean.time.values[-1])
        ax2.set_title("Global, Percentile Mean HWD")
        
        ax1.legend()
        ax2.legend()
        
        cmap = plt.get_cmap('winter')
        value = 0.5
        norm = Normalize(vmin=hwf_spatial_mean.percentile[0], vmax=hwf_spatial_mean.percentile[-1])
        
        for perc in hwf_spatial_mean.percentile:
        
            normalized_value = norm(perc)
            color = cmap(normalized_value)
            
            ax3.plot(
                hwf_spatial_mean.time,
                hwf_spatial_mean.sel(percentile=perc).mean(dim="definition"),
                color=color
            )
            ax4.plot(
                hwd_spatial_mean.time,
                hwd_spatial_mean.sel(percentile=perc).mean(dim="definition"),
                color=color
            )
        
        sm = ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar3 = plt.colorbar(sm, ax=ax3)
        cbar3.set_label('Percentile')
        
        cbar4 = plt.colorbar(sm, ax=ax4)
        cbar4.set_label('Percentile')
        
        ax3.set_xlim(hwf_spatial_mean.time.values[0], hwf_spatial_mean.time.values[-1])
        ax3.set_title("Global, Definition Mean HWF")
        
        ax4.set_xlim(hwd_spatial_mean.time.values[0], hwd_spatial_mean.time.values[-1])
        ax4.set_title("Global, Definition Mean HWD")
        
        return f
    
    
    def global_mean_implots(self):
        hwf_spatial_mean = self.HWF.weighted(np.cos(np.deg2rad(self.lat))).mean(dim=["lat", "lon"])
        hwd_spatial_mean = self.HWD.weighted(np.cos(np.deg2rad(self.lat))).mean(dim=["lat", "lon"])
        
        f = plt.figure(figsize=(8.6, 3.5), dpi=150)
        gridspec = f.add_gridspec(1, 2)
        
        ax1 = f.add_subplot(gridspec[0, 0])
        ax2 = f.add_subplot(gridspec[0, 1])
        
        im1 = hwf_spatial_mean.mean(dim="time").plot(ax=ax1, cmap="Reds")
        im1.colorbar.set_label('Days', rotation=270, labelpad=10)
        
        im2 = hwd_spatial_mean.mean(dim="time").plot(ax=ax2, cmap="Reds")
        im2.colorbar.set_label('Days', rotation=270, labelpad=10)
        
        ax1.set_title("Global, Temporal Mean HWF")
        ax2.set_title("Global, Temporal Mean HWD")
        
        ax1.set_xlabel("Definition")
        ax1.set_ylabel("Percentile")
        
        ax2.set_xlabel("Definition")
        ax2.set_ylabel("Percentile")
        
        return f

    def __single_hw_plots(self):
        f1 = plt.figure(figsize=(8.6, 7), dpi=150)
        gridspec = f1.add_gridspec(2, 2, height_ratios=[2, 2])
        
        ax1 = f1.add_subplot(gridspec[0, 0])
        ax2 = f1.add_subplot(gridspec[0, 1])
        ax3 = f1.add_subplot(gridspec[1, 0], projection=WinkelTripel())
        ax4 = f1.add_subplot(gridspec[1, 1], projection=WinkelTripel())
        
        hwf_spatial_mean = self.HWF.weighted(np.cos(np.deg2rad(self.lat))).mean(dim=["lat", "lon"])
        hwd_spatial_mean = self.HWD.weighted(np.cos(np.deg2rad(self.lat))).mean(dim=["lat", "lon"])
        
        time_start = hwf_spatial_mean.time.values[0]
        time_end = hwf_spatial_mean.time.values[-1]
        
        ax1.plot(hwf_spatial_mean.time, hwf_spatial_mean)
        ax1.set_xlim(time_start, time_end)
        ax1.set_xlabel("Time (Year)")
        ax1.set_ylabel("HWF (Days)")
        ax1.set_title("Global Mean HWF")
        
        ax2.plot(hwd_spatial_mean.time, hwd_spatial_mean)
        ax2.set_xlim(time_start, time_end)
        ax2.set_xlabel("Time (Year)")
        ax2.set_ylabel("HWD (Days)")
        ax2.set_title("Global Mean HWD")
        
        map3 = self.HWF.mean(dim="time").plot.contourf(ax=ax3, transform=ccrs.PlateCarree(), cmap="Reds")
        map4 = self.HWD.mean(dim="time").plot.contourf(ax=ax4, transform=ccrs.PlateCarree(), cmap="Reds")
        
        map3.colorbar.set_label('HWF (Days)', rotation=270, labelpad=10)
        map4.colorbar.set_label('HWD (Days)', rotation=270, labelpad=10)
        
        ax3.coastlines()
        ax4.coastlines()
        
        ax3.set_title(f"{time_start.year} to {time_end.year} Mean HWF")
        ax4.set_title(f"{time_start.year} to {time_end.year} Mean HWD")
        
        return [f1]

    def __single_perc_plots(self):
        pass

    def __single_def_plots(self):
        pass
    
    def plot(self):
        if self.percentile.size == 1 and self.definition.size == 1:
            figures = self.__single_hw_plots()
        elif self.percentile.size == 1:
            figures = self.__single_perc_plots()
        elif self.definition.size == 1:
            figures = self.__single_def_plots()
        else:
            figures = [
                self.global_mean_lineplots(),
                self.global_mean_implots()
            ]
        for f in figures:
            f.show()
        return figures