import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
import nc_time_axis
import numpy as np
import cartopy.crs as ccrs
from cartopy.util import add_cyclic_point
from matplotlib import cm


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

def create_global_mean_lineplot(hw_ds):
	hwf_spatial_mean = hw_ds.HWF.weighted(np.cos(np.deg2rad(hw_ds.lat))).mean(dim=["lat", "lon"])
	hwd_spatial_mean = hw_ds.HWD.weighted(np.cos(np.deg2rad(hw_ds.lat))).mean(dim=["lat", "lon"])
	
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
	
	cmap = cm.get_cmap('winter')
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


def create_global_mean_implot(hw_ds):
	hwf_spatial_mean = hw_ds.HWF.weighted(np.cos(np.deg2rad(hw_ds.lat))).mean(dim=["lat", "lon"])
	hwd_spatial_mean = hw_ds.HWD.weighted(np.cos(np.deg2rad(hw_ds.lat))).mean(dim=["lat", "lon"])
	
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

def create_single_hw_plot(hw_ds):
	f = plt.figure(figsize=(8.6, 7), dpi=150)
	gridspec = f.add_gridspec(2, 2, height_ratios=[2, 2])
	
	ax1 = f.add_subplot(gridspec[0, 0])
	ax2 = f.add_subplot(gridspec[0, 1])
	ax3 = f.add_subplot(gridspec[1, 0])
	ax4 = f.add_subplot(gridspec[1, 1])
	
	hwf_spatial_mean = hw_ds.HWF.weighted(np.cos(np.deg2rad(hw_ds.lat))).mean(dim=["lat", "lon"])
	hwd_spatial_mean = hw_ds.HWD.weighted(np.cos(np.deg2rad(hw_ds.lat))).mean(dim=["lat", "lon"])
	
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

	ax3.set_title(f"{time_start.year} to {time_end.year} Mean HWF")
	ax4.set_title(f"{time_start.year} to {time_end.year} Mean HWD")
	
	ax3.xaxis.set_visible(False)
	ax3.yaxis.set_visible(False)
	ax3.spines['top'].set_visible(False)
	ax3.spines['right'].set_visible(False)
	ax3.spines['left'].set_visible(False)
	ax3.spines['bottom'].set_visible(False)
	ax3.grid(False)
	ax3 = f.add_axes(ax3.get_position(), frameon=True, projection=WinkelTripel(), zorder=-1)

	ax4.xaxis.set_visible(False)
	ax4.yaxis.set_visible(False)
	ax4.spines['top'].set_visible(False)
	ax4.spines['right'].set_visible(False)
	ax4.spines['left'].set_visible(False)
	ax4.spines['bottom'].set_visible(False)
	ax4.grid(False)
	ax4 = f.add_axes(ax4.get_position(), frameon=True, projection=WinkelTripel(), zorder=-1)

	cyclic_values, cyclic_lons = add_cyclic_point(hw_ds.HWF.mean(dim=["time"]).values, hw_ds.lon.values, axis=-1)
	ax3_contour = ax3.contourf(cyclic_lons, hw_ds.lat.values, cyclic_values, transform=ccrs.PlateCarree(), cmap="Reds")
	ax3_cbar = f.colorbar(ax3_contour, location="bottom", anchor=(0.0, 0.0), pad=0)
	ax3_cbar.set_label("HWF (Days)")
	ax3.coastlines()
	
	cyclic_values, cyclic_lons = add_cyclic_point(hw_ds.HWD.mean(dim=["time"]).values, hw_ds.lon.values, axis=-1)
	ax4_contour = ax4.contourf(cyclic_lons, hw_ds.lat.values, cyclic_values, transform=ccrs.PlateCarree(), cmap="Reds")
	ax4_cbar = f.colorbar(ax4_contour, location="bottom", anchor=(0.0, 0.0), pad=0)
	ax4_cbar.set_label("HWD (Days)")
	ax4.coastlines()
	
	return f

def create_single_def_plot(hw_ds):
	f = plt.figure(figsize=(8.6, 7), dpi=150)
	gridspec = f.add_gridspec(2, 2)
	
	ax1 = f.add_subplot(gridspec[0, 0])
	ax2 = f.add_subplot(gridspec[0, 1])
	ax3 = f.add_subplot(gridspec[1, 0])
	ax4 = f.add_subplot(gridspec[1, 1])
	
	hwf_spatial_mean = hw_ds.HWF.weighted(np.cos(np.deg2rad(hw_ds.lat))).mean(dim=["lat", "lon"])
	hwd_spatial_mean = hw_ds.HWD.weighted(np.cos(np.deg2rad(hw_ds.lat))).mean(dim=["lat", "lon"])
	
	time_start = hwf_spatial_mean.time.values[0]
	time_end = hwf_spatial_mean.time.values[-1]
	
	for definition in hwf_spatial_mean.definition.values:
		hwf_spatial_mean.sel(definition=definition).plot(ax=ax1, label=definition)
		hwd_spatial_mean.sel(definition=definition).plot(ax=ax2, label=definition)
	
	ax3.set_title(f"{time_start.year} to {time_end.year}, Percentile Mean HWF")
	ax4.set_title(f"{time_start.year} to {time_end.year}, Percentile Mean HWD")
	
	ax3.xaxis.set_visible(False)
	ax3.yaxis.set_visible(False)
	ax3.spines['top'].set_visible(False)
	ax3.spines['right'].set_visible(False)
	ax3.spines['left'].set_visible(False)
	ax3.spines['bottom'].set_visible(False)
	ax3.grid(False)
	ax3 = f.add_axes(ax3.get_position(), frameon=True, projection=WinkelTripel(), zorder=-1)
	
	ax4.xaxis.set_visible(False)
	ax4.yaxis.set_visible(False)
	ax4.spines['top'].set_visible(False)
	ax4.spines['right'].set_visible(False)
	ax4.spines['left'].set_visible(False)
	ax4.spines['bottom'].set_visible(False)
	ax4.grid(False)
	ax4 = f.add_axes(ax4.get_position(), frameon=True, projection=WinkelTripel(), zorder=-1)
	
	ax1.set_title("Global Mean HWF")
	ax2.set_title("Global Mean HWD")
	
	ax1.set_xlim(time_start, time_end)
	ax2.set_xlim(time_start, time_end)
	
	ax1.legend()
	ax1.set_xlabel("Time (Year)")
	ax1.set_ylabel("HWF (Days)")
	ax1.set_title("Global Mean HWF")
	
	ax2.legend()
	ax2.set_xlabel("Time (Year)")
	ax2.set_ylabel("HWD (Days)")
	ax2.set_title("Global Mean HWD")
	
	cyclic_values, cyclic_lons = add_cyclic_point(hw_ds.HWF.mean(dim=["time", "percentile"]).values, hw_ds.lon.values, axis=-1)
	ax3_contour = ax3.contourf(cyclic_lons, hw_ds.lat.values, cyclic_values, transform=ccrs.PlateCarree(), cmap="Reds")
	ax3_cbar = f.colorbar(ax3_contour, location="bottom", anchor=(0.0, 0.0), pad=0)
	ax3_cbar.set_label("HWF (Days)")
	ax3.coastlines()
	
	cyclic_values, cyclic_lons = add_cyclic_point(hw_ds.HWD.mean(dim=["time", "percentile"]).values, hw_ds.lon.values, axis=-1)
	ax4_contour = ax4.contourf(cyclic_lons, hw_ds.lat.values, cyclic_values, transform=ccrs.PlateCarree(), cmap="Reds")
	ax4_cbar = f.colorbar(ax4_contour, location="bottom", anchor=(0.0, 0.0), pad=0)
	ax4_cbar.set_label("HWD (Days)")
	ax4.coastlines()
	return f

def create_single_perc_plot(hw_ds):
	f = plt.figure(figsize=(8.6, 7), dpi=150)
	gridspec = f.add_gridspec(2, 2)
	
	ax1 = f.add_subplot(gridspec[0, 0])
	ax2 = f.add_subplot(gridspec[0, 1])
	ax3 = f.add_subplot(gridspec[1, 0])
	ax4 = f.add_subplot(gridspec[1, 1])
	
	hwf_spatial_mean = hw_ds.HWF.weighted(np.cos(np.deg2rad(hw_ds.lat))).mean(dim=["lat", "lon"])
	hwd_spatial_mean = hw_ds.HWD.weighted(np.cos(np.deg2rad(hw_ds.lat))).mean(dim=["lat", "lon"])
	
	time_start = hwf_spatial_mean.time.values[0]
	time_end = hwf_spatial_mean.time.values[-1]
	
	for definition in hwf_spatial_mean.definition.values:
		hwf_spatial_mean.sel(definition=definition).plot(ax=ax1, label=definition)
		hwd_spatial_mean.sel(definition=definition).plot(ax=ax2, label=definition)
	
	
	ax3.set_title(f"{time_start.year} to {time_end.year}, Definition Mean HWF")
	ax4.set_title(f"{time_start.year} to {time_end.year}, Definition Mean HWD")
	
	ax3.xaxis.set_visible(False)
	ax3.yaxis.set_visible(False)
	ax3.spines['top'].set_visible(False)
	ax3.spines['right'].set_visible(False)
	ax3.spines['left'].set_visible(False)
	ax3.spines['bottom'].set_visible(False)
	ax3.grid(False)
	ax3 = f.add_axes(ax3.get_position(), frameon=True, projection=WinkelTripel(), zorder=-1)
	
	ax4.xaxis.set_visible(False)
	ax4.yaxis.set_visible(False)
	ax4.spines['top'].set_visible(False)
	ax4.spines['right'].set_visible(False)
	ax4.spines['left'].set_visible(False)
	ax4.spines['bottom'].set_visible(False)
	ax4.grid(False)
	ax4 = f.add_axes(ax4.get_position(), frameon=True, projection=WinkelTripel(), zorder=-1)
	
	ax1.set_title("Global Mean HWF")
	ax2.set_title("Global Mean HWD")
	
	ax1.set_xlim(time_start, time_end)
	ax2.set_xlim(time_start, time_end)
	
	ax1.legend()
	ax1.set_xlabel("Time (Year)")
	ax1.set_ylabel("HWF (Days)")
	ax1.set_title("Global Mean HWF")
	
	ax2.legend()
	ax2.set_xlabel("Time (Year)")
	ax2.set_ylabel("HWD (Days)")
	ax2.set_title("Global Mean HWD")
	
	cyclic_values, cyclic_lons = add_cyclic_point(hw_ds.HWF.mean(dim=["time", "definition"]).values, hw_ds.lon.values, axis=-1)
	ax3_contour = ax3.contourf(cyclic_lons, hw_ds.lat.values, cyclic_values, transform=ccrs.PlateCarree(), cmap="Reds")
	ax3_cbar = f.colorbar(ax3_contour, location="bottom", anchor=(0.0, 0.0), pad=0)
	ax3_cbar.set_label("HWF (Days)")
	ax3.coastlines()
	
	cyclic_values, cyclic_lons = add_cyclic_point(hw_ds.HWD.mean(dim=["time", "definition"]).values, hw_ds.lon.values, axis=-1)
	ax4_contour = ax4.contourf(cyclic_lons, hw_ds.lat.values, cyclic_values, transform=ccrs.PlateCarree(), cmap="Reds")
	ax4_cbar = f.colorbar(ax4_contour, location="bottom", anchor=(0.0, 0.0), pad=0)
	ax4_cbar.set_label("HWD (Days)")
	ax4.coastlines()
	return f

def create_multi_threshold_plot(hw_ds):
    f = plt.figure(figsize=(8.6, 3.5), dpi=150)
    gridspec = f.add_gridspec(1, 2)
    
    ax1 = f.add_subplot(gridspec[0, 0])
    ax2 = f.add_subplot(gridspec[0, 1])
    
    hw_ds_spatial = hw_ds.threshold.mean(dim=["lat", "lon"])
    
    ax1.set_xlim(hw_ds.day[0], hw_ds.day[-1])
    ax1.set_xlabel("Day Of Year")
    ax1.set_ylabel(f"Threshold ({hw_ds.get_threshold_units()})")
    ax1.set_title("Global Mean Threshold")
    
    cmap = plt.get_cmap('winter')
    value = 0.5
    norm = Normalize(vmin=hw_ds.percentile[0], vmax=hw_ds.percentile[-1])
    
    for perc in hw_ds.percentile:
        normalized_value = norm(perc)
        color = cmap(normalized_value)
        
        ax1.plot(
            hw_ds_spatial.day,
            hw_ds_spatial.sel(percentile=perc),
            color=color
        )
    
    sm = ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar1 = plt.colorbar(sm, ax=ax1)
    cbar1.set_label('Percentile')
    
    ax2.set_title("Mean DOY, Perc. Std. Threshold")
    ax2.xaxis.set_visible(False)
    ax2.yaxis.set_visible(False)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.spines['left'].set_visible(False)
    ax2.spines['bottom'].set_visible(False)
    ax2.grid(False)
    ax2 = f.add_axes(ax2.get_position(), frameon=True, projection=WinkelTripel(), zorder=-1)
    
    cyclic_values, cyclic_lons = add_cyclic_point(hw_ds.threshold.mean(dim="day").std(dim="percentile").values, hw_ds.lon.values, axis=-1)
    ax2_contour = ax2.contourf(cyclic_lons, hw_ds.lat.values, cyclic_values, transform=ccrs.PlateCarree(), cmap="Reds")
    ax2_cbar = f.colorbar(ax2_contour, location="bottom", anchor=(0.0, 0.0), pad=0)
    ax2_cbar.set_label(f"Threshold ({hw_ds.get_threshold_units()})")
    ax2.coastlines()
    return f

def create_single_threshold_plot(hw_ds):
	f = plt.figure(figsize=(8.6, 3.5), dpi=150)
	gridspec = f.add_gridspec(1, 2)
	
	ax1 = f.add_subplot(gridspec[0, 0])
	ax2 = f.add_subplot(gridspec[0, 1])
	
	ax1.plot(hw_ds.day, hw_ds.threshold.mean(dim=["lat", "lon"]).values)
	ax1.set_xlim(hw_ds.day[0], hw_ds.day[-1])
	ax1.set_xlabel("Day Of Year")
	ax1.set_ylabel(f"Threshold ({hw_ds.get_threshold_units()})")
	ax1.set_title("Global Mean Threshold")

	ax2.set_title("Mean Day of Year Threshold")
	ax2.xaxis.set_visible(False)
	ax2.yaxis.set_visible(False)
	ax2.spines['top'].set_visible(False)
	ax2.spines['right'].set_visible(False)
	ax2.spines['left'].set_visible(False)
	ax2.spines['bottom'].set_visible(False)
	ax2.grid(False)
	ax2 = f.add_axes(ax2.get_position(), frameon=True, projection=WinkelTripel(), zorder=-1)
	
	cyclic_values, cyclic_lons = add_cyclic_point(hw_ds.threshold.mean(dim=["day"]).values, hw_ds.lon.values, axis=-1)
	ax2_contour = ax2.contourf(cyclic_lons, hw_ds.lat.values, cyclic_values, transform=ccrs.PlateCarree(), cmap="Reds")
	ax2_cbar = f.colorbar(ax2_contour, location="bottom", anchor=(0.0, 0.0), pad=0)
	ax2_cbar.set_label(f"Threshold ({hw_ds.get_threshold_units()})")
	ax2.coastlines()
	return f

def create_threshold_plot(hw_ds):
    if hw_ds.percentile.size > 1:
        return create_multi_threshold_plot(hw_ds)
    else:
        return create_single_threshold_plot(hw_ds)