from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib import cm
from cartopy.util import add_cyclic_point
from hdp.definitions import PATH_MPL_STYLESHEET
from hdp.graphics.winkel_tripel import WinkelTripel
import cartopy.crs as ccrs
import numpy as np
import cftime


def compute_weighted_spatial_mean(da):
    return da.weighted(np.cos(np.deg2rad(da.lat))).mean(dim=["lat", "lon"])


def get_decadal_ranges(times):
    years = [t.year for t in times]
    
    start_year = int(np.floor(years[0] / 10)*10)
    end_year = int(np.ceil(years[-1] / 10)*10)
    
    decadal_ranges = []
    for period_start_year in np.arange(start_year, end_year, 10):
        decadal_ranges.append((int(period_start_year), int(period_start_year) + 9))
    return decadal_ranges


def generate_base_figure():
    plt.style.use(PATH_MPL_STYLESHEET)
    f = plt.figure()
    f.suptitle("Figure Title")
    return f


def convert_axis_to_map(ax):
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.grid(False)
    return ax.figure.add_axes(ax.get_position(), frameon=True, projection=WinkelTripel(), zorder=-1)


def add_four_panel(figure):
    gridspec = figure.add_gridspec(2, 2)
    ax1 = figure.add_subplot(gridspec[0, 0])
    ax2 = figure.add_subplot(gridspec[0, 1])
    ax3 = figure.add_subplot(gridspec[1, 0])
    ax4 = figure.add_subplot(gridspec[1, 1])

    axes = [ax1, ax2, ax3, ax4]

    for index, ax in enumerate(axes):
        ax.set_title(f"Axis {index} Title")
        ax.set_xlabel("X Label (Units)")
        ax.set_ylabel("Y Label (Units)")
    
    return axes


def get_color_for_value(cbar, value):
    """
    Get the color for a specific value from a colorbar.
    """
    cmap = cbar.mappable.get_cmap()
    norm = cbar.mappable.norm
    return cmap(norm(value))


def get_metric_axis_label(metric_name):
    if metric_name == "HWF":
        return "Days"
    elif metric_name == "HWD":
        return "Days"
    elif metric_name == "HWN":
        return "Heatwaves"
    elif metric_name == "HWA":
        return "Days"
    else:
        return None


def add_percentile_colorbar(ax, percentiles):
    cmap = plt.get_cmap('Reds', len(percentiles))
    norm = Normalize(vmin=-1, vmax=len(percentiles) - 1)
    sm = ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    
    cbar = plt.colorbar(sm, ax=ax, ticks=np.arange(len(percentiles)), pad=0.02)
    cbar.set_label('Percentile', rotation=270, labelpad=8)
    cbar.set_ticklabels(np.round(percentiles, 3))
    return cbar


def add_definitions_colorbar(ax, definitions):
    cmap = plt.get_cmap('tab20', len(definitions))
    norm = Normalize(vmin=0, vmax=len(definitions) - 1)
    sm = ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    
    cbar = plt.colorbar(sm, ax=ax, ticks=np.arange(len(definitions)), pad=0.02)
    cbar.set_label('Definition', rotation=270, labelpad=8)
    cbar.set_ticklabels(definitions)
    return cbar


def add_percentile_colorbar(ax, percentiles):
    cmap = plt.get_cmap('autumn', len(percentiles))
    norm = Normalize(vmin=0, vmax=len(percentiles))
    sm = ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    
    cbar = plt.colorbar(sm, ax=ax, ticks=np.arange(len(percentiles)) + 0.5, pad=0.02)
    cbar.set_label('Percentile', rotation=270, labelpad=8)
    cbar.set_ticklabels(np.round(percentiles, 3))
    return cbar


def get_metric_name(metric_da):
    return metric_da.name.split(".")[2]


def get_unique_metric_names(hw_ds):
    unique_metrics = []

    for name in list(hw_ds.data_vars):
        parsed = name.split(".")
        if len(parsed) >= 3:
            if parsed[2] not in unique_metrics:
                unique_metrics.append(parsed[2])
    return unique_metrics


def plot_map(metric_da, ax, cmap="hot"):
    cmap_obj = plt.get_cmap(cmap)
    metric_da = metric_da.transpose(..., "lon")

    cyclic_values, cyclic_lons = add_cyclic_point(metric_da.values, metric_da.lon.values, axis=-1)
    ax_contour = ax.contourf(cyclic_lons, metric_da.lat.values, cyclic_values, transform=ccrs.PlateCarree(), cmap=cmap_obj)
    ax.coastlines()
    return ax_contour


def plot_metric_decadal_maps(metric_da):
    """
    Multi-figure (4 panels per figure) plotting of decadal means for a particular metric.
    If an ensemble, set of percentiles, or set of definitions is supplied, the average will be used.
    Abbreviations include ensemble (ens), percentile (perc), and definition (def).

    :param metric_da: DataArray of the metric (with the metric name) to plot.
    :type metric_da: xarray.DataArray
    :return: Matplotlib Figure of the plot with the figure title containing a list of metrics displayed.
    :rtype: np.ndarray
    """
    metric_name = get_metric_name(metric_da)

    prefix = ""
    if "member" in metric_da.dims:
        metric_da = metric_da.mean(dim="member")
        prefix += " Ens."
    if "percentile" in metric_da.dims:
        metric_da = metric_da.mean(dim="percentile")
        prefix += " Perc."
    if "definition" in metric_da.dims:
        metric_da = metric_da.mean(dim="definition")
        prefix += " Def."
    
    decadal_ranges = get_decadal_ranges(metric_da.time.values)
    calendar = metric_da.time.values[0].calendar
    num_figs = int(np.ceil(len(decadal_ranges) / 4))
    
    figs = []

    for f_index in range(num_figs):
        fig = generate_base_figure()
        fig.suptitle(metric_name)
        axes = add_four_panel(fig)
    
        for ax_index, ax in enumerate(axes):
            ax.set_title(None)
            slice_index = ax_index + f_index * len(axes)
            if slice_index < len(decadal_ranges):
                start_year, end_year = decadal_ranges[slice_index]
                start_ts = cftime.datetime(start_year, 1, 1, calendar=calendar)
                end_ts = cftime.datetime(end_year, 1, 1, calendar=calendar)
                slice_da = metric_da.sel(time=slice(start_ts, end_ts))

                if start_year != slice_da.time.values[0].year:
                    start_year = slice_da.time.values[0].year
                if end_year != slice_da.time.values[-1].year:
                    end_year = slice_da.time.values[-1].year
                slice_da = slice_da.mean(dim=["time"])
                
                ax = convert_axis_to_map(ax)
                ax_contour = plot_map(slice_da, ax)
                ax_cbar = fig.colorbar(ax_contour, location="bottom", anchor=(0.0, 0.0), pad=0)
                ax_cbar.set_label(f"{metric_name} ({get_metric_axis_label(metric_name)})")
                ax.set_title(f"{start_year} to {end_year}{prefix} Mean")
            else:
                ax.set_axis_off()
        plt.close(fig)
        figs.append(fig)
    return figs


def plot_metric_timeseries(metric_da):
    """
    4-panel plot of time-evolution of metric, spatial-ensemble mean. 
    The top two panels are means taken across thresholds and definitions while the bottom two panels show standard deviations. 
    Abbreviations include ensemble (ens), percentile (perc), definition (def), and standard deviation (std).

    :param metric_da: DataArray of the metric (with the metric name) to plot.
    :type metric_da: xarray.DataArray
    :return: Matplotlib Figure of the plot with the figure title containing a list of metrics displayed.
    :rtype: np.ndarray
    """
    fig = generate_base_figure()
    axes = add_four_panel(fig)

    spatial_mean = compute_weighted_spatial_mean(metric_da).compute()
    metric_name = spatial_mean.name.split(".")[2]
    years = [ts.year for ts in spatial_mean.time.values]

    if "member" in spatial_mean.dims:
        spatial_mean = spatial_mean.mean(dim="member")
        axes[0].set_title(f"{metric_name}, Spatial-Ens-Perc. Mean")
        axes[1].set_title(f"{metric_name}, Spatial-Ens-Def. Mean")
        axes[2].set_title(f"{metric_name}, Spatial-Ens-Perc. Std.")
        axes[3].set_title(f"{metric_name}, Spatial-Ens-Def. Std.")
    else:
        axes[0].set_title(f"{metric_name}, Spatial-Perc. Mean")
        axes[1].set_title(f"{metric_name}, Spatial-Def. Mean")
        axes[2].set_title(f"{metric_name}, Spatial-Perc. Std.")
        axes[3].set_title(f"{metric_name}, Spatial-Def. Std.")

    def_cbar = add_definitions_colorbar(axes[0], spatial_mean.definition.values)
    def_cbar = add_definitions_colorbar(axes[2], spatial_mean.definition.values)

    perc_cbar = add_percentile_colorbar(axes[1], spatial_mean.percentile.values)
    perc_cbar = add_percentile_colorbar(axes[3], spatial_mean.percentile.values)

    for index, definition in enumerate(spatial_mean.definition.values):
        color = get_color_for_value(def_cbar, index)
        axes[0].plot(years, spatial_mean.sel(definition=definition).mean(dim="percentile").values, color=color)
        axes[2].plot(years, spatial_mean.sel(definition=definition).std(dim="percentile").values, color=color)

    for index, percentile in enumerate(spatial_mean.percentile.values):
        color = get_color_for_value(perc_cbar, index)
        axes[1].plot(years, spatial_mean.sel(percentile=percentile).mean(dim="definition").values, color=color)
        axes[3].plot(years, spatial_mean.sel(percentile=percentile).std(dim="definition").values, color=color)
    
    for ax in axes:
        ax.set_xlabel("Time (Year)")
        ax.set_ylabel(f"{metric_name} ({get_metric_axis_label(metric_name)})", labelpad=0.005)
        ax.set_xlim(years[0], years[-1])
    
    fig.suptitle(spatial_mean.name)
    plt.close(fig)
    return fig


def plot_metric_parameter_comparison(metric_da):
    """
    4-panel plot of metric visualizing various comparisons of percentiles against the definitions.
    If ensemble is supplied, the average is taken.

    :param metric_da: DataArray of the metric (with the metric name) to plot.
    :type metric_da: xarray.DataArray
    :return: Matplotlib Figure of the plot with the figure title containing a list of metrics displayed.
    :rtype: np.ndarray
    """
    fig = generate_base_figure()
    axes = add_four_panel(fig)
    
    metric_name = get_metric_name(metric_da)
    fig.suptitle(metric_name)

    if "member" in metric_da.dims:
        metric_da = metric_da.mean(dim="member")
    
    spatio_temporal_mean = compute_weighted_spatial_mean(metric_da).mean(dim="time")

    colormesh = axes[0].pcolormesh(spatio_temporal_mean.values, cmap="hot")
    axes[0].set_xticks(np.arange(spatio_temporal_mean.definition.size) + 0.5)
    axes[0].set_yticks(np.arange(spatio_temporal_mean.percentile.size) + 0.5)
    axes[0].set_xticklabels(spatio_temporal_mean.definition.values)
    axes[0].set_yticklabels(np.round(spatio_temporal_mean.percentile.values, 3))
    axes[0].set_xlabel("Definition")
    axes[0].set_ylabel("Percentile")
    cbar = fig.colorbar(colormesh, ax=axes[0], pad=0.02)
    cbar.set_label(get_metric_axis_label(metric_name), rotation=270, labelpad=12)
    axes[0].set_title(f"{metric_name} Spatial-Temporal Mean")

    lon_def_temporal_mean = metric_da.mean(dim=["time", "lon", "definition"])
    perc_cbar = add_percentile_colorbar(axes[1], lon_def_temporal_mean.percentile.values)

    for index, percentile in enumerate(lon_def_temporal_mean.percentile.values):
        color = get_color_for_value(perc_cbar, index)
        axes[1].plot(lon_def_temporal_mean.lat.values, lon_def_temporal_mean.sel(percentile=percentile).values, color=color)
    axes[1].set_xticks(np.arange(-90, 100, 30))
    axes[1].set_xlim(-90, 90)
    axes[1].set_ylabel(get_metric_axis_label(metric_name))
    axes[1].set_xlabel("Latitude")
    axes[1].set_title(f"{metric_name} Zonal-Temporal-Def. Means")

    param_mean = metric_da.mean(dim=["time", "percentile", "definition"])
    axes[2].set_title(None)
    axes[2] = convert_axis_to_map(axes[2])
    ax_contour = plot_map(param_mean, axes[2])
    ax_cbar = fig.colorbar(ax_contour, location="bottom", anchor=(0.0, 0.0), pad=0)
    ax_cbar.set_label(f"{metric_name} ({get_metric_axis_label(metric_name)})")
    axes[2].set_title(f"{metric_name} Temporal-Def.-Perc. Mean")
    
    param_std = metric_da.mean(dim=["time"]).std(dim=["percentile", "definition"])
    axes[3].set_title(None)
    axes[3] = convert_axis_to_map(axes[3])
    ax_contour = plot_map(param_std, axes[3])
    ax_cbar = fig.colorbar(ax_contour, location="bottom", anchor=(0.0, 0.0), pad=0)
    ax_cbar.set_label(f"{metric_name} ({get_metric_axis_label(metric_name)})")
    axes[3].set_title(f"{metric_name} Def.-Perc. Std. of Temporal Mean")
    
    plt.close(fig)
    return fig


def plot_multi_measure_metric_comparisons(hw_ds):
    """
    4-panel plot of time-evolution of heatwave metrics accross multiple measures.
    Spatially-weighted by latitide, mean taken spatially and over both percentiles and all definitions.
    If ensemble is used, the range of member values is shaded behind the mean line.
    Abbreviations include percentile (perc) and definition (def).

    :param metric_da: Dataset of the metrics (with the metric names) to plot.
    :type metric_da: xarray.Dataset
    :return: Matplotlib Figure of the plot with the figure title containing a list of metrics displayed.
    :rtype: np.ndarray
    """
    fig = generate_base_figure()
    axes = add_four_panel(fig)
    years = [ts.year for ts in hw_ds.time.values]
    metric_names = get_unique_metric_names(hw_ds)[:4]
    fig.suptitle(metric_names)
    
    for index, metric_name in enumerate(metric_names):
        metrics = [name for name in hw_ds.data_vars if metric_name in name]
        metrics_ds = compute_weighted_spatial_mean(hw_ds[metrics]).mean(dim=["percentile", "definition"]).compute()
        
        axes[index].spines['top'].set_visible(False)
        axes[index].spines['right'].set_visible(False)
        axes[index].grid(True)
    
        axes[index].set_xlabel("Time (Year)")
        axes[index].set_ylabel(f"{metric_name} ({get_metric_axis_label(metric_name)})", labelpad=2)
        axes[index].set_xlim(years[0], years[-1])
        axes[index].set_title(f"{metric_name} Spatial-Perc-Def Mean")
        
        for metric in metrics_ds:
            axes[index].plot(years, metrics_ds[metric].values, label=metric.split(".")[0])
    
        axes[index].legend()

    plt.close(fig)
    return fig

