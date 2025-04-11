# %%
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.collections import LineCollection
import matplotlib.cm as cm
from moist_thermodynamics import constants
from moist_thermodynamics import functions as mtfunc

# %%
# Global plot styling
SIZE = 15
plt.rcParams.update({
    "axes.labelsize": SIZE,
    "legend.fontsize": SIZE,
    "xtick.labelsize": SIZE,
    "ytick.labelsize": SIZE,
    "font.size": SIZE,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.major.size": 6,
    "ytick.major.size": 6
})

# %%
# Load dataset
ds = xr.open_dataset("ipns://latest.orcestra-campaign.org/products/Radiosondes/RAPSODI_RS_ORCESTRA_level2.zarr", engine="zarr")

# %%
# Subsets by platform
ds_BCO = ds.where(ds.platform == 'BCO', drop=True)
ds_INMG = ds.where(ds.platform == 'INMG', drop=True)
ds_MET = ds.where(ds.platform == 'RV_Meteor', drop=True)

# %%
# Max altitude calculation
valid_mask = ds["flight_lat"].notnull()
valid_mask_reversed = valid_mask.isel(alt=slice(None, None, -1))
max_alt_idx = valid_mask_reversed.argmax(dim="alt")
max_alt_idx = ds.sizes["alt"] - 1 - max_alt_idx
max_altitudes = ds["alt"].isel(alt=max_alt_idx)
median_max_altitude = np.nanmedian(max_altitudes)

print(f"Median of Maximum Altitude: {median_max_altitude:.2f} meters")

# %%
# Color sets
color_sets = [
    ["red", "blue", "green"], ["darkorange", "darkblue", "lime"],
    ["crimson", "royalblue", "darkgreen"], ["tomato", "mediumblue", "forestgreen"],
    ["firebrick", "cornflowerblue", "mediumseagreen"], ["darkred", "dodgerblue", "springgreen"],
    ["orangered", "blueviolet", "teal"], ["indianred", "deepskyblue", "mediumspringgreen"],
    ["chocolate", "navy", "chartreuse"], ["gold", "purple", "darkcyan"],
    ["gold", "darkblue", "crimson"], ["#BF312D", "darkblue", "#F6CA4C"]
]

active_colors = color_sets[11]
color_INMG, color_Meteor, color_BCO = active_colors

# %%
# Plot ascent-descent mean difference (all platforms)
bottom_cutoff = 200
variables_to_plot = ['p', 'ta', 'rh', 'wspd']
subplot_labels = ["(a)", "(b)", "(c)", "(d)"]

fig, axs = plt.subplots(1, len(variables_to_plot), figsize=(15, 6),
                        gridspec_kw={"wspace": 0.4, "hspace": 0.6}, sharey=True)
axs = axs.flatten()

for i, var in enumerate(variables_to_plot):
    ax = axs[i]
    ascent = ds[var].where(ds['ascent_flag'] == 0, drop=True).mean(dim='launch_time').sel(alt=slice(bottom_cutoff, 25000))
    descent = ds[var].where(ds['ascent_flag'] == 1, drop=True).mean(dim='launch_time').sel(alt=slice(bottom_cutoff, 25000))
    
    ax.plot(ascent - descent, ds.alt.sel(alt=slice(bottom_cutoff, 25000)) / 1000,
            color='black', label='Ascents$-$Descents')
    
    ax.axvline(0, color='black', linewidth=1, ls='dotted')
    ax.set_ylim(0, 25)
    ax.set_title(ds[var].attrs.get('standard_name', var), pad=20)
    ax.spines[["left", "bottom"]].set_position(("outward", 20))
    ax.spines[["right", "top"]].set_visible(False)
    ax.text(-0.2, 1.12, subplot_labels[i], transform=ax.transAxes,
            fontsize=SIZE + 4, verticalalignment="top")

    if i == 0:
        ax.set_ylabel("Altitude / km")
        ax.set_title('')
        ax.set_xlabel('Pressure / Pa')
    if ds[var].attrs.get('standard_name', var) == 'air_temperature':
        ax.set_title('')
        ax.set_xlabel('Air Temperature / K')
    if ds[var].attrs.get('standard_name', var) == 'relative_humidity':
        ax.set_title('')
        ax.set_xlabel('Relative Humidity / 1')
    if ds[var].attrs.get('standard_name', var) == 'wind_speed':
        ax.set_title('')
        ax.set_xlabel('Wind Speed / ms$^{-1}$')

handles, labels = ax.get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, -0.15), ncol=4, frameon=True, fontsize=SIZE)

plt.tight_layout()
plt.savefig('./Figures/Fig13_Appendix_Histogram_Difference_Ascent_Descent_Soundings.svg',
            format='svg', facecolor='white', bbox_inches="tight", dpi=300)
plt.show()

# %%
# Plot ascent-descent difference by platform group
vaisala_platforms = ['BCO', 'RV_Meteor']
meteomodem_platforms = ['INMG']

fig, axs = plt.subplots(1, len(variables_to_plot), figsize=(15, 6),
                        gridspec_kw={"wspace": 0.4, "hspace": 0.6}, sharey=True)
axs = axs.flatten()

for i, var in enumerate(variables_to_plot):
    ax = axs[i]

    # Vaisala
    ds_vaisala = ds.where(ds.platform.isin(vaisala_platforms), drop=True)
    ascent_v = ds_vaisala[var].where(ds_vaisala['ascent_flag'] == 0, drop=True).mean(dim='launch_time').sel(alt=slice(bottom_cutoff, 25000))
    descent_v = ds_vaisala[var].where(ds_vaisala['ascent_flag'] == 1, drop=True).mean(dim='launch_time').sel(alt=slice(bottom_cutoff, 25000))
    diff_v = ascent_v - descent_v
    ax.plot(diff_v, ds_vaisala.alt.sel(alt=slice(bottom_cutoff, 25000)) / 1000,
            label="Vaisala", color='black')

    mean_diff_v = diff_v.sel(alt=slice(bottom_cutoff, 15000)).mean()
    ax.vlines(mean_diff_v, ymin=-1.5, ymax=-0.8, color='black', linewidth=3, zorder=10, clip_on=False)

    # Meteomodem
    ds_meteomodem = ds.where(ds.platform.isin(meteomodem_platforms), drop=True)
    ascent_m = ds_meteomodem[var].where(ds_meteomodem['ascent_flag'] == 0, drop=True).mean(dim='launch_time').sel(alt=slice(bottom_cutoff, 25000))
    descent_m = ds_meteomodem[var].where(ds_meteomodem['ascent_flag'] == 1, drop=True).mean(dim='launch_time').sel(alt=slice(bottom_cutoff, 25000))
    diff_m = ascent_m - descent_m
    ax.plot(diff_m, ds_meteomodem.alt.sel(alt=slice(bottom_cutoff, 25000)) / 1000,
            label="Meteomodem", color='royalblue')

    mean_diff_m = diff_m.sel(alt=slice(bottom_cutoff, 15000)).mean()
    ax.vlines(mean_diff_m, ymin=-1.5, ymax=-0.8, color='royalblue', linewidth=3, zorder=10, clip_on=False)

    ax.axvline(0, color='black', linewidth=1, ls='dotted')
    ax.set_ylim(0, 25)
    ax.set_title(ds[var].attrs.get('standard_name', var), pad=20)
    ax.spines[["left", "bottom"]].set_position(("outward", 20))
    ax.spines[["right", "top"]].set_visible(False)
    ax.text(-0.2, 1.12, subplot_labels[i], transform=ax.transAxes,
            fontsize=SIZE + 4, verticalalignment="top")

    if i == 0:
        ax.set_ylabel("Altitude / km")
        ax.set_title('')
        ax.set_xlabel('Pressure / Pa')
    if ds[var].attrs.get('standard_name', var) == 'air_temperature':
        ax.set_title('')
        ax.set_xlabel('Air Temperature / K')
    if ds[var].attrs.get('standard_name', var) == 'relative_humidity':
        ax.set_title('')
        ax.set_xlabel('Relative Humidity / 1')
    if ds[var].attrs.get('standard_name', var) == 'wind_speed':
        ax.set_title('')
        ax.set_xlabel('Wind Speed / ms$^{-1}$')

handles, labels = ax.get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, -0.15), ncol=2, frameon=True, fontsize=SIZE)

plt.tight_layout()
plt.savefig('./Figures/Fig13_Appendix_Histogram_Diff_Ascent_Descent_ByPlatform.svg',
            format='svg', facecolor='white', bbox_inches="tight", dpi=300)
plt.savefig('./Figures/Fig13_Appendix_Histogram_Diff_Ascent_Descent_ByPlatform.png',
            format='png', facecolor='white', bbox_inches="tight", dpi=200)
plt.show()

# %%
