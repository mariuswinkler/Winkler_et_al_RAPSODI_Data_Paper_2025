# %%
import xarray as xr
import numpy as np
import pandas as pd
import matplotlib.pylab as plt
import matplotlib.colors as mcolors
from moist_thermodynamics.saturation_vapor_pressures import liq_wagner_pruss, ice_wagner_etal

# %%
def rh_mixed_phase(T, p, q):
    Rd = 287.05
    Rv = 461.5
    eps = Rd / Rv
    e = (q * p) / (eps + (1 - eps) * q)

    rh = np.full_like(T, np.nan)

    mask_liq = T > 273.15
    mask_ice = ~mask_liq

    rh[mask_liq] = e[mask_liq] / liq_wagner_pruss(T[mask_liq])
    rh[mask_ice] = e[mask_ice] / ice_wagner_etal(T[mask_ice])

    return rh

# %%
SIZE = 15
plt.rcParams["axes.labelsize"] = SIZE
plt.rcParams["legend.fontsize"] = SIZE
plt.rcParams["xtick.labelsize"] = SIZE
plt.rcParams["ytick.labelsize"] = SIZE
plt.rcParams["font.size"] = SIZE
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'
plt.rcParams['xtick.major.size'] = 6
plt.rcParams['ytick.major.size'] = 6
# %%
ds = xr.open_dataset("ipfs://bafybeid7cnw62zmzfgxcvc6q6fa267a7ivk2wcchbmkoyk4kdi5z2yj2w4", engine="zarr")

# %%
PLATFORM = 'BCO'
ds_BCO = ds.where(ds.platform == PLATFORM, drop=True)
PLATFORM = 'RV_Meteor'
ds_MET = ds.where(ds.platform == PLATFORM, drop=True)
PLATFORM = 'INMG'
ds_INMG = ds.where(ds.platform == PLATFORM, drop=True)
#ds_MET.sel(launch_time=(ds_MET['ascent_flag'] == 0)).sel(launch_time="2024-09-21").launch_time

# %%
# Compute relative humidity for each dataset
# Using the mixed-phase RH computation
for dset in [ds_BCO, ds_MET, ds_INMG]:
    dset["rh"] = xr.apply_ufunc(
        rh_mixed_phase,
        dset["ta"],
        dset["p"],
        dset["q"],
        dask="parallelized",
        output_dtypes=[float],
    )

# %%
valid_mask = ds["lat"].notnull()
valid_mask_reversed = valid_mask.isel(height=slice(None, None, -1))
max_height_idx = valid_mask_reversed.argmax(dim="height")
max_height_idx = ds.sizes["height"] - 1 - max_height_idx
max_heights = ds["height"].isel(height=max_height_idx)
median_max_height = np.nanmedian(max_heights)

print(f"Median of Maximum height: {median_max_height:.2f} meters")
# %%
## Color Sets
color_sets = [
    ["red", "blue", "green"], #0
    ["darkorange", "darkblue", "lime"], #1
    ["crimson", "royalblue", "darkgreen"], #2
    ["tomato", "mediumblue", "forestgreen"], #3
    ["firebrick", "cornflowerblue", "mediumseagreen"], #4
    ["darkred", "dodgerblue", "springgreen"], #5
    ["orangered", "blueviolet", "teal"], #6
    ["indianred", "deepskyblue", "mediumspringgreen"], #7
    ["chocolate", "navy", "chartreuse"], #8
    ["gold", "purple", "darkcyan"], #9
    ["gold", "darkblue", "crimson"], #10
    ["#BF312D", "darkblue", "#F6CA4C"] #11
]

active_color_set_index = 11

active_colors = color_sets[active_color_set_index]
color_INMG = active_colors[0]
color_Meteor = active_colors[1]
color_BCO = active_colors[2]

# %%
import matplotlib.colors as mcolors
import pandas as pd
from pandas import to_datetime

# Assuming you already have ds_INMG, ds_MET, ds_BCO loaded
# Make copies and adjust height to km
ds_INMG_plot = ds_INMG.copy()
ds_MET_plot  = ds_MET.copy()
ds_BCO_plot  = ds_BCO.copy()

ds_INMG_plot = ds_INMG_plot.assign_coords(height=ds_INMG_plot.height / 1000)
ds_MET_plot  = ds_MET_plot.assign_coords(height=ds_MET_plot.height / 1000)
ds_BCO_plot  = ds_BCO_plot.assign_coords(height=ds_BCO_plot.height / 1000)

SIZE = 12
VARIABLE = 'rh'
levels = np.arange(0, 1.1, 0.1)
norm = mcolors.BoundaryNorm(boundaries=levels, ncolors=256)
cmap = "YlGnBu"

fig, axs = plt.subplots(3, 1, figsize=(16, 15), gridspec_kw={'hspace': 0.4}, sharex=True)

mesh0 = axs[0].pcolormesh(ds_INMG_plot.launch_time, ds_INMG_plot.height, ds_INMG_plot[VARIABLE].T,
                          shading="auto", cmap=cmap, norm=norm, rasterized=True)
mesh1 = axs[1].pcolormesh(ds_MET_plot.launch_time, ds_MET_plot.height, ds_MET_plot[VARIABLE].T,
                          shading="auto", cmap=cmap, norm=norm, rasterized=True)
mesh2 = axs[2].pcolormesh(ds_BCO_plot.launch_time, ds_BCO_plot.height, ds_BCO_plot[VARIABLE].T,
                          shading="auto", cmap=cmap, norm=norm, rasterized=True)

for ax in axs:
    ax.set_ylabel("Height / km", fontsize=SIZE+5)
    ax.set_ylim(0, 25)
    ax.spines[["left", "bottom"]].set_position(("outward", 20))
    ax.spines[["right", "top"]].set_visible(False)

# Common x limits
axs[0].set_xlim(ds.launch_time[0].values, ds.launch_time[-1].values)

# Create ticks every 3 days at midday
times = ds.launch_time.to_pandas()
ticks = (
    times.groupby(times.dt.floor('D'))
    .apply(lambda group: group.iloc[(group - pd.Timestamp(group.dt.date.iloc[0]) - pd.Timedelta(hours=12)).abs().argmin()])
)

ticks = ticks[::3]
axs[2].set_xticks(ticks.values)
axs[2].set_xticklabels(ticks.dt.strftime('%Y-%m-%d'), rotation=45)

# Add twin x-axis only to axs[1] with longitude from ds_MET
twin = axs[1].twiny()

# Ensure the twin axis uses the same ticks as the bottom
twin.set_xticks(ticks.values)
twin.spines[["top"]].set_position(("outward", 0))

# Build longitude labels, but keep first two and last two empty
lon_labels = []
for i, t in enumerate(ticks):
    if i < 2 or i >= len(ticks) - 2:
        lon_labels.append(" ")
    else:
        idx = np.argmin(np.abs(ds_MET.launch_time.values - np.datetime64(t)))
        lon_val = ds_MET.launch_lon.isel(launch_time=idx).values
        lon_labels.append(f"{lon_val:.1f}")

# Apply labels and formatting
twin.set_xticklabels(lon_labels)
twin.set_xlim(ds.launch_time[0].values, ds.launch_time[-1].values)
twin.set_xlabel("Longitude / °W", fontsize=SIZE+5, labelpad=10)

# Hide unneeded spines
for spine in ["left", "bottom", "right"]:
    twin.spines[spine].set_visible(False)

# Add colorbar
cbar_ax = fig.add_axes([0.15, -0.02, 0.7, 0.02])
cbar = fig.colorbar(mesh2, cax=cbar_ax, orientation="horizontal", boundaries=levels, ticks=levels)
cbar.set_label("Relative Humidity / 1", fontsize=SIZE+5)

# Add subplot labels
axs[0].text(-0.1, 1.15, "(a)", transform=axs[0].transAxes, fontsize=SIZE+10, va='top', ha='left')
axs[1].text(-0.1, 1.15, "(b)", transform=axs[1].transAxes, fontsize=SIZE+10, va='top', ha='left')
axs[2].text(-0.1, 1.15, "(c)", transform=axs[2].transAxes, fontsize=SIZE+10, va='top', ha='left')

# Save and show
filepath = './Figures/'
filename = 'Fig11_Appendix1_rh_mixed_phase.svg'
plt.savefig(filepath + filename, format='svg', facecolor='white', bbox_inches="tight", dpi=300)
filename = 'Fig11_Appendix1_rh_mixed_phase.png'
plt.savefig(filepath + filename, format='png', facecolor='white', bbox_inches="tight", dpi=150)
plt.show()

# %%

# Assuming ds_INMG, ds_MET, ds_BCO are already loaded
# Prepare datasets
ds_INMG_plot = ds_INMG.copy().assign_coords(height=ds_INMG.height / 1000)
ds_MET_plot  = ds_MET.copy().assign_coords(height=ds_MET.height / 1000)
ds_BCO_plot  = ds_BCO.copy().assign_coords(height=ds_BCO.height / 1000)

# Plot settings
SIZE = 12
VARIABLE = 'ta'
levels = np.arange(-5, 6, 1)
norm = mcolors.BoundaryNorm(boundaries=levels, ncolors=256)
cmap = "coolwarm"

# Compute anomalies
ds_INMG_anom = ds_INMG_plot[VARIABLE] - ds_INMG_plot[VARIABLE].mean(dim='launch_time')
ds_MET_anom  = ds_MET_plot[VARIABLE]  - ds_MET_plot[VARIABLE].mean(dim='launch_time')
ds_BCO_anom  = ds_BCO_plot[VARIABLE]  - ds_BCO_plot[VARIABLE].mean(dim='launch_time')

# Plot
fig, axs = plt.subplots(3, 1, figsize=(16, 15), gridspec_kw={'hspace': 0.4}, sharex=True)

mesh0 = axs[0].pcolormesh(ds_INMG_plot.launch_time, ds_INMG_plot.height, ds_INMG_anom.T,
                          shading="auto", cmap=cmap, norm=norm, rasterized=True)
mesh1 = axs[1].pcolormesh(ds_MET_plot.launch_time, ds_MET_plot.height, ds_MET_anom.T,
                          shading="auto", cmap=cmap, norm=norm, rasterized=True)
mesh2 = axs[2].pcolormesh(ds_BCO_plot.launch_time, ds_BCO_plot.height, ds_BCO_anom.T,
                          shading="auto", cmap=cmap, norm=norm, rasterized=True)

# Axis formatting
for ax in axs:
    ax.set_ylabel("Height / km", fontsize=SIZE+5)
    ax.set_ylim(0, 25)
    ax.spines[["left", "bottom"]].set_position(("outward", 20))
    ax.spines[["right", "top"]].set_visible(False)

# Common x limits
axs[0].set_xlim(ds.launch_time[0].values, ds.launch_time[-1].values)

# Create ticks every 3 days at midday
times = ds.launch_time.to_pandas()
ticks = (
    times.groupby(times.dt.floor('D'))
    .apply(lambda group: group.iloc[(group - pd.Timestamp(group.dt.date.iloc[0]) - pd.Timedelta(hours=12)).abs().argmin()])
)
ticks = ticks[::3]

# Bottom ticks
axs[2].set_xticks(ticks.values)
axs[2].set_xticklabels(ticks.dt.strftime('%Y-%m-%d'), rotation=45)

# Add twin x-axis only to axs[1] with longitude from ds_MET
twin = axs[1].twiny()

# Ensure the twin axis uses the same ticks as the bottom
twin.set_xticks(ticks.values)
twin.spines[["top"]].set_position(("outward", 0))

# Build longitude labels, but keep first two and last two empty
lon_labels = []
for i, t in enumerate(ticks):
    if i < 2 or i >= len(ticks) - 2:
        lon_labels.append(" ")
    else:
        idx = np.argmin(np.abs(ds_MET.launch_time.values - np.datetime64(t)))
        lon_val = ds_MET.launch_lon.isel(launch_time=idx).values
        lon_labels.append(f"{lon_val:.1f}")

# Apply labels and formatting
twin.set_xticklabels(lon_labels)
twin.set_xlim(ds.launch_time[0].values, ds.launch_time[-1].values)
twin.set_xlabel("Longitude / °W", fontsize=SIZE+5, labelpad=10)

# Hide unneeded spines
for spine in ["left", "bottom", "right"]:
    twin.spines[spine].set_visible(False)

# Add colorbar
cbar_ax = fig.add_axes([0.15, -0.02, 0.7, 0.02])
cbar = fig.colorbar(mesh2, cax=cbar_ax, orientation="horizontal", boundaries=levels, ticks=levels)
cbar.set_label("Air Temperature Anomaly / K", fontsize=SIZE+5)

# Add subplot labels
axs[0].text(-0.1, 1.15, "(a)", transform=axs[0].transAxes, fontsize=SIZE+10, va='top', ha='left')
axs[1].text(-0.1, 1.15, "(b)", transform=axs[1].transAxes, fontsize=SIZE+10, va='top', ha='left')
axs[2].text(-0.1, 1.15, "(c)", transform=axs[2].transAxes, fontsize=SIZE+10, va='top', ha='left')

# Save and show
filepath = './Figures/'
filename = 'Fig12_Appendix2_ta.svg'
plt.savefig(filepath + filename, format='svg', facecolor='white', bbox_inches="tight", dpi=300)
filename = 'Fig12_Appendix2_ta.png'
plt.savefig(filepath + filename, format='png', facecolor='white', bbox_inches="tight", dpi=150)

plt.show()


# %%
