# %%
import xarray as xr
import numpy as np

import matplotlib.pylab as plt
from moist_thermodynamics import functions as mtfunc
from moist_thermodynamics.saturation_vapor_pressures import (
    liq_wagner_pruss,
    ice_wagner_etal,
)

# %%
SIZE = 22
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
def rh_mixed_phase(T, p, q):
    """
    Compute RH over liquid (T > 273.15K) and over ice (T <= 273.15K).
    T: temperature (K)
    p: pressure (Pa)
    q: specific humidity (kg/kg)
    Returns: RH (0 to 1)
    """
    Rd = 287.05
    Rv = 461.5
    eps = Rd / Rv
    e = (q * p) / (eps + (1 - eps) * q)

    rh = np.full_like(T, np.nan)

    mask_liq = T > 273.15
    mask_ice = ~mask_liq  # includes NaNs and T <= 273.15

    rh[mask_liq] = e[mask_liq] / liq_wagner_pruss(T[mask_liq])
    rh[mask_ice] = e[mask_ice] / ice_wagner_etal(T[mask_ice])

    return rh

# %%
PLATFORM = 'BCO'
ds_BCO = ds.where(ds.platform == PLATFORM, drop=True)
PLATFORM = 'INMG'
ds_INMG = ds.where(ds.platform == PLATFORM, drop=True)
PLATFORM = 'RV_Meteor'
ds_MET = ds.where(ds.platform == PLATFORM, drop=True)
#ds_MET.sel(launch_time=(ds_MET['ascent_flag'] == 0)).sel(launch_time="2024-09-21").launch_time
valid_mask = ds["lat"].notnull()
valid_mask_reversed = valid_mask.isel(height=slice(None, None, -1))
max_height_idx = valid_mask_reversed.argmax(dim="height")
max_height_idx = ds.sizes["height"] - 1 - max_height_idx
max_heights = ds["height"].isel(height=max_height_idx)
median_max_height = np.nanmedian(max_heights)

print(f"Median of Maximum height: {median_max_height:.2f} meters")
# %%
# Compute relative humidity over liquid and ice
for dset in [ds_BCO, ds_INMG, ds_MET]:
    dset["rh"] = xr.apply_ufunc(
        rh_mixed_phase,
        dset["ta"],
        dset["p"],
        dset["q"],
        dask="parallelized",
        output_dtypes=[float],
    )

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

# Split ds_MET into East and West Atlantic subsets
ds_MET_east = ds_MET.where(ds_MET.launch_lon > -40, drop=True) 
ds_MET_west = ds_MET.where(ds_MET.launch_lon <= -40, drop=True)  

# Add MSE to all datasets
for ds in [ds_BCO, ds_MET_east, ds_MET_west, ds_INMG]:
    ds["mse"] = mtfunc.moist_static_energy(ds.ta, ds.height, ds.q) / 1000  # in kJ/kg

# Define colors (adjust shades of blue as needed)
datasets = {
    "BCO": {"data": ds_BCO, "color": color_BCO},
    " R/V Meteor\n(West of 40°W)": {"data": ds_MET_west, "color": color_Meteor},
    " R/V Meteor\n(East of 40°W)": {"data": ds_MET_east, "color": "cornflowerblue"},
    "INMG": {"data": ds_INMG, "color": color_INMG},
}

subplot_labels = ["(a)", "(b)", "(c)", "(d)"]
variables = ["u", "v", "rh", "mse"]
titles = ["Zonal Wind", "Meridional Wind", "Relative\nHumidity", "Moist\nStatic Energy"]
x_labels = ["u / ms$^{-1}$", "v / ms$^{-1}$", "rh", r"MSE / kJ kg$^{-1}$"]

# %%


### LINEAR PRESSURE AXIS + BOTTOM MARKERS

fig, axes = plt.subplots(1, 4, figsize=(18, 10))

for i, (var, title, xlabel) in enumerate(zip(variables, titles, x_labels)):
    ax = axes[i]

    for dataset_name, dataset_info in datasets.items():
        datas = dataset_info["data"]
        color = dataset_info["color"]

        data = datas[var]
        pressure = datas["p"]
        
        valid_mask = ~np.all(np.isnan(data), axis=0)
        mean_data = np.nanmean(data[:, valid_mask], axis=0)
        mean_pressure = np.nanmean(pressure[:, valid_mask], axis=0) / 100  # convert to hPa
        valid = (mean_pressure > 100) & (mean_pressure < 1000)

        # Plot profile
        ax.plot(mean_data[valid], mean_pressure[valid], color=color, linewidth=4, label=dataset_name, zorder=10)

        # Compute statistics for top marker
        profile_data = mean_data[valid]
        mean_value = np.nanmean(profile_data)
        std_dev = np.nanstd(profile_data)

        # Plot bottom tick (just below x-axis)
        #ax.vlines(mean_value, ymin=1005, ymax=1036, color=color, linewidth=3, zorder=20, clip_on=False)

        # Optional: annotate value below
        ax.annotate(f"{mean_value:.1f}", (mean_value, 1035), color=color, fontsize=SIZE-6,
                    ha='center', va='bottom', rotation=90, clip_on=False)

        # Increase this for more separation between datasets
        offset_base = 80
        offset_step = 25  # ⬅️ increase this to separate further
        offset_index = list(datasets.keys()).index(dataset_name)
        offset = offset_base - offset_index * offset_step
        tick_height = 12  # height of vertical spread bars
        
        # Plot vertical bars for ±std
        for x in [mean_value - std_dev, mean_value + std_dev]:
            ax.vlines(x, ymin=offset - tick_height / 2, ymax=offset + tick_height / 2,
                      color=color, linewidth=3, zorder=20, clip_on=False)
        
        # Draw horizontal connector line between ±std
        ax.hlines(offset, xmin=mean_value - std_dev, xmax=mean_value + std_dev,
                  color=color, linewidth=1.5, zorder=20, clip_on=False)
        
        # Plot mean as a circle marker
        ax.plot(mean_value, offset, marker='o', color=color, markersize=8, zorder=21, clip_on=False)

    # Axis titles and layout
    ax.set_title(title, fontsize=SIZE, pad=80)
    ax.set_xlabel(xlabel)
    ax.invert_yaxis()
    ax.set_ylim(1000, 100)

    if i == 0:
        ax.set_ylabel("Pressure / hPa", fontsize=SIZE)
    else:
        ax.set_yticklabels([])

    if var == "rh":
        ax.set_xlim(0, 1)
    if var == "mse":
        ax.set_xlim(330, 370)
    if var in ["u", "v"]:
        ax.axvline(0, color='grey', ls='dashed')
    if var == "u":
        ax.set_xlim(-17, 17)
    if var == "v":
        ax.set_xlim(-7, 7)

    if i == 0:
        ax.set_yticks([1000, 800, 600, 400, 200])
        ax.set_yticklabels(["1000", "800", "600", "400", "200"])
        ax.set_ylabel("Pressure / hPa", fontsize=SIZE)
        ax.yaxis.set_visible(True)
    else:
        ax.set_yticks([1000, 800, 600, 400, 200])
        ax.set_yticklabels([])

    ax.text(-0.18, 1.25, subplot_labels[i], transform=ax.transAxes,
            fontsize=SIZE + 4, fontweight="normal", verticalalignment="top")

    ax.grid(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines[["left", "bottom"]].set_position(("outward", 20))

# Add legend
handles, labels = ax.get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, -0.04), ncol=4, frameon=True, fontsize=SIZE)

plt.tight_layout(rect=[0, 0.05, 1, 1])
#filepath = './Figures/'
#filename = f'Fig06_Overall_Means_pressure_rh_over_ice.svg'
#plt.savefig(filepath + filename, format='svg', facecolor='white', bbox_inches="tight", dpi=300)
#filename = f'Fig06_Overall_Means_pressure_rh_over_ice.png'
#plt.savefig(filepath + filename, format='png', facecolor='white', bbox_inches="tight", dpi=150)

plt.show()
# %%



### LOGARITHMIC PRESSURE AXIS + TOP MARKERS

from matplotlib import transforms
fig, axes = plt.subplots(1, 4, figsize=(18, 10))

for i, (var, title, xlabel) in enumerate(zip(variables, titles, x_labels)):
    ax = axes[i]

    # blended transform: x in data units, y in axes-fraction (0..1)
    trans = transforms.blended_transform_factory(ax.transData, ax.transAxes)

    for dataset_name, dataset_info in datasets.items():
        datas = dataset_info["data"]
        color = dataset_info["color"]

        data = datas[var]
        pressure = datas["p"]

        valid_mask = ~np.all(np.isnan(data), axis=0)
        mean_data = np.nanmean(data[:, valid_mask], axis=0)
        mean_pressure = np.nanmean(pressure[:, valid_mask], axis=0) / 100  # convert to hPa
        valid = (mean_pressure > 100) & (mean_pressure < 1000)

        # Plot mean vertical profile
        ax.plot(mean_data[valid], mean_pressure[valid],
                color=color, linewidth=4, label=dataset_name, zorder=10)

        # Compute statistics for top marker
        profile_data = mean_data[valid]
        mean_value = np.nanmean(profile_data)
        std_dev = np.nanstd(profile_data)

        # Vertically offset header markers per dataset (axes-fraction coordinates)
        offset_index = list(datasets.keys()).index(dataset_name)
        n_sets = len(datasets)
        ybar_base = 1.05      # baseline height above panel
        y_step = 0.03         # vertical separation between datasets
        ybar = ybar_base + (n_sets - 1 - offset_index) * y_step
        h = 0.02             # height of vertical tick (axes fraction)

        # Plot vertical bars for ±std
        for x in [mean_value - std_dev, mean_value + std_dev]:
            ax.vlines(x, ymin=ybar - h/2, ymax=ybar + h/2,
                      color=color, linewidth=3, transform=trans, clip_on=False)

        # Draw horizontal connector line between ±std
        ax.hlines(ybar, xmin=mean_value - std_dev, xmax=mean_value + std_dev,
                  color=color, linewidth=1.5, transform=trans, clip_on=False)

        # Plot mean as a circle marker
        ax.plot(mean_value, ybar, marker='o', color=color, markersize=8,
                transform=trans, clip_on=False)

    # Axis titles and layout
    ax.set_title(title, fontsize=SIZE, pad=80)
    ax.set_xlabel(xlabel)

    # Logarithmic pressure axis
    ax.set_yscale("log")
    ax.set_ylim(1000, 100)
    ticks = [1000, 800, 700, 600, 500, 400, 300, 200, 100]
    ax.set_yticks(ticks)

    if i == 0:
        ax.set_yticklabels([str(t) for t in ticks])
        ax.set_ylabel("Pressure / hPa", fontsize=SIZE)
    else:
        ax.set_yticklabels([])

    # X limits and guides
    if var == "rh":
        ax.set_xlim(0, 1)
    if var == "mse":
        ax.set_xlim(330, 370)
    if var in ["u", "v"]:
        ax.axvline(0, color='grey', ls='dashed')
    if var == "u":
        ax.set_xlim(-17, 17)
    if var == "v":
        ax.set_xlim(-7, 7)

    # Subplot labels (a), (b), ...
    ax.text(-0.18, 1.25, subplot_labels[i], transform=ax.transAxes,
            fontsize=SIZE + 4, fontweight="normal", verticalalignment="top")

    # Clean plot style
    ax.grid(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines[["left", "bottom"]].set_position(("outward", 20))

# Add legend below panels
handles, labels = axes[-1].get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center",
           bbox_to_anchor=(0.5, -0.04), ncol=4,
           frameon=True, fontsize=SIZE)

plt.tight_layout(rect=[0, 0.05, 1, 1])

# Save figure
filepath = './Figures/'
filename = 'Fig06_Overall_Means_pressure_rh_over_ice.svg'
plt.savefig(filepath + filename, format='svg', facecolor='white',
            bbox_inches="tight", dpi=300)
filename = 'Fig06_Overall_Means_pressure_rh_over_ice.png'
plt.savefig(filepath + filename, format='png', facecolor='white',
            bbox_inches="tight", dpi=150)

plt.show()

# %%
