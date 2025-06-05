# %%
import xarray as xr
import numpy as np
import re
import os
import pandas as pd

import matplotlib.pylab as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.dates as mdates
from matplotlib.patches import Polygon
from matplotlib.collections import LineCollection

import cartopy.crs as ccrs
import cartopy.feature as cfeature
from moist_thermodynamics import constants
from moist_thermodynamics import functions as mtfunc

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
ds = xr.open_dataset("ipns://latest.orcestra-campaign.org/products/Radiosondes/RAPSODI_RS_ORCESTRA_level2.zarr", engine="zarr")
#ds = xr.open_dataset("/Users/marius/ownCloud/PhD/12_Orcestra_Campaign/00_ORCESTRA_Radiosondes_Winkler/00_data_for_IPFS/RS_ORCESTRA_level2.zarr", engine="zarr")
#ds_osc = xr.open_dataset("ipfs://QmVsFoFCSU661EWukv5W2ponDVz46zmxgzXRpPZ1fXsbkp", engine="zarr")
PLATFORM = 'BCO'
ds_BCO = ds.where(ds.platform == PLATFORM, drop=True)
PLATFORM = 'INMG'
ds_INMG = ds.where(ds.platform == PLATFORM, drop=True)
PLATFORM = 'RV_Meteor'
ds_MET = ds.where(ds.platform == PLATFORM, drop=True)
#ds_MET.sel(launch_time=(ds_MET['ascent_flag'] == 0)).sel(launch_time="2024-09-21").launch_time
valid_mask = ds["flight_lat"].notnull()
valid_mask_reversed = valid_mask.isel(alt=slice(None, None, -1))
max_alt_idx = valid_mask_reversed.argmax(dim="alt")
max_alt_idx = ds.sizes["alt"] - 1 - max_alt_idx
max_altitudes = ds["alt"].isel(alt=max_alt_idx)
median_max_altitude = np.nanmedian(max_altitudes)

print(f"Median of Maximum Altitude: {median_max_altitude:.2f} meters")
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


for ds in [ds_BCO, ds_MET, ds_INMG]:
    ds["mse"] = mtfunc.moist_static_energy(ds.ta, ds.alt, ds.q) / 1000

datasets = {
    "BCO": {"data": ds_BCO, "color": color_BCO},
    "R/V Meteor": {"data": ds_MET, "color": color_Meteor},
    "INMG": {"data": ds_INMG, "color": color_INMG},
}

subplot_labels = ["(a)", "(b)", "(c)", "(d)"]

variables = ["mse", "rh", "u", "v"]
titles = [f"Moist\nStatic Energy", f"Relative\nHumidity", "Zonal Wind", "Meridional Wind"]
x_labels = [r"MSE / kJ kg$^{-1}$ / K", "rh", "u / ms$^{-1}$", "v / ms$^{-1}$"]

altitude_limit = 25000

fig, axes = plt.subplots(1, 4, figsize=(18, 10))

for i, (var, title, xlabel) in enumerate(zip(variables, titles, x_labels)):
    ax = axes[i]

    for dataset_name, dataset_info in datasets.items():
        datas = dataset_info["data"]
        color = dataset_info["color"]

        valid_indices = datas.alt <= altitude_limit
        data = datas[var][:, valid_indices]
        altitude = datas.alt[valid_indices]

        #for profile in data:
        #    ax.plot(profile, altitude, color=color, alpha=0.1, linewidth=0.5)
        # Reshape `data` to a 1D array
        flattened_data = data.values.flatten()
        
        # Repeat `altitude` for each profile
        repeated_altitude = np.tile(altitude.values, data.shape[0])
        
        # Scatter plot
        #ax.scatter(flattened_data, repeated_altitude, color=color, alpha=0.05, s=2, label=None, zorder=1)
        
        mean_profile = np.nanmean(data, axis=0)
        ax.plot(mean_profile, altitude, color=color, alpha=1, linewidth=4, label=dataset_name, zorder=10)

    ax.set_title(title, fontsize=SIZE, pad=20)
    ax.set_xlabel(xlabel)

    ax.set_ylim(bottom=0, top=altitude_limit)

#    if var == "theta":
#        ax.set_xlim(right=350)
    
    ax.text(-0.15, 1.15, subplot_labels[i], transform=ax.transAxes,
            fontsize=SIZE+4, fontweight="normal", verticalalignment="top")

    if var == "mse":
        ax.set_xlim(right=400)
        
    if var == "rh":
        ax.set_xlim(0,1)
        
    if var == "u" or var == "v":
        ax.axvline(0,color='grey',ls='dashed')

    if i == 0:
        ax.set_ylabel("Altitude / m", fontsize=SIZE)
        ax.set_yticks([5000, 10000, 15000, 20000, 25000]) 
        ax.set_yticklabels(["5000", "10000", "15000", "20000", "25000"], fontsize=SIZE)
        ax.yaxis.set_visible(True)
    else:
        ax.set_yticks([5000, 10000, 15000, 20000, 25000]) 
        ax.set_yticklabels([])

    ax.grid(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines[["left", "bottom"]].set_position(("outward", 20))

handles, labels = ax.get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, -0.03), ncol=4, frameon=True, fontsize=SIZE)

plt.tight_layout(rect=[0, 0.05, 1, 1])
#filepath = './Figures/'
#filename = '00.9_Overall_Means.png'
#plt.savefig(filepath+filename, facecolor='white', bbox_inches='tight', dpi=400)
plt.show()
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

# Add MSE to all datasets
for ds in [ds_BCO, ds_MET, ds_INMG]:
    ds["mse"] = mtfunc.moist_static_energy(ds.ta, ds.alt, ds.q) / 1000  # in kJ/kg

datasets = {
    "BCO": {"data": ds_BCO, "color": color_BCO},
    "R/V Meteor": {"data": ds_MET, "color": color_Meteor},
    "INMG": {"data": ds_INMG, "color": color_INMG},
}

subplot_labels = ["(a)", "(b)", "(c)", "(d)"]

variables = ["mse", "rh", "u", "v"]
titles = [f"Moist\nStatic Energy", f"Relative\nHumidity", "Zonal Wind", "Meridional Wind"]
x_labels = [r"MSE / kJ kg$^{-1}$", "rh", "u / ms$^{-1}$", "v / ms$^{-1}$"]

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
        
        # Only keep valid pressure values (between 1000 and 100 hPa)
        valid = (mean_pressure > 10) & (mean_pressure < 1000)
        
        ax.plot(mean_data[valid], mean_pressure[valid], color=color, linewidth=4, label=dataset_name, zorder=10)


    # Set labels and scale
    ax.set_title(title, fontsize=SIZE, pad=20)
    ax.set_xlabel(xlabel)

    #ax.set_yscale("log")
    ax.invert_yaxis()  # surface at bottom
    ax.set_ylim(1000, 100)  # in hPa

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
    
    if var in "u":
        ax.set_xlim(-15,15)
    
    if var in "v":
        ax.set_xlim(-7,7)

    if i == 0:
        ax.set_yticklabels([])
        ax.set_yticks([1000, 800, 600, 400, 200])
        ax.set_yticklabels(["1000", "800", "600", "400", "200"])
        ax.set_ylabel("Pressure / hPa", fontsize=SIZE)
        ax.yaxis.set_visible(True)
    else:
        ax.set_yticks([1000, 800, 600, 400, 200])
        ax.set_yticklabels([])

    ax.text(-0.15, 1.15, subplot_labels[i], transform=ax.transAxes,
            fontsize=SIZE + 4, fontweight="normal", verticalalignment="top")

    ax.grid(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines[["left", "bottom"]].set_position(("outward", 20))

# Add legend
handles, labels = ax.get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, -0.03), ncol=4, frameon=True, fontsize=SIZE)

plt.tight_layout(rect=[0, 0.05, 1, 1])
filepath = './Figures/'
filename = f'Fig06_Overall_Means_pressure.svg'
plt.savefig(filepath + filename, format='svg', facecolor='white', bbox_inches="tight", dpi=300)
plt.show()    

# %%
