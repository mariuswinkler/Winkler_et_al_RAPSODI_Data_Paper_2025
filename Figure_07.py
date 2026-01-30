# %%
import xarray as xr
import numpy as np
import pandas as pd
import matplotlib.pylab as plt

# %%
ds = xr.open_dataset("ipfs://bafybeid7cnw62zmzfgxcvc6q6fa267a7ivk2wcchbmkoyk4kdi5z2yj2w4", engine="zarr")

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
PLATFORM = 'BCO'
ds_BCO = ds.where(ds.platform == PLATFORM, drop=True)
PLATFORM = 'INMG'
ds_INMG = ds.where(ds.platform == PLATFORM, drop=True)
PLATFORM = 'RV_Meteor'
ds_MET = ds.where(ds.platform == PLATFORM, drop=True)
valid_mask = ds["lat"].notnull()
valid_mask_reversed = valid_mask.isel(height=slice(None, None, -1))
max_height_idx = valid_mask_reversed.argmax(dim="height")
max_height_idx = ds.sizes["height"] - 1 - max_height_idx
max_heights = ds["height"].isel(height=max_height_idx)
median_max_height = np.nanmedian(max_heights)

def max_height_per_sounding(ds_):
    valid = ds_["lat"].notnull()
    valid_rev = valid.isel(height=slice(None, None, -1))
    idx_from_top = valid_rev.argmax(dim="height")
    max_idx = ds_.sizes["height"] - 1 - idx_from_top
    return ds_["height"].isel(height=max_idx)

max_height_per_sounding_BCO  = max_height_per_sounding(ds_BCO)
max_height_per_sounding_MET  = max_height_per_sounding(ds_MET)
max_height_per_sounding_INMG = max_height_per_sounding(ds_INMG)

print(f"Median of Maximum height: {median_max_height:.2f} meters")

# %%
active_colors = ["#BF312D", "darkblue", "#F6CA4C"]
color_INMG = active_colors[0]
color_Meteor = active_colors[1]
color_BCO = active_colors[2]

# %%
fig, ax = plt.subplots(figsize=(10, 5))

bin_edges = np.linspace(0, 30, 31)

ax.axvline(median_max_height/1000, color='black', ls='dashed', lw=2)
ax.text(median_max_height/1000-1.5, 38, "Median", rotation=0,
        verticalalignment='top', color='black', fontsize=SIZE-2)

# BCO
n_bco = max_height_per_sounding_BCO.sizes.get("sounding", max_height_per_sounding_BCO.size)
weights_bco = np.ones(n_bco) * 100 / n_bco
ax.hist(max_height_per_sounding_BCO.values/1000, bins=bin_edges, weights=weights_bco, alpha=0.3, color=color_BCO, histtype='stepfilled')
ax.hist(max_height_per_sounding_BCO.values/1000, bins=bin_edges, weights=weights_bco, alpha=1, label='BCO',
        histtype='step', linewidth=2, color=color_BCO)

max_BCO_value = max_height_per_sounding_BCO.median().values 
ax.axvline(max_BCO_value / 1000, ymin=-0.07, ymax=-0.05, color=color_BCO, linewidth=2, clip_on=False)

# R/V Meteor
n_met = max_height_per_sounding_MET.sizes.get("sounding", max_height_per_sounding_MET.size)
weights_met = np.ones(n_met) * 100 / n_met
ax.hist(max_height_per_sounding_MET.values/1000, bins=bin_edges, weights=weights_met, alpha=0.3, color=color_Meteor, histtype='stepfilled')
ax.hist(max_height_per_sounding_MET.values/1000, bins=bin_edges, weights=weights_met, alpha=1, label='R/V Meteor',
        histtype='step', linewidth=2, color=color_Meteor)

max_MET_value = max_height_per_sounding_MET.median().values  
ax.axvline(max_MET_value / 1000, ymin=-0.07, ymax=-0.05, color=color_Meteor, linewidth=2, clip_on=False)

# INMG
n_inmg = max_height_per_sounding_INMG.sizes.get("sounding", max_height_per_sounding_INMG.size)
weights_inmg = np.ones(n_inmg) * 100 / n_inmg
ax.hist(max_height_per_sounding_INMG.values/1000, bins=bin_edges, weights=weights_inmg, alpha=0.3, color=color_INMG, histtype='stepfilled')
ax.hist(max_height_per_sounding_INMG.values/1000, bins=bin_edges, weights=weights_inmg, alpha=1, label='INMG',
        histtype='step', linewidth=2, color=color_INMG)

max_INMG_value = max_height_per_sounding_INMG.median().values  
ax.axvline(max_INMG_value / 1000, ymin=-0.07, ymax=-0.05, color=color_INMG, linewidth=2, clip_on=False)

ax.spines[["left", "bottom"]].set_position(("outward", 20))
ax.spines[["right", "top"]].set_visible(False)
ax.set_xlim(left=0, right=30)
ax.set_ylim(bottom=0, top=35)

ax.set_xlabel("Maximum height / km", fontsize=SIZE)
ax.set_ylabel("Soundings per Platform / %", fontsize=SIZE)

ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.25), fancybox=True, shadow=False, ncol=3, fontsize=SIZE)
filepath = './Figures/'
filename = f'Fig07_Max_Height_PDF.svg'
plt.savefig(filepath + filename, format='svg', facecolor='white', bbox_inches="tight", dpi=300)
filename = f'Fig07_Max_Height_PDF.png'
plt.savefig(filepath + filename, format='png', facecolor='white', bbox_inches="tight", dpi=150)

plt.show()

# %%
