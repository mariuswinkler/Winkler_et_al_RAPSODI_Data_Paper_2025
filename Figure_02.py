# %%
import xarray as xr
import numpy as np
import pandas as pd

import matplotlib.pylab as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Polygon

# %%
ds = xr.open_dataset("ipfs://bafybeid7cnw62zmzfgxcvc6q6fa267a7ivk2wcchbmkoyk4kdi5z2yj2w4", engine="zarr")

# %%
SIZE = 20
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

# %%
valid_mask = ds["lat"].notnull()
valid_mask_reversed = valid_mask.isel(height=slice(None, None, -1))
max_height_idx = valid_mask_reversed.argmax(dim="height")
max_height_idx = ds.sizes["height"] - 1 - max_height_idx
max_heights = ds["height"].isel(height=max_height_idx)
median_max_height = np.nanmedian(max_heights)

print(f"Median of Maximum height: {median_max_height:.2f} meters")

# %%
active_colors = ["#BF312D", "darkblue", "#F6CA4C"]
color_INMG = active_colors[0]
color_Meteor = active_colors[1]
color_BCO = active_colors[2]

# %%
## Ascent-descent statistics
data = pd.DataFrame({
    'launch_time': pd.to_datetime(ds['launch_time'].values),
    'platform': ds['platform'].values,
    'ascent_flag': ds['ascent_flag'].values
})
data['date'] = data['launch_time'].dt.date

platforms = ['BCO', 'RV_Meteor', 'INMG'] 
dates = pd.date_range(start=data['date'].min(), end=data['date'].max())

ascending_counts = data[data['ascent_flag'] == 0].groupby(['date', 'platform']).size().unstack(fill_value=0)
descending_counts = data[data['ascent_flag'] == 1].groupby(['date', 'platform']).size().unstack(fill_value=0)

ascending_counts = ascending_counts.reindex(index=dates, columns=platforms, fill_value=0)
descending_counts = descending_counts.reindex(index=dates, columns=platforms, fill_value=0)


fig, ax = plt.subplots(figsize=(3, 15))

levels = np.arange(0, 13) 
colors = [
    "white",           
    "navajowhite",    
    "darkgoldenrod",  
    "palegreen",      
    "mediumseagreen", 
    "seagreen",       
    "deepskyblue",    
    "royalblue",      
    "midnightblue",   
    "pink",           
    "crimson",        
    "darkred",         
]


cmap = mcolors.ListedColormap(colors, name="custom_colormap")
norm = mcolors.BoundaryNorm(boundaries=levels, ncolors=len(colors))

for y, date in enumerate(dates):
    for x, platform in enumerate(platforms):
        asc_value = ascending_counts.at[date, platform]
        desc_value = descending_counts.at[date, platform]

        asc_color = cmap(norm(asc_value))
        desc_color = cmap(norm(desc_value))

        asc_triangle = Polygon([[x, y], [x, y + 1], [x+1, y + 1]], color=asc_color)
        ax.add_patch(asc_triangle)

        desc_triangle = Polygon([[x, y], [x + 1, y], [x+1, y + 1]], color=desc_color)
        ax.add_patch(desc_triangle)

ax.hlines(np.arange(0, len(dates) + 1), xmin=0, xmax=len(platforms), color='black', linewidth=0.4)

for x in np.arange(len(platforms)):
    ax.vlines(x + 1, ymin=0, ymax=len(dates), color='black', linewidth=0.4)

ax.vlines(0, ymin=0, ymax=len(dates), color='black', linewidth=0.4)

ax.set_aspect('auto') 

ax.spines['left'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['bottom'].set_visible(False)

ax.set_xlim(0, len(platforms))  
ax.set_ylim(0, len(dates))     

ax.set_xticks(np.arange(len(platforms)) + 0.5)
ax.set_xticklabels(['BCO', 'R/V Meteor', 'INMG'], rotation=90)

yticks = np.arange(0, len(dates), 7) + 0.5
ax.set_yticks(yticks)
ax.set_yticklabels(dates[np.arange(0, len(dates), 7)].strftime('%b %d'))
ax.invert_yaxis()

sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])

cbar_ax = fig.add_axes([1.0, 0.2, 0.1, 0.6])  # [left, bottom, width, height]
cbar = plt.colorbar(sm, cax=cbar_ax) 
cbar.set_label('Soundings per day |ascent/descent|')

half_ticks = levels[:-1] + 0.5 
cbar.ax.set_yticks(half_ticks)
cbar.ax.set_yticklabels([f"{int(tick)}" for tick in half_ticks])

plt.tight_layout()
filepath = './Figures/'
filename = 'Fig02_Ascent_descent_statistics.svg'
plt.savefig(filepath + filename, format='svg', facecolor='white', bbox_inches="tight", dpi=300)
filename = 'Fig02_Ascent_descent_statistics.png'
plt.savefig(filepath + filename, format='png', facecolor='white', bbox_inches="tight", dpi=150)

plt.show()