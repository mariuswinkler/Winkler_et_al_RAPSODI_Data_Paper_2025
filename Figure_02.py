# %%
import xarray as xr
import numpy as np
import pandas as pd

import matplotlib.pylab as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Polygon
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
ds = xr.open_dataset("ipfs://bafybeid7cnw62zmzfgxcvc6q6fa267a7ivk2wcchbmkoyk4kdi5z2yj2w4", engine="zarr")

# %%
PLATFORM = 'BCO'
ds_BCO = ds.where(ds.platform == PLATFORM, drop=True)
PLATFORM = 'INMG'
ds_INMG = ds.where(ds.platform == PLATFORM, drop=True)
PLATFORM = 'RV_Meteor'
ds_MET = ds.where(ds.platform == PLATFORM, drop=True)
#ds_MET.sel(launch_time=(ds_MET['ascent_flag'] == 0)).sel(launch_time="2024-09-21").launch_time
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
## Ascent-descent statistics
data = pd.DataFrame({
    'launch_time': pd.to_datetime(ds['launch_time'].values),
    'platform': ds['platform'].values,
    'ascent_flag': ds['ascent_flag'].values
})
data['date'] = data['launch_time'].dt.date

# List of unique platforms and dates
platforms = ['BCO', 'RV_Meteor', 'INMG']  # Update based on your dataset
dates = pd.date_range(start=data['date'].min(), end=data['date'].max())

ascending_counts = data[data['ascent_flag'] == 0].groupby(['date', 'platform']).size().unstack(fill_value=0)
descending_counts = data[data['ascent_flag'] == 1].groupby(['date', 'platform']).size().unstack(fill_value=0)

ascending_counts = ascending_counts.reindex(index=dates, columns=platforms, fill_value=0)
descending_counts = descending_counts.reindex(index=dates, columns=platforms, fill_value=0)


fig, ax = plt.subplots(figsize=(3, 15))  # Adjusted figure size for larger squares

# Define discrete color levels and boundaries for the colormap
levels = np.arange(0, 13)  # 9 bins (0 to 8)
#colors = ["white", "navajowhite", "palegreen", "mediumseagreen", "seagreen", "aquamarine", "lightseagreen", "royalblue", "midnightblue", "black", "pink", "crimson", "darkred"]  # Added 'navy' as an extra color
colors = [
    "white",          # Lightest
    "navajowhite",    # Light yellow
    "darkgoldenrod",  # Dark yellow
    "palegreen",      # Light green
    "mediumseagreen", # Medium green
    "seagreen",       # Dark green
    "deepskyblue",    # Medium blue
    "royalblue",      # Dark blue
    "midnightblue",   # Darkest blue
    "pink",           # Pink
    "crimson",        # Crimson
    "darkred",         # Dark red
]

# Create the colormap
cmap = mcolors.ListedColormap(colors, name="custom_colormap")

# Ensure correct mapping
norm = mcolors.BoundaryNorm(boundaries=levels, ncolors=len(colors))

# Iterate over the grid and create diagonal half-squares
for y, date in enumerate(dates):
    for x, platform in enumerate(platforms):
        # Get the number of soundings for ascending and descending
        asc_value = ascending_counts.at[date, platform]
        desc_value = descending_counts.at[date, platform]

        # Get the colors from the colormap
        asc_color = cmap(norm(asc_value))
        desc_color = cmap(norm(desc_value))

        # Create a square divided into two triangles for each platform-day combination
        # Bottom-left triangle (ascending)
        asc_triangle = Polygon([[x, y], [x, y + 1], [x+1, y + 1]], color=asc_color)
        ax.add_patch(asc_triangle)

        # Top-right triangle (descending)
        desc_triangle = Polygon([[x, y], [x + 1, y], [x+1, y + 1]], color=desc_color)
        ax.add_patch(desc_triangle)

# Add horizontal lines to separate the days
ax.hlines(np.arange(0, len(dates) + 1), xmin=0, xmax=len(platforms), color='black', linewidth=0.4)

# Add vertical lines to separate the ascending/descending soundings within each platform column
for x in np.arange(len(platforms)):
    ax.vlines(x + 1, ymin=0, ymax=len(dates), color='black', linewidth=0.4)

ax.vlines(0, ymin=0, ymax=len(dates), color='black', linewidth=0.4)

# Set aspect ratio to ensure squares are square
#ax.set_aspect('equal')
ax.set_aspect('auto') 

# Set x and y limits to ensure the squares fit well within the plot
ax.set_xlim(0, len(platforms))  # Extend the x-axis limits
ax.set_ylim(0, len(dates))       # Extend the y-axis limits

# Remove plot spines
ax.spines['left'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['bottom'].set_visible(False)

# Formatting the x and y ticks
ax.set_xticks(np.arange(len(platforms)) + 0.5)
ax.set_xticklabels(['BCO', 'R/V Meteor', 'INMG'], rotation=90)

# Show every 7th date on the y-axis, centered in the boxes
yticks = np.arange(0, len(dates), 7) + 0.5
ax.set_yticks(yticks)
ax.set_yticklabels(dates[np.arange(0, len(dates), 7)].strftime('%b %d'))
ax.invert_yaxis()


# Add a thicker line where the dates are written
#for y in yticks:
#    ax.axhline(y=y, color='black', linewidth=0.8)  # Adjust the linewidth as needed

# Create the colorbar on the right, vertical
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])

# Adjust figure layout to add a colorbar on the right
cbar_ax = fig.add_axes([1.0, 0.2, 0.1, 0.6])  # [left, bottom, width, height]
cbar = plt.colorbar(sm, cax=cbar_ax) 
cbar.set_label('Soundings per day |ascent/descent|')

# Explicitly set only half-level ticks
half_ticks = levels[:-1] + 0.5  # Midpoints between levels
cbar.ax.set_yticks(half_ticks)
cbar.ax.set_yticklabels([f"{int(tick)}" for tick in half_ticks])  # Label half ticks as integers

plt.tight_layout()
filepath = './Figures/'
filename = 'Fig02_Ascent_descent_statistics.svg'
plt.savefig(filepath + filename, format='svg', facecolor='white', bbox_inches="tight", dpi=300)
filename = 'Fig02_Ascent_descent_statistics.png'
plt.savefig(filepath + filename, format='png', facecolor='white', bbox_inches="tight", dpi=150)


plt.show()

# %%
