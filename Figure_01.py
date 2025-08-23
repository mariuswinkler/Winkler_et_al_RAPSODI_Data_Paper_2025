# %%
import xarray as xr
import numpy as np
import matplotlib.pylab as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Rectangle
from matplotlib.ticker import FixedLocator
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
import cartopy.crs as ccrs
import cartopy.feature as cfeature

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
ds = xr.open_dataset("ipfs://bafybeihd6kyscsf7vzjnlivdtdd4fh5epuqqfqk7ldj6d2k634fuse2lay", engine="zarr")

# %%
PLATFORM = 'BCO'
ds_BCO = ds.where(ds.platform == PLATFORM, drop=True)
PLATFORM = 'INMG'
ds_INMG = ds.where(ds.platform == PLATFORM, drop=True)
PLATFORM = 'RV_Meteor'
ds_MET = ds.where(ds.platform == PLATFORM, drop=True)
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
    ["#008080", "#DAA520", "#003366"] #11
]

active_color_set_index = 11

active_colors = color_sets[active_color_set_index]
color_INMG = active_colors[0]
color_Meteor = active_colors[1]
color_BCO = active_colors[2]

# %%
# Altitudes (merged main + extra)
altitudes = np.arange(0, 31000, 1000)
extra_point = 30990
altitudes_merged = np.unique(np.append(altitudes, extra_point))

# Colormap and normalization
cmap = plt.get_cmap("winter") #PiYG, 
norm = mcolors.Normalize(vmin=0, vmax=25)

# Figure setup: main + two subplots (manually placed)
fig = plt.figure(figsize=(14, 10))

# === MAIN AXIS ===
main_ax = fig.add_axes([0.05, 0.05, 0.75, 0.5], projection=ccrs.PlateCarree())
main_ax.set_extent([-65, -15, 0, 23], crs=ccrs.PlateCarree())
main_ax.add_feature(cfeature.LAND, edgecolor='black', facecolor='lightgray')
main_ax.add_feature(cfeature.COASTLINE, edgecolor="black", linewidth=1)
main_ax.set_facecolor("white")

# Gridlines for main axis
gl = main_ax.gridlines(draw_labels=True, linestyle="dotted", alpha=0.7, color='gray', linewidth=0.6)
gl.top_labels = False
gl.right_labels = False

INSET_EXTRA = 0.5

# === INSET 1: Barbados ===
inset1 = fig.add_axes([0.033, 0.6, 0.35, 0.35], projection=ccrs.PlateCarree())
inset1_extent = [-60.15 - INSET_EXTRA, -59 + INSET_EXTRA, 12.9 - INSET_EXTRA, 13.6 + INSET_EXTRA]
inset1.set_extent(inset1_extent, crs=ccrs.PlateCarree())
inset1.add_feature(cfeature.LAND, edgecolor='black', facecolor='lightgray')
inset1.add_feature(cfeature.COASTLINE, edgecolor="black", linewidth=1, zorder=12)

inset1.set_xticks([-60, -59], crs=ccrs.PlateCarree())
inset1.set_yticks([13, 14], crs=ccrs.PlateCarree())
gl1 = inset1.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                       linestyle='dotted', linewidth=0.6, color='gray')
gl1.top_labels = False
gl1.right_labels = False
gl1.bottom_labels = True
gl1.left_labels = True
gl1.xlocator = FixedLocator([-60, -59])
gl1.ylocator = FixedLocator([13, 14])
gl1.xformatter = LongitudeFormatter()
gl1.yformatter = LatitudeFormatter()
inset1.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)

for spine in ['left', 'bottom', 'right', 'top']:
    inset1.spines[spine].set_linestyle('dotted')
    inset1.spines[spine].set_linewidth(0.6)
    inset1.spines[spine].set_edgecolor('black')


# === INSET 2: Cape Verde ===
inset2 = fig.add_axes([0.467, 0.6, 0.35, 0.35], projection=ccrs.PlateCarree())
inset2_extent = [-24 - INSET_EXTRA, -22.85 + INSET_EXTRA, 16.4 - INSET_EXTRA, 17.1 + INSET_EXTRA]
inset2.set_extent(inset2_extent, crs=ccrs.PlateCarree())
inset2.add_feature(cfeature.LAND, edgecolor='black', facecolor='lightgray')
inset2.add_feature(cfeature.COASTLINE, edgecolor="black", linewidth=1, zorder=12)

inset2.set_xticks([-24, -23], crs=ccrs.PlateCarree())
inset2.set_yticks([16, 17], crs=ccrs.PlateCarree())
gl2 = inset2.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                       linestyle='dotted', linewidth=0.6, color='gray')
gl2.top_labels = False
gl2.right_labels = False
gl2.bottom_labels = True
gl2.left_labels = True
gl2.xlocator = FixedLocator([-24, -23])
gl2.ylocator = FixedLocator([16, 17])
gl2.xformatter = LongitudeFormatter()
gl2.yformatter = LatitudeFormatter()
inset2.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)

# === Plotting function ===
def plot_the_dot(ds, ax):
    flight_lon = ds.lon
    flight_lat = ds.lat
    altitude = ds.alt / 1000  # Convert to km

    out_of_bounds_mask = (
        (flight_lon < -65) | (flight_lon > -15) |
        (flight_lat < 0)   | (flight_lat > 23)
    )
    flight_lon = flight_lon.where(~out_of_bounds_mask, np.nan)
    flight_lat = flight_lat.where(~out_of_bounds_mask, np.nan)
    altitude = altitude.where(~out_of_bounds_mask, np.nan)

    mask = ~np.isnan(flight_lon) & ~np.isnan(flight_lat) & ~np.isnan(altitude)
    flight_lon, flight_lat, altitude = flight_lon[mask], flight_lat[mask], altitude[mask]

    sc = ax.scatter(flight_lon, flight_lat, c=altitude, cmap=cmap, norm=norm,
                    s=10, edgecolor=None, alpha=0.4, transform=ccrs.PlateCarree(), zorder=10)
    return sc

# === Loop over altitudes and plot ===
sc = None
for alt in altitudes_merged:
    ds_slice_main = ds.where(ds['ascent_flag'] == 0, drop=True).sel(alt=alt, method='nearest').compute()
    sc = plot_the_dot(ds_slice_main, main_ax)

    ds_slice_bco = ds.where(ds['ascent_flag'] == 0, drop=True).sel(alt=alt, method='nearest').compute() #ds_BCO.sel(alt=alt, method='nearest').compute()
    sc = plot_the_dot(ds_slice_bco, inset1)

    ds_slice_inmg = ds_INMG.where(ds_INMG['ascent_flag'] == 0, drop=True).sel(alt=alt, method='nearest').compute()
    plot_the_dot(ds_slice_inmg, inset2)
    
    ds_slice_main = ds.where(ds['ascent_flag'] == 1, drop=True).sel(alt=alt, method='nearest').compute()
    sc = plot_the_dot(ds_slice_main, main_ax)

    ds_slice_bco = ds.where(ds['ascent_flag'] == 1, drop=True).sel(alt=alt, method='nearest').compute() #ds_BCO.sel(alt=alt, method='nearest').compute()
    plot_the_dot(ds_slice_bco, inset1)

    ds_slice_inmg = ds_INMG.where(ds_INMG['ascent_flag'] == 1, drop=True).sel(alt=alt, method='nearest').compute()
    plot_the_dot(ds_slice_inmg, inset2)

# === Bounding boxes on main_ax ===
barbados_box = Rectangle((inset1_extent[0], inset1_extent[2]),
                         width=inset1_extent[1] - inset1_extent[0],
                         height=inset1_extent[3] - inset1_extent[2],
                         linewidth=2, edgecolor='black', facecolor='none',
                         linestyle='dotted', transform=ccrs.PlateCarree(), zorder=5)
main_ax.add_patch(barbados_box)

cv_box = Rectangle((inset2_extent[0], inset2_extent[2]),
                   width=inset2_extent[1] - inset2_extent[0],
                   height=inset2_extent[3] - inset2_extent[2],
                   linewidth=2, edgecolor='black', facecolor='none',
                   linestyle='dotted', transform=ccrs.PlateCarree(), zorder=5)
main_ax.add_patch(cv_box)


main_ax.text(-0.1, 1.05, "(c)", transform=main_ax.transAxes, fontsize=SIZE+5)
inset1.text(-0.1, 1.9, "(a) BCO", transform=main_ax.transAxes, fontsize=SIZE+5)
inset2.text(0.47, 1.9, "(b) INMG", transform=main_ax.transAxes, fontsize=SIZE+5)

# === Colorbar on the far right ===
cbar_ax = fig.add_axes([0.82, 0.058, 0.015, 0.89])
cbar = plt.colorbar(mappable=sc, cax=cbar_ax)
cbar.set_label("Altitude / km")

# Save or show
plt.savefig('./Figures/Fig01_scatter_and_insets.png', dpi=400, bbox_inches='tight', facecolor='white')
plt.show()


# %%
