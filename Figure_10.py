# %%
import xarray as xr
import numpy as np
import matplotlib.pylab as plt

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
ds = xr.open_dataset("ipfs://bafybeihd6kyscsf7vzjnlivdtdd4fh5epuqqfqk7ldj6d2k634fuse2lay", engine="zarr")

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
def plot_mean_dz_with_percentiles(ax, ds, ascent_flag, alt, title, cmap_ascent, cmap_descent):
    # Filter data based on ascent_flag (0 for ascent, 1 for descent)
    dz_ascent = ds['dz'].where(ds['ascent_flag'] == 0, drop=True)
    dz_descent = ds['dz'].where(ds['ascent_flag'] == 1, drop=True)

    ALT = 50
    
    # Compute mean, 10th, and 90th percentiles for ascent and descent
    dz_ascent_min  = dz_ascent.min(dim='launch_time').rolling(alt=ALT, center=True).mean()
    dz_ascent_max  = dz_ascent.max(dim='launch_time').rolling(alt=ALT, center=True).mean()
    dz_ascent_mean = dz_ascent.mean(dim='launch_time')
    dz_ascent_10th = dz_ascent.quantile(0.1, dim='launch_time')
    dz_ascent_90th = dz_ascent.quantile(0.9, dim='launch_time')

    dz_descent_min  = dz_descent.min(dim='launch_time').rolling(alt=ALT, center=True).mean()
    dz_descent_max  = dz_descent.max(dim='launch_time').rolling(alt=ALT, center=True).mean()
    dz_descent_mean = dz_descent.mean(dim='launch_time')
    dz_descent_10th = dz_descent.quantile(0.1, dim='launch_time')
    dz_descent_90th = dz_descent.quantile(0.9, dim='launch_time')

    # Plot ascent (on the left)
    ascent_altitudes = alt
    ax[0].plot(dz_ascent_mean, alt, color='#1F77B4')
    ax[0].plot(dz_ascent_min, alt, color='grey', alpha=0.5)
    ax[0].plot(dz_ascent_max, alt, color='grey', alpha=0.5)
    ax[0].set_xlim(0, 10)
    ax[0].set_ylim(0, 25)
    ax[0].plot(dz_ascent_10th, ascent_altitudes, linestyle='solid', color='#6FD08C')
    ax[0].plot(dz_ascent_90th, ascent_altitudes, linestyle='solid', color='#6FD08C')
    ax[0].set_title(f'{title} Ascent', pad=20, fontsize=SIZE)
    ax[0].set_ylabel('Altitude / km')
    ax[0].set_xlabel(r'Ascent rate / ${\rm m\, s}^{-1}$')
    
    # Plot descent (on the right)
    descent_altitudes = alt
    ax[1].plot(dz_descent_mean, alt, color='#1F77B4')
    ax[1].plot(dz_descent_min, alt, label='Max/Min', color='grey', alpha=0.5)
    ax[1].plot(dz_descent_max, alt, color='grey', alpha=0.5)
    ax[1].set_xlim(-70, 0)
    ax[1].set_ylim(0, 25)
    ax[1].plot(dz_descent_10th, descent_altitudes, linestyle='solid', color='#6FD08C')
    ax[1].plot(dz_descent_90th, descent_altitudes, linestyle='solid', color='#6FD08C')
    ax[1].set_title(f'{title} Descent', pad=20, fontsize=SIZE)
    #ax[1].set_ylabel('Altitude / m')
    ax[1].set_xlabel(r'Descent rate / ${\rm m\, s}^{-1}$')

    ax[1].axvline(1000, linestyle='solid', lw=4, label='Mean rate', color='#1F77B4')
    ax[1].axvline(1000, linestyle='solid', lw=4, label='10th/90th percentile', color='#6FD08C')
    ax[1].axvline(1000, linestyle='solid', lw=4, color='grey')

    # Compute mean dz below 15 km (in km units)
    alt_limit_km = 25
    below_25km = alt < alt_limit_km

    mean_dz_ascent_below_25km = dz_ascent.where(below_25km, drop=True).mean()
    mean_dz_descent_below_25km = dz_descent.where(below_25km, drop=True).mean()

    # Add short tick marks to indicate mean below 15 km
    ax[0].axvline(mean_dz_ascent_below_25km, ymin=-0.06, ymax=-0.04, color='#1F77B4', linewidth=4, label='_nolegend_', clip_on=False)
    ax[1].axvline(mean_dz_descent_below_25km, ymin=-0.06, ymax=-0.04, color='#1F77B4', linewidth=4, label='_nolegend_', clip_on=False)
    print("mean_dz_ascent_below_25km", mean_dz_ascent_below_25km.values)
    print("mean_dz_descent_below_25km", mean_dz_descent_below_25km.values)
    
    for axis in ax:
        axis.spines[["left", "bottom"]].set_position(("outward", 20))
        axis.spines[["right", "top"]].set_visible(False)
        axis.grid(True, linestyle="dotted", alpha=0.5, color='grey')



# %%
fig, axs = plt.subplots(2, 2, figsize=(10, 12), sharex='col', sharey=True)

cmap_ascent = plt.get_cmap('Grays')
cmap_descent = plt.get_cmap('Grays')

plot_mean_dz_with_percentiles(axs[0], ds_INMG, ascent_flag='ascent_flag', alt=ds_INMG['alt']/1000, title="Meteomodem", cmap_ascent=cmap_ascent, cmap_descent=cmap_descent)
plot_mean_dz_with_percentiles(axs[1], xr.merge([ds_MET, ds_BCO]), ascent_flag='ascent_flag', alt=xr.merge([ds_MET, ds_BCO])['alt']/1000, title="Vaisala", cmap_ascent=cmap_ascent, cmap_descent=cmap_descent)

axs[0, 0].set_xlabel("")
axs[0, 1].set_xlabel("")

desired_order = ["Mean rate", "10th/90th percentile", "Max/Min"]  
handles, labels = [], []
for ax in axs.flat:
    for h, l in zip(*ax.get_legend_handles_labels()):
        if l not in labels:
            handles.append(h)
            labels.append(l)

sorted_indices = [labels.index(l) for l in desired_order if l in labels]
handles = [handles[i] for i in sorted_indices]
labels = [labels[i] for i in sorted_indices]

fig.legend(handles, labels, loc='lower center', ncol=3, bbox_to_anchor=(0.53, -0.055))

axs[0, 0].text(-0.2, 1.15, "(a)", transform=axs[0, 0].transAxes, fontsize=SIZE+2, verticalalignment='top', horizontalalignment='left')
axs[0, 1].text(-0.2, 1.15, "(b)", transform=axs[0, 1].transAxes, fontsize=SIZE+2, verticalalignment='top', horizontalalignment='left')
axs[1, 0].text(-0.2, 1.15, "(c)", transform=axs[1, 0].transAxes, fontsize=SIZE+2, verticalalignment='top', horizontalalignment='left')
axs[1, 1].text(-0.2, 1.15, "(d)", transform=axs[1, 1].transAxes, fontsize=SIZE+2, verticalalignment='top', horizontalalignment='left')

plt.tight_layout()
filepath = './Figures/'
filename = 'Fig10_ascent_descent_rate.svg'
plt.savefig(filepath + filename, format='svg', facecolor='white', bbox_inches="tight", dpi=300)
filename = 'Fig10_ascent_descent_rate.png'
plt.savefig(filepath + filename, format='png', facecolor='white', bbox_inches="tight", dpi=150)

plt.show()
# %%
