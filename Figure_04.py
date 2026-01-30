# %%
import numpy as np
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.cm as cm
from matplotlib.colors import ListedColormap

# %%
osc_RS = xr.open_dataset("ipfs://bafybeidtfcyurvbw5obbhl5zlyfaoamnlhu6rrxyrmg2r2wwwidxq32oeq", engine="zarr")
pr_data = xr.open_dataset("ipfs://bafybeihjcwsecgpmsjxoo5peqafnuqfnalu3ya3vtibwl7qkm76izsnuei", engine="zarr")

# %%
def truncate_colormap(cmap_name='Spectral', minval=0.05, maxval=0.95, n=256):
    base = plt.get_cmap(cmap_name)
    new_colors = base(np.linspace(minval, maxval, n))
    return ListedColormap(new_colors)

SIZE = 11
plt.rcParams["axes.labelsize"] = SIZE
plt.rcParams["legend.fontsize"] = SIZE
plt.rcParams["xtick.labelsize"] = SIZE
plt.rcParams["ytick.labelsize"] = SIZE
plt.rcParams["font.size"] = SIZE
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'
plt.rcParams['xtick.major.size'] = 6
plt.rcParams['ytick.major.size'] = 6


width_cm = 17
height_cm = 12
fig, ax = plt.subplots(figsize=(width_cm / 2.54, height_cm / 2.54))

# Collect all rain means
rain_means = []
for i in range(len(osc_RS.release_time)):
    lt = osc_RS.release_time[i].values
    start_time = np.datetime64(lt)
    end_time = start_time + np.timedelta64(20, 'm')
    rain_slice = pr_data.rain_rate_qc.sel(time=slice(start_time, end_time))
    rain_mean = float(rain_slice.mean(skipna=True).values)
    rain_means.append(rain_mean)
    print(lt, rain_mean)

# Map rain rates to categorical bins
sorted_unique_rain = sorted(set(rain_means))
bin_index_map = {val: i for i, val in enumerate(sorted_unique_rain)}
bin_indices = [bin_index_map[val] for val in rain_means]

# Create colormap and norm
num_bins = len(sorted_unique_rain)
boundaries = np.arange(-0.5, num_bins + 0.5, 1)
cmap = truncate_colormap('jet_r', minval=0.3, maxval=0.99, n=num_bins)
norm = mcolors.BoundaryNorm(boundaries, ncolors=num_bins)

trajectory_colors = [cmap(norm(idx)) for idx in bin_indices]

for i in range(len(osc_RS.sonde_id)):
    ds_single = osc_RS.sortby("release_time").isel(sonde_id=i)
    altitude = ds_single["alt"].values
    flight_time = osc_RS.isel(sonde_id=4)["flight_time"].values
    time_offset = (flight_time - flight_time[0]) / np.timedelta64(1, 'm')
    release_time = np.datetime_as_string(ds_single["release_time"].values, unit='m')

    ax.plot(time_offset, altitude / 1000, label=release_time,
            color=trajectory_colors[i], linewidth=2)

    # Freezing level
    temperature = ds_single["ta"].values
    freezing_idx = np.where(temperature <= 273.15)[0]
    if len(freezing_idx) > 0:
        freezing_alt = altitude[freezing_idx[0]]
        ax.axhline(y=freezing_alt / 1000, color=trajectory_colors[i],
                   linestyle='--', linewidth=1, alpha=0.8)

ax.set_xlabel("Time Since Launch / hh:mm", fontsize=SIZE)
ax.set_ylabel("Altitude / km", fontsize=SIZE)
ax.spines[["top", "right"]].set_visible(False)
ax.spines[["left", "bottom"]].set_position(("outward", 20))
tick_intervals = np.arange(0, max(time_offset) + 1, 30)
ax.set_xticks(tick_intervals)
ax.set_xticklabels([f"{int(t // 60):02}:{int(t % 60):02}" for t in tick_intervals])
ax.set_xlim(0, 300)
ax.set_ylim(2, 10)

ax.legend(fontsize=SIZE - 1, title_fontsize=SIZE,
          loc="lower center", bbox_to_anchor=(0.5, -0.55),
          frameon=True, ncol=3)

sm = cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
cbar_ax = fig.add_axes([0.95, 0.45, 0.02, 0.5])
cbar = fig.colorbar(sm, cax=cbar_ax, orientation='vertical', pad=0.01, shrink=0.8)
cbar.set_label("20min Rain Rate / mm h⁻¹", fontsize=SIZE)

tick_locs = np.arange(num_bins)
tick_labels = [f"{val:.2f}" for val in sorted_unique_rain]
cbar.set_ticks(tick_locs)
cbar.set_ticklabels(tick_labels)

plt.tight_layout()
filepath = './Figures/'
filename = 'Fig04_Oscillator_Sondes_ColoredByRain.svg'
plt.savefig(filepath + filename, format='svg', facecolor='white', bbox_inches="tight", dpi=300)

filename = 'Fig04_Oscillator_Sondes_ColoredByRain.png'
plt.savefig(filepath + filename, format='png', facecolor='white', bbox_inches="tight", dpi=150)
plt.show()


# %%
