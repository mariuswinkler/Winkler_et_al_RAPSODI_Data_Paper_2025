# %%
import xarray as xr
import numpy as np
import matplotlib.pylab as plt
import pandas as pd
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
ds_BCO = ds.where(ds.platform == PLATFORM, drop=True).where(ds['ascent_flag'] == 0, drop=True)
PLATFORM = 'INMG'
ds_INMG = ds.where(ds.platform == PLATFORM, drop=True).where(ds['ascent_flag'] == 0, drop=True)
PLATFORM = 'RV_Meteor'
ds_MET = ds.where(ds.platform == PLATFORM, drop=True).where(ds['ascent_flag'] == 0, drop=True)
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

# --- Helper: get flight time in seconds from ds['bin_average_time'] and ds['launch_time']
def bin_time_seconds_since_launch(ds):
    """
    Interpret ds['bin_average_time'] as epoch nanoseconds and return
    seconds since ds['launch_time'] (shape: launch_time × alt).
    """
    bat = ds['bin_average_time']  # (launch_time, alt), float with NaNs

    # mask finite values
    mask = np.isfinite(bat)

    # float ns -> int64 ns (fill NaNs with 0 just for casting), then -> datetime64[ns]
    i64 = bat.where(mask, other=0).astype('int64')
    bin_ts = i64.astype('datetime64[ns]').where(mask, other=np.datetime64('NaT'))

    # broadcast launch_time to 2D and subtract
    launch2d = xr.broadcast(ds['launch_time'].astype('datetime64[ns]'), bat)[0]
    sec_since_launch = (bin_ts - launch2d) / np.timedelta64(1, 's')

    return sec_since_launch




# %%
## All
# Target time thresholds in minutes
target_minutes = [5, 10, 15, 20, 25, 80]
target_seconds = np.array(target_minutes) * 60

# --- All platforms together ---
bin_average_time_sec = bin_time_seconds_since_launch(ds)
  # convert to seconds

results = {
    "time_min": [],
    "mean_altitude_m": [],
    "mean_pressure_hPa": []
}

for t_sec in target_seconds:
    # Find the index along "alt" that is closest to target time per profile
    time_diff = np.abs(bin_average_time_sec - t_sec)
    idx = time_diff.argmin(dim='alt')  # shape: (launch_time,)

    # Select altitude and pressure at this index
    alt_at_t = ds['alt'].isel(alt=idx)
    p_at_t = ds['p'].isel(alt=idx)

    # Mean across all soundings
    mean_alt = alt_at_t.mean(dim='launch_time').item()
    mean_p = p_at_t.mean(dim='launch_time').item()

    results["time_min"].append(t_sec // 60)
    results["mean_altitude_m"].append(mean_alt)
    results["mean_pressure_hPa"].append(mean_p / 100)

# Print as DataFrame
df = pd.DataFrame(results)
print(df)

## INMG
# Target time thresholds in minutes
target_minutes = [5, 10, 15, 20, 25, 80]
target_seconds = np.array(target_minutes) * 60

# --- INMG ---
bin_average_time_sec = bin_time_seconds_since_launch(ds_INMG)
  # convert to seconds

results = {
    "time_min": [],
    "mean_altitude_m": [],
    "mean_pressure_hPa": []
}

for t_sec in target_seconds:
    # Find the index along "alt" that is closest to target time per profile
    time_diff = np.abs(bin_average_time_sec - t_sec)
    idx = time_diff.argmin(dim='alt')  # shape: (launch_time,)

    # Select altitude and pressure at this index
    alt_at_t = ds_INMG['alt'].isel(alt=idx)
    p_at_t = ds_INMG['p'].isel(alt=idx)

    # Mean across all soundings
    mean_alt = alt_at_t.mean(dim='launch_time').item()
    mean_p = p_at_t.mean(dim='launch_time').item()

    results["time_min"].append(t_sec // 60)
    results["mean_altitude_m"].append(mean_alt)
    results["mean_pressure_hPa"].append(mean_p / 100)

# Print as DataFrame
df = pd.DataFrame(results)
print(df)

## R/V Meteor
# Target time thresholds in minutes
target_minutes = [5, 10, 15, 20, 25, 80]
target_seconds = np.array(target_minutes) * 60

# --- R/V Meteor ---
bin_average_time_sec = bin_time_seconds_since_launch(ds_MET) # convert to seconds

results = {
    "time_min": [],
    "mean_altitude_m": [],
    "mean_pressure_hPa": []
}

for t_sec in target_seconds:
    # Find the index along "alt" that is closest to target time per profile
    time_diff = np.abs(bin_average_time_sec - t_sec)
    idx = time_diff.argmin(dim='alt')  # shape: (launch_time,)

    # Select altitude and pressure at this index
    alt_at_t = ds_MET['alt'].isel(alt=idx)
    p_at_t = ds_MET['p'].isel(alt=idx)

    # Mean across all soundings
    mean_alt = alt_at_t.mean(dim='launch_time').item()
    mean_p = p_at_t.mean(dim='launch_time').item()

    results["time_min"].append(t_sec // 60)
    results["mean_altitude_m"].append(mean_alt)
    results["mean_pressure_hPa"].append(mean_p / 100)

# Print as DataFrame
df = pd.DataFrame(results)
print(df)

## BCO
# Target time thresholds in minutes
target_minutes = [5, 10, 15, 20, 25, 80]
target_seconds = np.array(target_minutes) * 60

# --- BCO ---
bin_average_time_sec = bin_time_seconds_since_launch(ds_BCO)
  # convert to seconds

results = {
    "time_min": [],
    "mean_altitude_m": [],
    "mean_pressure_hPa": []
}

for t_sec in target_seconds:
    # Find the index along "alt" that is closest to target time per profile
    time_diff = np.abs(bin_average_time_sec - t_sec)
    idx = time_diff.argmin(dim='alt')  # shape: (launch_time,)

    # Select altitude and pressure at this index
    alt_at_t = ds_BCO['alt'].isel(alt=idx)
    p_at_t = ds_BCO['p'].isel(alt=idx)

    # Mean across all soundings
    mean_alt = alt_at_t.mean(dim='launch_time').item()
    mean_p = p_at_t.mean(dim='launch_time').item()

    results["time_min"].append(t_sec // 60)
    results["mean_altitude_m"].append(mean_alt)
    results["mean_pressure_hPa"].append(mean_p / 100)

# Print as DataFrame
df = pd.DataFrame(results)
print(df)





# %%
# Launch Clock
# Font size
SIZE = 20

# Convert launch times to pandas datetime
times_BCO  = pd.to_datetime(ds_BCO['launch_time'].values)
times_INMG = pd.to_datetime(ds_INMG['launch_time'].values)
times_MET  = pd.to_datetime(ds_MET['launch_time'].values)

def plot_clock(ax, times, color, alpha=0.6, label=None, radius=0.3, zorder=1):
    hours = times.hour + times.minute / 60
    angles = 2 * np.pi * hours / 24
    r_start = radius - 0.15
    r_end   = radius

    for i, theta in enumerate(angles):
        ax.plot([theta, theta], [r_start, r_end], color=color, alpha=alpha, lw=2,
                label=label if i == 0 else None, zorder=zorder)

# Create polar plot
fig, ax = plt.subplots(subplot_kw={'projection': 'polar'}, figsize=(8, 8))

# Plot each platform with distinct radius

plot_clock(ax, times_INMG, color_INMG,   alpha=0.5, label="INMG",       radius=0.54)
plot_clock(ax, times_MET,  color_Meteor, alpha=0.5, label="R/V Meteor", radius=0.37)
plot_clock(ax, times_BCO,  color_BCO,    alpha=0.5, label="BCO",        radius=0.2, zorder=10)

# Clock formatting
ax.set_facecolor('white')
ax.grid(False)
ax.set_theta_zero_location("N")
ax.set_theta_direction(-1)
ax.set_ylim(0, 0.6)  # Keep everything close to center
ax.spines['polar'].set_visible(False)  # ← Removes the black ring

# Hour labels every 3 hours starting from 2:00
hours = np.arange(0, 24, 3)
angles = 2 * np.pi * hours / 24
ax.set_xticks(angles)
ax.set_xticklabels([f"{h:02d}:00" for h in hours], fontsize=SIZE)

# Hour tick marks (pointing outward beyond ticks)
for angle in angles:
    ax.plot([angle, angle], [0., 0.57], color='grey', lw=1, ls='dotted',zorder=20, clip_on=False)

# Remove radius labels
ax.set_yticklabels([])

# Legend and title
ax.legend(loc='lower center', bbox_to_anchor=(0.5, -0.18), ncol=3, frameon=False, fontsize=SIZE)
#plt.title("Radiosonde Launch Times on a 24-Hour Clock", fontsize=SIZE + 2, pad=30)

# === Save and show ===
plt.tight_layout()
filepath = './Figures/'
filename = 'Fig15_Launch_Time_Watch.svg'
plt.savefig(filepath + filename, format='svg', facecolor='white', bbox_inches="tight", dpi=300)
filename = 'Fig15_Launch_Time_Watch.png'
plt.savefig(filepath + filename, format='png', facecolor='white', bbox_inches="tight", dpi=150)

plt.show()

times_INMG

# %%
