# %%
import xarray as xr
import numpy as np
import matplotlib.pylab as plt
import pandas as pd

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
ds_BCO = ds.where(ds.platform == PLATFORM, drop=True).where(ds['ascent_flag'] == 0, drop=True)
PLATFORM = 'INMG'
ds_INMG = ds.where(ds.platform == PLATFORM, drop=True).where(ds['ascent_flag'] == 0, drop=True)
PLATFORM = 'RV_Meteor'
ds_MET = ds.where(ds.platform == PLATFORM, drop=True).where(ds['ascent_flag'] == 0, drop=True)

# %%

active_colors = ["#BF312D", "darkblue", "#F6CA4C"]
color_INMG = active_colors[0]
color_Meteor = active_colors[1]
color_BCO = active_colors[2]


# %%
def bin_time_seconds_since_launch(ds):
    """
    Return seconds since launch with shape (launch_time, height).
    Works whether ds['interpolated_time'] is datetime64[ns] or
    a numeric value representing ns since epoch.
    """
    bat = ds['interpolated_time']

    if np.issubdtype(bat.dtype, np.datetime64):
        bin_ts = bat.astype('datetime64[ns]')
    else:
        mask = np.isfinite(bat)
        i64 = bat.where(mask, other=0).astype('int64')
        bin_ts = i64.astype('datetime64[ns]').where(mask, other=np.datetime64('NaT'))

    launch2d = xr.broadcast(ds['launch_time'].astype('datetime64[ns]'), bin_ts)[0]
    sec_since_launch = (bin_ts - launch2d) / np.timedelta64(1, 's')
    return sec_since_launch

# %%
target_minutes = [5, 10, 15, 20, 25, 80]
target_seconds = np.array(target_minutes) * 60

interpolated_time_sec = bin_time_seconds_since_launch(ds)

results = {
    "time_min": [],
    "mean_height_m": [],
    "mean_pressure_hPa": []
}

for t_sec in target_seconds:
    time_diff = np.abs(interpolated_time_sec - t_sec)
    idx = time_diff.argmin(dim='height')  

    height_at_t = ds['height'].isel(height=idx)
    p_at_t = ds['p'].isel(height=idx)

    mean_height = height_at_t.mean(dim='launch_time').item()
    mean_p = p_at_t.mean(dim='launch_time').item()

    results["time_min"].append(t_sec // 60)
    results["mean_height_m"].append(mean_height)
    results["mean_pressure_hPa"].append(mean_p / 100)

df = pd.DataFrame(results)
print(df)

## INMG
target_minutes = [5, 10, 15, 20, 25, 80]
target_seconds = np.array(target_minutes) * 60

interpolated_time_sec = bin_time_seconds_since_launch(ds_INMG)

results = {
    "time_min": [],
    "mean_height_m": [],
    "mean_pressure_hPa": []
}

for t_sec in target_seconds:
    time_diff = np.abs(interpolated_time_sec - t_sec)
    idx = time_diff.argmin(dim='height') 

    height_at_t = ds_INMG['height'].isel(height=idx)
    p_at_t = ds_INMG['p'].isel(height=idx)

    mean_height = height_at_t.mean(dim='launch_time').item()
    mean_p = p_at_t.mean(dim='launch_time').item()

    results["time_min"].append(t_sec // 60)
    results["mean_height_m"].append(mean_height)
    results["mean_pressure_hPa"].append(mean_p / 100)

df = pd.DataFrame(results)
print(df)

## R/V Meteor
target_minutes = [5, 10, 15, 20, 25, 80]
target_seconds = np.array(target_minutes) * 60

interpolated_time_sec = bin_time_seconds_since_launch(ds_MET) 

results = {
    "time_min": [],
    "mean_height_m": [],
    "mean_pressure_hPa": []
}

for t_sec in target_seconds:
    time_diff = np.abs(interpolated_time_sec - t_sec)
    idx = time_diff.argmin(dim='height')  

    height_at_t = ds_MET['height'].isel(height=idx)
    p_at_t = ds_MET['p'].isel(height=idx)

    mean_height = height_at_t.mean(dim='launch_time').item()
    mean_p = p_at_t.mean(dim='launch_time').item()

    results["time_min"].append(t_sec // 60)
    results["mean_height_m"].append(mean_height)
    results["mean_pressure_hPa"].append(mean_p / 100)

df = pd.DataFrame(results)
print(df)

## BCO
target_minutes = [5, 10, 15, 20, 25, 80]
target_seconds = np.array(target_minutes) * 60

interpolated_time_sec = bin_time_seconds_since_launch(ds_BCO)

results = {
    "time_min": [],
    "mean_height_m": [],
    "mean_pressure_hPa": []
}

for t_sec in target_seconds:
    time_diff = np.abs(interpolated_time_sec - t_sec)
    idx = time_diff.argmin(dim='height') 

    height_at_t = ds_BCO['height'].isel(height=idx)
    p_at_t = ds_BCO['p'].isel(height=idx)

    mean_height = height_at_t.mean(dim='launch_time').item()
    mean_p = p_at_t.mean(dim='launch_time').item()

    results["time_min"].append(t_sec // 60)
    results["mean_height_m"].append(mean_height)
    results["mean_pressure_hPa"].append(mean_p / 100)

df = pd.DataFrame(results)
print(df)

# %%
# Launch Clock
SIZE = 20

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

fig, ax = plt.subplots(subplot_kw={'projection': 'polar'}, figsize=(8, 8))

plot_clock(ax, times_INMG, color_INMG,   alpha=0.5, label="INMG",       radius=0.54)
plot_clock(ax, times_MET,  color_Meteor, alpha=0.5, label="R/V Meteor", radius=0.37)
plot_clock(ax, times_BCO,  color_BCO,    alpha=0.5, label="BCO",        radius=0.2, zorder=10)

ax.set_facecolor('white')
ax.grid(False)
ax.set_theta_zero_location("N")
ax.set_theta_direction(-1)
ax.set_ylim(0, 0.6)  
ax.spines['polar'].set_visible(False) 

# Hour labels every 3 hours starting from 2:00
hours = np.arange(0, 24, 3)
angles = 2 * np.pi * hours / 24
ax.set_xticks(angles)
ax.set_xticklabels([f"{h:02d}:00" for h in hours], fontsize=SIZE)

for angle in angles:
    ax.plot([angle, angle], [0., 0.57], color='grey', lw=1, ls='dotted',zorder=20, clip_on=False)

ax.set_yticklabels([])

ax.legend(loc='lower center', bbox_to_anchor=(0.5, -0.18), ncol=3, frameon=False, fontsize=SIZE)

plt.tight_layout()
filepath = './Figures/'
filename = 'Fig15_Launch_Time_Watch.svg'
plt.savefig(filepath + filename, format='svg', facecolor='white', bbox_inches="tight", dpi=300)
filename = 'Fig15_Launch_Time_Watch.png'
plt.savefig(filepath + filename, format='png', facecolor='white', bbox_inches="tight", dpi=150)

plt.show()
# %%
