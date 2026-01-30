# %%
import xarray as xr
import numpy as np
import matplotlib.pylab as plt
from matplotlib.patches import Patch
import matplotlib.image as mpimg
from scipy.interpolate import interp1d
from metpy.plots import SkewT
from metpy.units import units

# %%
ds = xr.open_dataset("ipfs://bafybeid7cnw62zmzfgxcvc6q6fa267a7ivk2wcchbmkoyk4kdi5z2yj2w4", engine="zarr")

# %%
SIZE = 24
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
MET_ARRIVAL = '2024-09-22T20:30'
MET_DEPARTURE = '2024-09-24T03:00'
ds_BCO_meets_MET = ds_BCO.where(ds_BCO.ascent_flag==0, drop=True).sel(launch_time=slice(MET_ARRIVAL,MET_DEPARTURE))
ds_BCO_meets_MET;
times_to_remove = [
    '2024-09-23T09:50:01.200030016',
    '2024-09-23T11:40:08.420030016',
    '2024-09-23T12:10:18.733000992'
] 
times_to_remove = np.array(times_to_remove, dtype="datetime64[ns]")

ds_MET_at_BCO = ds_MET.where(ds_MET.ascent_flag==0, drop=True).sel(launch_time=slice(MET_ARRIVAL,MET_DEPARTURE)).drop_sel(launch_time=times_to_remove)
ds_MET_at_BCO;
ds_BCO_meets_MET_adapted_time = ds_BCO_meets_MET.assign_coords({"launch_time": ("launch_time", ds_MET_at_BCO.launch_time.values)})

# %%
T_met_raw = ds_MET_at_BCO.ta.values
dp_met_raw = ds_MET_at_BCO.dp.values
p_met_raw = ds_MET_at_BCO.p.values / 100  # Pa to hPa

T_bco_raw = ds_BCO_meets_MET_adapted_time.ta.values
dp_bco_raw = ds_BCO_meets_MET_adapted_time.dp.values
p_bco_raw = ds_BCO_meets_MET_adapted_time.p.values / 100

# Mean and std profiles
T_met = np.nanmean(T_met_raw, axis=0) * units.kelvin
T_met_std = np.nanstd(T_met_raw, axis=0) * units.kelvin
dp_met = np.nanmean(dp_met_raw, axis=0) * units.kelvin
dp_met_std = np.nanstd(dp_met_raw, axis=0) * units.kelvin
p_met = p_met_raw[0, :] * units.hPa

T_bco = np.nanmean(T_bco_raw, axis=0) * units.kelvin
T_bco_std = np.nanstd(T_bco_raw, axis=0) * units.kelvin
dp_bco = np.nanmean(dp_bco_raw, axis=0) * units.kelvin
dp_bco_std = np.nanstd(dp_bco_raw, axis=0) * units.kelvin
p_bco = p_bco_raw[0, :] * units.hPa

# Absolute Temperature Difference in Kelvin
interp_T_met = interp1d(p_met.m, T_met.m, bounds_error=False, fill_value=np.nan)
interp_dp_met = interp1d(p_met.m, dp_met.m, bounds_error=False, fill_value=np.nan)

T_abs_diff = np.abs(interp_T_met(p_bco.m) - T_bco.m)  # in Kelvin
dp_abs_diff = np.abs(interp_dp_met(p_bco.m) - dp_bco.m)  # in Kelvin

# Plotting
LW = 2.5

fig = plt.figure(figsize=(14, 20))  

img = mpimg.imread('./Figures/Fig14_Meteor_meets_BCO.jpg')
aspect = img.shape[0] / img.shape[1]

img_height_frac = aspect * 0.7  
img_bottom = 1.0 - img_height_frac
img_width_frac = 0.7  
img_left = 0.15

ax_img = fig.add_axes([img_left, img_bottom, img_width_frac, img_height_frac])

ax_img.imshow(img)
ax_img.axis('off')

plot_top = img_bottom + 0.0  # + padding below the image
plot_height = plot_top        

gs_subplots = fig.add_gridspec(
    1, 2,                     
    width_ratios=[3, 1],
    left=0.0,
    right=1,
    bottom=0.05,
    top=plot_top,
    wspace=0.1
)

skew = SkewT(fig, rotation=45)
skew.ax.set_position(gs_subplots[0].get_position(fig))
skew.ax.set_subplotspec(gs_subplots[0])

skew.plot(p_met*100, T_met*2, color='black', linewidth=LW, label="Mean Temperature $\overline{\mathrm{T}}$")
skew.plot(p_met*100, dp_met*2, color='black', linewidth=LW, linestyle='--', label="Mean Dew Point $\overline{\mathrm{T}}_{\mathrm{d}}$")
skew.plot(p_met*100, T_met*2, color=color_BCO, linewidth=LW, label="BCO")
skew.plot(p_met*100, dp_met*2, color=color_Meteor, linewidth=LW, label="R/V Meteor")

skew.plot(p_met, T_met, color=color_Meteor, linewidth=LW)
skew.plot(p_met, dp_met, color=color_Meteor, linestyle='--', linewidth=LW)
skew.plot(p_bco, T_bco, color=color_BCO, linewidth=LW)
skew.plot(p_bco, dp_bco, color=color_BCO, linestyle='--', linewidth=LW)

T_met_C = T_met.to('degC').magnitude
T_met_std_C = T_met_std.to('delta_degC').magnitude
dp_met_C = dp_met.to('degC').magnitude
dp_met_std_C = dp_met_std.to('delta_degC').magnitude

T_bco_C = T_bco.to('degC').magnitude
T_bco_std_C = T_bco_std.to('delta_degC').magnitude
dp_bco_C = dp_bco.to('degC').magnitude
dp_bco_std_C = dp_bco_std.to('delta_degC').magnitude

skew.ax.fill_betweenx(p_met.m, T_met_C - T_met_std_C, T_met_C + T_met_std_C,
                      color=color_Meteor, alpha=0.2, linewidth=0)
skew.ax.fill_betweenx(p_met.m, dp_met_C - dp_met_std_C, dp_met_C + dp_met_std_C,
                      color=color_Meteor, alpha=0.2, linewidth=0)

skew.ax.fill_betweenx(p_bco.m, T_bco_C - T_bco_std_C, T_bco_C + T_bco_std_C,
                      color=color_BCO, alpha=0.2, linewidth=0)
skew.ax.fill_betweenx(p_bco.m, dp_bco_C - dp_bco_std_C, dp_bco_C + dp_bco_std_C,
                      color=color_BCO, alpha=0.2, linewidth=0)


skew.plot_dry_adiabats(alpha=0.2, color='gray', linestyle='--')
skew.plot_moist_adiabats(alpha=0.2, color='gray', linestyle='--')
skew.plot_mixing_lines(alpha=0.2, color='gray', linestyle='--')

skew.ax.set_ylim(1000, 100)
skew.ax.set_xlim(-20, 40)
skew.ax.set_xlabel("Temperature / °C", fontsize=SIZE)
skew.ax.set_ylabel("Pressure / hPa", fontsize=SIZE)
skew.ax.spines[["top", "right"]].set_visible(False)
skew.ax.spines[["left", "bottom"]].set_position(("outward", 20))
skew.ax.tick_params(top=False)
skew.ax.yaxis.grid(False)

ax_diff = fig.add_subplot(gs_subplots[1], sharey=skew.ax)
ax_diff.plot(T_abs_diff, p_bco, color='black', linewidth=LW, label='|$\overline{\mathrm{T}}$ (R/V Meteor - BCO)|', zorder=10)
ax_diff.plot(dp_abs_diff, p_bco, color='gray', linewidth=LW, label='|$\overline{\mathrm{T}}_{\mathrm{d}}$ (R/V Meteor - BCO)|')


ax_diff.set_xlim(0, 4)
ax_diff.set_xticks([0,1, 2, 3, 4])
ax_diff.set_xlabel("Absolute\nTemperature Difference / K", fontsize=SIZE)
ax_diff.set_ylabel("", fontsize=SIZE)
ax_diff.spines[['top', 'right']].set_visible(False)
ax_diff.spines[['left', 'bottom']].set_position(("outward", 20))
ax_diff.tick_params(labelleft=False)

handles1, labels1 = skew.ax.get_legend_handles_labels()
handles2, labels2 = ax_diff.get_legend_handles_labels()

shading_meteor = Patch(facecolor=color_Meteor, alpha=0.2, label=r'±1$\sigma$ (R/V Meteor)')
shading_bco = Patch(facecolor=color_BCO, alpha=0.2, label=r'±1$\sigma$ (BCO)')

handles1 += [shading_bco, shading_meteor]
labels1 += [r'±1$\sigma$ (BCO)', r'±1$\sigma$ (R/V Meteor)']

skew.ax.legend(handles1, labels1,
               loc="lower center", bbox_to_anchor=(0.5, -0.35), 
               ncol=2, fontsize=SIZE, frameon=True)

ax_diff.legend(handles2, labels2,
               loc="lower center", bbox_to_anchor=(0.5, -0.35), 
               ncol=1, fontsize=SIZE, frameon=True)

ax_img.text(-0.05, 1.02, "(a)", transform=ax_img.transAxes,  fontsize=SIZE+5, va='bottom', ha='left')
skew.ax.text(-0.2, 1.05, "(b)", transform=skew.ax.transAxes, fontsize=SIZE+5)
ax_diff.text(-0.2, 1.05, "(c)", transform=ax_diff.transAxes, fontsize=SIZE+5)

filepath = './Figures/'
filename = 'Fig08_SkewT_with_Diff.svg'
plt.savefig(filepath + filename, format='svg', facecolor='white', bbox_inches="tight", dpi=300)
filename = 'Fig08_SkewT_with_Diff.png'
plt.savefig(filepath + filename, format='png', facecolor='white', bbox_inches="tight", dpi=150)
plt.show()

# %%
