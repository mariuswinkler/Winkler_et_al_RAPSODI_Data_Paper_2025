# %%
import xarray as xr
import numpy as np
import os
import matplotlib.pyplot as plt
from moist_thermodynamics import constants

from RAPSODI_functions import filter_profiles, calc_iwv

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
ds = xr.open_dataset("ipfs://bafybeid7cnw62zmzfgxcvc6q6fa267a7ivk2wcchbmkoyk4kdi5z2yj2w4", engine="zarr")
# %%
# Colors
color_INMG = "#BF312D"
color_Meteor = "darkblue"
color_BCO = "#F6CA4C"

# %%
def plot_iwv_histograms(
    iwv_da,                 # 1D DataArray of IWV (coords include launch_time)
    platform_da,            # 1D DataArray of platform aligned with iwv_da
    valid_height_threshold=8000,
    near_surface_min_pts=50,
    max_missing_frac=0.20,
    color_BCO="#F6CA4C",
    color_Meteor="darkblue",
    color_INMG="#BF312D",
    filepath="./Figures/"
):
    """
    Make per-platform IWV histograms with mean markers and save SVG/PNG.

    Parameters
    ----------
    iwv_da : xr.DataArray
        1D, name 'IWV' preferred. Units: kg m^-2.
    platform_da : xr.DataArray
        1D, same length/order as iwv_da. Values: 'BCO', 'RV_Meteor', 'INMG'.
    """

    os.makedirs(filepath, exist_ok=True)

    platform_colors = {"BCO": color_BCO, "RV_Meteor": color_Meteor, "INMG": color_INMG}
    platform_names  = {"BCO": "BCO", "RV_Meteor": "R/V Meteor", "INMG": "INMG"}

    IWV_values = iwv_da
    filtered_ds_platform = platform_da

    fig, ax = plt.subplots(figsize=(10, 5))
    for platform in platform_colors:
        mask = (filtered_ds_platform == platform)
        platform_iwv = IWV_values.where(mask, drop=True)

        ax.hist(platform_iwv, bins=25, density=True, alpha=0.1,
                color=platform_colors[platform], histtype="stepfilled")
        ax.hist(platform_iwv, bins=25, density=True, alpha=1,
                label=f"{platform_names[platform]}",
                histtype="step", linewidth=2, color=platform_colors[platform], clip_on=False)

        if platform_iwv.size > 0:
            mean_iwv = float(platform_iwv.mean().item())
            ax.vlines(mean_iwv, ymin=-0.0095, ymax=-0.004,
                      color=platform_colors[platform], linewidth=3, zorder=10, clip_on=False)
            print(platform, "Mean IWV", mean_iwv)

    ax.set_xlim(30, 75)
    ax.set_ylim(0, 0.12)
    ax.set_xlabel("IWV / kgm$^{-2}$")
    ax.set_ylabel("Probability Density")
    ax.spines[["left", "bottom"]].set_position(("outward", 20))
    ax.spines[["right", "top"]].set_visible(False)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.25),
              fancybox=True, shadow=False, ncol=5)

    fname_base = f'Fig05_histo_valid_altitude_threshold={valid_height_threshold//1000}km_ns{near_surface_min_pts}_miss{int(max_missing_frac*100)}'
    plt.savefig(os.path.join(filepath, fname_base + ".svg"),
                format="svg", facecolor="white", bbox_inches="tight", dpi=200)
    plt.savefig(os.path.join(filepath, fname_base + ".png"),
                format="png", facecolor="white", bbox_inches="tight", dpi=200)
    plt.show()
    plt.close()

# %%
# 1) Filter first (Steps 1–4)
ds_filtered = filter_profiles(
    ds,
    valid_height_threshold=8000,  # Step 2 threshold in meters
    near_surface_h=1000,            # Step 4: near-surface height range (m)
    near_surface_min_pts=50,        # Step 4: min points below 1 km
    max_missing_frac=0.20,          # Step 3: allow up to 20% missing
    alt_dim="height"                   # Name of height dimension
)

# %%
# 2) Compute IWV on the filtered set
iwv_ds = calc_iwv(
    ds_filtered,
    sonde_dim="launch_time",
    alt_dim="height",
    max_surface_gap_m=300,
    vertical_resolution_m=10,
)

# %%
plot_iwv_histograms(
    iwv_da=iwv_ds["iwv"],
    platform_da=ds_filtered["platform"],
    valid_height_threshold=8000,
    near_surface_min_pts=50,
    max_missing_frac=0.20,
    color_BCO="#F6CA4C",
    color_Meteor="darkblue",
    color_INMG="#BF312D",
    filepath="./Figures/"
)
# %%
