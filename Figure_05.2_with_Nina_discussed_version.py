# %%
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from moist_thermodynamics import constants

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
ds = xr.open_dataset("ipns://latest.orcestra-campaign.org/products/Radiosondes/RAPSODI_RS_ORCESTRA_level2.zarr", engine="zarr")

# %%
# Colors
color_INMG = "#BF312D"
color_Meteor = "darkblue"
color_BCO = "#F6CA4C"

# %%
def IWV_computation(
    ds,
    valid_altitude_threshold=8000,   # Step 1: must reach ≥ this altitude (m)
    near_surface_h=1000,             # Step 2 window (m)
    near_surface_min_pts=50,         # Step 2: at least this many valid levels in first 1 km
    max_missing_frac=0.20            # Step 3: ≤ 20% missing (0–valid_altitude_threshold)
):

    # -----------------------------
    # Step 0: Filter out R/V Meteor launches before Aug 16 (Mindelo port)
    # -----------------------------
    early_meteor = (ds.platform == "RV_Meteor") & (ds.launch_time < np.datetime64("2024-08-16"))
    ds = ds.where(~early_meteor, drop=True)

    # -----------------------------
    # Build masks for Steps 1–3
    # -----------------------------

    # Step 1: reaches valid_altitude_threshold with ANY valid P/T/q value above that level
    above = ds.sel(alt=slice(valid_altitude_threshold, None))
    reaches_top = (
        above.q.notnull().any("alt") |
        above.p.notnull().any("alt") |
        above.ta.notnull().any("alt")
    )

    # Step 2: near-surface coverage — at least N valid triplets (q, p, ta) in first 1 km
    low = ds.sel(alt=slice(0, near_surface_h))
    low_triplet_valid = low.q.notnull() & low.p.notnull() & low.ta.notnull()
    near_surface_ok = (low_triplet_valid.sum("alt") >= near_surface_min_pts)

    # Step 3: profile sparsity — within 0 to valid_altitude_threshold, ≤ 20% missing triplets
    rng = ds.sel(alt=slice(0, valid_altitude_threshold))
    rng_triplet_valid = rng.q.notnull() & rng.p.notnull() & rng.ta.notnull()
    total_bins = rng_triplet_valid.sizes.get("alt", 0)
    valid_counts = rng_triplet_valid.sum("alt")
    missing_frac = 1.0 - (valid_counts / total_bins)
    sparsity_ok = (missing_frac <= max_missing_frac)

    # Combine all masks
    keep = reaches_top & near_surface_ok & sparsity_ok

    filtered_ds = ds.where(keep, drop=True)

    # Interpolate remaining small gaps along the full profile
    for var in ["q", "p", "ta", "rh"]:
        if var in filtered_ds:
            filtered_ds[var] = filtered_ds[var].interpolate_na(dim="alt", method="cubic")

    # -----------------------------
    # IWV computation
    # -----------------------------
    g = constants.gravity_earth
    Rd = constants.Rd

    Tv = filtered_ds.ta * (1.0 + 0.61 * filtered_ds.q)
    rho = filtered_ds.p / (Rd * Tv)
    rho = rho.interpolate_na(dim="alt", method="cubic")

    IWV_values = ((filtered_ds.q * rho).fillna(0)).integrate("alt")  # kg m^-2

    # Report how many remain
    n_initial = ds.launch_time.size
    n_remaining = filtered_ds.launch_time.size
    remaining_percentage = 100.0 * n_remaining / n_initial if n_initial else 0.0
    print(f"Remaining soundings after filtering: {n_remaining} / {n_initial} ({remaining_percentage:.1f}%)")

    # -----------------------------
    # Plot
    # -----------------------------
    platform_colors = {"BCO": color_BCO, "RV_Meteor": color_Meteor, "INMG": color_INMG}
    platform_names = {"BCO": "BCO", "RV_Meteor": "R/V Meteor", "INMG": "INMG"}

    fig, ax = plt.subplots(figsize=(10, 5))
    for platform in platform_colors:
        mask = filtered_ds.platform == platform
        platform_iwv = IWV_values.where(mask, drop=True)

        ax.hist(platform_iwv, bins=25, density=True, alpha=0.1,
                color=platform_colors[platform], histtype='stepfilled')
        ax.hist(platform_iwv, bins=25, density=True, alpha=1,
                label=f"{platform_names[platform]}",
                histtype='step', linewidth=2, color=platform_colors[platform])

        # Bold x-axis tick for mean
        if platform_iwv.size > 0:
            mean_iwv = platform_iwv.mean().item()
            ax.vlines(mean_iwv, ymin=-0.0095, ymax=-0.004,
                      color=platform_colors[platform], linewidth=3, zorder=10, clip_on=False)
            print(platform, "Mean IWV", mean_iwv)

    ax.set_xlim(30, 70)
    ax.set_ylim(0, 0.13)
    ax.set_xlabel("IWV / kgm$^{-2}$")
    ax.set_ylabel("Probability Density")
    ax.spines[["left", "bottom"]].set_position(("outward", 20))
    ax.spines[["right", "top"]].set_visible(False)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.25), fancybox=True, shadow=False, ncol=5)

    filepath = './Figures/'
    filename = f'Fig05_histo_valid_altitude_threshold={valid_altitude_threshold//1000}km_ns{near_surface_min_pts}_miss{int(max_missing_frac*100)}.svg'
    plt.savefig(filepath + filename, format='svg', facecolor='white', bbox_inches="tight", dpi=200)
    filename = f'Fig05_histo_valid_altitude_threshold={valid_altitude_threshold//1000}km_ns{near_surface_min_pts}_miss{int(max_missing_frac*100)}.png'
    plt.savefig(filepath + filename, format='png', facecolor='white', bbox_inches="tight", dpi=200)
    plt.show()
    plt.close()

# %%
IWV_computation(
    ds,
    valid_altitude_threshold=8000,  # Step 1
    near_surface_h=1000,            # Step 2
    near_surface_min_pts=50,        # Step 2
    max_missing_frac=0.20           # Step 3
)
# %%
