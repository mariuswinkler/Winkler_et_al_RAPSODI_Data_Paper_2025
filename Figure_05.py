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
def IWV_computation(ds, valid_altitude_threshold=15000, approx_alt_max=3000, gap_threshold=500):

    # Step 0: Filter out R/V Meteor soundings before August 16 (private communication with Allison Wing. These soundings were launched in the port of Mindelo)
    if "RV_Meteor" in ds.platform:
        ds = ds.where(~((ds.platform == "RV_Meteor") & (ds.launch_time < np.datetime64("2024-08-16"))), drop=True)

    # Step 1: Filter soundings that reach valid_altitude_threshold and above.
    above_threshold = ds.sel(alt=slice(valid_altitude_threshold, None))
    has_valid = (
        above_threshold.q.notnull().any(dim="alt") |
        above_threshold.p.notnull().any(dim="alt") |
        above_threshold.ta.notnull().any(dim="alt")
    )
    top_cut_ds = ds.where(has_valid, drop=False)

    # Helper function: identify first valid value
    def get_first_valid(var):
        idx = var.notnull().argmax(dim="alt")
        alt = top_cut_ds.alt.isel(alt=idx)
        val = var.isel(alt=idx)
        return idx, alt, val

    idx_q, alt_q, val_q = get_first_valid(top_cut_ds.q)
    idx_p, alt_p, val_p = get_first_valid(top_cut_ds.p)
    idx_ta, alt_ta, val_ta = get_first_valid(top_cut_ds.ta)

    mask_q = alt_q < approx_alt_max
    mask_p = alt_p < approx_alt_max
    mask_ta = alt_ta < approx_alt_max

    val_q = val_q.where(mask_q)
    val_p = val_p.where(mask_p)
    val_ta = val_ta.where(mask_ta)

    dz = top_cut_ds.alt - alt_ta

    g = constants.gravity_earth
    Rd = constants.Rd

    lapse_rate = (
        (top_cut_ds.ta.isel(alt=idx_ta + 1) - val_ta) /
        (top_cut_ds.alt.isel(alt=idx_ta + 1) - alt_ta)
    ).where(mask_ta)

    # Step 2: Fills q, p, ta from at the surface for nan-values below approx_alt_max.
    top_cut_ds["q"] = xr.where((top_cut_ds.alt < alt_q) & mask_q, val_q, top_cut_ds.q)
    top_cut_ds["p"] = xr.where((top_cut_ds.alt < alt_p) & mask_p,
                                val_p * np.exp(- (g / (Rd * val_ta)) * dz), top_cut_ds.p)
    top_cut_ds["ta"] = xr.where((top_cut_ds.alt < alt_ta) & mask_ta,
                                 val_ta - lapse_rate * dz, top_cut_ds.ta)

    # Step 3: Identify and accumulate gaps below valid_altitude_threshold. Throw out profiles with gaps larger than gap_threshold.
    below_threshold = top_cut_ds.sel(alt=slice(0, valid_altitude_threshold))
    nan_mask = below_threshold.q.isnull() | below_threshold.p.isnull() | below_threshold.ta.isnull()
    alt_diff = below_threshold.alt.diff(dim="alt", label="lower").fillna(0)
    nan_gaps = (nan_mask * alt_diff).where(nan_mask)

    def accumulate_gaps_1d(arr):
        out = np.zeros_like(arr)
        acc = 0
        for i, val in enumerate(arr):
            if np.isnan(val): acc = 0
            elif val == 10: acc += 10
            else: acc = 0
            out[i] = acc
        return out

    accumulated = xr.apply_ufunc(
        accumulate_gaps_1d, nan_gaps,
        input_core_dims=[["alt"]], output_core_dims=[["alt"]],
        vectorize=True, dask="parallelized", output_dtypes=[np.float32]
    )
    
    mask = (accumulated > gap_threshold).any(dim="alt")
    filtered_ds = top_cut_ds.where(~mask, drop=False)
 
    # Step 4: Interpolate remaining small gaps along full profiles.
    for var in ["q", "p", "ta", "rh"]:
        if var in filtered_ds:
            filtered_ds[var] = filtered_ds[var].interpolate_na(dim="alt", method="cubic")

    # Step 5: Compute IWV
    Tv = filtered_ds.ta * (1 + 0.61 * filtered_ds.q)
    rho = filtered_ds.p / (Rd * Tv)
    rho = rho.interpolate_na(dim="alt", method="cubic")
    IWV_values = ((filtered_ds.q * rho).fillna(0)).integrate("alt")

    # Step 5.5: Report percentage of remaining soundings
    n_initial = ds.launch_time.size
    n_remaining = filtered_ds.q.notnull().any(dim="alt").sum().item()
    remaining_percentage = 100 * n_remaining / n_initial
    
    print(f"Remaining soundings after filtering: {n_remaining} / {n_initial} ({remaining_percentage:.1f}%)")

    # Step 6: Plot
    platform_colors = {"BCO": color_BCO, "RV_Meteor": color_Meteor, "INMG": color_INMG}
    platform_names = {"BCO": "BCO", "RV_Meteor": "R/V Meteor", "INMG": "INMG"}

    fig, ax = plt.subplots(figsize=(10, 5))
    for platform in platform_colors:
        mask = filtered_ds.platform == platform
        platform_iwv = IWV_values.where(mask, drop=True)
        percentage = 100 * len(platform_iwv.launch_time) / len(ds.launch_time)

        ax.hist(platform_iwv, bins=25, density=True, alpha=0.3,
                color=platform_colors[platform], histtype='stepfilled')
        ax.hist(platform_iwv, bins=25, density=True, alpha=1,
                label=f"{platform_names[platform]}",
                histtype='step', linewidth=2, color=platform_colors[platform])
        # Add bold x-tick marker for the mean of each distribution
        mean_iwv = platform_iwv.mean().item()
        ax.vlines(mean_iwv, ymin=-0.0095, ymax=-0.004, color=platform_colors[platform],
                  linewidth=3, zorder=10, clip_on=False)
        print(platform, "Mean IWV", mean_iwv)

    #ax.axvline(48, color='black', ls='dashed', lw=2)
    ax.set_xlim(30, 70)
    ax.set_ylim(0, 0.13)
    ax.set_xlabel("IWV / kgm$^{-2}$")
    ax.set_ylabel("Probability Density")
    ax.spines[["left", "bottom"]].set_position(("outward", 20))
    ax.spines[["right", "top"]].set_visible(False)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.25), fancybox=True, shadow=False, ncol=5)

    filepath = './Figures/'
    filename = f'Fig05_histo_valid_altitude_threshold={valid_altitude_threshold//1000}km_approx_alt_max={approx_alt_max}m_gap_threshold={gap_threshold}m.svg'
    plt.savefig(filepath + filename, format='svg', facecolor='white', bbox_inches="tight", dpi=200)
    filename = f'Fig05_histo_valid_altitude_threshold={valid_altitude_threshold//1000}km_approx_alt_max={approx_alt_max}m_gap_threshold={gap_threshold}m.png'
    plt.savefig(filepath + filename, format='png', facecolor='white', bbox_inches="tight", dpi=200)
    plt.show()
    plt.close()
# %%
IWV_computation(
    ds,
    valid_altitude_threshold=8000,
    approx_alt_max=600,
    gap_threshold=500
)
# %%