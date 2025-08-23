# %%
import xarray as xr
import numpy as np
import pandas as pd

ds = xr.open_dataset("ipfs://bafybeihd6kyscsf7vzjnlivdtdd4fh5epuqqfqk7ldj6d2k634fuse2lay", engine="zarr")

# %%
def platform_summary(ds, platform):
    ds_plat = ds.where(ds.platform == platform, drop=True)

    # number of ascents and descents
    n_ascent = int((ds_plat.ascent_flag == 0).sum().item())
    n_descent = int((ds_plat.ascent_flag == 1).sum().item())

    # start and end date
    start_date = str(pd.to_datetime(ds_plat.launch_time.values.min()).date())
    end_date   = str(pd.to_datetime(ds_plat.launch_time.values.max()).date())

    # platform altitude (mean over all soundings)
    plat_alt = float(ds_plat.alt.isel(alt=0).mean().item())

    return {
        "Platform": platform,
        "Number of ascents": n_ascent,
        "Number of descents": n_descent,
        "Start date": start_date,
        "End date": end_date,
        "Platform altitude / m": plat_alt,
    }

# Apply for each platform
summary_INMG   = platform_summary(ds, "INMG")
summary_METEOR = platform_summary(ds, "RV_Meteor")
summary_BCO    = platform_summary(ds, "BCO")

summary_INMG, summary_METEOR, summary_BCO

# %%

def overall_descent_stats(ds, threshold_hpa=980.0):
    """
    Prints:
      (1) Successful descents in % relative to all ascents across all platforms.
      (2) % of descent profiles that reached at least 'threshold_hpa' near-surface pressure.
    """
    # Counts for success rate
    n_ascent = int((ds.ascent_flag == 0).sum().item())
    n_descent = int((ds.ascent_flag == 1).sum().item())

    success_pct = float("nan") if n_ascent == 0 else (n_descent / n_ascent) * 100.0

    # Consider only descent profiles
    ds_desc = ds.where(ds.ascent_flag == 1, drop=True)

    # Number of descent profiles
    n_desc_profiles = ds_desc.sizes.get("launch_time", 0)

    # % of descents that reached >= threshold_hpa (near-surface) pressure at any level
    reached_mask = (ds_desc["p"].max(dim="alt") >= threshold_hpa * 100.0)
    n_reached = int(reached_mask.sum().item()) if n_desc_profiles > 0 else 0
    reached_pct = float("nan") if n_desc_profiles == 0 else (n_reached / n_desc_profiles) * 100.0

    print(f"[Total] successful descents relative to ascents: {success_pct:.1f}% "
          f"({n_descent}/{n_ascent})")
    print(f"[Total] descents reaching >= {threshold_hpa:.0f} hPa: {reached_pct:.1f}% "
          f"({n_reached}/{n_desc_profiles})")


# --- Apply for all platforms combined ---
overall_descent_stats(ds, threshold_hpa=980.0)

# %%
