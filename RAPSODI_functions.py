import numpy as np
import xarray as xr

Rd = 287.05  # J/(kg*K)

def filter_profiles(
    ds,
    valid_height_threshold=8000,   # m
    near_surface_h=1000,             # m
    near_surface_min_pts=50,         # at least this many valid levels in first 1 km
    max_missing_frac=0.20,           # ≤ 20% missing in 0–8 km
    alt_dim="height",
):
    """
    Apply Steps 1–4 profile filters and return the filtered dataset.
    Prints how many profiles are left (absolute + percentage) after each step.
    """

    initial_count = ds.sizes.get("launch_time", 0)
    print(f"Initial profiles: {initial_count} (100.0%)")

    def pct(count):
        return f"{(count / initial_count * 100):.1f}%" if initial_count > 0 else "n/a"

    # ---- Step 1: remove early RV_Meteor profiles
    early_meteor = (ds.platform == "RV_Meteor") & (ds.launch_time < np.datetime64("2024-08-16"))
    ds1 = ds.where(~early_meteor, drop=True)
    c1 = ds1.sizes.get("launch_time", 0)
    print(f"After Step 1 (remove early RV_Meteor): {c1} ({pct(c1)})")

    # Helper: OR across available variables for "has any data"
    def has_any_valid(dsub):
        masks = []
        for v in ("q", "p", "ta"):
            if v in dsub:
                masks.append(dsub[v].notnull().any(alt_dim))
        if not masks:
            raise ValueError("Dataset must contain at least one of 'q', 'p', or 'ta'.")
        out = masks[0]
        for m in masks[1:]:
            out = out | m
        return out

    # ---- Step 2: profile-extent
    below = ds1.sel({alt_dim: slice(0, valid_height_threshold)})
    above = ds1.sel({alt_dim: slice(valid_height_threshold, None)})
    has_below = has_any_valid(below)
    has_above = has_any_valid(above)
    extent_ok = has_below & has_above
    ds2 = ds1.where(extent_ok, drop=True)
    c2 = ds2.sizes.get("launch_time", 0)
    print(f"After Step 2 (extent >= {valid_height_threshold} m): {c2} ({pct(c2)})")

    # ---- Step 3: profile-sparsity in 0–8 km
    rng = ds2.sel({alt_dim: slice(0, valid_height_threshold)})
    triplet_valid = rng.q.notnull() & rng.p.notnull() & rng.ta.notnull()
    valid_counts = triplet_valid.sum(alt_dim)  # per-profile valid count

    total_levels = rng[alt_dim].notnull().sum(alt_dim)
    sparsity_ok = valid_counts >= ((1 - max_missing_frac) * total_levels)
    ds3 = ds2.where(sparsity_ok, drop=True)
    c3 = ds3.sizes.get("launch_time", 0)
    print(f"After Step 3 (≤ {max_missing_frac*100:.0f}% missing in 0–8 km): {c3} ({pct(c3)})")

    # ---- Step 4: near-surface-coverage
    low = ds3.sel({alt_dim: slice(0, near_surface_h)})
    low_triplet_valid = low.q.notnull() & low.p.notnull() & low.ta.notnull()
    near_surface_ok = (low_triplet_valid.sum(alt_dim) >= near_surface_min_pts)
    ds4 = ds3.where(near_surface_ok, drop=True)
    c4 = ds4.sizes.get("launch_time", 0)
    print(f"After Step 4 (>= {near_surface_min_pts} points in first {near_surface_h} m): {c4} ({pct(c4)})")

    return ds4




def calc_iwv(
    ds,
    sonde_dim="launch_time",
    alt_dim="height",
    max_surface_gap_m=300,
    vertical_resolution_m=10,
):
    """
    Compute Integrated Water Vapor (IWV) from radiosonde profiles
    and merge it back into the input dataset.
    """

    # 1) Interpolate q (linear)
    q_interp = ds.q.interpolate_na(dim=alt_dim, method="linear")

    # 2) Interpolate p (log-linear), then extrapolate ends
    log_p = np.log(ds.p)
    p_interp = np.exp(log_p.interpolate_na(dim=alt_dim, method="linear"))
    p_interp = p_interp.ffill(alt_dim).bfill(alt_dim)

    # 3) Interpolate ta (linear), then extrapolate ends
    ta_interp = ds.ta.interpolate_na(dim=alt_dim, method="linear")
    ta_interp = ta_interp.ffill(alt_dim).bfill(alt_dim)

    # 4) Backfill q near surface up to max_surface_gap_m
    limit_levels = int(max(0, np.floor(max_surface_gap_m / float(vertical_resolution_m))))
    q_fill = q_interp.bfill(dim=alt_dim, limit=limit_levels)

    # 5) Identify profiles where surface is still NaN after fill
    surface_nan = q_fill.isel({alt_dim: 0}).isnull()

    # 6) Compute Tv, density, and IWV
    Tv = ta_interp * (1.0 + 0.61 * q_fill)
    rho = p_interp / (Rd * Tv)

    # Integrate q * rho over height (ignoring NaNs)
    iwv_vals = (q_fill * rho).fillna(0).integrate(alt_dim)

    # Mask IWV for profiles with bad surface
    iwv_vals = iwv_vals.where(~surface_nan)

    # Add metadata
    iwv_vals.name = "iwv"
    iwv_vals.attrs.update({
        "standard_name": "atmosphere_mass_content_of_water_vapor",
        "long_name": "Integrated water vapor",
        "units": "kg m-2",
        "cell_methods": f"{alt_dim}: sum",
        "ancillary_variables": "q p ta",
        "coordinates": "launch_time lat lon",
        "comment": (
            "Computed as vertical integral of specific humidity times air density "
            "over height. q, ta interpolated linearly; p interpolated log-linearly "
            "with extrapolation. Near-surface q backfilled up to max_surface_gap_m; "
            "profiles with missing near-surface q are masked."
        ),
    })

    # Merge into the original dataset
    ds_out = ds.copy()
    ds_out["iwv"] = iwv_vals

    return ds_out



def add_IWV(
    ds,
    valid_height_threshold=8000,   # m
    near_surface_h=1000,             # m
    near_surface_min_pts=50,         # at least this many valid levels in first 1 km
    max_missing_frac=0.20,           # ≤ 20% missing in 0–8 km
    alt_dim="height",
    sonde_dim="launch_time",         # <- match your dataset
    max_surface_gap_m=300,
    vertical_resolution_m=10,
):
    """
    Apply the exact same filtering (Steps 1–4) and IWV computation
    as used in the plotting workflow, then merge 'iwv' back into `ds`.
    """

    # 1) Filter with the same routine
    ds_filtered = filter_profiles(
        ds,
        valid_height_threshold=valid_height_threshold,
        near_surface_h=near_surface_h,
        near_surface_min_pts=near_surface_min_pts,
        max_missing_frac=max_missing_frac,
        alt_dim=alt_dim,
    )

    # If nothing left after filtering, just return the original ds
    if ds_filtered.sizes.get(sonde_dim, 0) == 0:
        print("No profiles left after filtering. Returning original dataset without IWV.")
        return ds

    # 2) Compute IWV with the same routine
    iwv_ds = calc_iwv(
        ds_filtered,
        sonde_dim=sonde_dim,
        alt_dim=alt_dim,
        max_surface_gap_m=max_surface_gap_m,
        vertical_resolution_m=vertical_resolution_m,
    )

    # 3) Merge IWV back into the original dataset (aligned by coords)
    ds_out = ds.copy()
    ds_out["iwv"] = iwv_ds["iwv"]

    return ds_out


def filter_oscillating_launch_times(ds, time_var="launch_time", oscillating_times=None):
    oscillating_launch_times = [
        "2024-08-18T01:50:36", # "FS_METEOR_20240818_015036.mwx"
        "2024-08-21T13:50:33", # "ZVQEQCM_20240821_135033.mwx"
        "2024-08-25T19:50:19", # "FS_METEOR_20240825_195019.mwx"
        "2024-09-07T13:50:07", # "FS_METEOR_20240907_135007.mwx"
        "2024-09-11T07:50:20", # "FS_METEOR_20240911_075020.mwx"
    ] # those are the times Hauke gave me. They don't match exactly.
    launch_times_truncated = np.datetime_as_string(ds[time_var], unit="s")
    mask = ~np.isin(launch_times_truncated, oscillating_launch_times) 
    return ds.sel(sounding=mask)

