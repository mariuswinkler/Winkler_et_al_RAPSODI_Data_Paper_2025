# %%
import xarray as xr
import numpy as np

ds = xr.open_dataset("ipfs://bafybeid7cnw62zmzfgxcvc6q6fa267a7ivk2wcchbmkoyk4kdi5z2yj2w4", engine="zarr")

HEIGHT_NAME  = "height"          # meters
TIME_NAME = "launch_time"  # datetime64
PLAT_NAME = "platform"
PLATFORMS = ["BCO", "RV_Meteor", "INMG"]

def fmt_time(tval):
    return np.datetime_as_string(np.asarray(tval).astype("datetime64[s]"), unit="s")

def global_extreme_with_height(sub, da_name, extreme="min"):
    """Return (value, time, height_km) for global min/max of sub[da_name].
       Works for 2-D variables with dims (launch_time, height)."""
    da = sub[da_name]
    # ensure 2-D over (time, height)
    if not ({TIME_NAME, HEIGHT_NAME} <= set(da.dims)):
        raise ValueError(f"{da_name} must have dims ({TIME_NAME}, {HEIGHT_NAME})")
    m = np.isfinite(da)
    if extreme == "min":
        flat = da.where(m).stack(z=(TIME_NAME, HEIGHT_NAME)).argmin("z").item()
    else:
        flat = da.where(m).stack(z=(TIME_NAME, HEIGHT_NAME)).argmax("z").item()
    it, iz = np.unravel_index(flat, (sub.sizes[TIME_NAME], sub.sizes[HEIGHT_NAME]))
    val = float(da.isel({TIME_NAME: it, HEIGHT_NAME: iz}))
    t   = sub[TIME_NAME].values[it]
    height_km = float(sub[HEIGHT_NAME].isel({HEIGHT_NAME: iz})) / 1000.0
    return val, t, height_km

def longest_distance_for_profile(lat, lon):
    """Great-circle distance (km) from first valid point to all points; returns (max_km, iz)."""
    ok0 = np.isfinite(lat) & np.isfinite(lon)
    if not np.any(ok0):
        return np.nan, None
    i0 = np.argmax(ok0)  # first valid index
    lat0 = np.deg2rad(lat[i0]); lon0 = np.deg2rad(lon[i0])
    latr = np.deg2rad(lat[ok0]); lonr = np.deg2rad(lon[ok0])
    dlat = latr - lat0
    dlon = lonr - lon0
    a = np.sin(dlat/2.0)**2 + np.cos(lat0) * np.cos(latr) * np.sin(dlon/2.0)**2
    c = 2.0 * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a))
    R_earth_km = 6371.0
    dkm_valid = R_earth_km * c
    jmax = int(np.nanargmax(dkm_valid))
    dkm = float(dkm_valid[jmax])
    iz = np.arange(len(lat))[ok0][jmax]
    return dkm, int(iz)

def longest_distance_global(sub):
    """Scan each profile; return (best_km, time, height_km)."""
    best_dist = np.nan
    best_time = None
    best_height_km = np.nan
    for it in range(sub.sizes[TIME_NAME]):
        lat = sub["lat"].isel({TIME_NAME: it}).values
        lon = sub["lon"].isel({TIME_NAME: it}).values
        dkm, iz = longest_distance_for_profile(lat, lon)
        if np.isfinite(dkm) and (np.isnan(best_dist) or dkm > best_dist):
            best_dist = dkm
            best_time = sub[TIME_NAME].values[it]
            best_height_km = float(sub[HEIGHT_NAME].isel({HEIGHT_NAME: iz})) / 1000.0 if iz is not None else np.nan
    return best_dist, best_time, best_height_km

def highest_height_global(sub):
    """Find, for each profile, its maximum valid height, then return the overall max and time."""
    # use presence of lat as validity mask (adjust if you prefer p/notnull)
    valid = sub["lat"].notnull()
    # last valid index along height for each profile
    rev = valid.isel(height=slice(None, None, -1))
    last_valid_from_top = rev.argmax("height")
    iz_max = sub.sizes["height"] - 1 - last_valid_from_top
    # height at that index for each profile
    prof_max_height_m = sub[HEIGHT_NAME].isel(height=iz_max)
    # pick profile with largest max
    it = int(prof_max_height_m.argmax().item())
    height_m = float(prof_max_height_m.isel({TIME_NAME: it}))
    t = sub[TIME_NAME].values[it]
    return height_m / 1000.0, t

def bin_time_seconds_since_launch(sub):
    """
    Return time since launch in seconds for each profile.
    Expects a time coordinate varying along 'height' for each 'launch_time'.
    """
    if "time" not in sub:
        raise ValueError("Dataset must contain a 'time' variable for computing seconds since launch.")
    
    # Convert to numpy datetime64[s]
    time_vals = sub["time"].values.astype("datetime64[s]")
    # Get the launch times (one per profile)
    launch_vals = sub[TIME_NAME].values.astype("datetime64[s]")
    
    # Broadcast to shape (launch_time, height)
    t = np.broadcast_to(time_vals, (sub.sizes[TIME_NAME], sub.sizes[HEIGHT_NAME]))
    t0 = launch_vals[:, np.newaxis]
    
    # Compute seconds difference
    dt_sec = (t - t0).astype("timedelta64[s]").astype(float)
    return xr.DataArray(dt_sec, dims=(TIME_NAME, HEIGHT_NAME))


print("\nEXTREMES BY PLATFORM\n" + "-"*60)
for plat in PLATFORMS:
    sub = ds.where(ds[PLAT_NAME] == plat, drop=True)

    # Coldest temperature (K → °C)
    t_min_K, t_time, t_height_km = global_extreme_with_height(sub, "ta", "min")
    t_min_C = t_min_K - 273.15

    # Max & Min wind speed (m/s)
    w_max, wmax_time, wmax_height_km = global_extreme_with_height(sub, "wspd", "max")
    w_min, wmin_time, wmin_height_km = global_extreme_with_height(sub, "wspd", "min")

    # Longest distance from launch (km)
    dist_km, dist_time, dist_height_km = longest_distance_global(sub)

    # Highest height achieved (km)
    height_max_km, height_time = highest_height_global(sub)

    # Lowest pressure (Pa → hPa)
    p_min_Pa, p_time, p_height_km = global_extreme_with_height(sub, "p", "min")
    p_min_hPa = p_min_Pa / 100.0

    print(f"\nPlatform: {plat}")
    print(f"  Coldest Temperature : {t_min_C:.1f} °C  at {fmt_time(t_time)}  ({t_height_km:.1f} km)")
    print(f"  Max. Wind Speed     : {w_max:.1f} m/s  at {fmt_time(wmax_time)}  ({wmax_height_km:.1f} km)")
    print(f"  Min. Wind Speed     : {w_min:.1f} m/s  at {fmt_time(wmin_time)}  ({wmin_height_km:.1f} km)")
    print(f"  Longest Distance    : {dist_km:.1f} km  at {fmt_time(dist_time)}  ({dist_height_km:.1f} km)")
    print(f"  Highest Height    : {height_max_km:.1f} km at {fmt_time(height_time)}")
    print(f"  Lowest Pressure     : {p_min_hPa:.1f} hPa at {fmt_time(p_time)}  ({p_height_km:.1f} km)")


# %%
# --- Medians of flight time and distance per platform, split asc/desc ---

def flight_time_for_profile_minutes(sub):
    """
    Per-profile flight duration (minutes) for the given subset.
    Uses max(interpolated_time - launch_time) along height.
    Returns 1D DataArray over launch_time.
    """
    if "interpolated_time" not in sub:
        raise ValueError("Expected 'interpolated_time' with dims (launch_time, height).")

    it = sub["interpolated_time"]                     # datetime64, (launch_time, height)
    lt = sub[TIME_NAME]                               # datetime64, (launch_time,)

    # Broadcast launch_time to the same shape as interpolated_time
    lt2d = xr.DataArray(lt).broadcast_like(it)

    # Timedelta array (no astype to datetime64[...]!)
    dt = it - lt2d                                    # timedelta64, (launch_time, height)

    # Convert to seconds WITHOUT astype: divide by 1 second
    dt_sec = dt / np.timedelta64(1, "s")              # float seconds

    # Duration per profile = max along height (ignore NaNs)
    dur_sec = dt_sec.max(dim=HEIGHT_NAME, skipna=True)

    return (dur_sec / 60.0).astype(float)             # minutes




def distance_for_profile_km(sub):
    """Per-profile max great-circle distance (km) from first valid point."""
    dists = []
    for it in range(sub.sizes[TIME_NAME]):
        lat = sub["lat"].isel({TIME_NAME: it}).values
        lon = sub["lon"].isel({TIME_NAME: it}).values
        dkm, _ = longest_distance_for_profile(lat, lon)
        dists.append(dkm)
    return np.array(dists)

print("\nMEDIAN FLIGHT TIMES AND DISTANCES (asc/desc)\n" + "-"*60)
for plat in PLATFORMS:
    for flag, label in [(0, "ascending"), (1, "descending")]:
        sub = ds.where((ds[PLAT_NAME] == plat) & (ds["ascent_flag"] == flag), drop=True)

        if sub[TIME_NAME].size == 0:
            continue  # no profiles of this type

        durations_min = flight_time_for_profile_minutes(sub).values
        distances_km  = distance_for_profile_km(sub)

        med_time_min = np.nanmedian(durations_min)
        med_dist_km  = np.nanmedian(distances_km)

        print(f"{plat} ({label}): median flight time = {med_time_min:.1f} min, "
              f"median distance = {med_dist_km:.1f} km")

# %%
