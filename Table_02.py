# %%
import xarray as xr
import numpy as np

ds = xr.open_dataset("ipfs://bafybeihd6kyscsf7vzjnlivdtdd4fh5epuqqfqk7ldj6d2k634fuse2lay", engine="zarr")

ALT_NAME  = "alt"          # meters
TIME_NAME = "launch_time"  # datetime64
PLAT_NAME = "platform"
PLATFORMS = ["BCO", "RV_Meteor", "INMG"]

def fmt_time(tval):
    return np.datetime_as_string(np.asarray(tval).astype("datetime64[s]"), unit="s")

def global_extreme_with_alt(sub, da_name, extreme="min"):
    """Return (value, time, alt_km) for global min/max of sub[da_name].
       Works for 2-D variables with dims (launch_time, alt)."""
    da = sub[da_name]
    # ensure 2-D over (time, alt)
    if not ({TIME_NAME, ALT_NAME} <= set(da.dims)):
        raise ValueError(f"{da_name} must have dims ({TIME_NAME}, {ALT_NAME})")
    m = np.isfinite(da)
    if extreme == "min":
        flat = da.where(m).stack(z=(TIME_NAME, ALT_NAME)).argmin("z").item()
    else:
        flat = da.where(m).stack(z=(TIME_NAME, ALT_NAME)).argmax("z").item()
    it, iz = np.unravel_index(flat, (sub.sizes[TIME_NAME], sub.sizes[ALT_NAME]))
    val = float(da.isel({TIME_NAME: it, ALT_NAME: iz}))
    t   = sub[TIME_NAME].values[it]
    alt_km = float(sub[ALT_NAME].isel({ALT_NAME: iz})) / 1000.0
    return val, t, alt_km

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
    """Scan each profile; return (best_km, time, alt_km)."""
    best_dist = np.nan
    best_time = None
    best_alt_km = np.nan
    for it in range(sub.sizes[TIME_NAME]):
        lat = sub["lat"].isel({TIME_NAME: it}).values
        lon = sub["lon"].isel({TIME_NAME: it}).values
        dkm, iz = longest_distance_for_profile(lat, lon)
        if np.isfinite(dkm) and (np.isnan(best_dist) or dkm > best_dist):
            best_dist = dkm
            best_time = sub[TIME_NAME].values[it]
            best_alt_km = float(sub[ALT_NAME].isel({ALT_NAME: iz})) / 1000.0 if iz is not None else np.nan
    return best_dist, best_time, best_alt_km

def highest_altitude_global(sub):
    """Find, for each profile, its maximum valid altitude, then return the overall max and time."""
    # use presence of lat as validity mask (adjust if you prefer p/notnull)
    valid = sub["lat"].notnull()
    # last valid index along alt for each profile
    rev = valid.isel(alt=slice(None, None, -1))
    last_valid_from_top = rev.argmax("alt")
    iz_max = sub.sizes["alt"] - 1 - last_valid_from_top
    # altitude at that index for each profile
    prof_max_alt_m = sub[ALT_NAME].isel(alt=iz_max)
    # pick profile with largest max
    it = int(prof_max_alt_m.argmax().item())
    alt_m = float(prof_max_alt_m.isel({TIME_NAME: it}))
    t = sub[TIME_NAME].values[it]
    return alt_m / 1000.0, t

print("\nEXTREMES BY PLATFORM\n" + "-"*60)
for plat in PLATFORMS:
    sub = ds.where(ds[PLAT_NAME] == plat, drop=True)

    # Coldest temperature (K → °C)
    t_min_K, t_time, t_alt_km = global_extreme_with_alt(sub, "ta", "min")
    t_min_C = t_min_K - 273.15

    # Max & Min wind speed (m/s)
    w_max, wmax_time, wmax_alt_km = global_extreme_with_alt(sub, "wspd", "max")
    w_min, wmin_time, wmin_alt_km = global_extreme_with_alt(sub, "wspd", "min")

    # Longest distance from launch (km)
    dist_km, dist_time, dist_alt_km = longest_distance_global(sub)

    # Highest altitude achieved (km)
    alt_max_km, alt_time = highest_altitude_global(sub)

    # Lowest pressure (Pa → hPa)
    p_min_Pa, p_time, p_alt_km = global_extreme_with_alt(sub, "p", "min")
    p_min_hPa = p_min_Pa / 100.0

    print(f"\nPlatform: {plat}")
    print(f"  Coldest Temperature : {t_min_C:.1f} °C  at {fmt_time(t_time)}  ({t_alt_km:.1f} km)")
    print(f"  Max. Wind Speed     : {w_max:.1f} m/s  at {fmt_time(wmax_time)}  ({wmax_alt_km:.1f} km)")
    print(f"  Min. Wind Speed     : {w_min:.1f} m/s  at {fmt_time(wmin_time)}  ({wmin_alt_km:.1f} km)")
    print(f"  Longest Distance    : {dist_km:.1f} km  at {fmt_time(dist_time)}  ({dist_alt_km:.1f} km)")
    print(f"  Highest Altitude    : {alt_max_km:.1f} km at {fmt_time(alt_time)}")
    print(f"  Lowest Pressure     : {p_min_hPa:.1f} hPa at {fmt_time(p_time)}  ({p_alt_km:.1f} km)")


# %%
# --- Medians of flight time and distance per platform, split asc/desc ---

def flight_time_for_profile_minutes(sub):
    """Per-profile flight duration in minutes from seconds-since-launch."""
    sec_since_launch = bin_time_seconds_since_launch(sub)  # (launch_time, alt)
    # last (max) finite value along alt per profile
    dur_sec = sec_since_launch.max(dim="alt", skipna=True)
    return dur_sec / 60.0  # minutes

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
