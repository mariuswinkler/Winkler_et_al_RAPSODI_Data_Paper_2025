# %%
import os
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(SCRIPT_DIR)

# %%
import re
import numpy as np
import xarray as xr
from RAPSODI_functions import add_IWV

# Load the datasets
DS_VERSION = 'v4.0.7'
DS_directory = '../level2/merged_dataset/'
DS = xr.open_dataset(DS_directory+f"RS_ORCESTRA_level2_{DS_VERSION}_raw.nc")
DS

# %%
def normalize_sonde_id(ds: xr.Dataset) -> xr.Dataset:
    """
    Shorten and normalize sonde_id IDs of the form:
      "<platform>__<ascent|descent>__<lat>_<lon>__<YYYYMMDDhhmm>"
    → "<platform>_<ascent|descent>_<YYYYMMDDhhmm>"

    Robust to extra/missing segments; ignores whatever is between
    direction and timestamp. Ensures unique IDs if collisions occur.
    """
    if "sonde_id" not in ds.coords:
        return ds

    old_attrs = dict(ds["sonde_id"].attrs)
    vals = ds["sonde_id"].values.astype(str)

    def _shorten(s: str) -> str:
        parts = s.strip().split("__")
        plat_raw = parts[0] if parts else s
        dir_chunk = next((p for p in parts if p.lower() in ("ascent", "descent")), None)
        ts = None
        for p in reversed(parts):
            m = re.search(r"(\d{12})", p)
            if m:
                ts = m.group(1)
                break

        if dir_chunk and ts:
            plat = re.sub(r"\s+", "_", plat_raw.strip())
            out = f"{plat}_{dir_chunk}_{ts}"
        else:
            out = re.sub(r"\s+", "_", s.strip())

        out = re.sub(r"_+", "_", out)
        return out

    seen = {}
    out = []
    for s in vals:
        new = _shorten(s)
        if new in seen:
            seen[new] += 1
            new = f"{new}_{seen[new]}"
        else:
            seen[new] = 0
        out.append(new)

    maxlen = max(len(x) for x in out) if out else 1
    dtypeU = f"U{maxlen}"
    ds = ds.assign_coords(sonde_id=("launch_time", np.array(out, dtype=dtypeU)))
    ds["sonde_id"].attrs.update(old_attrs)
    return ds


def demote_aux_coords_strict(ds, names=("alt", "p")):
    ds = ds.reset_coords([n for n in names if n in ds.coords], drop=False)

    for var in ds.variables: 
        anc = ds[var].attrs.get("coordinates")
        if not anc:
            continue
        toks = anc.split()
        keep = [t for t in toks if t not in names]
        if keep:
            ds[var].attrs["coordinates"] = " ".join(keep)
        else:
            ds[var].attrs.pop("coordinates", None)

    if "coordinates" in ds.attrs:
        toks = ds.attrs["coordinates"].split()
        keep = [t for t in toks if t not in names]
        if keep:
            ds.attrs["coordinates"] = " ".join(keep)
        else:
            ds.attrs.pop("coordinates", None)

    return ds


_COORD_SPLIT = re.compile(r"[\s,]+")

def _scrub_cf_coordinate_references(ds: xr.Dataset, demoted: set[str]) -> xr.Dataset:
    demoted = set(map(str, demoted))

    for name in list(ds.variables):
        da = ds[name]
        val = da.attrs.get("coordinates")
        if val:
            toks = [t for t in _COORD_SPLIT.split(str(val).strip()) if t]
            kept = [t for t in toks if t not in demoted]
            if kept:
                da.attrs["coordinates"] = " ".join(kept)
            else:
                da.attrs.pop("coordinates", None)

        val = da.encoding.get("coordinates")
        if val:
            toks = [t for t in _COORD_SPLIT.split(str(val).strip()) if t]
            kept = [t for t in toks if t not in demoted]
            if kept:
                da.encoding["coordinates"] = " ".join(kept)
            else:
                da.encoding.pop("coordinates", None)

    dval = ds.attrs.get("coordinates")
    if dval:
        toks = [t for t in _COORD_SPLIT.split(str(dval).strip()) if t]
        kept = [t for t in toks if t not in demoted]
        if kept:
            ds.attrs["coordinates"] = " ".join(kept)
        else:
            ds.attrs.pop("coordinates", None)

    return ds

def drop_count_ancillary(ds,
                         banned=("N_gps", "m_gps", "m_gps_", "N_ptu", "m_ptu")):
    """
    Remove 'ancillary_variables' entries that are only N_gps/m_gps(/m_gps_)/N_ptu/m_ptu.
    If 'ancillary_variables' contains more items, drop the banned ones and keep others.
    Returns the same Dataset (modified).
    """
    banned_norm = {b.rstrip("_") for b in banned}  

    for name in ds.variables:
        anc = ds[name].attrs.get("ancillary_variables")
        if not anc:
            continue

        toks = re.findall(r"[A-Za-z0-9_]+", anc)
        keep = [t for t in toks if t.rstrip("_") not in banned_norm]

        if keep:
            ds[name].attrs["ancillary_variables"] = " ".join(keep)
        else:
            ds[name].attrs.pop("ancillary_variables", None)

    return ds


def _apply_attrs(ds, attr_map, overwrite_keys=("cell_methods",)):
    for name, add in attr_map.items():
        if name in ds.variables or name in ds.coords:
            tgt = ds[name]
            for k, v in add.items():
                if (overwrite_keys is None) or (k in overwrite_keys) or (k not in tgt.attrs):
                    tgt.attrs[k] = v
    return ds

def _strip_coord_references(ds, demoted={"p","sonde_id","alt"}):
    for name, da in ds.data_vars.items():
        coords_attr = da.attrs.get("coordinates")
        if coords_attr:
            coords = coords_attr.split()
            kept = [c for c in coords if c not in demoted]
            if kept:
                da.attrs["coordinates"] = " ".join(kept)
            else:
                da.attrs.pop("coordinates", None)
    return ds

def prep_for_ipfs(ds):
    """
    Prepare a (launch_time, height) radiosonde dataset for publishing:
      - swap to launch_time, sort
      - keep along-track lat/lon and add launch_lat/lon (1-D per profile)
      - rename flight_time->bin_average_time (seconds since launch)
      - ensure CF-ish attributes on coords and data variables
      - drop unused bounds/aux dims
    """
    if "launch_time" in ds.coords and "sounding" in ds.dims and "launch_time" not in ds.dims:
        ds = ds.swap_dims({"sounding": "launch_time"})
    if "launch_time" in ds.coords:
        ds = ds.sortby("launch_time")

    if "sounding" in ds.dims or "sounding" in ds.coords or "sounding" in ds.variables:
        ds = ds.rename({"sounding": "sonde_id"})

    if "sonde_id" in ds and "sonde_id" not in ds.coords:
        ds = ds.set_coords("sonde_id")

    ds["sonde_id"] = ds["sonde_id"].assign_attrs(
        dict(
            cf_role="profile_id",
            long_name="sonde identifier",
            description="Unique string describing the sounding's origin (PLATFORM_DIRECTION_TIME)",
        )
    )

    print(ds.sonde_id.values)
    ds = normalize_sonde_id(ds)

    if "flight_time" in ds and "interpolated_time" not in ds:
        ds = ds.rename({"flight_time": "interpolated_time"})
    if "interpolated_time" in ds:
        if "interpolated_time" not in ds.coords:
            ds = ds.set_coords(["interpolated_time"])
        ds["interpolated_time"].attrs.update(dict(
            standard_name="time",
            long_name="time of recorded measurement",
            time_zone="UTC",
            cell_methods="height: interpolated (interval: 10 m)",
            comment="seconds since launch (per-profile relative time)"
        ))

    if {"lat", "lon"}.issubset(ds.coords):
        if "height" in ds.dims:
            first_valid_lvl = ds["lat"].isnull().argmin("height")
            launch_lat = ds["lat"].isel(height=first_valid_lvl).rename(None)
            launch_lon = ds["lon"].isel(height=first_valid_lvl).rename(None)
            ds = ds.assign(launch_lat=launch_lat, launch_lon=launch_lon)
            ds["launch_lat"].attrs.update(dict(
                standard_name="latitude", long_name="launch latitude", units="degrees_north", axis="Y"
            ))
            ds["launch_lon"].attrs.update(dict(
                standard_name="longitude", long_name="launch longitude", units="degrees_east", axis="X"
            ))

    if "alt" in ds.coords:
        ds["alt"].attrs.setdefault("standard_name", "height_above_reference_ellipsoid")
        ds["alt"].attrs.setdefault("long_name", "altitude height above reference ellipsoid (WGS84)")
        ds["alt"].attrs.setdefault("units", "m")
        ds["alt"].attrs.setdefault("axis", "Z")
        ds["alt"].attrs.setdefault("positive", "up")

    for v in ("alt_bnds",):
        if v in ds.variables: ds = ds.drop_vars(v)
    if "nv" in ds.dims:
        ds = ds.drop_dims("nv")

    attr_map = {
        "launch_time": dict(standard_name="time", long_name="time at which the radiosonde was launched", time_zone="UTC"),
        "launch_lon": dict(cell_methods="height: interpolated (interval: 10 m)"),
        "launch_lat": dict(cell_methods="height: interpolated (interval: 10 m)"),
        "lat":         dict(standard_name="latitude", long_name="latitude", axis="Y",
                            description="Latitudinal position during flight", units="degree_north", cell_methods="height: interpolated (interval: 10 m)"),
        "lon":         dict(standard_name="longitude", long_name="longitude", axis="X",
                            description="Longitudinal position during flight", units="degree_east", cell_methods="height: interpolated (interval: 10 m)"),
        "p":           dict(standard_name="air_pressure", long_name="air pressure",
                            description="Air Pressure during flight", units="Pa", cell_methods="height: interpolated (interval: 10 m)"),
        "ta":    dict(standard_name="air_temperature", long_name="air temperature", units="K"),
        "dp":    dict(standard_name="dew_point_temperature", long_name="dew point temperature", units="K", cell_methods="height: interpolated (interval: 10 m)"),
        "theta": dict(standard_name="potential_temperature", long_name="potential temperature", units="K", cell_methods="height: interpolated (interval: 10 m)"),
        "q":     dict(standard_name="specific_humidity", long_name="specific humidity", units="1", cell_methods="height: interpolated (interval: 10 m)"),
        "mr":    dict(standard_name="mixing_ratio", long_name="water vapor mixing ratio", units="1"),
        "rh":    dict(standard_name="relative_humidity", long_name="relative humidity", units="1"),
        "u":     dict(standard_name="eastward_wind", long_name="u-component of the wind", units="m s-1", cell_methods="height: interpolated (interval: 10 m)"),
        "v":     dict(standard_name="northward_wind", long_name="v-component of the wind", units="m s-1", cell_methods="height: interpolated (interval: 10 m)"),
        "wspd":  dict(standard_name="wind_speed", long_name="wind speed", units="m s-1", cell_methods="height: point (derived from interpolated u, v)"),
        "wdir":  dict(standard_name="wind_direction", long_name="wind from direction", units="degree", cell_methods="height: point (derived from interpolated u, v)"),
        "height": dict(standard_name="geopotential_height", long_name="geopotential height from PTU (barometric)", positive="up", units="m"),
        "dz": dict(standard_name="platform_vertical_velocity", long_name="ascent/descent rate of measuring device", description="ascent rate is positive/ decent rate is negative", units="m s-1", cell_methods="height: interpolated (interval: 10 m)"),
        "platform": dict(long_name="launching platform"),
        "iwv": dict(standard_name="atmosphere_mass_content_of_water_vapor", long_name="Integrated water vapor", units="kg m-2"),
    }

    ds = _apply_attrs(ds, attr_map, overwrite_keys=("cell_methods",))

    if "dz" in ds:
        ds["dz"].attrs.setdefault("comment",
            "Derived from vertical changes of geopotential height; describes platform motion, not air vertical velocity.")
    if "wspd" in ds:
        ds["wspd"].attrs.setdefault("cell_methods", "height: point (derived from averaged u, v)")
    if "ta" in ds:
        ds["ta"].attrs.setdefault("cell_methods", "height: point (derived from averaged theta)")

    candidates = {"p", "sonde_id", "alt"}
    to_demote = [v for v in candidates if v in ds.coords and v not in ds.dims]
    if to_demote:
        ds = ds.reset_coords(to_demote, drop=False)
        ds = _scrub_cf_coordinate_references(ds, set(to_demote))

    to_drop = [v for v in ("N_ptu", "m_ptu", "N_gps", "m_gps", "alt") if v in ds]
    if to_drop:
        ds = ds.drop_vars(to_drop)
    return ds

# %%
# ============================================================
# Prepare the dataset for IPFS
# ============================================================

DS = prep_for_ipfs(DS)
DS

# %%
# ============================================================
# Add Integrated Water Vapor (IWV)
# ============================================================

DS = add_IWV(DS)
DS

# %%
# ============================================================
# Demote Aux Coords for "p" and "alt" to stay data_vars
# ============================================================

DS = demote_aux_coords_strict(DS, names=("alt","p"))
DS

# %%
# ============================================================
# Drop Ancillary N_ptu, N_gps, m_ptu, m_gps
# ============================================================

DS = drop_count_ancillary(DS)
DS

# %%
DS = DS.sortby(DS.launch_time)
DS

# %%
DS.to_netcdf(DS_directory+f"for_IPFS/RS_ORCESTRA_level2_{DS_VERSION}_for_IPFS.nc")
