# %%
#!/usr/bin/env python3
"""
RAPSODI ORCESTRA Level-1 merger & publisher
-------------------------------------------

Part 1: Gather & merge level-1 files (pad to common `level`, concat along `sounding`)
Part 2: Amend dataset (rename, normalize IDs, add launch coords, attrs, demote coords)
"""

from pathlib import Path
import os
import re
import numpy as np
import xarray as xr
import pandas as pd
from tqdm import tqdm

# %%
DS_VERSION = "v4.0.7"

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(SCRIPT_DIR)

INPUT_DIRS = [
    "../level1/BCO",
    "../level1/INMG",
    "../level1/Meteor",
]

OUT_DIR_RAW = Path(
    "../level1/00_merged_datasets_for_IPFS"
)
OUT_DIR_RAW.mkdir(parents=True, exist_ok=True)
OUT_FILE_RAW = OUT_DIR_RAW / f"RS_ORCESTRA_level1_{DS_VERSION}_raw.nc"

OUT_DIR_FINAL = OUT_DIR_RAW / "for_IPFS"
OUT_DIR_FINAL.mkdir(parents=True, exist_ok=True)
OUT_FILE_FINAL = OUT_DIR_FINAL / f"RS_ORCESTRA_level1_{DS_VERSION}_for_IPFS.nc"

SORT_BY_LAUNCH_TIME = True 

# %%
# =============================================================================
# Part 1 — Gather & merge
# =============================================================================
def _find_nc_files(dirs) -> list[Path]:
    files = []
    for d in dirs:
        files.extend(sorted(Path(d).glob("**/*.nc")))
    return files


def _get_max_level(files: list[Path]) -> int:
    maxlvl = 0
    for f in tqdm(files, desc="Scanning for max 'level'"):
        with xr.open_dataset(f, decode_cf=True) as ds:
            if "level" not in ds.sizes:
                raise ValueError(f"'level' dimension missing in {f}")
            maxlvl = max(maxlvl, int(ds.sizes["level"]))
    return maxlvl


def _pad_to_level(ds: xr.Dataset, target_len: int) -> xr.Dataset:
    """Reindex along 'level' to a common integer coordinate [0..target_len-1]."""
    dtype = "int16" if target_len - 1 <= np.iinfo(np.int16).max else "int32"
    target_level = xr.DataArray(np.arange(target_len, dtype=dtype), dims=("level",), name="level")
    return ds.reindex(level=target_level, copy=False)


def run_part1_gather_and_merge(
    input_dirs=INPUT_DIRS,
    out_file=OUT_FILE_RAW,
    sort_by_launch_time=SORT_BY_LAUNCH_TIME,
) -> Path:
    files = _find_nc_files(input_dirs)
    if not files:
        raise SystemExit("No NetCDF files found in the provided directories.")

    max_level = _get_max_level(files)

    datasets = []
    for f in tqdm(files, desc="Loading & padding"):
        ds = xr.open_dataset(f, decode_cf=True)
        ds = _pad_to_level(ds, max_level)
        if "sounding" not in ds.sizes:
            ds.close()
            raise ValueError(f"'sounding' dimension missing in {f}")
        datasets.append(ds)

    merged = xr.concat(
        datasets,
        dim="sounding",
        join="outer",
        data_vars="all",
        coords="minimal",
        compat="override",
    )

    if sort_by_launch_time and "launch_time" in merged.coords:
        merged = merged.sortby("launch_time")

    merged = merged.assign_attrs({
        **{k: v for k, v in merged.attrs.items() if isinstance(v, (str, int, float))},
        "title": "RAPSODI Radiosonde Measurements during ORCESTRA (Level 1) (merged, padded to common vertical levels)",
        "dataset_version": DS_VERSION,
        "history": (merged.attrs.get("history", "") + f"\nMerged & padded to level={max_level}").strip(),
        "Conventions": "CF-1.10",
    })

    comp = dict(zlib=True, complevel=4)
    encoding = {v: comp for v in merged.data_vars if str(merged[v].dtype).startswith(("float", "int"))}

    merged.to_netcdf(out_file, encoding=encoding)

    for ds in datasets:
        ds.close()

    return Path(out_file)


# %%
# >>> Run this cell for Part 1
merged_path = run_part1_gather_and_merge()
merged_path








# %%
# =============================================================================
# Part 2 — Amend dataset (prep for IPFS)
# =============================================================================
def _apply_attrs(ds, attr_map, overwrite_keys=("cell_methods",)):
    for name, add in attr_map.items():
        if name in ds.variables or name in ds.coords:
            tgt = ds[name]
            for k, v in add.items():
                if (overwrite_keys is None) or (k in overwrite_keys) or (k not in tgt.attrs):
                    tgt.attrs[k] = v
    return ds

def normalize_sonde_id(ds: xr.Dataset) -> xr.Dataset:
    if "sonde_id" not in ds.coords or "launch_time" not in ds.coords:
        return ds

    old_attrs = dict(ds["sonde_id"].attrs)
    vals = ds["sonde_id"].values.astype(str)

    lt = pd.to_datetime(ds["launch_time"].values)
    ts12 = np.array([t.strftime("%Y%m%d%H%M") for t in lt], dtype=object)

    def _shorten(s: str, ts: str) -> str:
        mdir = re.search(r"(ascent|descent)", s, flags=re.IGNORECASE)
        if not mdir:
            return re.sub(r"_+", "_", s.strip())
        direction = mdir.group(1).lower()

        plat = re.split(r"__|_", s.strip(), maxsplit=1)[0]
        plat = re.sub(r"\s+", "_", plat.strip())

        out = f"{plat}_{direction}_{ts}"
        out = re.sub(r"_+", "_", out)
        return out

    out = []
    seen = {}
    for s, ts in zip(vals, ts12):
        new = _shorten(s, ts)
        if new in seen:
            seen[new] += 1
            new = f"{new}_{seen[new]}"
        else:
            seen[new] = 0
        out.append(new)

    maxlen = max(len(x) for x in out) if out else 1
    ds = ds.assign_coords(sonde_id=("sonde_id", np.array(out, dtype=f"U{maxlen}")))
    ds["sonde_id"].attrs.update(old_attrs)
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


def prep_level1_for_ipfs(ds: xr.Dataset) -> xr.Dataset:
    """
    Prepare a merged Level-1 dataset for publishing:
      - rename dim 'sounding'->'sonde_id'
      - normalize 'sonde_id' values
      - add 1-D launch_lat/launch_lon (per sonde_id) safely
      - preserve all (sonde_id, level) shapes
      - set global/variable attributes
      - demote all coords except: sonde_id, level, launch_time, lat, lon, flight_time
    """

    if 'platform' in ds:
        da = ds['platform'].astype('U4')
        da = xr.where(da == 'INM', 'INMG', da)
        ds['platform'] = da

    if "sounding" not in ds.dims or "level" not in ds.dims:
        return ds

    must_keep_2d = [n for n in ds.variables if ("level" in getattr(ds[n], "dims", ()))]
    orig_2d = {n: ds[n].copy(deep=True) for n in must_keep_2d}

    ds = ds.rename({"sounding": "sonde_id"})
    if "sonde_id" not in ds.coords:
        ds = ds.set_coords("sonde_id")
    ds["sonde_id"] = ds["sonde_id"].assign_attrs(
        dict(cf_role="profile_id", long_name="sonde identifier",
             description="Unique string describing the sounding's origin (PLATFORM_DIRECTION_TIME)"))
    
    ds = normalize_sonde_id(ds)

    if "launch_time" in ds.coords:
        ds = ds.sortby("launch_time")

    if {"lat", "lon"}.issubset(ds.coords):
        first_valid_lvl = ds["lat"].isnull().argmin("level")   
        la = ds["lat"].isel(level=first_valid_lvl).data        
        lo = ds["lon"].isel(level=first_valid_lvl).data        
        ds = ds.assign(
            launch_lat=(("sonde_id",), la),
            launch_lon=(("sonde_id",), lo),
        )
        ds["launch_lat"].attrs.update(dict(standard_name="latitude", long_name="launch latitude",
                                           units="degrees_north", axis="Y"))
        ds["launch_lon"].attrs.update(dict(standard_name="longitude", long_name="launch longitude",
                                           units="degrees_east", axis="X"))

    if "flight_time" in ds.coords:
        ds["flight_time"].attrs.update(dict(standard_name="time",
                                            long_name="time of recorded measurement",
                                            time_zone="UTC", cell_methods="level: point"))
    if "launch_time" in ds.coords:
        ds["launch_time"].attrs.update(dict(standard_name="time",
                                            long_name="time at which the radiosonde was launched",
                                            time_zone="UTC"))
    for name, meta in {
        "height": dict(standard_name="geopotential_height",
                       long_name="geopotential height from PTU (barometric)", units="m", positive="up", axis="Z"),
        "alt":    dict(standard_name="height_above_reference_ellipsoid",
                       long_name="altitude height above reference ellipsoid (WGS84)", units="m", positive="up", axis="Z"),
        "p":      dict(standard_name="air_pressure", long_name="air pressure", units="Pa", cell_methods="level: point"),
        "lat":    dict(standard_name="latitude", units="degrees_north", axis="Y", cell_methods="level: point"),
        "lon":    dict(standard_name="longitude", units="degrees_east", axis="X", cell_methods="level: point"),
    }.items():
        if name in ds:
            for k, v in meta.items():
                ds[name].attrs.setdefault(k, v)
    if "level" in ds.coords:
        ds["level"].attrs.setdefault("long_name", "sample index within profile")
        ds["level"].attrs.setdefault("units", "1")

    attr_map = {
        "ta":   dict(standard_name="air_temperature", long_name="air temperature", units="K", cell_methods="level: point"),
        "dp":   dict(standard_name="dew_point_temperature", long_name="dew point temperature", units="K", cell_methods="level: point"),
        "rh":   dict(standard_name="relative_humidity", long_name="relative humidity", units="1", cell_methods="level: point"),
        "mr":   dict(standard_name="mixing_ratio", long_name="water vapor mixing ratio", units="1", cell_methods="level: point"),
        "wspd": dict(standard_name="wind_speed", long_name="wind speed", units="m s-1", cell_methods="level: point"),
        "wdir": dict(standard_name="wind_direction", long_name="wind from direction", units="degree", cell_methods="level: point"),
        "dz":   dict(standard_name="platform_vertical_velocity",
                    long_name="ascent/descent rate of measuring device",
                    description="ascent rate is positive / descent rate is negative",
                    units="m s-1", cell_methods="level: point"),
        "platform": dict(long_name="launching platform"),
    }
    ds = _apply_attrs(ds, attr_map, overwrite_keys=("cell_methods",))

    if "dz" in ds:
        ds["dz"].attrs.setdefault("comment",
            "Derived from vertical changes of geopotential height; describes platform motion, not air vertical velocity.")

    for name, arr in orig_2d.items():
        if name in ds:
            expected = tuple("sonde_id" if d == "sounding" else d for d in arr.dims)
            if ds[name].dims != expected:
                arr_fixed = arr.rename({"sounding": "sonde_id"})
                ds[name] = xr.DataArray(arr_fixed.data, dims=expected)

    for key in ("Conventions", "dataset_version"):
        ds.attrs.pop(key, None)
        
    ds.attrs["platform"] = "INMG, RV Meteor, BCO"
    ds.attrs["title"] = (
        "RAPSODI Radiosonde Measurements during ORCESTRA (Level 1) "
        "(merged, padded to common vertical levels)"
    )
    ds.attrs["keywords"] = "ORCESTRA, RAPSODI, Radiosonde Profiles, Sounding, INMG, RV Meteor, BCO"
    ds.attrs["summary"] = (
    "Vertical atmospheric profile, retrieved from atmospheric sounding "
    "attached to a helium-filled balloon."
    )

    keep_coords = {"sonde_id", "level", "launch_time", "lat", "lon", "flight_time"}
    to_demote = sorted((set(ds.coords) - keep_coords) - set(ds.dims))
    if to_demote:
        ds = ds.reset_coords(to_demote, drop=False)
        ds = _scrub_cf_coordinate_references(ds, set(to_demote))


    return ds


def run_part2_amend_dataset(in_file=OUT_FILE_RAW, out_file=OUT_FILE_FINAL) -> Path:
    with xr.open_dataset(in_file, decode_cf=True) as src:
        ds = prep_level1_for_ipfs(src)

    comp = dict(zlib=True, complevel=4)
    encoding = {v: comp for v in ds.data_vars if str(ds[v].dtype).startswith(("float", "int"))}
    ds.to_netcdf(out_file, encoding=encoding)
    return ds, out_file


# %%
# >>> Run this cell for Part 2
final_ds, outfile_path = run_part2_amend_dataset()
final_ds

# %%
