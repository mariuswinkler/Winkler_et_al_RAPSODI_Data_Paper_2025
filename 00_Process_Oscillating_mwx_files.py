# %%
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import xarray as xr
import re
from pathlib import Path
import zipfile
import xml.etree.ElementTree as ET
from typing import Union, Optional, List, Dict

# ============================================================
# Paths (relative to this script)
# ============================================================
SCRIPT_DIR = Path(__file__).resolve().parent
BASE_DIR = (SCRIPT_DIR / "../level0/Meteor_Oscillating").resolve()  
OUT_NC   = (SCRIPT_DIR / "../level1/Meteor_Oscillating/RS_ORCESTRA_Meteor_Oscillating_level1_for_IPFS.nc").resolve()

XML_NAMES = {
    "ptu": "PtuResults.xml",
    "gps": "GpsResults.xml",
    "snd": "Soundings.xml",
}

def _zip_find_member(z: zipfile.ZipFile, wanted: str) -> Optional[str]:
    wl = wanted.lower()
    for name in z.namelist():
        if name.lower().endswith("/" + wl) or name.lower().endswith(wl):
            return name
    return None

def _load_xml_text(source: Union[str, Path], filename: str) -> str:
    source = Path(source)
    if source.is_dir():
        p = source / filename
        if not p.exists():
            cands = [pp for pp in source.rglob("*.xml") if pp.name.lower() == filename.lower()]
            if not cands:
                raise FileNotFoundError(f"{filename} not found under {source}")
            p = cands[0]
        return p.read_text(encoding="utf-8", errors="ignore")

    if source.suffix.lower() in (".mwx", ".zip"):
        with zipfile.ZipFile(source) as z:
            member = _zip_find_member(z, filename)
            if member is None:
                raise FileNotFoundError(f"{filename} not found in archive {source}")
            return z.read(member).decode("utf-8", errors="ignore")

    raise ValueError(f"Unsupported source: {source}")

def _parse_xml_to_df(xml_text: str) -> pd.DataFrame:
    root = ET.fromstring(xml_text)
    rows = [dict(row.attrib) for row in root]
    df = pd.DataFrame(rows)
    if "DataSrvTime" in df.columns:
        df["DataSrvTime"] = pd.to_datetime(df["DataSrvTime"], errors="coerce")
    for c in df.columns:
        if c != "DataSrvTime":
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

def _read_soundings_meta(xml_text: str) -> Dict:
    root = ET.fromstring(xml_text)
    row = root[0].attrib if len(root) else {}
    meta = {
        "IdPk": row.get("IdPk"),
        "begin_time": pd.to_datetime(row.get("BeginTime")) if row.get("BeginTime") else None,
        "launch_time_s": float(row.get("LaunchTime")) if row.get("LaunchTime") not in (None, "NULL") else None,
        "duration_s": float(row.get("Duration")) if row.get("Duration") not in (None, "NULL") else None,
        "software_version": row.get("SoftwareVersion"),
        "status": int(row.get("Status")) if row.get("Status") not in (None, "NULL") else None,
    }
    return meta

# ============================================================
# QC + merge + physics
# ============================================================
def qc_filter_ptu(df_ptu: pd.DataFrame) -> pd.DataFrame:
    out = df_ptu[df_ptu.get("Status", 0) == 0].copy()
    out = out[(out["Temperature"].between(150, 330)) & (out["Humidity"].between(0, 120))]
    out = out[(out["SensorPressure"].between(1, 1100))]
    out = out.sort_values("DataSrvTime").drop_duplicates("DataSrvTime", keep="last")
    return out

def qc_filter_gps(df_gps: pd.DataFrame) -> pd.DataFrame:
    out = df_gps.copy()
    req = (out["Wgs84Latitude"].between(-90, 90)) & \
          (out["Wgs84Longitude"].between(-180, 180)) & \
          (out["Wgs84Altitude"].between(-500, 50000))
    if "HDOP" in out and "VDOP" in out:
        req &= out["HDOP"].between(0, 10, inclusive="both") & out["VDOP"].between(0, 10, inclusive="both")
    out = out[req].sort_values("DataSrvTime").drop_duplicates("DataSrvTime", keep="last")
    if {"VelocityNorth","VelocityEast"}.issubset(out.columns):
        spd = np.sqrt(out["VelocityNorth"]**2 + out["VelocityEast"]**2)
        out = out[spd < 65.0]
    return out

def resample_and_merge(ptu: pd.DataFrame, gps: pd.DataFrame, freq="1S", tolerance="600ms") -> pd.DataFrame:
    t0 = max(ptu["DataSrvTime"].min(), gps["DataSrvTime"].min())
    t1 = min(ptu["DataSrvTime"].max(), gps["DataSrvTime"].max())
    idx = pd.date_range(t0, t1, freq=freq)
    ptu_i = pd.merge_asof(pd.DataFrame({"DataSrvTime": idx}),
                          ptu.sort_values("DataSrvTime"),
                          on="DataSrvTime",
                          tolerance=pd.Timedelta(tolerance),
                          direction="nearest")
    merged = pd.merge_asof(ptu_i.sort_values("DataSrvTime"),
                           gps.sort_values("DataSrvTime"),
                           on="DataSrvTime",
                           tolerance=pd.Timedelta(tolerance),
                           direction="nearest",
                           suffixes=("_ptu","_gps"))
    keep = ["SensorPressure","Temperature","Humidity", "Wgs84Latitude","Wgs84Longitude","Wgs84Altitude"]
    merged = merged.dropna(subset=keep)
    return merged

def esat_pa(T_K: np.ndarray) -> np.ndarray:
    try:
        from moist_thermodynamics import functions as mtfunc
        for name in ("esat_water","es_w","p_ws","saturation_vapor_pressure_water"):
            if hasattr(mtfunc, name):
                return getattr(mtfunc, name)(T_K)
    except Exception:
        pass
    T_C = T_K - 273.15
    return 611.2 * np.exp(17.67 * T_C / (T_C + 243.5))

def dewpoint_vaisala_MW41(T_K: np.ndarray, RH_frac: np.ndarray) -> np.ndarray:
    """
    Dew point from T [K] and RH [0..1] using the formula used by Vaisala MW41.
    (Adapted to plain numpy; unit-aware bits removed.)
    """
    rh_pct = np.clip(RH_frac * 100.0, 1e-3, 100.0 - 1e-6)
    kelvin = 15.0 * np.log(100.0 / rh_pct) - 2.0 * (T_K - 273.15) + 2711.5
    t_dew = T_K * 2.0 * kelvin / (T_K * np.log(100.0 / rh_pct) + 2.0 * kelvin)
    return t_dew

def esat_wagner_pruss(T_K):
    """
    Saturation vapor pressure over liquid water [Pa] via Wagner & Pruss (2002).
    Tries mtsvp.liq_wagner_pruss first; falls back to a local implementation.
    """
    T = np.asarray(T_K, dtype=float)

    try:
        from moist_thermodynamics import mtsvp as _mtsvp
        e = _mtsvp.liq_wagner_pruss(T)
        return np.asarray(getattr(e, "magnitude", e))
    except Exception:
        pass

    try:
        import mtsvp as _mtsvp
        e = _mtsvp.liq_wagner_pruss(T)
        return np.asarray(getattr(e, "magnitude", e))
    except Exception:
        pass

    Tc = 647.096        # K
    pc = 22_064_000.0   # Pa
    tau = 1.0 - (T / Tc)
    coeffs = [
        (-7.85951783, 1.0),
        ( 1.84408259, 1.5),
        (-11.7866497, 3.0),
        (22.6807411, 3.5),
        (-15.9618719, 4.0),
        ( 1.80122502, 7.5),
    ]
    ln_p = (Tc / T) * sum(n * tau**t for n, t in coeffs)
    return pc * np.exp(ln_p)


def mixing_ratio_from_p_T_RH(p_Pa: np.ndarray, T_K: np.ndarray, RH_frac: np.ndarray) -> np.ndarray:
    """Mixing ratio using Wagner–Pruss saturation pressure."""
    e_s = esat_wagner_pruss(T_K)                   # Pa (saturation vapor pressure)
    e   = np.clip(RH_frac, 0.0, 1.0) * e_s         # Pa (actual vapor pressure)
    mr  = 0.62198 * e / np.maximum(p_Pa - e, 1.0)  # kg/kg
    return mr


def make_alt_bounds(alt: np.ndarray) -> np.ndarray:
    n = alt.size
    b = np.empty((n, 2), dtype=float)
    if n == 1:
        b[0, :] = [alt[0]-0.5, alt[0]+0.5]
        return b
    mids = (alt[:-1] + alt[1:]) / 2.0
    b[1:-1, 0] = mids[:-1]
    b[1:-1, 1] = mids[1:]
    # edges: extrapolate
    b[0, 0] = alt[0] - (mids[0] - alt[0])
    b[0, 1] = mids[0]
    b[-1, 0] = mids[-1]
    b[-1, 1] = alt[-1] + (alt[-1] - mids[-1])
    return b

# ============================================================
# Single-sounding builder (from folder or .mwx)
# ============================================================
def build_dataset_from_source(source: Union[str, Path]) -> xr.Dataset:
    ptu_text = _load_xml_text(source, XML_NAMES["ptu"])
    gps_text = _load_xml_text(source, XML_NAMES["gps"])
    snd_text = _load_xml_text(source, XML_NAMES["snd"])

    df_ptu = _parse_xml_to_df(ptu_text)
    df_gps = _parse_xml_to_df(gps_text)
    meta   = _read_soundings_meta(snd_text)

    ptu = qc_filter_ptu(df_ptu)
    gps = qc_filter_gps(df_gps)
    merged = resample_and_merge(ptu, gps, freq="1S", tolerance="600ms")

    time = merged["DataSrvTime"].to_numpy()
    N = len(time)
    p_Pa = merged["SensorPressure"].to_numpy() * 100.0
    T_K  = merged["Temperature"].to_numpy()
    RHf  = merged["Humidity"].to_numpy() / 100.0
    lat  = merged["Wgs84Latitude"].to_numpy().astype(float)
    lon  = merged["Wgs84Longitude"].to_numpy().astype(float)
    alt_gps = merged["Wgs84Altitude"].to_numpy().astype(float)  

    if "VelocityUp" in merged:
        dz = merged["VelocityUp"].to_numpy().astype(float)
    else:
        dz = np.gradient(alt_gps, initial=0.0)

    colmap = {c.lower(): c for c in merged.columns}
    if "dewpointtemperature" in colmap:
        dp = merged[colmap["dewpointtemperature"]].to_numpy().astype(float)
    elif "dewpoint" in colmap:
        dp = merged[colmap["dewpoint"]].to_numpy().astype(float)
    else:
        dp = dewpoint_vaisala_MW41(T_K, RHf)

    mr = mixing_ratio_from_p_T_RH(p_Pa, T_K, RHf)

    if {"VelocityNorth","VelocityEast"}.issubset(merged.columns):
        vN = merged["VelocityNorth"].to_numpy().astype(float)
        vE = merged["VelocityEast"].to_numpy().astype(float)
        wspd = np.sqrt(vN**2 + vE**2)
        track_dir = (np.degrees(np.arctan2(vE, vN)) + 360.0) % 360.0
        wdir = (track_dir + 180.0) % 360.0
    else:
        wspd = np.full(N, np.nan)
        wdir = np.full(N, np.nan)

    k0 = int(np.nanargmin(alt_gps)) if N > 0 else 0
    launch_lat = float(lat[k0])
    launch_lon = float(lon[k0])

    rt = meta.get("begin_time") or (pd.Timestamp(time[0]) if N else pd.NaT)
    id_str = f"RV Meteor__{launch_lat:.5f}__{launch_lon:.5f}__{rt.strftime('%Y%m%d%H%M') if pd.notnull(rt) else 'NA'}"

    ds = xr.Dataset(
        data_vars=dict(
            ta   = (("sounding","sample"), T_K.reshape(1, N)),
            rh   = (("sounding","sample"), RHf.reshape(1, N)),
            dp   = (("sounding","sample"), dp.reshape(1, N)),
            mr   = (("sounding","sample"), mr.reshape(1, N)),
            dz   = (("sounding","sample"), dz.reshape(1, N)),
            wspd = (("sounding","sample"), wspd.reshape(1, N)),
            wdir = (("sounding","sample"), wdir.reshape(1, N)),
            launch_lat = (("sounding",), np.array([launch_lat])),
            launch_lon = (("sounding",), np.array([launch_lon])),
        ),
        coords=dict(
            sounding     = ("sounding", [id_str]),
            sample       = ("sample", np.arange(N, dtype=np.int32)),
            release_time = ("sounding", np.array([np.datetime64(rt)])),
            flight_time  = (("sounding","sample"), time.reshape(1, N)),
            p   = (("sounding","sample"), p_Pa.reshape(1, N)),
            lat = (("sounding","sample"), lat.reshape(1, N)),
            lon = (("sounding","sample"), lon.reshape(1, N)),
            alt = (("sounding","sample"), alt_gps.reshape(1, N)),
        ),
        attrs={}
    )

    ds["alt"].attrs.update(dict(standard_name="altitude", long_name="ellipsoidal vertical position (WGS84, from GPS)", axis="Z", positive="up", description="Altitude measured by radisonde", units="m"))
    ds["sample"].attrs.update(dict(long_name="Individual sounding level", standard_name="sample", description=""))
    ds["sounding"].attrs.update(dict(cf_role="profile_id", standard_name="sounding", long_name="sounding identifier", description="Unique string describing the soundings origin (PLATFORM_SND-DIRECTION_LAT_LON_TIME)"))        
    ds["release_time"].attrs.update(dict(standard_name="release time", long_name="time at which the radiosonde was released", axis="T"))
    ds["flight_time"].attrs.update(dict(standard_name="time", long_name="time at sample level", axis="T"))
    ds["p"].attrs.update(dict(standard_name="air_pressure", long_name="air pressure", description="Air Pressure during flight", units="Pa"))
    ds["lat"].attrs.update(dict(standard_name="latitude", long_name="latitude", axis="Y", description="Latitudinal position during flight", units="degree_north"))
    ds["lon"].attrs.update(dict(standard_name="longitude", long_name="longitude", axis="X", description="Longitudinal position during flight", units="degree_east"))
    ds["dz"].attrs.update(dict(standard_name="ascent_descent_rate", long_name="ascent/descent rate of measuring device", description="ascent rate is positive/ decent rate is negative", units="m s-1"))
    ds["ta"].attrs.update(dict(standard_name="air_temperature", long_name="air temperature", units="K"))
    ds["rh"].attrs.update(dict(standard_name="relative_humidity", long_name="relative humidity", units="1"))
    ds["dp"].attrs.update(dict(standard_name="dew_point_temperature", long_name="dew point temperature", units="K"))
    ds["mr"].attrs.update(dict(standard_name="mixing_ratio", long_name="water vapor mixing ratio", units="1"))
    ds["wspd"].attrs.update(dict(standard_name="wind_speed", long_name="wind speed", units="m s-1"))
    ds["wdir"].attrs.update(dict(standard_name="wind_direction", long_name="wind from direction", units="degree"))
    ds["launch_lat"].attrs.update(dict(standard_name="launch_latitude", long_name="launch latitude", axis="Y", units="degree_north"))
    ds["launch_lon"].attrs.update(dict(standard_name="launch_longitude", long_name="launch longitude", axis="X", units="degree_east"))

    return ds


# ============================================================
# Padding to max sample length, concatenation, and globals
# ============================================================
def pad_to_sample_len(ds: xr.Dataset, target_len: int) -> xr.Dataset:
    """
    Pad a single-sounding dataset to target_len along 'sample' by reindexing.
    This avoids coordinate-alignment conflicts and fills with NaN/NaT.
    """
    cur = ds.sizes["sample"]
    if cur == target_len:
        return ds

    new_sample = np.arange(target_len, dtype=np.int32)

    if "sample" not in ds.coords:
        ds = ds.assign_coords(sample=np.arange(cur, dtype=np.int32))
    ds_padded = ds.reindex(sample=new_sample)

    return ds_padded

def _normalize_sonde_id_ids(ds: xr.Dataset) -> xr.Dataset:
    if "sonde_id" not in ds.coords:
        return ds

    old_attrs = dict(ds["sonde_id"].attrs)
    vals = ds["sonde_id"].values.astype(str)

    def _shorten(s: str) -> str:
        s = s.strip()
        parts = s.split("__")
        plat_raw = parts[0] if parts else s

        ts = None
        for p in reversed(parts):
            m = re.search(r"(\d{12})", p)
            if m:
                ts = m.group(1)
                break

        plat = re.sub(r"\s+", "_", plat_raw.strip())

        out = f"{plat}_{ts}" if ts else plat
        return re.sub(r"_+", "_", out)  

    seen, out = {}, []
    for s in vals:
        new = _shorten(s)
        if new in seen:
            seen[new] += 1
            new = f"{new}_{seen[new]}"
        else:
            seen[new] = 0
        out.append(new)

    arr = np.array(out, dtype=f"U{max(1, max(map(len, out)))}")

    sonde_dim = ds["sonde_id"].dims[0] if ds["sonde_id"].ndim == 1 else None
    if sonde_dim is not None:
        ds = ds.assign_coords(sonde_id=(sonde_dim, arr))
    else:
        ds["sonde_id"].data = arr

    ds["sonde_id"].attrs.update({k: v for k, v in old_attrs.items() if k != "standard_name"})
    ds["sonde_id"].attrs["description"] = "Unique string describing the sounding's origin (PLATFORM_TIME)"

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

def finalize_level1_before_save(ds: xr.Dataset) -> xr.Dataset:
    if "sounding" in ds.dims:
        ds = ds.rename({"sounding": "sonde_id"})
    if "sonde_id" not in ds.coords and "sonde_id" in ds.dims:
        ds = ds.set_coords("sonde_id")

    ds = _normalize_sonde_id_ids(ds)

    if "release_time" in ds:
        ds = ds.set_coords("release_time")

    if "release_time" in ds.coords:
        ds = ds.sortby("release_time")

    vertical_idx = "sample" if "sample" in ds.dims else ("level" if "level" in ds.dims else None)

    keep_coords = {"sonde_id", "lat", "lon", "flight_time", "release_time", "sample"}
    if vertical_idx:
        keep_coords.add(vertical_idx)

    to_demote = sorted((set(map(str, ds.coords)) - keep_coords) - set(ds.dims))
    if to_demote:
        ds = ds.reset_coords(to_demote, drop=False)
        ds = _scrub_cf_coordinate_references(ds, set(to_demote))

    return ds


def main():
    mwx_files = sorted(BASE_DIR.glob("*.mwx"))
    if not mwx_files:
        raise FileNotFoundError(f"No .mwx files found under {BASE_DIR}")

    dsets: List[xr.Dataset] = []
    sample_lens = []
    for fp in mwx_files:
        ds = build_dataset_from_source(fp)
        dsets.append(ds)
        sample_lens.append(ds.sizes["sample"])

    target_len = int(np.max(sample_lens))
    dsets = [pad_to_sample_len(ds, target_len) for ds in dsets]

    ds_all = xr.concat(dsets, dim="sounding")

    ds_all = ds_all.assign_attrs(dict(
        title="RAPSODI Oscillating Radiosonde Measurements during ORCESTRA (Level 1)",
        summary=("Radiosondes launched from R/V Meteor during intense rain exhibited repeated "
             "ascent–descent cycles near the freezing level (~5000 m) before ultimately reaching "
             "burst altitude. One case began above the freezing level. A plausible explanation is "
             "temporary icing on a moistened balloon that melts during descent, though a strong "
             "stable layer could also contribute. Similar oscillatory behavior has been reported previously."),
        creator_name="Marius Winkler, Karl-Hermann Wieners, Marius Rixen",
        creator_email="marius.winkler@mpimet.mpg.de, karl-hermann.wieners@mpimet.mpg.de, marius.rixen@mpimet.mpg.de",
        project="ORCESTRA; BOW-TIE; PICCOLO",
        platform="RV Meteor",
        source="Radiosondes",
        history="Processed from Vaisala .mwx files using 00_Process_Oscillating_mwx_files.py; aligned with functions from pysonde",
        license="CC-BY-4.0",
        references="https://github.com/mariuswinkler/Winkler_et_al_RAPSODI_Data_paper_2025.git",
        keywords="ORCESTRA, RAPSODI, Radiosonde Profiles, Sounding, RV Meteor",
        featureType="profile",
    ))

    ds_all = finalize_level1_before_save(ds_all)

    OUT_NC.parent.mkdir(parents=True, exist_ok=True)
    ds_all.to_netcdf(OUT_NC)
    print(f"Wrote {OUT_NC} with dims {dict(ds_all.sizes)}")

if __name__ == "__main__":
    main()
# %%
ds = xr.open_dataset(OUT_NC)
ds